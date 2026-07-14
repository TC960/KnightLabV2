"""How much of the 'floor' is just full text sitting OUTSIDE PMC/EPMC?

Phase A: query Unpaywall by DOI for every floor row; count OA copies at a URL we never tried.
Phase B: fetch those OA copies (PDF via pdftotext, HTML/XML via tag-strip), run through the
         dictionary, count how many yield an accession.

Run:  python3 -m accession.unpaywall_pass
"""
import json, os, re, hashlib, subprocess, tempfile, time, urllib.request, urllib.error, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config, cache, sources, dictionary

UA = {"User-Agent": config.USER_AGENT}
PMC_HOST = re.compile(r"(ncbi\.nlm\.nih\.gov/pmc|europepmc\.org|/PMC\d)", re.I)


def _fillable(acc):
    """A cell holds no real, human-entered code — safe to fill/re-triage."""
    v = (acc or "").strip().upper()
    return v in ("", "N/A", "NA", "ACCESSION_NOT_FOUND")


def floor_rows():
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    out = []
    for i, e in ext.items():
        # EVERY code-less row with a DOI, regardless of blank-target vs prod-N/A. The old
        # target-only boundary was an artifact of which pipeline filled the sheet first; two rows
        # in the same state should get the same triage. Guard on acc_existing so we never touch a
        # row that already holds a real code (never-overwrite).
        if e["accession_col"] == "N/A" and e["flag"] != "HUMAN_CAN_GET" \
                and recs[i].get("doi") and _fillable(recs[i].get("acc_existing")):
            out.append((i, recs[i]["doi"]))
    return out


def pick_oa_url(data):
    """Return (url, host_type) for an OA full-text copy NOT on PMC/EPMC, else (None, None)."""
    if not data or not data.get("is_oa"):
        return None, None
    locs = []
    if data.get("best_oa_location"):
        locs.append(data["best_oa_location"])
    locs += data.get("oa_locations", []) or []
    for loc in locs:
        for key in ("url_for_pdf", "url"):
            u = loc.get(key)
            if u and not PMC_HOST.search(u):
                return u, loc.get("host_type")
    return None, None


def oa_fetch(url):
    """Binary fetch with its own disk cache. Returns (status, bytes, content_type)."""
    os.makedirs(config.OA_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    binp = os.path.join(config.OA_CACHE_DIR, key + ".bin")
    metp = os.path.join(config.OA_CACHE_DIR, key + ".meta")
    if os.path.exists(metp):
        meta = json.load(open(metp))
        data = open(binp, "rb").read() if os.path.exists(binp) else b""
        return meta["status"], data, meta.get("ctype", "")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
            status = r.status
            ctype = r.headers.get("Content-Type", "")
            data = r.read(config.OA_MAX_BYTES)
    except urllib.error.HTTPError as e:
        status, ctype, data = e.code, "", b""
    except Exception:
        status, ctype, data = -1, "", b""
    json.dump({"status": status, "ctype": ctype, "url": url}, open(metp, "w"))
    if data:
        open(binp, "wb").write(data)
    return status, data, ctype


def text_from_oa(data, ctype):
    if not data:
        return ""
    if data[:4] == b"%PDF":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data); p = f.name
        try:
            out = subprocess.run(["pdftotext", "-q", p, "-"], capture_output=True, timeout=60).stdout
        except Exception:
            out = b""
        finally:
            os.unlink(p)
        return re.sub(r"\s+", " ", out.decode("utf-8", "replace"))
    head = data[:2000].lower()
    if b"html" in head or b"<article" in head or b"<?xml" in head or "html" in ctype.lower() or "xml" in ctype.lower():
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", data.decode("utf-8", "replace")))
    return ""


OUT = os.path.join(config.HERE, "unpaywall.jsonl")


def main():
    rows = floor_rows()
    print("floor rows with a DOI: %d" % len(rows), flush=True)

    # per-row result for every floor row queried (persisted for writeback)
    res = {}  # row -> dict

    # ---- Phase A: Unpaywall query (concurrent; token bucket caps the real rate) ----
    A = dict(found=0, is_oa=0, candidate=0)
    hosts = collections.Counter()
    candidates = []  # (row, doi, url, host)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(sources.unpaywall, doi): (i, doi) for i, doi in rows}
        for k, fut in enumerate(as_completed(futs), 1):
            i, doi = futs[fut]
            data = fut.result()
            url, host = pick_oa_url(data)
            res[i] = {"row_index": i, "doi": doi, "found": bool(data),
                      "is_oa": bool(data.get("is_oa")), "oa_status": data.get("oa_status"),
                      "oa_url": url, "host": host, "fetched": False, "codes": []}
            if data:
                A["found"] += 1
            if data.get("is_oa"):
                A["is_oa"] += 1
            if url:
                A["candidate"] += 1
                hosts[host or "?"] += 1
                candidates.append((i, doi, url, host))
            if k % 400 == 0:
                print("  [A %d/%d] found=%d is_oa=%d candidate=%d (%.0fs)"
                      % (k, len(rows), A["found"], A["is_oa"], A["candidate"], time.time() - t0), flush=True)
    print("\n=== PHASE A: Unpaywall ===")
    print("  Unpaywall record found : %d/%d" % (A["found"], len(rows)))
    print("  is_oa = true           : %d" % A["is_oa"])
    print("  OA full-text URL we NEVER tried (not PMC/EPMC): %d" % A["candidate"])
    print("  by host_type:", dict(hosts.most_common()))

    # ---- Phase B: fetch OA copies + extract (WITH provenance for the recovered codes) ----
    print("\n=== PHASE B: fetch %d OA copies + extract ===" % len(candidates), flush=True)
    B = dict(fetched=0, parsed=0, yielded=0)
    hits = []
    t1 = time.time()

    def work(item):
        i, doi, url, host = item
        st, data, ctype = oa_fetch(url)
        if st != 200 or not data:
            return (i, "fetch_fail", [])
        txt = text_from_oa(data, ctype)
        if not txt:
            return (i, "unparseable", [])
        seen, codes = set(), []
        for c in dictionary.extract_codes(txt, with_provenance=True):
            if dictionary.in_accession_column(c["repo"]) and c["code"] not in seen:
                seen.add(c["code"]); codes.append(c)
        return (i, "ok", codes)

    with ThreadPoolExecutor(max_workers=config.OA_WORKERS) as ex:
        futs = [ex.submit(work, it) for it in candidates]
        for k, fut in enumerate(as_completed(futs), 1):
            i, stt, codes = fut.result()
            if stt in ("ok", "unparseable"):
                B["fetched"] += 1; res[i]["fetched"] = True
            if stt == "ok":
                B["parsed"] += 1
            if codes:
                res[i]["codes"] = codes
                B["yielded"] += 1
                if len(hits) < 20:
                    hits.append((i, res[i]["host"], [c["code"] for c in codes][:5], res[i]["doi"]))
            if k % 100 == 0:
                print("  [B %d/%d] fetched=%d parsed=%d yielded=%d (%.0fs)"
                      % (k, len(candidates), B["fetched"], B["parsed"], B["yielded"], time.time() - t1), flush=True)

    with open(OUT, "w") as f:
        for i in sorted(res):
            f.write(json.dumps(res[i]) + "\n")

    print("\n=== RESULTS ===")
    print("  OA candidates                 : %d" % len(candidates))
    print("  fetched OK                     : %d" % B["fetched"])
    print("  parsed to text                 : %d" % B["parsed"])
    print("  YIELDED >=1 accession          : %d" % B["yielded"])
    print("  wrote %s" % OUT)
    print("\n  sample hits (row, host, codes, doi):")
    for h in hits:
        print("   ", h)


if __name__ == "__main__":
    main()
