"""Supplement-file accession recovery for SUPPLEMENT-flagged rows.

For each row flagged SUPPLEMENT with a PMCID, pull EPMC's supplementaryFiles ZIP,
extract text from every member file (pdf/xlsx/docx/csv/txt/xml), and run the
dictionary over it. Writes accession/supplement.jsonl (one JSON array).

Only ~15/166 rows have an OA supplement ZIP on EPMC; the rest are 'not open access'
and are recorded with status='no-oa' so we know they were tried, not skipped.

Run:  python3 -m accession.supplement_pass
"""
import csv, io, json, os, re, subprocess, tempfile, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config, dictionary

OUT = os.path.join(config.HERE, "supplement.jsonl")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/%s/supplementaryFiles"
UA = {"User-Agent": config.USER_AGENT}


def _text_from_member(name, blob):
    """Best-effort text extraction from one supplement file (by extension/magic)."""
    lo = name.lower()
    if blob[:4] == b"%PDF" or lo.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(blob); p = f.name
        try:
            out = subprocess.run(["pdftotext", "-q", p, "-"], capture_output=True, timeout=60).stdout
        except Exception:
            out = b""
        finally:
            os.unlink(p)
        return out.decode("utf-8", "replace")
    # OpenXML (xlsx/docx/pptx) and ODF are ZIPs of XML — read the inner XML, strip tags.
    if blob[:2] == b"PK":
        try:
            z = zipfile.ZipFile(io.BytesIO(blob))
            parts = []
            for n in z.namelist():
                if n.endswith(".xml"):
                    parts.append(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8", "replace")))
            return " ".join(parts)
        except Exception:
            return ""
    # csv/tsv/txt/xml/html and anything else: decode as text.
    return blob.decode("utf-8", "replace")


def _fetch_zip(pmc):
    try:
        req = urllib.request.Request(EPMC % pmc, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(config.OA_MAX_BYTES if hasattr(config, "OA_MAX_BYTES") else 50_000_000)
    except Exception as e:
        return None, "err:%s" % type(e).__name__
    if data[:2] != b"PK":
        return None, "no-oa"
    return data, "zip"


def work(item):
    i, pmc, doi = item
    data, status = _fetch_zip(pmc)
    if status != "zip":
        return {"row_index": i, "pmc": pmc, "doi": doi, "status": status,
                "files": [], "codes": []}
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except Exception:
        return {"row_index": i, "pmc": pmc, "doi": doi, "status": "bad-zip",
                "files": [], "codes": []}
    seen, codes, files = set(), [], []
    for name in z.namelist():
        if name.endswith("/"):
            continue
        files.append(name)
        try:
            blob = z.read(name)
        except Exception:
            continue
        txt = _text_from_member(name, blob)
        if not txt:
            continue
        for c in dictionary.extract_codes(txt, with_provenance=True):
            if dictionary.in_accession_column(c["repo"]) and c["code"] not in seen:
                seen.add(c["code"])
                codes.append({**c, "file": name})
    return {"row_index": i, "pmc": pmc, "doi": doi, "status": "ok",
            "files": files, "codes": codes}


def main():
    rows = list(csv.DictReader(open(config.CSV_PATH, newline="", encoding="utf-8")))
    # match on the WRITTEN output flag so we target the same 189 the audit found
    out_path = os.path.join(config.REPO, "articles.out.csv")
    out_rows = list(csv.DictReader(open(out_path, newline="", encoding="utf-8")))
    targets = [(i, r["pmc_id"].strip(), (r.get("doi") or "").strip())
               for i, r in enumerate(out_rows)
               if (r.get("flag") or "").strip() == "SUPPLEMENT" and (r.get("pmc_id") or "").strip()]
    print("SUPPLEMENT rows with a PMCID: %d" % len(targets), flush=True)

    results = []
    n_ok = n_noa = n_err = n_yield = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, t) for t in targets]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                n_ok += 1
                if r["codes"]:
                    n_yield += 1
            elif r["status"] == "no-oa":
                n_noa += 1
            else:
                n_err += 1
            if k % 25 == 0:
                print("  [%d/%d] ok=%d no-oa=%d err=%d yielded=%d"
                      % (k, len(targets), n_ok, n_noa, n_err, n_yield), flush=True)

    results.sort(key=lambda r: r["row_index"])
    json.dump(results, open(OUT, "w"), indent=1)

    print("\n=== SUPPLEMENT RESULTS ===")
    print("  supplement ZIP fetched & parsed : %d" % n_ok)
    print("  not open access (no EPMC supp)  : %d" % n_noa)
    print("  error                           : %d" % n_err)
    print("  YIELDED >=1 accession           : %d" % n_yield)
    print("  wrote %s" % OUT)
    print("\n  hits (row, pmc, codes):")
    for r in results:
        if r["codes"]:
            print("   ", r["row_index"], r["pmc"], [c["code"] for c in r["codes"]][:8])


if __name__ == "__main__":
    main()
