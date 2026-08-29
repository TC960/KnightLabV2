"""Deterministic certainty, strict rule:  a no-code row is 'verified-empty' ONLY IF we have on
record a full text of >= 40,000 chars that we read and found no accession. Everything else that
lacks a code is 'needs-review' (flag for a human).

read-length per row = max over every channel we actually pulled text from:
  * EPMC / NCBI full-text XML   (replayed from cache)
  * tier-1 alternate OA copy    (recomputed from oa_cache: pdftotext / tag-strip)
  * tier-2 headful browser      (nchars stored per row)

Rewrites the `certainty` column with: coded | verified-empty | needs-review.

Run:  python3 -m accession.reclassify_40k
"""
import csv, json, os
from . import config
from .extract import _fulltext
from .unpaywall_pass import oa_fetch, text_from_oa

OUT = os.path.join(config.REPO, "articles.out.csv")
THRESHOLD = 40000


def _coded(r):
    return (r.get("Accession Code") or "").strip().upper() not in ("", "N/A", "NA", "ACCESSION_NOT_FOUND")


def main():
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}

    # every OA copy we ever fetched (main Unpaywall pass + 403 tier-1 alternate), keyed by row.
    # measure the cached text so a genuinely-read full OA paper counts toward the 40k rule.
    oa_url = {}
    up_path = os.path.join(config.HERE, "unpaywall.jsonl")
    if os.path.exists(up_path):
        for l in open(up_path):
            d = json.loads(l)
            if d.get("oa_url"):
                oa_url[d["row_index"]] = d["oa_url"]
    for r in json.load(open(os.path.join(config.HERE, "pipe403_tier1.jsonl"))):
        if r["status"] == "read-no-code" and r.get("url"):
            oa_url.setdefault(r["row"], r["url"])
    t1_url = oa_url

    # tier-2 browser + headful CDP browser nchars (a >=40k read there counts as verified-empty)
    import glob
    t2_len = {}
    for p2 in ([os.path.join(config.HERE, "pipe403_tier2.jsonl")]
               + glob.glob(os.path.join(config.HERE, "browser_recover*.jsonl"))):
        if os.path.exists(p2):
            for line in open(p2):
                d = json.loads(line)
                t2_len[d["row"]] = max(t2_len.get(d["row"], 0), d.get("nchars", 0))

    # rows the TOLERANT re-audit read in full (>=40k) — swept for ws-split + lookalike misses, so
    # a clean read here is a genuinely-audited empty (earns verified-empty like EPMC/NCBI does).
    tolerant_audited = set()
    for fp in (glob.glob(os.path.join(config.HERE, "browser_reaudit*.jsonl"))
               + glob.glob(os.path.join(config.HERE, "browser_nr*.jsonl"))):
        for line in open(fp):
            d = json.loads(line)
            if d.get("nchars", 0) >= THRESHOLD:
                tolerant_audited.add(d["row"])

    rows = list(csv.DictReader(open(OUT, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    if "certainty" not in fields:
        fields.append("certainty")

    n = {"coded": 0, "verified-empty": 0, "needs-review": 0}
    reaudit = []          # rows we READ >=40k but only via an UN-audited channel (OA/browser)
    done = 0
    for i, r in enumerate(rows):
        if _coded(r):
            r["certainty"] = "coded"; n["coded"] += 1
            continue
        # AUDITED read = EPMC/NCBI full text (the only text that went through the independent
        # whitespace + lookalike sweeps). ONLY this earns 'verified-empty'.
        t, _, _ = _fulltext(recs.get(i, {}))
        audited_len = len(t) if t else 0
        # UN-audited reads (OA copy / browser) — >=40k here means "read but not independently swept"
        other_len = 0
        if i in t1_url:
            st, blob, ctype = oa_fetch(t1_url[i])
            if st == 200 and blob:
                other_len = max(other_len, len(text_from_oa(blob, ctype)))
        if i in t2_len:
            other_len = max(other_len, t2_len[i])

        if audited_len >= THRESHOLD or i in tolerant_audited:
            r["certainty"] = "verified-empty"; n["verified-empty"] += 1
        else:
            r["certainty"] = "needs-review"; n["needs-review"] += 1
            if other_len >= THRESHOLD:
                reaudit.append(i)     # read a full paper, but with the buggy extractor -> re-audit
        done += 1
        if done % 1000 == 0:
            print("  ...%d non-coded rows classified" % done, flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    json.dump(reaudit, open(os.path.join(config.HERE, "reaudit_rows.json"), "w"))
    print("  -> %d rows need re-audit (read >=40k via OA/browser, unswept)" % len(reaudit))

    print("\nrewrote `certainty` (strict >=%dk full-text rule) in %s" % (THRESHOLD // 1000, OUT))
    for k in ("coded", "verified-empty", "needs-review"):
        print("  %-15s %d" % (k, n[k]))
    print("  total %d" % sum(n.values()))


if __name__ == "__main__":
    main()
