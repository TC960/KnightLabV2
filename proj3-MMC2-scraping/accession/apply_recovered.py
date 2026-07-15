"""Merge this session's recovered accession codes into articles.out.csv.

Sources (all side-files produced this session; writeback.py does not know them):
  * pipe403_tier1.jsonl  status==coded   -> repository-copy codes for 403-blocked rows
  * audit_whitespace.jsonl               -> codes recovered from whitespace-split accessions
  * pmc_restricted_unpaywall.jsonl       -> codes from restricted-PMC rows via Unpaywall

Idempotent: only fills rows that are currently code-less (never overwrites a real code).
Rows in DOUBTFUL are still written but tagged [review] in flag_detail.

Run:  python3 -m accession.apply_recovered
"""
import csv, json, os, re
from . import config

OUT = os.path.join(config.REPO, "articles.out.csv")
# repository/landing-page DOIs not stated as data accessions in the paper body — verify by hand.
DOUBTFUL = {12106, 12252, 9601}   # 12106/12252: Zenodo dup pair; 9601: figshare DOI not relocated


def _fillable(acc):
    return (acc or "").strip().upper() in ("", "N/A", "NA", "ACCESSION_NOT_FOUND")


def gather():
    rec = {}   # row -> {"codes":[...], "source":str}

    def add(row, codes, source):
        codes = [c for c in codes if c]
        if not codes:
            return
        rec.setdefault(row, {"codes": [], "source": source})
        for c in codes:
            if c not in rec[row]["codes"]:
                rec[row]["codes"].append(c)

    p1 = os.path.join(config.HERE, "pipe403_tier1.jsonl")
    for r in json.load(open(p1)):
        if r["status"] == "coded":
            add(r["row"], [c["code"] for c in r["codes"]], "403 repository copy (%s)" % r.get("host", "?"))

    pw = os.path.join(config.HERE, "audit_whitespace.jsonl")
    for w in json.load(open(pw)):
        add(w["row_index"], [c["code"] for c in w["codes"]], "whitespace-split fix")

    pm = os.path.join(config.HERE, "pmc_restricted_unpaywall.jsonl")
    for r in json.load(open(pm)):
        if r.get("codes"):
            add(r["row_index"], [c["code"] for c in r["codes"]], "restricted-PMC Unpaywall copy")

    # tier-2 headful browser run (line-delimited jsonl)
    p2 = os.path.join(config.HERE, "pipe403_tier2.jsonl")
    if os.path.exists(p2):
        for line in open(p2):
            r = json.loads(line)
            if r.get("status") == "coded" and r.get("codes"):
                add(r["row"], [c["code"] for c in r["codes"]], "403 headful browser read")

    # lookalike-collision sweep (real codes the ±60 lookalike filter wrongly dropped)
    pl = os.path.join(config.HERE, "audit_lookalike.jsonl")
    if os.path.exists(pl):
        for r in json.load(open(pl)):
            add(r["row_index"], [c["code"] for c in r["codes"]], "lookalike-collision fix")

    # headful real-Chrome (CDP) browser recovery + tolerant re-audit — all browser_re* shard files
    import glob
    for fp in glob.glob(os.path.join(config.HERE, "browser_*.jsonl")):
        for line in open(fp):
            r = json.loads(line)
            if r.get("status") == "coded" and r.get("codes"):
                add(r["row"], [c["code"] for c in r["codes"]], "real-Chrome browser (CDP)")

    return rec


def main():
    rows = list(csv.DictReader(open(OUT, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    rec = gather()

    n_merged = n_skip = n_review = n_append = 0
    for row, info in sorted(rec.items()):
        if row >= len(rows):
            continue
        r = rows[row]
        if not _fillable(r["Accession Code"]):
            # never overwrite — but a lookalike-collision fix may ADD a code the pipeline dropped
            if info["source"].startswith("lookalike"):
                have = {c.strip().upper() for c in re.split(r"[;,]", r["Accession Code"])}
                extra = [c for c in info["codes"] if c.upper() not in have]
                if extra:
                    r["Accession Code"] = r["Accession Code"].rstrip() + "; " + "; ".join(extra)
                    r["flag_detail"] = (r.get("flag_detail") or "").rstrip() + \
                        "  [added] lookalike-collision fix: " + ", ".join(extra)
                    n_append += 1
            n_skip += 1
            continue
        codes = info["codes"]
        r["Accession Code"] = "; ".join(codes)
        r["Notes"] = "[auto] recovered (%s): %s" % (info["source"], "; ".join(codes))
        r["flag"] = "CLEAN" if len(codes) == 1 else "MULTI_CODE"
        detail = "recovered this session via %s" % info["source"]
        if row in DOUBTFUL:
            detail += "  [review] Zenodo id scraped from landing page, not stated in paper — VERIFY"
            r["flag"] = "PROV_UNCLEAR"
            n_review += 1
        r["flag_detail"] = detail
        n_merged += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    coded = sum(1 for r in rows if not _fillable(r["Accession Code"]))
    print("merged recovered codes into %s" % OUT)
    print("  rows newly coded : %d" % n_merged)
    print("  tagged [review]  : %d" % n_review)
    print("  extra code appended to already-coded row: %d" % n_append)
    print("  skipped (already had a code): %d" % n_skip)
    print("  TOTAL coded rows now: %d" % coded)


if __name__ == "__main__":
    main()
