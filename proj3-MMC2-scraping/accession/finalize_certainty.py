"""Add a `certainty` column to articles.out.csv classifying every row's no-code confidence.

  coded          — an accession is present
  verified-empty — we READ the full text (EPMC/NCBI body, or a fetched OA copy) and the
                   dictionary + independent audit found no accession. High confidence (~99.6%).
  unverified     — we did NOT read the full text (paywalled / restricted PMC / abstract- or
                   landing-page only). We cannot assert there is no accession. THIS is the
                   'not 100% sure' set the curator should treat as open.

Idempotent. Run:  python3 -m accession.finalize_certainty
"""
import csv, json, os
from . import config

OUT = os.path.join(config.REPO, "articles.out.csv")


def _load():
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ncbi = {}
    np = os.path.join(config.HERE, "ncbi.jsonl")
    if os.path.exists(np):
        for l in open(np):
            d = json.loads(l); ncbi[d["row_index"]] = d
    t1read = {r["row"] for r in json.load(open(os.path.join(config.HERE, "pipe403_tier1.jsonl")))
              if r["status"] == "read-no-code"}
    return recs, ncbi, t1read


def main():
    recs, ncbi, t1read = _load()

    def read_ft(i):
        if recs.get(i, {}).get("ft_status") == 200:
            return True
        n = ncbi.get(i)
        if n and n.get("ncbi_status") == 200 and not n.get("ncbi_non_oa"):
            return True
        return i in t1read

    rows = list(csv.DictReader(open(OUT, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    if "certainty" not in fields:
        fields.append("certainty")

    def coded(r):
        return (r.get("Accession Code") or "").strip().upper() not in ("", "N/A", "NA", "ACCESSION_NOT_FOUND")

    n = {"coded": 0, "verified-empty": 0, "unverified": 0}
    for i, r in enumerate(rows):
        if coded(r):
            c = "coded"
        elif read_ft(i):
            c = "verified-empty"
        else:
            c = "unverified"
        r["certainty"] = c
        n[c] += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("added `certainty` column to %s" % OUT)
    for k in ("coded", "verified-empty", "unverified"):
        print("  %-15s %d" % (k, n[k]))
    print("  total %d" % sum(n.values()))


if __name__ == "__main__":
    main()
