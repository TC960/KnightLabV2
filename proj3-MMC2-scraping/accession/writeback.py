"""Writeback to CSV (never the live sheet). Produces articles.out.csv.

Policy (honours 'never overwrite a row where a real code already exists'):
  * TARGET rows (blank / ACCESSION_NOT_FOUND): filled with pipeline output, with dbGaP
    version-collapse and provenance corrections applied.
  * ANSWERED rows: Accession Code left UNTOUCHED. Provenance/dbGaP corrections that land on
    them are appended to flag_detail as '[review] ...' suggestions — surfaced, never applied.

Run:  python3 -m accession.writeback
"""
import csv, json, os, re, collections
from . import config, dictionary
from .provenance_pass import dbgap_key

OUT = os.path.join(config.REPO, "articles.out.csv")


def _is_target(acc):
    v = (acc or "").strip().upper()
    return v == "" or v == "ACCESSION_NOT_FOUND"


def _collapse_dbgap(codes):
    """Within one row: if multiple variants of a phs study appear, keep the versioned form."""
    by_key = collections.defaultdict(list)
    order = []
    for c in codes:
        k = dbgap_key(c["code"]) or c["code"]
        if k not in by_key:
            order.append(k)
        by_key[k].append(c)
    out = []
    for k in order:
        variants = by_key[k]
        if len(variants) == 1:
            out.append(variants[0])
        else:  # keep the versioned form (has .V..P..)
            out.append(sorted(variants, key=lambda c: (".V" not in c["code"].upper(), -len(c["code"])))[0])
    return out


def main():
    rows = list(csv.DictReader(open(config.CSV_PATH, newline="", encoding="utf-8")))
    fields = list(rows[0].keys())
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}
    corr = [json.loads(l) for l in open(os.path.join(config.HERE, "provenance_corrections.jsonl"))]
    corr_by_row = collections.defaultdict(dict)  # row -> {code: correction}
    for c in corr:
        corr_by_row[c["row"]][c["code"].upper()] = c
    up_path = os.path.join(config.HERE, "unpaywall.jsonl")
    up_map = {json.loads(l)["row_index"]: json.loads(l) for l in open(up_path)} if os.path.exists(up_path) else {}

    n_filled = n_na = n_hcg = n_applied = n_review = 0
    n_up_filled = n_oa = n_paywall = 0
    for i, r in enumerate(rows):
        if _is_target(r["Accession Code"]):
            e = ext.get(i, {})
            codes = [c for c in e.get("codes", []) if dictionary.in_accession_column(c["repo"])]
            if codes:
                codes = _collapse_dbgap(codes)
                parts = []
                for c in codes:
                    prov = c.get("prov") or "unclear"
                    fix = corr_by_row.get(i, {}).get(c["code"].upper())
                    if fix:
                        prov = fix["new_prov"] + " [corrected]"
                        n_applied += 1
                    parts.append((c["code"], prov))
                r["Accession Code"] = "; ".join(p[0] for p in parts)
                r["Notes"] = "[auto] " + "; ".join("%s=%s" % p for p in parts)
                r["flag"] = e["flag"]
                r["flag_detail"] = "auto-extracted via %s" % e["channel"]
                n_filled += 1
            else:
                up = up_map.get(i)
                if up and up.get("codes"):
                    # recovered from an OA copy Unpaywall found outside PMC/EPMC
                    codes = _collapse_dbgap(up["codes"])
                    parts = [(c["code"], c.get("prov") or "unclear") for c in codes]
                    r["Accession Code"] = "; ".join(p[0] for p in parts)
                    r["Notes"] = "[auto] " + "; ".join("%s=%s" % p for p in parts)
                    r["flag"] = "CLEAN" if len(parts) == 1 else "MULTI_CODE"
                    r["flag_detail"] = "auto-extracted via Unpaywall OA copy (%s): %s" % (up["host"], up["oa_url"])
                    n_up_filled += 1
                else:
                    r["Accession Code"] = "N/A"
                    r["Notes"] = e.get("notes", "[auto] N/A")
                    if e.get("flag") == "HUMAN_CAN_GET":
                        r["flag"] = e["flag"]; r["flag_detail"] = e.get("floor", ""); n_hcg += 1
                    elif up and up.get("oa_url"):
                        # full text exists free outside PMC — no accession in it, but a curator can open it
                        r["flag"] = "OA_AVAILABLE"
                        r["flag_detail"] = "OA full text (%s), no accession in it — %s" % (up["host"], up["oa_url"])
                        n_oa += 1
                    elif up and up.get("found") and not up.get("is_oa"):
                        r["flag"] = "PAYWALLED"
                        r["flag_detail"] = "paywalled — full text exists at publisher, retrievable with library access"
                        n_paywall += 1
                    else:
                        r["flag"] = e.get("flag", "NO_ACCESSION"); r["flag_detail"] = e.get("floor", ""); n_na += 1
        else:
            # answered — never overwrite the code; surface corrections as review notes only
            fixes = corr_by_row.get(i, {})
            if fixes:
                sug = "; ".join("%s %s->%s (%s)" % (c["code"], c["old_prov"], c["new_prov"], c["reason"])
                                for c in fixes.values())
                r["flag_detail"] = (r.get("flag_detail") or "").rstrip() + "  [review] provenance: " + sug
                n_review += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("wrote %s" % OUT)
    print("  target rows filled (EPMC/NCBI full text): %d" % n_filled)
    print("  target rows filled (Unpaywall OA copy)  : %d" % n_up_filled)
    print("  target rows -> HUMAN_CAN_GET            : %d" % n_hcg)
    print("  target rows -> OA_AVAILABLE (free, click-through url): %d" % n_oa)
    print("  target rows -> PAYWALLED (library access)            : %d" % n_paywall)
    print("  target rows -> N/A (residual true floor): %d" % n_na)
    print("  provenance corrections applied (target) : %d" % n_applied)
    print("  answered rows with [review] suggestions : %d" % n_review)
    print("  --- total codes now: %d ---" % (n_filled + n_up_filled))


if __name__ == "__main__":
    main()
