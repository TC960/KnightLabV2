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


def _is_fillable(acc):
    # blank / ACCESSION_NOT_FOUND / N/A — none is a real human-entered code, so safe to fill or
    # re-triage. Rows holding a real code never enter this branch (never-overwrite is preserved).
    v = (acc or "").strip().upper()
    return v in ("", "N/A", "NA", "ACCESSION_NOT_FOUND")


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
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ncbi_path = os.path.join(config.HERE, "ncbi.jsonl")
    ncbi_body = {json.loads(l)["row_index"] for l in open(ncbi_path)
                 if json.loads(l).get("ncbi_status") == 200 and not json.loads(l).get("ncbi_non_oa")} \
        if os.path.exists(ncbi_path) else set()

    def read_fulltext(i):
        return recs.get(i, {}).get("ft_status") == 200 or i in ncbi_body

    n_filled = n_na = n_hcg = n_applied = n_review = 0
    n_up_filled = n_oa = n_paywall = n_floor_kept = 0
    # Detail phrasing for definitive floor flags — reads an OA copy can't improve on.
    DEFIN_DETAIL = {"ON_REQUEST": "available on request",
                    "NO_DATA": "no data generated",
                    "NO_ACCESSION": "full text read, no accession"}

    def definitive_floor(i, our_f, orig_flag):
        # ON_REQUEST / NO_DATA are explicit "don't expect a deposited accession" signals from either
        # our read or prod's original flag. NO_ACCESSION is definitive ONLY when we actually read the
        # full text THIS run (our catch-all NO_ACCESSION otherwise includes abstract-only rows, where
        # an OA full-text copy genuinely is new information).
        for f in (our_f, orig_flag):
            if f in ("ON_REQUEST", "NO_DATA"):
                return f
        if our_f == "NO_ACCESSION" and read_fulltext(i):
            return "NO_ACCESSION"
        return None

    for i, r in enumerate(rows):
        orig_flag = (r.get("flag") or "").strip()  # prod's determination, before we overwrite it
        if _is_fillable(r["Accession Code"]):
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
                    our_f = e.get("flag")
                    oa = up.get("oa_url") if up else None
                    floor_flag = definitive_floor(i, our_f, orig_flag)
                    if our_f == "HUMAN_CAN_GET":
                        r["flag"] = "HUMAN_CAN_GET"; r["flag_detail"] = e.get("floor", ""); n_hcg += 1
                    elif floor_flag:
                        # keep the determination; append the OA copy so a curator can still verify
                        r["flag"] = floor_flag
                        r["flag_detail"] = DEFIN_DETAIL[floor_flag] + ((" -- OA copy: %s" % oa) if oa else "")
                        n_floor_kept += 1
                    elif oa:
                        r["flag"] = "OA_AVAILABLE"
                        r["flag_detail"] = "OA full text (%s) -- %s" % (up["host"], oa)
                        n_oa += 1
                    elif up and up.get("found") and not up.get("is_oa"):
                        r["flag"] = "PAYWALLED"
                        r["flag_detail"] = "paywalled -- full text exists at publisher, retrievable with library access"
                        n_paywall += 1
                    else:
                        r["flag"] = our_f or "NO_ACCESSION"; r["flag_detail"] = e.get("floor", ""); n_na += 1
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
    print("  code-less rows -> OA_AVAILABLE (free, click-through url): %d" % n_oa)
    print("  code-less rows -> PAYWALLED (library access)            : %d" % n_paywall)
    print("  code-less rows -> definitive floor kept (on request / no data / read-none): %d" % n_floor_kept)
    print("  code-less rows -> N/A (residual, no signal): %d" % n_na)
    print("  provenance corrections applied (target) : %d" % n_applied)
    print("  answered rows with [review] suggestions : %d" % n_review)
    print("  --- total codes now: %d ---" % (n_filled + n_up_filled))


if __name__ == "__main__":
    main()
