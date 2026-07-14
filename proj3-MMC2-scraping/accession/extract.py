"""Phase 2 — EXTRACT. Fast, re-runnable, NETWORK-FREE (reads only the cache + records.jsonl).

Tweak dictionary.py, re-run this, get new results in seconds over the whole corpus. This is the
capability the Apps Script version could never have.

Run:  python3 -m accession.extract
"""
import csv, json, os, re
from collections import Counter
from . import config, sources, dictionary


def strip_xml(x):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()


def _fulltext(rec):
    """Return (plain_text, source, non_oa). EPMC 200 preferred; NCBI efetch fallback on EPMC-404.
    Cache-only in practice — the NCBI fallback pass pre-populates these responses."""
    if rec.get("pmcid") and rec.get("ft_status") == 200:
        _, xml = sources.epmc_fulltext(rec["pmcid"])
        if xml:
            return strip_xml(xml), "fulltext", False
    if rec.get("pmcid"):
        st, xml = sources.ncbi_efetch(rec["pmcid"])
        if st == 200 and xml:
            return strip_xml(xml), "ncbi_fulltext", sources.is_non_oa(xml)
    return "", None, False


def _record_dict(rec):
    """Re-read the cached EPMC core record (cache hit, no network) for abstractText."""
    via = rec.get("resolved_via")
    if via == "pmid" and rec.get("pmid"):
        return sources.epmc_record_by_pmid(rec["pmid"])
    if via == "doi" and rec.get("doi"):
        return sources.epmc_record_by_doi(rec["doi"])
    return {}


def _crossref_abstract(rec):
    if not rec.get("crossref") or not rec.get("doi"):
        return ""
    ab = sources.crossref_work(rec["doi"]).get("abstract", "")
    return strip_xml(ab) if ab else ""


def extract_row(rec):
    """Return dict: codes[], channel, accession_col, notes, flag, floor. Cache-only."""
    if rec.get("error"):
        return {"codes": [], "channel": "error", "floor": "fetch error",
                "accession_col": "", "notes": "[auto] fetch error", "flag": "FETCH_ERROR"}

    codes, seen = [], set()
    channel = None
    fulltext_plain = abstract_plain = ""
    non_oa = False

    # 1) full text (strongest). EPMC first; on EPMC-404 fall back to NCBI efetch (cached by the
    #    fallback pass). NCBI non-OA papers return 200 with front-matter only — mine it anyway.
    ft_plain, ft_src, non_oa = _fulltext(rec)
    if ft_plain:
        fulltext_plain = ft_plain
        for c in dictionary.extract_codes(fulltext_plain):
            if c["code"] not in seen:
                seen.add(c["code"]); c["src"] = ft_src; codes.append(c); channel = channel or ft_src

    # 2) mined annotations (only channel for the pmid-only bucket)
    if rec.get("ann_id"):
        exacts = sources.epmc_annotations(rec["ann_id"])
        for c in dictionary.extract_from_annotations(exacts):
            if c["code"] not in seen:
                seen.add(c["code"]); c["src"] = "annotation"; codes.append(c); channel = channel or "annotation"

    # 3) abstract fallback (EPMC core, else Crossref) — only if nothing yet
    if not codes:
        rd = _record_dict(rec)
        abstract_plain = strip_xml(rd.get("abstractText", "")) or _crossref_abstract(rec)
        if abstract_plain:
            for c in dictionary.extract_codes(abstract_plain):
                if c["code"] not in seen:
                    seen.add(c["code"]); c["src"] = "abstract"; codes.append(c); channel = channel or "abstract"

    # column string honours the GCA capture-vs-column split
    col_codes = [c for c in codes if dictionary.in_accession_column(c["repo"])]

    if col_codes:
        accession_col = "; ".join(c["code"] for c in col_codes)
        notes = "[auto] " + "; ".join("%s=%s" % (c["code"], c["prov"]) for c in col_codes)
        flag = _flag(col_codes)
        floor = ""
    else:
        text_for_floor = fulltext_plain or abstract_plain
        if non_oa:
            # publisher-restricted full text EXISTS — a human can retrieve it. Not floor.
            floor = "non-OA: full text exists but is not machine-accessible"
            flag = "HUMAN_CAN_GET"
        elif text_for_floor:
            floor = dictionary.floor_reason(text_for_floor)
            flag = _floor_flag(floor)
        elif rec.get("pmcid"):
            floor = "no accession in text"
            flag = _floor_flag(floor)
        elif rec.get("pmid") or rec.get("doi"):
            floor = "no full text (publisher restricted)"
            flag = _floor_flag(floor)
        else:
            floor = "no identifier"
            flag = _floor_flag(floor)
        accession_col = "N/A"
        notes = "[auto] N/A - " + floor

    return {"codes": codes, "channel": channel or "none", "accession_col": accession_col,
            "notes": notes, "flag": flag, "floor": floor, "non_oa": non_oa,
            "gca_captured": [c["code"] for c in codes if c["repo"] == "GCA"]}


def _flag(col_codes):
    provs = {c["prov"] for c in col_codes}
    unclear = any("unclear" in p for p in provs)
    mined = all(c["src"] == "annotation" for c in col_codes)
    if mined:
        return "MINED_NO_CONTEXT"
    if len(col_codes) > 1 and unclear:
        return "PROV_UNCLEAR+MULTI_CODE"
    if len(col_codes) > 1:
        return "MULTI_CODE"
    if unclear:
        return "PROV_UNCLEAR"
    return "CLEAN"


def _floor_flag(floor):
    return {"available on request": "ON_REQUEST",
            "no data generated": "NO_DATA",
            "data in article/supplement": "SUPPLEMENT",
            "no accession in text": "NO_ACCESSION",
            "no full text (publisher restricted)": "NO_FULLTEXT",
            "no identifier": "NO_IDENTIFIER"}.get(floor, "NO_ACCESSION")


def main():
    if not os.path.exists(config.RECORDS_PATH):
        print("no records.jsonl yet — run fetch_corpus first"); return
    recs = [json.loads(l) for l in open(config.RECORDS_PATH)]
    out_path = os.path.join(config.HERE, "extracted.jsonl")
    fout = open(out_path, "w")
    flag_counts, chan_counts = Counter(), Counter()
    n_codes = n_target_codes = n_gca = 0
    for rec in recs:
        r = extract_row(rec)
        rec_out = {"row_index": rec.get("row_index"), "target": rec.get("target"), **r}
        fout.write(json.dumps(rec_out) + "\n")
        flag_counts[r["flag"]] += 1
        chan_counts[r["channel"]] += 1
        if r["accession_col"] not in ("", "N/A"):
            n_codes += 1
            if rec.get("target"):
                n_target_codes += 1
        n_gca += len(r.get("gca_captured", []))
    fout.close()
    print("extracted %d rows -> %s" % (len(recs), out_path))
    print("rows with >=1 column code:", n_codes, " (of which target:", n_target_codes, ")")
    print("GCA codes captured (not in column):", n_gca)
    print("\nflags:", dict(flag_counts.most_common()))
    print("channels:", dict(chan_counts.most_common()))


if __name__ == "__main__":
    main()
