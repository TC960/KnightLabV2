"""Offline provenance correction — fixes the two biases prod.gs has, using ZERO network.

1. Earliest-depositor rule: an accession that appears on multiple papers can only have been
   DEPOSITED by the earliest one; later papers citing it are REUSING. This corrects prod.gs's
   systematic 'own' bias on controlled-access repos (EGA/dbGaP), where deposit and reuse are
   phrased identically and the regex cannot tell them apart.
2. dbGaP dedup: PHS001768 (bare) and PHS001768.V1.P1 (versioned) are the same study; collapse
   on the phs number, keep the versioned form.

Reads: the sheet (answered rows' codes+provenance) + extracted.jsonl (new target codes) + the
cache (publication dates). Writes: provenance_corrections.jsonl + dbgap_collapse.jsonl.

Run:  python3 -m accession.provenance_pass
"""
import csv, json, os, re, collections
from . import config, sources, dictionary


def _year_from_xml(xml):
    m = re.search(r"<pub-date[^>]*>.*?<year>(\d{4})</year>", xml or "", re.S)
    return m.group(1) if m else None


def get_year(rec):
    """Publication year from cache only. EPMC core record -> fulltext XML -> NCBI XML."""
    via = rec.get("resolved_via")
    if via == "pmid" and rec.get("pmid"):
        d = sources.epmc_record_by_pmid(rec["pmid"])
        if d.get("firstPublicationDate"):
            return d["firstPublicationDate"][:4]
        if d.get("pubYear"):
            return str(d["pubYear"])
    elif via == "doi" and rec.get("doi"):
        d = sources.epmc_record_by_doi(rec["doi"])
        if d.get("firstPublicationDate"):
            return d["firstPublicationDate"][:4]
        if d.get("pubYear"):
            return str(d["pubYear"])
    if rec.get("pmcid"):
        if rec.get("ft_status") == 200:
            _, xml = sources.epmc_fulltext(rec["pmcid"])
            y = _year_from_xml(xml)
            if y:
                return y
        st, xml = sources.ncbi_efetch(rec["pmcid"])
        if st == 200:
            return _year_from_xml(xml)
    return None


def dbgap_key(code):
    m = re.match(r"(PHS\d+)", code, re.I)
    return m.group(1).upper() if m else None


def _is_own(prov):
    # anchored, NOT substring: must not match 'unknown' (which contains 'own').
    return bool(prov) and re.match(r"own\b", prov.strip().lower()) is not None


def parse_sheet_codes(notes, acc):
    """Answered rows: recover (code, provenance) from Notes '[auto] CODE=prov; ...', falling back
    to the accession column with unknown provenance."""
    pairs = []
    for piece in re.split(r"[;,]", notes or ""):
        m = re.match(r"\s*\[?auto\]?\s*([A-Za-z0-9._-]+)\s*=\s*([a-z][a-z ()\-]*)", piece, re.I)
        if m and re.search(r"\d", m.group(1)):
            pairs.append((m.group(1).upper(), m.group(2).strip().lower()))
    if pairs:
        return pairs
    return [(c.upper(), "unknown") for c in re.split(r"[;,\s]+", (acc or "").strip())
            if re.search(r"\d", c)]


def main():
    rows = list(csv.DictReader(open(config.CSV_PATH, newline="", encoding="utf-8")))
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}

    # occurrences[key] = [ {row, code, year, prov, kind} ] ; key groups dbGaP versions together
    occ = collections.defaultdict(list)
    year_cache = {}

    def is_target(r):
        v = (r["Accession Code"] or "").strip().upper()
        return v == "" or v == "ACCESSION_NOT_FOUND"

    for i, r in enumerate(rows):
        if i not in recs:
            continue
        if i not in year_cache:
            year_cache[i] = get_year(recs[i])
        yr = year_cache[i]
        if is_target(r):
            for c in ext.get(i, {}).get("codes", []):
                if dictionary.in_accession_column(c["repo"]):
                    key = dbgap_key(c["code"]) or c["code"]
                    occ[key].append({"row": i, "code": c["code"], "year": yr,
                                     "prov": c.get("prov") or "unclear", "kind": "new"})
        else:
            for code, prov in parse_sheet_codes(r.get("Notes"), r.get("Accession Code")):
                key = dbgap_key(code) or code
                occ[key].append({"row": i, "code": code, "year": yr, "prov": prov, "kind": "existing"})

    # --- correction 1: earliest-depositor ---
    corrections = []
    by_repo = collections.Counter()
    for key, os_ in occ.items():
        rows_seen = {o["row"] for o in os_}
        dated = [o for o in os_ if o["year"]]
        if len(rows_seen) < 2 or not dated:
            continue
        earliest = min(o["year"] for o in dated)
        for o in os_:
            if _is_own(o["prov"]) and o["year"] and o["year"] > earliest:
                repo = next((r for r, rx, _ in dictionary.DICT if rx.match(o["code"])), "?")
                corrections.append({"row": o["row"], "code": o["code"], "key": key,
                                    "old_prov": o["prov"], "new_prov": "reused",
                                    "reason": "corpus: earlier paper (%s) deposited this accession" % earliest,
                                    "this_year": o["year"], "earliest_year": earliest,
                                    "kind": o["kind"], "repo": repo})
                by_repo[repo] += 1

    # --- correction 2: dbGaP version collapse ---
    collapses = []
    for key, os_ in occ.items():
        if not key.startswith("PHS"):
            continue
        variants = {o["code"] for o in os_}
        if len(variants) > 1:
            canonical = sorted(variants, key=lambda c: (".V" not in c.upper(), -len(c)))[0]
            collapses.append({"phs": key, "variants": sorted(variants), "canonical": canonical,
                              "rows": sorted({o["row"] for o in os_})})

    with open(os.path.join(config.HERE, "provenance_corrections.jsonl"), "w") as f:
        for c in corrections:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(config.HERE, "dbgap_collapse.jsonl"), "w") as f:
        for c in collapses:
            f.write(json.dumps(c) + "\n")

    multi = sum(1 for k, v in occ.items() if len({o["row"] for o in v}) >= 2)
    print("distinct accession keys: %d  (appearing on >=2 papers: %d)" % (len(occ), multi))
    print("\n=== correction 1: earliest-depositor (own -> reused) ===")
    print("  total corrections: %d" % len(corrections))
    print("  by repo:", dict(by_repo.most_common()))
    print("  (prod.gs cross-check found 23 such conflicts, mostly EGA/dbGaP — same direction)")
    print("\n=== correction 2: dbGaP version collapse ===")
    print("  phs numbers with multiple variant forms: %d" % len(collapses))
    for c in collapses[:8]:
        print("   ", c["phs"], c["variants"], "-> keep", c["canonical"])
    print("\n  sample earliest-depositor corrections:")
    for c in corrections[:8]:
        print("    row %-5d %-18s %s->reused (this=%s earliest=%s)"
              % (c["row"], c["code"], c["old_prov"], c["this_year"], c["earliest_year"]))


if __name__ == "__main__":
    main()
