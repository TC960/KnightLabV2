"""Size the whitespace-split false-negative across ALL code-less rows.

Production's INSDC/GEO regexes have no \\s, so 'PRJNA 795467' (a space or newline injected by
PDF/XML extraction) is invisible to them. This scans every code-less row's replayed text for a
whitespace-split accession, keeps only hits with DEPOSIT/ARCHIVE language nearby (so we count
real accessions, not grant/ethics numbers), and confirms production's contiguous regex genuinely
missed it. Output = a high-precision recovery list.

Run:  python3 -m accession.audit_whitespace
"""
import csv, json, os, re, collections
from . import config, dictionary
from .extract import _fulltext, _record_dict, _crossref_abstract, strip_xml

OUT = os.path.join(config.HERE, "audit_whitespace.jsonl")

# split patterns: prefix, 1-3 whitespace, digits. Capture the full thing to normalize.
SPLIT = [
    ("BioProject",   re.compile(r"(PRJ(?:EB|NA|DB|CA))\s{1,3}(\d{4,})", re.I)),
    ("BioSample",    re.compile(r"(SAM(?:EA|N|D))\s{1,3}(\d{6,})", re.I)),
    ("SRA/ENA/DDBJ", re.compile(r"([SED]R[APRSX])\s{1,3}(\d{5,})", re.I)),
    ("GEO",          re.compile(r"(G(?:SE|SM|PL|DS))\s{1,3}(\d{4,})", re.I)),
    ("dbGaP",        re.compile(r"(phs)\s{1,3}(\d{6})", re.I)),
    ("PRIDE",        re.compile(r"(PXD)\s{1,3}(\d{6})", re.I)),
    ("EGA",          re.compile(r"(EGA[SD])\s{1,3}(\d{6,})", re.I)),
    ("MetaboLights", re.compile(r"(MTBLS)\s{1,3}(\d+)", re.I)),
]
# require real deposit/archive vocabulary within the window — filters grant/ethics/OTU numbers
DEPOSIT = re.compile(
    r"deposit|archiv|accession|bioproject|biosample|sequence read archive|\bsra\b|\bena\b|"
    r"\bgeo\b|ddbj|european nucleotide|genome[- ]?phenome|repositor|available (in|at|under|through)|"
    r"data availability|submitted to|under the (project|accession)", re.I)


def _text(rec):
    t, _, _ = _fulltext(rec)
    if t:
        return t
    rd = _record_dict(rec)
    return strip_xml(rd.get("abstractText", "")) or _crossref_abstract(rec)


def main():
    out_rows = list(csv.DictReader(open(os.path.join(config.REPO, "articles.out.csv"),
                                        newline="", encoding="utf-8")))
    final_flag = {i: (r.get("flag") or "").strip() for i, r in enumerate(out_rows)}
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}

    codeless = [i for i, e in ext.items() if e.get("accession_col") in ("N/A", "")]
    print("code-less rows to scan: %d" % len(codeless), flush=True)

    found = []
    by_flag = collections.Counter()
    for n, i in enumerate(codeless, 1):
        text = _text(recs.get(i, {}))
        if not text:
            continue
        row_hits = {}
        for repo, rx in SPLIT:
            for m in rx.finditer(text):
                norm = (m.group(1) + m.group(2)).upper()
                at, L = m.start(), len(m.group(0))
                if not DEPOSIT.search(text[max(0, at - 120): at + L + 120]):
                    continue
                # confirm production's CONTIGUOUS regex truly missed it (not already extracted elsewhere)
                if re.search(re.escape(norm), text):
                    continue  # a contiguous copy exists -> production had a fair shot; skip
                if norm not in row_hits:
                    row_hits[norm] = {"code": norm, "repo": repo,
                                      "context": re.sub(r"\s+", " ", text[max(0, at - 110): at + L + 90])}
        if row_hits:
            found.append({"row_index": i, "flag": final_flag.get(i), "doi": recs[i].get("doi"),
                          "pmcid": recs[i].get("pmcid"), "codes": list(row_hits.values())})
            by_flag[final_flag.get(i)] += 1
        if n % 1000 == 0:
            print("  ...scanned %d/%d, recoverable rows %d" % (n, len(codeless), len(found)), flush=True)

    json.dump(found, open(OUT, "w"), indent=1)
    print("\n==============  WHITESPACE-SPLIT RECOVERY  ==============")
    print("code-less rows scanned            : %d" % len(codeless))
    print("rows with a RECOVERABLE accession : %d" % len(found))
    print("  by current (wrong) flag: %s" % dict(by_flag.most_common()))
    print("  wrote %s" % OUT)
    print("\n  every recovered row (row / flag / code / context):")
    for f in found:
        for c in f["codes"]:
            print("   r%-6d %-13s %-13s | ...%s..." % (f["row_index"], f["flag"], c["code"], c["context"][:78]))


if __name__ == "__main__":
    main()
