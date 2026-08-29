"""Corpus-wide sweep for the lookalike-collision false negative.

Production rejects any dictionary match with a lookalike word (catalog/primer/assay/...) within
+/-60 chars. That window is too wide: a legit accession beside an unrelated 'Catalog' or 'kit' is
wrongly dropped (e.g. EGAD00001007035 next to 'Polygenic Score Catalog'). This finds every
code-less row where a guarded dictionary match was killed ONLY by the lookalike filter AND has
deposit/archive language nearby (high precision -> a real accession we missed).

Run:  python3 -m accession.audit_lookalike
"""
import csv, json, os, re
from . import config, dictionary
from .extract import _fulltext, _record_dict, _crossref_abstract, strip_xml

OUT = os.path.join(config.HERE, "audit_lookalike.jsonl")
DEPOSIT = re.compile(
    r"deposit|archiv|accession|bioproject|biosample|sequence read archive|\bsra\b|\bena\b|ddbj|"
    r"european nucleotide|genome[- ]?phenome|\bega\b|repositor|"
    r"(data|reads|sequences).{0,30}(available|published|submitted)", re.I)


def _text(rec):
    t, _, _ = _fulltext(rec)
    if t:
        return t
    rd = _record_dict(rec)
    return strip_xml(rd.get("abstractText", "")) or _crossref_abstract(rec)


def main():
    rows = list(csv.DictReader(open(os.path.join(config.REPO, "articles.out.csv"),
                                     newline="", encoding="utf-8")))
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}
    codeless = [i for i, e in ext.items() if e.get("accession_col") in ("N/A", "")]
    print("code-less rows to sweep: %d" % len(codeless), flush=True)

    found = []
    for n, i in enumerate(codeless, 1):
        text = _text(recs.get(i, {}))
        if not text:
            continue
        hits = {}
        for repo, rx, guard in dictionary.DICT:
            if repo in ("figshare", "Zenodo", "Dryad", "GCA"):
                continue
            for m in rx.finditer(text):
                at, L = m.start(), len(m.group(0))
                code = m.group(0).upper()
                # must PASS the standalone guard (a genuine standalone token)...
                before = text[at - 1] if at > 0 else " "
                after = text[at + L] if at + L < len(text) else " "
                if guard and (re.match(r"[A-Za-z0-9]", before) or re.match(r"[A-Za-z]", after)):
                    continue
                # ...and be killed ONLY by the lookalike filter...
                if not dictionary.LOOKALIKE.search(text[max(0, at - 60): at + L + 60]):
                    continue
                # ...and have deposit language nearby (high precision).
                if not DEPOSIT.search(text[max(0, at - 120): at + L + 120]):
                    continue
                if code not in hits:
                    hits[code] = {"code": code, "repo": repo,
                                  "context": re.sub(r"\s+", " ", text[max(0, at - 120): at + L + 90])}
        if hits:
            found.append({"row_index": i, "flag": (rows[i].get("flag") or "").strip(),
                          "doi": recs[i].get("doi"), "codes": list(hits.values())})
        if n % 1000 == 0:
            print("  ...%d/%d, found %d" % (n, len(codeless), len(found)), flush=True)

    json.dump(found, open(OUT, "w"), indent=1)
    print("\n=== LOOKALIKE-COLLISION FALSE NEGATIVES ===")
    print("rows recovered: %d" % len(found))
    print("wrote %s" % OUT)
    for f in found:
        for c in f["codes"]:
            print("  r%-6d %-13s %-16s | ...%s..." % (f["row_index"], f["flag"], c["code"], c["context"][:80]))


if __name__ == "__main__":
    main()
