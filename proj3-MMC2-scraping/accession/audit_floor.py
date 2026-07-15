"""AUDIT the code-less 'floor' declarations. Independent false-negative hunt.

For every row the pipeline declared code-less (accession_col == N/A), we REPLAY the exact
text it was judged on (cache-only, no network), then run an INDEPENDENT detector that is
deliberately LOOSER than production in the three places a real accession could have been
wrongly dropped:

  A) unguarded  — raw DICT pattern with the standalone-token guard REMOVED
  B) delookalike— raw DICT pattern with the LOOKALIKE ±60 filter REMOVED
  C) spaced     — whitespace-tolerant pattern: catches 'PRJNA 123456' / 'GSE\\n12345'
                  that the production regex (no \\s) cannot see at all

Every independent hit in a code-less row is a CANDIDATE FALSE NEGATIVE. We bucket by which
relaxation surfaced it and by final flag, then dump a context window for manual verdict.

Run:  python3 -m accession.audit_floor
"""
import csv, json, os, re, collections
from . import config, sources, dictionary
from .extract import _fulltext, _record_dict, _crossref_abstract, strip_xml

OUT = os.path.join(config.HERE, "audit_floor.jsonl")

# The floor flags where we ASSERT 'no deposited accession' — the audit targets.
FLOOR_FLAGS = {"ON_REQUEST", "NO_DATA", "NO_ACCESSION"}

# Whitespace-tolerant variants of the high-value INSDC/GEO patterns (spaces/newlines allowed
# between the prefix letters and the digits — the exact failure PDF/XML extraction produces).
SPACED = [
    ("BioProject",   re.compile(r"PRJ(?:EB|NA|DB|CA)\s{1,3}\d{4,}", re.I)),
    ("BioSample",    re.compile(r"SAM(?:EA|N|D)\s{1,3}\d{6,}", re.I)),
    ("SRA/ENA/DDBJ", re.compile(r"[SED]R[APRSX]\s{1,3}\d{5,}", re.I)),
    ("GEO",          re.compile(r"G(?:SE|SM|PL|DS)\s{1,3}\d{4,}", re.I)),
    ("dbGaP",        re.compile(r"phs\s{1,3}\d{6}", re.I)),
    ("PRIDE",        re.compile(r"PXD\s{1,3}\d{6}", re.I)),
    ("EGA",          re.compile(r"EGA[SD]\s{1,3}\d{6,}", re.I)),
    ("MetaboLights", re.compile(r"MTBLS\s{1,3}\d+", re.I)),
]


def _text_and_source(rec):
    """Replay the exact text the pipeline judged this row on. Returns (text, source)."""
    ft_plain, ft_src, _ = _fulltext(rec)
    if ft_plain:
        return ft_plain, ft_src           # 'fulltext' or 'ncbi_fulltext'
    rd = _record_dict(rec)
    ab = strip_xml(rd.get("abstractText", "")) or _crossref_abstract(rec)
    if ab:
        return ab, "abstract"
    return "", "none"


def independent_hits(text):
    """Return list of candidate accessions the LOOSE detector finds. Each: (code, repo, kind, at)."""
    hits = []
    for repo, rx, guard in dictionary.DICT:
        if repo in ("figshare", "Zenodo", "Dryad", "GCA"):
            continue  # DOI repos / capture-only — not the floor concern
        for m in rx.finditer(text):
            at, L = m.start(), len(m.group(0))
            code = m.group(0).upper()
            before = text[at - 1] if at > 0 else " "
            after = text[at + L] if at + L < len(text) else " "
            guard_fail = guard and (re.match(r"[A-Za-z0-9]", before) or re.match(r"[A-Za-z]", after))
            look_fail = bool(dictionary.LOOKALIKE.search(text[max(0, at - 60): at + L + 60]))
            if not guard_fail and not look_fail:
                continue  # production would have KEPT this — not a miss (shouldn't happen on code-less rows)
            kind = "unguarded" if guard_fail else "delookalike"
            hits.append((code, repo, kind, at))
    for repo, rx in SPACED:
        for m in rx.finditer(text):
            hits.append((re.sub(r"\s+", "", m.group(0)).upper(), repo, "spaced", m.start()))
    return hits


def main():
    out_rows = list(csv.DictReader(open(os.path.join(config.REPO, "articles.out.csv"),
                                        newline="", encoding="utf-8")))
    final_flag = {i: (r.get("flag") or "").strip() for i, r in enumerate(out_rows)}
    recs = {json.loads(l)["row_index"]: json.loads(l) for l in open(config.RECORDS_PATH)}
    ext = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "extracted.jsonl"))}

    # audit population: code-less rows whose FINAL flag is a floor assertion
    targets = [i for i, e in ext.items()
               if e.get("accession_col") in ("N/A", "") and final_flag.get(i) in FLOOR_FLAGS]

    src_by_flag = collections.defaultdict(collections.Counter)   # flag -> {source: n}
    candidates = []                                              # rows with >=1 independent hit
    kind_tally = collections.Counter()
    repo_tally = collections.Counter()

    for n, i in enumerate(targets, 1):
        rec = recs.get(i, {})
        text, source = _text_and_source(rec)
        src_by_flag[final_flag[i]][source] += 1
        if not text:
            continue
        hits = independent_hits(text)
        if hits:
            # dedup by code
            uniq = {}
            for code, repo, kind, at in hits:
                if code not in uniq:
                    uniq[code] = (repo, kind, at)
                    kind_tally[kind] += 1
                    repo_tally[repo] += 1
            ctx = []
            for code, (repo, kind, at) in list(uniq.items())[:6]:
                ctx.append({"code": code, "repo": repo, "kind": kind,
                            "context": re.sub(r"\s+", " ", text[max(0, at - 90): at + 90])})
            candidates.append({"row_index": i, "flag": final_flag[i], "source": source,
                               "doi": rec.get("doi"), "pmcid": rec.get("pmcid"),
                               "n_hits": len(uniq), "hits": ctx})
        if n % 500 == 0:
            print("  ...replayed %d/%d, candidates so far %d" % (n, len(targets), len(candidates)), flush=True)

    json.dump(candidates, open(OUT, "w"), indent=1)

    print("\n==================  FLOOR AUDIT  ==================")
    print("audited rows (final flag in %s): %d\n" % (sorted(FLOOR_FLAGS), len(targets)))
    print("--- 1) What text was each floor label based on? ---")
    print("    (a floor label on ABSTRACT-ONLY text is weaker: the code could be in unread full text)")
    for flag in sorted(src_by_flag):
        d = src_by_flag[flag]
        tot = sum(d.values())
        ft = d.get("fulltext", 0) + d.get("ncbi_fulltext", 0)
        print("    %-13s n=%-5d  fulltext=%-5d abstract-only=%-4d none=%-4d"
              % (flag, tot, ft, d.get("abstract", 0), d.get("none", 0)))
    print("\n--- 2) Independent re-detection (candidate FALSE NEGATIVES) ---")
    print("    rows where the LOOSE detector found an accession production dropped: %d" % len(candidates))
    print("    by relaxation that surfaced it:", dict(kind_tally))
    print("    by repo:", dict(repo_tally.most_common()))
    print("\n    wrote per-row context to %s" % OUT)
    print("\n    first 12 candidates (row / flag / source / code / kind / context):")
    for c in candidates[:12]:
        for h in c["hits"][:2]:
            print("     r%-5d %-12s %-9s %-14s %-11s | ...%s..."
                  % (c["row_index"], c["flag"], c["source"], h["code"], h["kind"], h["context"][:70]))


if __name__ == "__main__":
    main()
