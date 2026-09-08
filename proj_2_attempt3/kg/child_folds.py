#!/usr/bin/env python3
"""Which surface strings still EXTEND the name they resolve to -> child_folds_current.json.

WHY THIS EXISTS, AND WHY IT DOES NOT OVERWRITE `child_folds.json`.
`child_folds.json` has been the project's map of where the resolver silently
collapses a rank -- it found the SILVA placeholder collapse and then the species
collapse -- but it was produced ad hoc and **no generator was ever committed**, so
it could never be refreshed and by 2026-09-08 described a graph three corrections
old. This script answers the same question reproducibly, in the PRESENT tense.

It writes `child_folds_current.json`, deliberately, because `child_folds.json` is
now a frozen INPUT: `species_synonyms.py` reads it to know which strings folded
BEFORE the fix. Overwrite it against today's graph and the 24 split species
disappear from it, the synonym table regenerates EMPTY, and the next
`build_kg.py` silently un-splits them while printing success -- the self-erasing
fix this repo has already shipped twice. `species_synonyms.candidates()` asserts
against exactly that.

This reimplementation is NOT byte-faithful to the 2026-09-01 artifact (146 rows
vs 115, and ~38 rows classified differently), because the original generator does
not exist to compare against. Treat the classes as advisory, as below.

THE QUESTION IT ASKS. If a mention resolved to a taxon whose scientific name is a
strict PREFIX of the string the paper actually wrote, then the resolver threw a
qualifier away to get there -- "Prevotella copri" landing on *Prevotella*,
"Erysipelotrichaceae UCG-003" on *Erysipelotrichaceae*. Sometimes that is right
and sometimes it is a rank collapse, so the classes below are advisory: they say
what KIND of tail was discarded, and the decision about each kind is recorded in
FINDINGS_rank_collapse.md and FINDINGS_species_split.md.

  placeholder_child   16S pipeline rank labels (UCG-003, ND3007 group, "Prevotella
                      9", "Clostridium IV"). SPLIT since 2026-09-01.
  unspecified_member  "X sp.", "X spp.", "X unclassified". Folding is CORRECT --
                      the paper named no species. Do not touch.
  named_child         everything else. NOT a synonym for "a species": of the 54
                      once in this class, 24 named a species and are now split,
                      while the rest name two taxa at once ("Escherichia /
                      Shigella"), are cluster labels ("Clostridium_XlVa"), or name
                      no organism ("Neisseria multispecies").

`resolved_species` marks entries that `species_synonyms.json` now splits, so the
file distinguishes "still folding" from "handled" instead of implying everything
listed is outstanding.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "child_folds_current.json")

PLACEHOLDER = re.compile(
    r"(UCG[-_ ]?\d+|_?group$|ND\d{3,}|R-\d+\b|incertae[ _]sedis|"
    r"sensu[ _]stricto|\bAD\d{3,}\b|\b[A-Z]{1,3}\d{2,}\b|"
    r"[ _]\d{1,3}$|[ _][IVXL]+$|\bcluster[ _]|\bFamily[ _][IVXL]+\b)")
UNSPECIFIED = re.compile(r"^(sp|spp|species|unclassified|unidentified)\b\.?", re.I)


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split())


def classify(surface, tail):
    """The two tests read different strings, and swapping them silently mis-buckets.

    `unspecified_member` is about what the qualifier IS, so it anchors on the TAIL
    ("sp.", "spp."). `placeholder_child` is about the SILVA/RDP shape of the whole
    label, and those patterns anchor on a separator -- "Prevotella 9" matches
    `[ _]\\d{1,3}$` only while the space is still attached, which the bare tail
    "9" has already lost. Reading the tail for both put 30 SILVA placeholders in
    `named_child`; reading the full string for both emptied `unspecified_member`
    entirely, since every surface starts with its genus.
    """
    if UNSPECIFIED.match(tail.strip()):
        return "unspecified_member"
    if PLACEHOLDER.search(surface):
        return "placeholder_child"
    return "named_child"


def main(out=OUT, no_supplement=False):
    """Run over the RAW extraction strings, not over graph.json's aliases.

    The distinction is not cosmetic and cost one wrong reimplementation. A
    placeholder that has already been SPLIT is its own node, so its string is no
    longer an alias of the parent and vanishes from an alias-based scan -- which
    silently hides the 32 placeholder folds the diagnostic exists to show. Asking
    the resolver directly, string by string, is the question actually being asked:
    'does resolving this drop a qualifier?'
    """
    import build_kg as BK
    from taxonomy_cache import load_taxonomy
    tax = load_taxonomy(verbose=False)
    if no_supplement:
        # reproduce the pre-2026-09-08 file, before the species split existed
        tax.sup = {}
    try:
        from species_synonyms import load_table
        split = set(load_table())
    except Exception:
        split = set()

    rows, seen = [], set()
    for r in json.load(open(os.path.join(HERE, "extractions_screened.json"))):
        for col in ("predicted_enriched", "predicted_depleted"):
            for raw in BK.parse_taxa(r.get(col)):
                if raw in seen:
                    continue
                seen.add(raw)
                tid, sci, rank, how = tax.resolve(raw)
                if not tid or not sci:
                    continue
                s, label = norm(raw), norm(sci)
                if s.lower() == label.lower():
                    continue
                # the string must EXTEND the resolved name, not merely contain it
                if not s.lower().startswith(label.lower() + " "):
                    continue
                tail = s[len(label):].strip()
                rows.append({"parent": sci, "parent_rank": rank,
                             "node": f"t:ncbi:{tid}", "surface": raw, "tail": tail,
                             # classify on the FULL string: the placeholder patterns
                             # ("Prevotella 9", "Clostridium IV") anchor on a leading
                             # separator that the bare tail has already lost
                             "cls": classify(s, tail),
                             "resolved_species": norm(raw).lower() in split})
    rows.sort(key=lambda r: (r["parent"] or "", r["surface"]))
    json.dump(rows, open(out, "w"), indent=1)
    from collections import Counter
    print(f"child folds: {len(rows)} over {len({r['node'] for r in rows})} nodes"
          f"{'  [supplement suppressed]' if no_supplement else ''}")
    for k, v in Counter(r["cls"] for r in rows).most_common():
        print(f"  {k:20} {v}")
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else OUT, no_supplement="--no-supplement" in a)
