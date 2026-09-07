#!/usr/bin/env python3
"""PROBE / DEAD END -- do NOT use this to resolve taxa. See the warning below.

The real resolver is `resolve_named_children.py` on the `claude/kg-species-split`
branch, which gets all 54 right. This file is kept because its FAILURE is the
result: it shows that the obvious way to resolve a renamed species -- fuzzy
matching inside the genus its obsolete binomial names -- is systematically wrong
for exactly the species that were reclassified, and wrong in a way that looks
like a clean hit. Five of its twenty accepted matches are a different real
organism (Prevotella copri -> "Prevotella corporis", Bacteroides vulgatus ->
"Bacteroides ovatus"). Write-up: FINDINGS_containment_provenance.md.

Resolve the `named_child` surface strings in child_folds.json against NCBI.

WHY THIS EXISTS -- read before trusting it.

`child_folds.json` classifies 115 taxon strings that were folded into a coarser
parent node. 54 are marked `named_child`: a *named* child collapsed into its
parent, which the project forbids (rank collapse). Fixing them needs an NCBI
lookup, and this environment's network policy denies ftp.ncbi.nih.gov outright
(CONNECT -> 403), so the taxdump is unavailable and `taxonomy.py` degrades to
`ok=False`.

The way around it is the `ncbi-taxon-db` wheel behind the `taxoniq` package,
which is installed from PyPI (reachable) and *bundles* a prebuilt NCBI taxonomy
snapshot -- 2.6M scientific names with ranks and full lineages. It is a real
NCBI export, not a heuristic.

THE SCOPE CONDITION -- this is not a drop-in for names.dmp:

  The bundled DB indexes SCIENTIFIC NAMES ONLY. It has no synonym or
  equivalent-name table.

That matters concretely. `Bacteroidetes` and `Firmicutes` -- the two folds
`taxonomy.py` exists to perform -- are *synonyms* of `Bacteroidota`/`Bacillota`
and DO NOT RESOLVE here. So this module must never be used as the project's
general resolver; it would silently break the phylum folds that the whole graph
rests on. It is used only to answer one bounded question about 54 known strings,
each of which is then reported individually for audit.

It also means a renamed species resolves only under its CURRENT name. That is a
feature for this task -- it is how we learn that `Prevotella copri` is now
`Segatella copri` in a DIFFERENT genus -- but it is why strategy `epithet_global`
below exists at all.

STRATEGIES, applied in order, most trustworthy first. Every hit records which
one fired so a reader can discount the weak ones:

  exact          surface is verbatim an NCBI scientific name.
  normalized     same after collapsing underscores/brackets/whitespace.
  parent_epithet parent genus + trailing token, e.g. "Klebsiella pneumonia"
                 -> "Klebsiella pneumoniae". Catches misspellings in place.
  fuzzy_in_genus closest scientific name *within the parent genus*, difflib
                 ratio >= FUZZY_MIN. Catches "Bacteroides uniforms".
  epithet_global the species epithet matched against every NCBI species with
                 that epithet, then filtered to the parent's FAMILY. Catches
                 genus renames (Prevotella copri -> Segatella copri) without
                 letting an unrelated organism in.

THE CHECK THAT MATTERS. For every hit we ask whether the parent taxid the graph
folded the string into actually appears in the resolved organism's lineage
(`parent_in_lineage`). That is what makes the fold correct or incorrect, and it
is computed from NCBI's own lineage rather than from string shape. A hit with
`parent_in_lineage == False` must NOT simply be split out as a child of that
parent -- it is not one.

Output: named_children_resolved.json  (one record per surface, plus a summary)
"""
import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD_FOLDS = os.path.join(HERE, "child_folds.json")
OUT = os.path.join(HERE, "probe_fuzzy_genus_matching.json")

FUZZY_MIN = 0.86

# Strings whose tail marks them as NOT a named species at all, so "split the
# named child out of its parent" is the wrong operation regardless of whether
# something resolves. Kept explicit and separate from the resolution logic so
# the two judgements can be audited apart.
NONSPECIES_TAIL = re.compile(
    r"^(sp|spp|sp\.|bacterium|multispecies|species|phage|virus)\b", re.I)


def norm(s):
    s = (s or "").replace("_", " ").replace("[", "").replace("]", "")
    return " ".join(s.strip().split())


def load_db():
    try:
        import marisa_trie
        import ncbi_taxon_db
        import taxoniq
    except ImportError as e:
        sys.exit(f"need taxoniq + ncbi-taxon-db: pip install taxoniq  ({e})")
    p = os.path.dirname(ncbi_taxon_db.__file__)
    trie = marisa_trie.BytesTrie()
    trie.load(os.path.join(p, "sn2taxid.marisa"))
    return taxoniq, trie


def taxon(taxoniq, name):
    try:
        return taxoniq.Taxon(scientific_name=name)
    except Exception:
        return None


def lineage_ids(t):
    try:
        return [str(a.tax_id) for a in t.ranked_lineage]
    except Exception:
        return []


def describe(t):
    try:
        return {
            "taxid": str(t.tax_id),
            "scientific_name": t.scientific_name,
            "rank": t.rank.name if t.rank else "no rank",
            "lineage": [(str(a.tax_id), a.rank.name, a.scientific_name)
                        for a in t.ranked_lineage],
        }
    except Exception:
        return None


def main():
    taxoniq, trie = load_db()
    folds = json.load(open(CHILD_FOLDS))
    named = [f for f in folds if f["cls"] == "named_child"]
    print(f"named_child entries: {len(named)}")

    # index every scientific name by its final token, for epithet_global
    by_epithet = defaultdict(list)
    by_stem = defaultdict(list)
    for k in trie.iterkeys():
        toks = k.rsplit(" ", 1)
        if len(toks) == 2:
            e = toks[1]
            by_epithet[e].append(k)
            if len(e) >= 5 and e[0].islower():
                by_stem[e.lower()[:5]].append(k)
    print(f"indexed {len(by_epithet)} distinct epithets, {len(by_stem)} stems")

    records = []
    for f in named:
        surface = f["surface"]
        parent = f["parent"]
        parent_tid = f["node"].split(":")[-1]
        tail = norm(f.get("tail", ""))
        n = norm(surface)

        rec = {
            "surface": surface,
            "parent": parent,
            "parent_rank": f["parent_rank"],
            "parent_taxid": parent_tid,
            "tail": f.get("tail", ""),
            "nonspecies_tail": bool(NONSPECIES_TAIL.match(tail)),
            "how": None,
            "resolved": None,
            "parent_in_lineage": None,
            "candidates": [],
        }

        hit, how = None, None

        # 1 exact / 2 normalized
        for cand, label in ((surface, "exact"), (n, "normalized")):
            t = taxon(taxoniq, cand)
            if t:
                hit, how = t, label
                break

        # 3 parent + trailing token, repairing a misspelled epithet in place
        if hit is None and tail and " " not in tail and f["parent_rank"] == "genus":
            t = taxon(taxoniq, f"{parent} {tail.lower()}")
            if t:
                hit, how = t, "parent_epithet"

        # 4 closest name inside the parent genus
        if hit is None and f["parent_rank"] == "genus":
            best, score = None, 0.0
            for k in trie.iterkeys(parent + " "):
                r = SequenceMatcher(None, n.lower(), k.lower()).ratio()
                if r > score:
                    best, score = k, r
            if best and score >= FUZZY_MIN:
                t = taxon(taxoniq, best)
                if t:
                    hit, how = t, "fuzzy_in_genus"
                    rec["fuzzy_score"] = round(score, 3)

        # 5 same epithet anywhere, then restricted to the parent's family
        if hit is None and tail and " " not in tail:
            pt = taxon(taxoniq, parent)
            fam = None
            if pt:
                fam = next((str(a.tax_id) for a in pt.ranked_lineage
                            if a.rank and a.rank.name == "family"), None)
            cands = by_epithet.get(tail.lower(), [])
            keep = []
            for c in cands:
                t = taxon(taxoniq, c)
                if not t:
                    continue
                if fam and fam in lineage_ids(t):
                    keep.append((c, t))
            rec["candidates"] = [c for c, _ in keep]
            if len(keep) == 1:
                hit, how = keep[0][1], "epithet_global"

        # 6 A GENUS RENAME USUALLY CHANGES THE EPITHET TOO, because the epithet
        #   must agree in gender with the new genus: Eubacterium rectale ->
        #   Agathobacter rectalis, Clostridium clostridioforme -> Enterocloster
        #   clostridioformis, Bacteroides plebeus -> Phocaeicola plebeius. An
        #   exact-epithet index cannot see any of those, and the new genus is
        #   often in a different family (Eubacteriaceae -> Lachnospiraceae), so
        #   family restriction cannot either.
        #
        #   So match on the epithet STEM across all of NCBI and report the top
        #   candidates WITHOUT auto-accepting them. Every hit here is a proposal
        #   for a human to confirm, never a resolution: a stem is weak evidence
        #   and a wrong genus reassignment would invent a false containment edge,
        #   which is the exact failure this whole exercise exists to remove.
        if hit is None and tail and " " not in tail and len(tail) >= 5:
            stem = tail.lower()[:5]
            scored = []
            for c in by_stem.get(stem, ()):
                ce = c.rsplit(" ", 1)[-1].lower()
                r = SequenceMatcher(None, tail.lower(), ce).ratio()
                if r >= 0.80:
                    scored.append((round(r, 3), c))
            scored.sort(reverse=True)
            rec["stem_candidates"] = [
                {"name": c, "epithet_similarity": r,
                 "genus": c.rsplit(" ", 1)[0]}
                for r, c in scored[:6]]

        if hit is not None:
            rec["how"] = how
            rec["resolved"] = describe(hit)
            rec["parent_in_lineage"] = parent_tid in lineage_ids(hit)
        records.append(rec)

    res = [r for r in records if r["resolved"]]
    summary = {
        "n_named_child": len(named),
        "n_resolved": len(res),
        "n_unresolved": len(named) - len(res),
        "by_how": {},
        "n_parent_in_lineage": sum(1 for r in res if r["parent_in_lineage"]),
        "n_parent_NOT_in_lineage": sum(1 for r in res if not r["parent_in_lineage"]),
        "n_nonspecies_tail": sum(1 for r in records if r["nonspecies_tail"]),
        "by_rank": {},
    }
    for r in res:
        summary["by_how"][r["how"]] = summary["by_how"].get(r["how"], 0) + 1
        rk = r["resolved"]["rank"]
        summary["by_rank"][rk] = summary["by_rank"].get(rk, 0) + 1

    json.dump({"summary": summary, "records": records}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}")

    print("\n--- resolved but parent NOT an ancestor (fold is taxonomically wrong) ---")
    for r in res:
        if not r["parent_in_lineage"]:
            print(f"  {r['surface']:38} folded into {r['parent']:20} "
                  f"but is {r['resolved']['scientific_name']} "
                  f"({r['resolved']['rank']}, {r['resolved']['taxid']}) [{r['how']}]")

    print("\n--- unresolved, with stem candidates for audit (NOT accepted) ---")
    for r in records:
        if r["resolved"]:
            continue
        flag = " [non-species tail]" if r["nonspecies_tail"] else ""
        print(f"  {r['surface']}{flag}")
        for c in r.get("stem_candidates", [])[:4]:
            print(f"        ? {c['name']}  (epithet sim {c['epithet_similarity']})")

    print("\n--- fuzzy_in_genus hits, for audit (weakest accepted strategy) ---")
    for r in res:
        if r["how"] == "fuzzy_in_genus":
            print(f"  {r['surface']:38} -> {r['resolved']['scientific_name']:42} "
                  f"score {r.get('fuzzy_score')}")


if __name__ == "__main__":
    main()
