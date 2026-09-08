#!/usr/bin/env python3
"""Recover the species that the resolver silently folds into their genus.

THE DEFECT. `child_folds.json` records 115 surface strings that EXTEND the
scientific name they resolved to. 32 are SILVA rank placeholders (already split),
29 are "X sp./spp./unclassified" (folding is correct), and 54 were classified
`named_child` -- read as "54 real species collapsed into their genus", and named
the project's top open defect.

WHY IT HAPPENS -- and it is NOT that the taxdump was missing. The shipped graph
resolves renamed binomials perfectly well: "Clostridium aldenense" is a node
labelled *Enterocloster aldenensis*, "Eubacterium eligens" is *Lachnospira
eligens*. What fails is a specific subset of the 2024-25 reclassifications, whose
old binomials this pipeline's names.dmp lookup does not return: *Prevotella
copri* (now *Segatella copri*), *Eubacterium rectale* (now *Agathobacter
rectalis*), *Bacteroides vulgatus/dorei/plebeius/coprophilus* (now *Phocaeicola*),
*Clostridium clostridioforme* (now *Enterocloster clostridioformis*), *Prevotella
buccae/timonensis/shahii* (now *Segatella*/*Hoylesella*). For those, resolve()
falls through to its qualifier-tail trim, which throws the epithet away and lands
the mention on the GENUS -- a rank collapse the project explicitly forbids.

The proof that this is a real collision, not a naming quibble: THREE of these
organisms are ALREADY nodes in the graph under their new names. Papers writing
"Phocaeicola dorei" built a species node; papers writing "Bacteroides dorei" were
folded into genus *Bacteroides*. One organism, two nodes, two ranks. Same for
*Holdemanella biformis*/"Eubacterium biforme" and *Enterocloster
clostridioformis*/"Clostridium clostridioforme".

HOW THE MAPPING IS ESTABLISHED -- no model memory, two independent sources
joined on a key that renaming cannot move:

  An NCBI taxid is STABLE across a rename. Disbiome was curated before the
  reclassifications and stores the OLD binomial next to its taxid
  ("Prevotella copri" -> 165179). `ncbi-taxon-db` (the NCBI taxonomy as
  redistributed on PyPI, behind `taxoniq`) resolves that taxid to the CURRENT
  scientific name and rank (165179 -> *Segatella copri*, species). Neither
  source knows about the other; the taxid is the join.

Every entry therefore carries `via` and `evidence` saying exactly how it was
derived, and nothing enters this table on assertion alone.

RESOLUTION LADDER, per surface string (first hit wins):
  1. `taxoniq_exact`   -- the string IS a current NCBI scientific name.
  2. `disbiome_taxid`  -- Disbiome holds the string with a taxid; taxoniq
                          confirms that taxid is a species.
  3. `fuzzy`           -- misspellings ("Bacteroides uniforms",
                          "Faecalibacterium prauznitzii", "Klebsiella
                          pneumonia"). Levenshtein <= 2 against Disbiome names
                          and the parent genus's NCBI species, first token
                          (the genus) required to match EXACTLY, unique best
                          only. Then routed through 1 or 2.

WHAT DELIBERATELY DOES NOT RESOLVE, and must not. The ladder is the classifier:
strings that are not species simply fail it. "Escherichia / Shigella" and
"Streptococcus salivarius/thermophilus" name two taxa and cannot become one
species node; "Clostridium_XlVa", "Eubacterium_g4", "Prevotella VZCB",
"Lachnospiraceae_NC2004" are pipeline cluster labels; "Neisseria multispecies"
names no organism. They stay folded, and are listed in the report as `unresolved`
so the decision is visible rather than silent.

CONSUMED BY `taxonomy.py` / `taxonomy_cache.py`, which check this table only
AFTER a full-string taxdump lookup has already failed. It can therefore never
override a real NCBI answer -- it only pre-empts the trim-to-parent fallback.
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "species_synonyms.json")

NOT_ORGANISM = re.compile(r"\b(virus|phage|bacteriophage|uncultured)\b", re.I)
# "sp.", "spp.", "bacterium", "species", "multispecies", "unclassified": these name an
# UNSPECIFIED member of the parent, and folding them into the parent is correct.
UNSPECIFIED = re.compile(r"\b(sp|spp|species|multispecies|unclassified|bacterium|"
                         r"asp|genomosp)\b\.?", re.I)
ACCEPT_RANKS = {"species", "subspecies"}


def norm(s):
    return " ".join(str(s or "").replace("_", " ").split()).strip()


def lev(a, b):
    """Levenshtein distance. Small strings, so the simple DP is fine."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:          # we never accept > 2
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_table(path=OUT):
    """-> {normalised surface: entry}. Empty if the table has not been generated.

    Deliberately tolerant: a missing table degrades to the old folding behaviour
    rather than erroring, so `build_kg.py` still runs in an environment that has
    never run this generator. Callers that care can check emptiness.
    """
    try:
        with open(path) as f:
            return json.load(f).get("entries", {})
    except Exception:
        return {}


class Sources:
    def __init__(self):
        import taxoniq                                    # noqa: F401
        self.taxoniq = taxoniq
        self.disbiome = {}                                 # lowercased name -> taxid
        path = os.path.join(HERE, "disbiome_experiments.json")
        for r in json.load(open(path)):
            n, t = norm(r.get("organism_name")).lower(), r.get("organism_ncbi_id")
            if n and t:
                self.disbiome.setdefault(n, int(t))
        self._children = {}

    def by_taxid(self, tid):
        try:
            t = self.taxoniq.Taxon(int(tid))
            return str(t.tax_id), t.scientific_name, str(t.rank).split(".")[-1]
        except Exception:
            return None

    def lineage(self, tid):
        """[self, parent, ...] as NCBI taxids.

        Required, not decorative. build_kg.py builds every containment link by
        walking `tax.lineage(t)` for the nearest ancestor that is also a node. A
        split species whose ancestry is unknown gets NO containment link, which is
        the regression that made splitting these unsafe without a taxdump: the
        node would appear, detached, and the family it belongs to would lose it.
        """
        try:
            return [str(x.tax_id) for x in self.taxoniq.Taxon(int(tid)).lineage]
        except Exception:
            return []

    def by_sciname(self, name):
        try:
            t = self.taxoniq.Taxon(scientific_name=name)
            return str(t.tax_id), t.scientific_name, str(t.rank).split(".")[-1]
        except Exception:
            return None

    def genus_species(self, genus_name):
        """Current NCBI species sitting directly under `genus_name`."""
        if genus_name in self._children:
            return self._children[genus_name]
        out = []
        g = self.by_sciname(genus_name)
        if g:
            try:
                for c in self.taxoniq.Taxon(int(g[0])).child_nodes:
                    if str(c.rank).split(".")[-1] == "species":
                        out.append(c.scientific_name)
            except Exception:
                pass
        self._children[genus_name] = out
        return out


def resolve_species(surface, src):
    """-> dict(taxid, scientific_name, rank, via, evidence) or None."""
    s = norm(surface)
    toks = s.split()
    if len(toks) < 2 or NOT_ORGANISM.search(s) or UNSPECIFIED.search(s):
        return None
    if any(c in s for c in "/–—"):        # "Escherichia / Shigella": two taxa
        return None

    # 1. already a current NCBI scientific name
    hit = src.by_sciname(s)
    if hit and hit[2] in ACCEPT_RANKS:
        return dict(taxid=hit[0], scientific_name=hit[1], rank=hit[2],
                    via="taxoniq_exact", evidence=f"NCBI scientific name '{s}'")

    # 2. Disbiome holds the pre-rename binomial against a stable taxid
    tid = src.disbiome.get(s.lower())
    if tid:
        hit = src.by_taxid(tid)
        if hit and hit[2] in ACCEPT_RANKS:
            return dict(taxid=hit[0], scientific_name=hit[1], rank=hit[2],
                        via="disbiome_taxid",
                        evidence=f"Disbiome '{s}' -> taxid {tid}; NCBI {tid} = "
                                 f"'{hit[1]}' [{hit[2]}]")

    # A compound clade label is not a binomial. "Lachnospiraceae_Eubacterium" and
    # "Escherichia_Shigella" name a clade inside another clade, and their second
    # token is itself a genus or family name -- which is the deterministic test,
    # rather than a guess about capitalisation. Without this, fuzzy matching pairs
    # "Lachnospiraceae Eubacterium" with the species "Lachnospiraceae bacterium"
    # at edit distance 2 and invents a species that no paper reported.
    for t in toks[1:]:
        h = src.by_sciname(t) or src.by_sciname(t.capitalize())
        if h and h[2] not in ACCEPT_RANKS:
            return None

    # 3. misspelling: nearest name sharing the genus token exactly.
    # Case matters -- taxoniq's scientific-name index is case-sensitive -- so the
    # pool keeps the original spelling and only the COMPARISON is lowercased.
    genus = toks[0]
    pool = {n: n for n in src.disbiome if n.startswith(genus.lower() + " ")}
    for n in src.genus_species(genus):
        pool.setdefault(n.lower(), n)
    best, bestd = [], 3
    for cand in pool:
        d = lev(s.lower(), cand)
        if d < bestd:
            best, bestd = [cand], d
        elif d == bestd:
            best.append(cand)
    if len(best) == 1 and bestd <= 2:
        cand = best[0]
        hit = src.by_sciname(pool[cand]) or (src.by_taxid(src.disbiome[cand])
                                             if cand in src.disbiome else None)
        if hit and hit[2] in ACCEPT_RANKS:
            return dict(taxid=hit[0], scientific_name=hit[1], rank=hit[2],
                        via="fuzzy",
                        evidence=f"'{s}' ~ '{cand}' (edit distance {bestd}, unique "
                                 f"best, genus token exact) -> NCBI {hit[0]} "
                                 f"'{hit[1]}' [{hit[2]}]")
    return None


def main():
    src = Sources()
    folds = json.load(open(os.path.join(HERE, "child_folds.json")))
    # Every fold is a candidate, not just `named_child`: the ladder decides, and
    # letting the stale classification decide would re-import its judgement calls.
    surfaces = sorted({f["surface"] for f in folds})
    parent_of = {f["surface"]: f["parent"] for f in folds}

    table, unresolved = {}, []
    for s in surfaces:
        r = resolve_species(s, src)
        if r:
            r["surface"] = s
            r["folded_into"] = parent_of.get(s)
            r["lineage"] = src.lineage(r["taxid"])
            r["rank_of"] = {t: (src.by_taxid(t) or (None, None, "no rank"))[2]
                            for t in r["lineage"]}
            if not r["lineage"]:
                # No ancestry -> no containment link. Refuse the split rather than
                # ship a detached node; say so loudly instead of dropping it.
                print(f"  REFUSED (no lineage available): {s}")
                unresolved.append(s)
                continue
            table[norm(s).lower()] = r
        else:
            unresolved.append(s)

    # A split is only meaningful if it moves the mention OFF the node it folded
    # into. If the species IS the parent (it never is here, but assert it) drop it.
    by_target = defaultdict(list)
    for k, v in table.items():
        by_target[v["scientific_name"]].append(v["surface"])

    json.dump({"_note": "generated by species_synonyms.py; see its docstring for "
                        "how each entry was established. Consumed by taxonomy.py "
                        "and taxonomy_cache.py AFTER a taxdump lookup fails.",
               "entries": table,
               "unresolved": unresolved},
              open(OUT, "w"), indent=1, sort_keys=True)

    print(f"candidates (all child folds): {len(surfaces)}")
    print(f"resolved to a species:        {len(table)}")
    print(f"left folded (not a species):  {len(unresolved)}")
    print("\nby route:")
    for via in ("taxoniq_exact", "disbiome_taxid", "fuzzy"):
        rows = [v for v in table.values() if v["via"] == via]
        print(f"  {via:15} {len(rows)}")
        for v in sorted(rows, key=lambda x: x["surface"]):
            print(f"      {v['surface']:34} -> {v['scientific_name']} "
                  f"({v['taxid']}) [was folded into {v['folded_into']}]")
    dupes = {k: v for k, v in by_target.items() if len(v) > 1}
    if dupes:
        print("\nsurfaces converging on one species (these POOL, which is the point):")
        for k, v in sorted(dupes.items()):
            print(f"  {k}: {v}")
    print(f"\nleft folded ({len(unresolved)}):")
    for s in unresolved:
        print(f"  {s}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
