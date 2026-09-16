#!/usr/bin/env python3
"""Orphan taxon nodes whose own label names their parent.

The gap
-------
170 of the graph's 883 taxon nodes have no parent containment link; 118 of those
are unresolved (no NCBI taxid, so `taxonomy.py` can give them no lineage). For
**24 of them the parent is written in the label itself** — `unclassified_f_`
`Lachnospiraceae`, `norank_f_Christensenellaceae`, `gut_metagenome_g_`
`Faecalibacterium` — and in each case that parent already exists as a resolved
node in this graph. They sit detached anyway, because the placeholder branch of
`norm_taxon` only fires for labels the taxonomy *did* resolve, and these resolve
to nothing at all.

This matters for the reason `build_kg.py` gives for modelling containment in the
first place: containment is not redundancy. In this corpus *Lachnospiraceae* is
depleted in 8 of 9 papers while *Hungatella* inside it is enriched in 6 of 7, and
that only survives if the nesting is represented. An orphan node is invisible to
that layer entirely.

Why this is a curated table and not a rule
------------------------------------------
Because substring matching gets it **wrong**, and the wrong answers are not
subtle. `Klebsiella virus KP36` contains the name of a genus it is emphatically
NOT contained in — a bacteriophage is a virus that *infects* Klebsiella. Four of
the 24 candidates are phages, and a generic "link to the genus named in the
label" rule would assert four false taxonomic relations. This is the same lesson
as `taxon_typos.py`, where edit distance would have merged `Oscillospirales`
into `Oscillospira`: **the refusals are the product**, so they are recorded here
with reasons rather than left implicit.

The joint two-genus labels are deliberately REFUSED too, and that is a finding
rather than a gap — see `REFUSED` below and `FINDINGS_disease_ontology.md`.

Output: `orphan_parents.json`, consumed by `build_kg.py`. Re-run after any
change to taxon normalisation.
"""
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "orphan_parents.json")

# ---------------------------------------------------------------------------
# LINK: the label names exactly one parent, the relation is a genuine is-a, and
# folding an "unclassified X" / "uncultured X" into X is the operation CLAUDE.md
# already calls CORRECT and explicitly not a rank collapse.
# ---------------------------------------------------------------------------
LINK = {
    "unclassified_f_Lachnospiraceae": ("Lachnospiraceae", "unclassified member of the family"),
    "Lachnospiraceae-unclassified": ("Lachnospiraceae", "same, suffix word order"),
    "uncultured Lachnospiraceae feature": ("Lachnospiraceae", "uncultured member of the family"),
    "norank_f_Christensenellaceae": ("Christensenellaceae", "SILVA 'no rank' child of the family"),
    "norank_f_Erysipelotrichaceae": ("Erysipelotrichaceae", "SILVA 'no rank' child of the family"),
    "norank_p_Flavobacteriaceae": ("Flavobacteriaceae", "SILVA 'no rank' child; the p_ prefix "
                                                        "is mislabelled in the source, the named taxon is a family"),
    "Enterobacteriaceae-unclassified": ("Enterobacteriaceae", "unclassified member of the family"),
    "unclassified_c_Bacilli": ("Bacilli", "unclassified member of the class"),
    "unclassified_g_Clostridium_sensu_stricto_1": (
        "Clostridium", "Clostridium sensu stricto 1 is an RDP/SILVA cluster inside Clostridium"),
    "uncultured Clostridium sp. 1": ("Clostridium", "uncultured species in the genus"),
    "gut_metagenome_g_Faecalibacterium": (
        "Faecalibacterium", "metagenome-assembled entry under the genus; the g_ tag names it"),
    "EC_Eubacterium_xylanophilumgroup": (
        "Eubacterium", "the [Eubacterium] xylanophilum group sits under Eubacterium"),
    "Ruminiclostridium-5": ("Ruminiclostridium", "numbered RDP cluster inside the genus"),
    "Ruminiclostridium-1": ("Ruminiclostridium", "numbered RDP cluster inside the genus"),
}

# ---------------------------------------------------------------------------
# REFUSED, with the reason. Do not re-propose these.
# ---------------------------------------------------------------------------
REFUSED = {
    # --- the four that prove a generic rule would be wrong ---
    "Klebsiella virus KP36":
        "a BACTERIOPHAGE. It infects Klebsiella; it is not contained in it. "
        "Linking it would assert a false taxonomic relation, and a query rolling "
        "up Klebsiella would absorb virus evidence.",
    "Streptococcus phage EJ 1": "bacteriophage, as above.",
    "Enterococcus phage EFAP 1": "bacteriophage, as above.",
    "Escherichia virus JES2013": "bacteriophage, as above.",

    # --- joint labels from an assay that cannot separate two taxa ---
    # This is the open modelling call, and containment is the WRONG primitive
    # for it. "Escherichia-Shigella" denotes 16S reads that could be EITHER
    # genus, so it is not a subset of Escherichia: rolling it up into
    # Escherichia would absorb evidence that may be Shigella. Refusing the link
    # is not a gap -- it sharpens the decision. Making the joint node reachable
    # from both parents WITHOUT asserting containment needs a different edge
    # type ("ambiguous assay", not "parent_of"), which is a schema decision for
    # a human. It is now the last genuinely open modelling call in the graph.
    "Escherichia-Shigella":
        "joint two-genus 16S label (10 edges, the largest orphan). NOT contained "
        "in either genus -- see the note above. Needs an 'ambiguous assay' edge "
        "type, not parent_of. PI decision.",
    "unclassified_g__Escherichia-Shigella":
        "same, and its natural parent is the joint node, which is itself refused.",
    "Bacteroides-Prevotella": "joint two-genus label, as above.",
    "Hafnia-Obesumbacterium":
        "joint two-genus label; Obesumbacterium is not even a node here, so only "
        "half the join is representable.",
    "Lachnospiraceae_Eubacterium":
        "the SILVA 'Lachnospiraceae [Eubacterium] group' label. Family + a "
        "bracketed misplaced genus, not a clean is-a, and multi_taxon.py already "
        "holds it as a joint concept deliberately.",

    # --- parent not representable ---
    "unclassified genus belonging to Barnesiellaceae and Lachnospiraceae":
        "names TWO families; the paper itself could not place it. Ambiguous.",
    "unclassified genus belonging to order SHA-98":
        "SHA-98 is not a node in this graph, so there is no parent to link to.",
    "Unclassified Absconditabacteriales":
        "Absconditabacteriales is not a node in this graph.",
    "norank_p_Parcubacteria":
        "Parcubacteria is not a node in this graph.",
}


def sep_key(s):
    """Mirror build_kg's separator collapse so keys match node ids."""
    return re.sub(r"[\s\-/_–—]+", " ", (s or "").lower()).strip()


def main():
    g = json.load(open(GRAPH))
    tn = [n for n in g["nodes"] if n["id"].startswith("t:")]
    by_label = {n["label"]: n for n in tn}
    par = defaultdict(list)
    for h in g["hierarchy"]:
        par[h["child"]].append(h["parent"])

    orphans = [n for n in tn if not par.get(n["id"])]
    unres = [n for n in orphans if not n.get("resolved")]
    print(f"taxon nodes {len(tn)}; orphans (no parent link) {len(orphans)}; "
          f"of those unresolved {len(unres)}")

    links, problems = [], []
    for child_label, (parent_label, why) in LINK.items():
        c = by_label.get(child_label)
        p = by_label.get(parent_label)
        if not c:
            problems.append(f"LINK child not a node: {child_label!r}")
            continue
        if not p:
            problems.append(f"LINK parent not a node: {parent_label!r}")
            continue
        if par.get(c["id"]):
            problems.append(f"{child_label!r} already has a parent link; drop it from LINK")
            continue
        links.append({
            "child_label": child_label, "child_id": c["id"],
            "child_key": sep_key(child_label),
            "parent_label": parent_label, "parent_id": p["id"],
            "parent_taxid": p.get("taxid"),
            "parent_rank": p.get("rank"), "n_edges": c.get("degree"),
            "why": why,
        })

    for r in REFUSED:
        if r not in by_label:
            problems.append(f"REFUSED entry is not a node (stale?): {r!r}")

    covered = set(LINK) | set(REFUSED)
    uncovered = [n["label"] for n in unres if n["label"] not in covered]
    print(f"\nLINK   {len(links)} orphans -> a named parent "
          f"({sum(l['n_edges'] or 0 for l in links)} edges gain containment)")
    for l in sorted(links, key=lambda x: -(x["n_edges"] or 0)):
        print(f"    {l['child_label'][:46]:48s} -> {l['parent_label']:22s} "
              f"({l['parent_rank']}, {l['n_edges']}e)")
    print(f"\nREFUSE {len(REFUSED)}, with reasons recorded:")
    for k, v in REFUSED.items():
        print(f"    {k[:46]:48s} {v.split('.')[0]}")

    if problems:
        print("\n!! PROBLEMS -- the table disagrees with the graph:")
        for p in problems:
            print("   ", p)

    print(f"\n{len(uncovered)} unresolved orphans are not in either table "
          f"(no parent recoverable from the label; left detached, correctly)")

    json.dump({"_note": "Curated parents for orphan taxon nodes whose label names "
                        "their parent. The REFUSALS are part of the product: a "
                        "generic substring rule would link four bacteriophages "
                        "into the genus they infect. See orphan_parents.py.",
               "links": links, "refused": REFUSED,
               "n_orphans": len(orphans), "n_unresolved_orphans": len(unres),
               "n_uncovered": len(uncovered), "problems": problems},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
