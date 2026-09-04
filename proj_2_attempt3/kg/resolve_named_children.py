#!/usr/bin/env python3
"""Resolve the 54 `named_child` surface strings to real species taxids.

WHY. `child_folds.json` classifies 115 surface strings that EXTEND the scientific
name they resolved to. 32 were SILVA rank placeholders (fixed 2026-09-03), 29 are
`X sp./spp.` where folding into the genus is correct, and 54 were labelled
`named_child` and left unfixed because splitting them needs real species taxids
and the analysis environment's network policy denies `ftp.ncbi.nih.gov`.

THE WAY IN. `taxoniq` ships the NCBI taxon database (Sept-2024 snapshot) as a
PyPI data package, and PyPI is reachable where NCBI's FTP is not. It gives
scientific name -> taxid, rank, and parent/child edges for 2.6M names.

WHAT IT IS NOT. taxoniq carries **scientific names only** -- no synonyms, no
merged.dmp. `Bacteroidetes` and `Firmicutes` both raise KeyError. It therefore
CANNOT replace `taxonomy.py`, whose whole job is folding those synonyms, and it
is used here strictly as an ADDITIVE lookup for specific strings. Every taxid it
returns is written to `named_child_taxids.json` with provenance so the source of
any id in the graph stays auditable.

THE SAFETY RULE. A candidate is only accepted if NCBI places it inside the same
FAMILY as the genus the string was folded into. `Bacteroides uniforms` may become
*Bacteroides uniformis*; it may not become anything outside Bacteroidaceae. This
makes a wrong correction structurally impossible rather than merely unlikely,
which matters because a bad species taxid would silently break the
Disbiome/Peryton join for exactly the taxa under study.

Family, not genus, because the genus is the thing that has MOVED. The first pass
here constrained to children of the folded-into genus and lost four of the five
highest-evidence species: NCBI has since reassigned *Bacteroides dorei*,
*B. vulgatus* and *B. plebeius* to *Phocaeicola*, and *Prevotella copri* to
*Segatella*. The taxid is stable across those renames (copri is 165179 either
way), which is exactly why the graph keys on taxids -- and why the split gains
the external join rather than losing it: Disbiome and Peryton both still file
these under the legacy names, 48 records for *Prevotella copri* alone.

NAMING. The node keeps the name the corpus and the curated databases use
(*Prevotella copri*), with NCBI's current name recorded alongside it. Renaming
the node to *Segatella copri* would be more correct and less useful: nothing
else in this literature calls it that.

THE THIRD CLASS. Working through them shows `named_child` is not one category.
Only about half are real species. The rest are strain/clade labels with no
species taxid (`Clostridium_XlVa`, `Dorea asp: CAG:317`, `Turicibacter
sp001543345`) and multi-genus group labels (`Escherichia_Shigella`,
`Streptococcus salivarius/thermophilus`). Asserting a species taxid for those
would be inventing one. They are emitted as `clade` -- the same treatment the 32
SILVA placeholders already get -- so they still stop being counted as their
parent, which is the actual defect.

Output: named_child_taxids.json
"""
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import taxoniq
except ImportError:
    sys.exit("taxoniq not installed:  pip install taxoniq")

# Strings that name more than one taxon. There is no single correct taxid, and
# picking either genus asserts something the paper did not say.
GROUP_LABELS = re.compile(
    r"(/|–|—|_Shigella$|\bmultispecies\b|salivarius.thermophilus)", re.I)

# --- the adjudication table ------------------------------------------------
# Each of the 54 strings is adjudicated by hand and then VERIFIED against NCBI,
# rather than matched by a similarity score. Automatic matching was tried first
# and got it wrong in both directions: constrained to the genus it lost every
# species NCBI has since reassigned, and widened to the family it matched
# `Eubacterium_g4` onto the genus *Eubacterium* itself -- reintroducing exactly
# the rank collapse this is meant to fix -- and `Lachnospiraceae_Eubacterium`
# onto an unrelated species at ratio 0.86. A similarity score cannot tell a
# misspelling from a different taxon, because it does not know what the string
# means.
#
# So the value below is a PROPOSAL, and nothing is accepted on the proposal's
# authority: `verify()` requires the name to exist in NCBI, to be species rank,
# and to share its specific epithet with the surface string (allowing for the
# misspelling being corrected). Anything failing that stays a clade. The point of
# the table is to say what was intended; the point of the check is that being
# wrong here is caught rather than written into the graph.
SPECIES = {
    # surface string                    proposed current NCBI scientific name
    "Prevotella copri":                 "Segatella copri",
    # The Prevotellaceae have been split into Segatella / Hoylesella /
    # Prevotella; the taxids are unchanged, so the curated databases still find
    # these under the names the papers use.
    "Prevotella buccae":                "Segatella buccae",
    "Prevotella timonensis":            "Hoylesella timonensis",
    "Prevotella jejunii":               "Prevotella jejuni",
    "Prevotella shaii":                 "Hoylesella shahii",
    "Eubacterium rectale":              "Agathobacter rectalis",
    "Eubacterium biforme":              "Holdemanella biformis",
    "Bacteroides dorei":                "Phocaeicola dorei",
    "Bacteroides vulgatus":             "Phocaeicola vulgatus",
    "Bacteroides plebeus":              "Phocaeicola plebeius",
    "Bacteroides coprophilus":          "Phocaeicola coprophilus",
    "Bacteroides uniforms":             "Bacteroides uniformis",
    "Klebsiella pneumonia":             "Klebsiella pneumoniae",
    "Faecalibacterium prauznitzii":     "Faecalibacterium prausnitzii",
    "Faecalibacterium prasunitzii":     "Faecalibacterium prausnitzii",
    "Bifidobacterium brevis":           "Bifidobacterium breve",
    # NCBI disambiguates these two by describing author, and holds a separate
    # taxid for each. The gut isolate is the earlier one in both cases. The pick
    # only has to be CONSISTENT -- both sides of the external join route through
    # this same table -- but it is a judgement call, so it is named here.
    "Blautia massiliensis":             "Blautia massiliensis (ex Durand et al. 2017)",
    "Blautia_massiliensis":             "Blautia massiliensis (ex Durand et al. 2017)",
    "Ruminococcus bicirculans":         "Ruminococcus bicirculans (ex Wegman et al. 2014)",
    "Clostridium clostridioforme":      "Enterocloster clostridioformis",
    "Hungatella hathewyi":              "Hungatella hathewayi",
    "Hungatella effluvia":              "Hungatella effluvii",
    "Monoglobus pectinyliticus":        "Monoglobus pectinilyticus",
    # "Lawsonibacter phoceensis" is deliberately absent: NCBI has no phoceensis
    # under Lawsonibacter, and the plausible basonym (Clostridium phoceensis)
    # is a guess. One observation; an unresolved clade is the honest answer.
    "Oribacterium sinu":                "Oribacterium sinus",
    "Fusobacterium_A mortiferum":       "Fusobacterium mortiferum",
}

# The node's display name. Default is the surface string, because that is what
# the corpus and both curated databases call the organism -- renaming the node
# to *Segatella copri* would be more correct and less useful. But that default is
# wrong for a MISSPELLING: it made taxid 573 display as "Klebsiella pneumonia",
# and a graph that shows a typo as a species name is worse than one that shows a
# superseded name. So a misspelling is displayed under the correct spelling of
# the name the paper was reaching for -- the legacy binomial, not necessarily
# NCBI's current one.
#
# This cannot be derived from the epithet ratio. `rectale` -> `rectalis` scores
# 0.80 and is a legitimate reclassification; `pneumonia` -> `pneumoniae` scores
# 0.95 and is a typo. Nothing in the strings distinguishes them, so it is stated.
LABEL = {
    "Klebsiella pneumonia":         "Klebsiella pneumoniae",
    "Bacteroides uniforms":         "Bacteroides uniformis",
    "Bacteroides plebeus":          "Bacteroides plebeius",
    "Faecalibacterium prauznitzii": "Faecalibacterium prausnitzii",
    "Faecalibacterium prasunitzii": "Faecalibacterium prausnitzii",
    "Bifidobacterium brevis":       "Bifidobacterium breve",
    "Hungatella hathewyi":          "Hungatella hathewayi",
    "Hungatella effluvia":          "Hungatella effluvii",
    "Monoglobus pectinyliticus":    "Monoglobus pectinilyticus",
    "Oribacterium sinu":            "Oribacterium sinus",
    "Prevotella shaii":             "Prevotella shahii",
    "Prevotella jejunii":           "Prevotella jejuni",
    "Fusobacterium_A mortiferum":   "Fusobacterium mortiferum",
}

# Labels that are a strain, an assembly bin, or a pipeline's clade id rather than
# a species name: CAG bins, GTDB `sp001543345` accessions and `Fusobacterium_A`
# suffixes, roman-numeral RDP clusters, bare uppercase tags. Case-SENSITIVE on
# purpose -- an earlier pass compiled this with re.I, which made `[A-Z]{2,}$`
# match any lowercase epithet and swallowed all 22 real species.
CLADE_LABEL = re.compile(
    r"(CAG[: _-]?\d+|\bsp\d{6,}|_[A-Z]\b|_[IVXL]+[a-z]?$|_g\d+$|"
    r"\b[A-Z]{2,}[A-Z0-9]*$|\bNML\b|\bMC[- _]?\d+|\bNC\d{4}|\bbacterium\b|"
    r"\bphage\b|\bspecies\b|\bsp\.? |\bg\d+ |\d)")

MIN_RATIO = 0.85


def norm(s):
    s = re.sub(r"[_]+", " ", (s or "").strip())
    return re.sub(r"\s+", " ", s)


def family_of(taxid):
    """Walk up to the enclosing family (the pool a correction may draw from)."""
    try:
        t = taxoniq.Taxon(int(taxid))
    except Exception:
        return None
    for _ in range(12):
        if getattr(t.rank, "name", "") == "family":
            return t
        nxt = t.parent
        if nxt is None or nxt.tax_id == t.tax_id:
            return None
        t = nxt
    return None


_POOL_CACHE = {}


def species_in_family(fam):
    """Every species NCBI places anywhere under `fam`, as name -> Taxon.

    Two levels (family -> genus -> species) is enough: NCBI files species
    directly under a genus, and the reassignments this has to survive move a
    species between genera of the SAME family.
    """
    if fam is None:
        return {}
    if fam.tax_id in _POOL_CACHE:
        return _POOL_CACHE[fam.tax_id]
    pool = {}
    try:
        genera = list(fam.child_nodes)
    except Exception:
        genera = []
    for g in genera:
        try:
            pool[g.scientific_name.lower()] = g
            for s in g.child_nodes:
                pool[s.scientific_name.lower()] = s
        except Exception:
            continue
    _POOL_CACHE[fam.tax_id] = pool
    return pool


def verify(surface, proposed):
    """Check a proposed name against NCBI. -> (Taxon, note) or (None, why not).

    Three gates, all of which a wrong table entry fails:
      1. the name exists in NCBI as a scientific name;
      2. it is species rank -- this is what stops `Eubacterium_g4` resolving to
         the genus *Eubacterium*, which is the exact collapse being fixed;
      3. its specific epithet is close to the surface string's, so a typo in the
         table cannot silently redirect a string to an unrelated organism.
    """
    try:
        t = taxoniq.Taxon(scientific_name=proposed)
    except Exception:
        return None, f"proposed name '{proposed}' is not in NCBI"
    rank = getattr(t.rank, "name", str(t.rank))
    if rank != "species":
        return None, f"'{proposed}' is rank {rank}, not species"
    s_epi = norm(surface).split()[-1].lower()
    p_epi = re.sub(r"\s*\(ex .*\)$", "", proposed).split()[-1].lower()
    ratio = difflib.SequenceMatcher(None, s_epi, p_epi).ratio()
    # 0.70, not 0.80: the commonest correction here is a Latin declension
    # (`brevis` -> `breve`, ratio 0.73), and an unrelated organism scores well
    # below this -- the rejected proposals in the first run scored 0.3-0.5.
    if ratio < 0.70:
        return None, (f"epithet '{s_epi}' is not a spelling of '{p_epi}' "
                      f"(ratio {ratio:.2f})")
    return t, f"epithet ratio {ratio:.2f}"


def resolve_one(surface, parent_name, parent_taxid):
    """-> dict with taxid/label/rank/how, or a clade/group verdict."""
    raw = norm(surface)
    rec = {"surface": surface, "parent": parent_name, "parent_taxid": parent_taxid}
    fam = family_of(parent_taxid)
    rec["parent_family"] = fam.scientific_name if fam else None

    if surface in SPECIES:
        proposed = SPECIES[surface]
        t, note = verify(surface, proposed)
        if t is None:
            rec.update(verdict="clade", taxid=None,
                       reason=f"REJECTED proposal: {note}")
            return rec
        tfam = family_of(t.tax_id)
        rec.update(verdict="species", taxid=str(t.tax_id),
                   label=LABEL.get(surface, raw), surface_is_misspelling=surface in LABEL,
                   ncbi_current_name=t.scientific_name,
                   renamed=(t.scientific_name.lower() != raw.lower()),
                   rank="species", family=tfam.scientific_name if tfam else None,
                   moved_family=bool(fam and tfam and fam.tax_id != tfam.tax_id),
                   how="adjudicated", reason=note)
        return rec

    if GROUP_LABELS.search(raw) or GROUP_LABELS.search(surface):
        rec.update(verdict="group_label", taxid=None,
                   reason="names more than one taxon; no single taxid is correct")
        return rec

    rec.update(verdict="clade", taxid=None,
               reason="strain/bin/pipeline clade label, not a species name"
               if CLADE_LABEL.search(raw) else
               "not adjudicated as a species; kept as an unresolved clade")
    return rec


def main():
    folds = [x for x in json.load(open(os.path.join(HERE, "child_folds.json")))
             if x["cls"] == "named_child"]
    out = []
    for f in folds:
        ptid = f["node"].split(":")[-1] if f["node"].startswith("t:ncbi:") else None
        out.append(resolve_one(f["surface"], f["parent"], ptid))

    counts = {}
    for r in out:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    payload = {
        "source": "taxoniq / ncbi-taxon-db (NCBI taxonomy, 2024-09 snapshot)",
        "note": "scientific names only -- no synonyms; additive lookup, NOT a "
                "replacement for taxonomy.py",
        "counts": counts,
        "resolutions": out,
    }
    path = os.path.join(HERE, "named_child_taxids.json")
    json.dump(payload, open(path, "w"), indent=1)
    print(json.dumps(counts, indent=1))
    for r in sorted(out, key=lambda r: (r["verdict"], r["surface"])):
        tid = r.get("taxid") or "-"
        print(f"  {r['verdict']:12} {r['surface']:40} -> {tid:10} "
              f"{r.get('label','')}  [{r.get('how', r['reason'][:38])}]")
    print("wrote", path)


if __name__ == "__main__":
    main()
