#!/usr/bin/env python3
"""Repair the genus-abbreviation misattributions in relation_sentences.json.

THE DEFECT (found 2026-09-12, fixed at source in relation_sentences.py the same
day). `filter_paper` built a per-paper map from a single INITIAL LETTER to a
genus with `alias.setdefault(sci[0].upper(), sci)` -- first genus seen wins, no
collision check. In a paper naming Bacteroides, Bifidobacterium and Blautia,
every later "B. <epithet>" was credited to Blautia. And `TaxonMatcher.find`
compounded it: when "<that genus> <epithet>" failed to resolve it fell back to
`self._lookup(g)`, crediting the bare genus anyway -- so a failed expansion still
produced a confident, wrong hit rather than no hit.

Worked example, and the one that surfaced it: an oral-vs-gut Alzheimer's paper
files *P. gingivalis* -- Porphyromonas gingivalis, the periodontal pathogen the
paper is largely about -- under **Phascolarctobacterium**, a gut genus, because
Phascolarctobacterium was the first P-genus the matcher saw.

Scope: 774 of 18,436 taxon mentions came from the abbreviation path; 362 of those
sit in a paper where two or more genera share the initial. 78 of 348 papers are
affected. The KNOWLEDGE GRAPH IS NOT AFFECTED -- `build_kg.py` never reads this
file; it builds from `extractions_screened.json`. What is affected is everything
that treats these sentences as the corpus substrate: `cooccur_direction.py` and
`cooccur_diagnostics.py` (the FINDINGS_cooccurrence.md null), the adjudication
packets, and the RAG chunks.

WHY REPAIR RATHER THAN REBUILD. Rebuilding needs `taxonomy.py`, which needs the
NCBI taxdump, and `ftp.ncbi.nih.gov` is blocked in this environment (probed in
six sessions). But the repair does not need it: when a paper says "B.
adolescentis" and ALSO names "Bifidobacterium" in full somewhere, that full
mention already carries its resolved taxid in this very file. So the correct
taxid is recoverable offline from the paper's own other mentions.

Three outcomes per ambiguous mention:
  REASSIGN -- the true genus is named in full elsewhere in the same paper, so its
              taxid is known; rewrite the mention to it.
  KEEP     -- the assigned genus was already the true one (the clash was
              harmless for this mention).
  DROP     -- the true genus is unknown, or is not named in full in this paper,
              so no taxid is available. A mention we cannot attribute is worse
              than no mention.

The surface -> true-genus table is CURATED (`ABBREV_TRUE`), not derived by edit
distance and not taken on an agent's word: every entry is a binomial whose genus
is settled nomenclature. Entries deliberately refused are listed in REFUSED with
the reason, the same convention `taxon_typos.py` uses.

TWO GUARDS, both added after the first run of this script tried to corrupt data
it was written to repair -- the table is an instrument and was briefly weaker
than the thing it audited:

  1. *A species-rank resolution is never overridden.* "L. salivarius" had already
     resolved to **Ligilactobacillus salivarius** and "R. hominis" to **Roseburia
     hominis** -- full binomials, strictly better evidence than any epithet
     table. A first draft keyed on the epithet alone would have rewritten them to
     *Streptococcus* and *Dialister*, because `salivarius` and `hominis` are
     epithets several genera share. Species-rank mentions are now skipped.
  2. *The candidate genus must start with the mention's own initial.* That is the
     one piece of evidence the abbreviation itself carries, and it is what keeps
     a shared epithet from crossing genera: `salivarius` may be Streptococcus or
     Ligilactobacillus, but "L." settles it.

Each epithet therefore maps to a LIST of acceptable genera, not one, so that a
renamed taxon matches under either name (*Eubacterium rectale* and *Agathobacter
rectalis* are the same organism; a paper writing "E. rectale" and naming
*Eubacterium* in full is not an error to be "corrected" to Agathobacter).

Writes relation_sentences_clean.json and clean_abbrev.json.
"""
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "relation_sentences.json")
OUT = os.path.join(HERE, "relation_sentences_clean.json")
REPORT = os.path.join(HERE, "clean_abbrev.json")

ABBREV = re.compile(r"^([A-Z])\.\s*([a-z][a-z\-]{2,})$")

# Curated: species epithet -> the genus/genera the binomial may belong to.
# A LIST, so a renamed organism matches under either name. Reassignment also
# requires the genus to start with the mention's own initial (see module docstring).
ABBREV_TRUE = {
    "adolescentis": ["Bifidobacterium"],
    "animalis": ["Bifidobacterium"],
    "bifidum": ["Bifidobacterium"],
    "breve": ["Bifidobacterium"],
    "longum": ["Bifidobacterium"],
    "catenulatum": ["Bifidobacterium"],
    "dentium": ["Bifidobacterium"],
    "pseudocatenulatum": ["Bifidobacterium"],
    "fragilis": ["Bacteroides"],
    "vulgatus": ["Bacteroides", "Phocaeicola"],
    "uniformis": ["Bacteroides"],
    "ovatus": ["Bacteroides"],
    "thetaiotaomicron": ["Bacteroides"],
    "xylanisolvens": ["Bacteroides"],
    "dorei": ["Bacteroides", "Phocaeicola"],
    "caccae": ["Bacteroides"],
    "eggerthii": ["Bacteroides"],
    "stercoris": ["Bacteroides"],
    "plebeius": ["Bacteroides", "Phocaeicola"],
    "muciniphila": ["Akkermansia"],
    "glycaniphila": ["Akkermansia"],
    "prausnitzii": ["Faecalibacterium"],
    "gingivalis": ["Porphyromonas"],
    "copri": ["Prevotella", "Segatella"],
    "melaninogenica": ["Prevotella"],
    "nigrescens": ["Prevotella"],
    "difficile": ["Clostridioides", "Clostridium"],
    "perfringens": ["Clostridium"],
    "butyricum": ["Clostridium"],
    "ramosum": ["Clostridium", "Thomasclavelia", "Erysipelatoclostridium"],
    "leptum": ["Clostridium"],
    "coli": ["Escherichia"],
    "pylori": ["Helicobacter"],
    "aureus": ["Staphylococcus"],
    "epidermidis": ["Staphylococcus"],
    "mutans": ["Streptococcus"],
    "sanguinis": ["Streptococcus"],
    "mitis": ["Streptococcus"],
    "gordonii": ["Streptococcus"],
    "thermophilus": ["Streptococcus"],
    "rhamnosus": ["Lacticaseibacillus", "Lactobacillus"],
    "plantarum": ["Lactiplantibacillus", "Lactobacillus"],
    "reuteri": ["Limosilactobacillus", "Lactobacillus"],
    "casei": ["Lacticaseibacillus", "Lactobacillus"],
    "acidophilus": ["Lactobacillus"],
    "gasseri": ["Lactobacillus"],
    "faecalis": ["Enterococcus"],
    "faecium": ["Enterococcus"],
    "aeruginosa": ["Pseudomonas"],
    "nucleatum": ["Fusobacterium"],
    "hallii": ["Anaerobutyricum", "Eubacterium"],
    "rectale": ["Agathobacter", "Eubacterium"],
    "eligens": ["Lachnospira", "Eubacterium"],
    "onderdonkii": ["Alistipes"],
    "putredinis": ["Alistipes"],
    "shahii": ["Alistipes"],
    "finegoldii": ["Alistipes"],
    "bromii": ["Ruminococcus"],
    "gnavus": ["Mediterraneibacter", "Ruminococcus"],
    "torques": ["Mediterraneibacter", "Ruminococcus"],
    "wadsworthia": ["Bilophila"],
    "acnes": ["Cutibacterium", "Propionibacterium"],
    "merdae": ["Parabacteroides"],
    "distasonis": ["Parabacteroides"],
    "intestinalis": ["Roseburia", "Bacteroides", "Odoribacter"],
    "wexlerae": ["Blautia"],
    "moorei": ["Solobacterium"],
    "asaccharolytica": ["Pyramidobacter", "Porphyromonas"],
    "sputigena": ["Selenomonas"],
    "concisus": ["Campylobacter"],
    "rectus": ["Campylobacter"],
    "forsythia": ["Tannerella"],
    "denticola": ["Treponema"],
    "actinomycetemcomitans": ["Aggregatibacter"],
    "parvula": ["Veillonella"],
    "dispar": ["Veillonella"],
    "atypica": ["Veillonella"],
}

# Refused on purpose -- recorded so the next session does not "fix" them.
REFUSED = {
    "bacterium": "not a species epithet; 'E. bacterium' is a parse artefact of "
                 "'...Eubacterium bacterium...' or a truncated label",
    "sp": "'X. sp' is a genus-level placeholder, already handled by norm_taxon",
    "spp": "as above",
    "species": "generic word, not an epithet",
    "difficile_toxin": "names a toxin, not an organism",
    "pneumoniae_ambiguous": "'K. pneumoniae' and 'S. pneumoniae' are different "
                            "organisms sharing an epithet; resolved by the "
                            "paper's own initial, never by the epithet alone",
}


def genus_index(rec):
    """initial -> {genus: taxid} from this paper's FULL (non-abbreviated) mentions."""
    idx = defaultdict(dict)
    for s in rec["kept"]:
        for t in s["taxa"]:
            surface, tid, sci, rank = t[0], t[1], t[2], t[3]
            if rank == "genus" and sci and not ABBREV.match(surface.strip()):
                idx[sci[0].upper()][sci] = tid
    return idx


def main():
    raw = json.load(open(SRC))
    papers = raw["papers"]

    stats = {"REASSIGN": 0, "KEEP": 0, "DROP": 0, "unambiguous": 0, "species_protected": 0}
    actions = []
    dropped_sentences = 0
    kept_sentences_before = sum(len(v["kept"]) for v in papers.values())

    for title, rec in papers.items():
        idx = genus_index(rec)
        new_kept = []
        for s in rec["kept"]:
            new_taxa = []
            for t in s["taxa"]:
                surface, tid, sci, rank = t[0], t[1], t[2], t[3]
                m = ABBREV.match(surface.strip())
                if not m:
                    new_taxa.append(t)
                    continue
                initial, epithet = m.group(1).upper(), m.group(2)

                # GUARD 1: a full binomial already resolved at species rank is
                # better evidence than any epithet table. Never override it.
                if rank == "species":
                    stats["species_protected"] += 1
                    new_taxa.append(t)
                    continue

                candidates = idx.get(initial, {})
                if len(candidates) <= 1:
                    stats["unambiguous"] += 1
                    new_taxa.append(t)
                    continue

                allowed = ABBREV_TRUE.get(epithet) or []
                assigned_genus = (sci or "").split()[0] if sci else ""

                # GUARD 2: only a genus starting with the mention's own initial
                # can be the referent -- that is what the abbreviation asserts.
                targets = [gname for gname in allowed
                           if gname[0].upper() == initial and gname in candidates]

                if allowed and assigned_genus in allowed:
                    stats["KEEP"] += 1
                    new_taxa.append(t)
                elif len(targets) == 1:
                    g = targets[0]
                    new_t = list(t)
                    new_t[1] = candidates[g]
                    new_t[2] = g
                    new_t[3] = "genus"
                    new_taxa.append(new_t)
                    stats["REASSIGN"] += 1
                    actions.append({"action": "REASSIGN", "surface": surface,
                                    "from": sci, "to": g,
                                    "taxid": candidates[g], "paper": title})
                else:
                    stats["DROP"] += 1
                    if not allowed:
                        reason = "epithet not in curated table"
                    elif not targets:
                        reason = "no acceptable genus with this initial named in full in this paper"
                    else:
                        reason = "several acceptable genera with this initial in this paper"
                    actions.append({"action": "DROP", "surface": surface,
                                    "from": sci, "reason": reason,
                                    "allowed": allowed,
                                    "candidates": sorted(candidates),
                                    "paper": title})
            if new_taxa:
                s = dict(s)
                s["taxa"] = new_taxa
                new_kept.append(s)
            else:
                dropped_sentences += 1
        rec["kept"] = new_kept

    raw["stats"]["abbrev_cleaned"] = stats
    raw["stats"]["sentences_dropped_by_clean"] = dropped_sentences
    json.dump(raw, open(OUT, "w"))

    report = {
        "mentions_reassigned": stats["REASSIGN"],
        "mentions_kept_already_correct": stats["KEEP"],
        "mentions_dropped": stats["DROP"],
        "mentions_unambiguous_untouched": stats["unambiguous"],
        "mentions_species_protected": stats["species_protected"],
        "sentences_before": kept_sentences_before,
        "sentences_after": sum(len(v["kept"]) for v in papers.values()),
        "sentences_dropped": dropped_sentences,
        "n_curated_epithets": len(ABBREV_TRUE),
        "refused": REFUSED,
        "actions": actions,
    }
    json.dump(report, open(REPORT, "w"), indent=1)

    print(f"ambiguous mentions reassigned : {stats['REASSIGN']}")
    print(f"  already correct (kept)      : {stats['KEEP']}")
    print(f"  dropped (unattributable)    : {stats['DROP']}")
    print(f"unambiguous, left alone       : {stats['unambiguous']}")
    print(f"sentences {kept_sentences_before} -> {report['sentences_after']} "
          f"({dropped_sentences} lost all taxa)")
    print(f"wrote {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
