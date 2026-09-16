#!/usr/bin/env python3
"""Strings that name TWO taxa, and must not be filed under either one.

THE DEFECT, found 2026-09-08 by asking which concepts land on more than one node.
"Escherichia-Shigella" is the standard SILVA/QIIME label for a pair of genera that
16S cannot separate. The corpus writes it seven ways, and the graph files those 27
mentions under **four different nodes**:

    11x  Escherichia-Shigella     -> t:escherichia-shigella   (unresolved)
     6x  Escherichia_Shigella     -> t:ncbi:561  *** Escherichia ***
     5x  Escherichia/Shigella     -> t:escherichia/shigella   (unresolved)
     2x  Escherichia–Shigella     -> t:escherichia–shigella   (unresolved, EN DASH)
     1x  Escherichia / Shigella   -> t:ncbi:561  *** Escherichia ***
     1x  Escherichia-shigella     -> t:escherichia-shigella
     1x  Escherichia – Shigella   -> t:ncbi:561  *** Escherichia ***

Two separate bugs, visible in one place. **Fragmentation**: the same concept is
split by punctuation alone, so its evidence never pools. **Misattribution**: the
split is not even consistent — when the separator happens to be a space or an
underscore, `resolve()` converts it to a space, fails to match, then trims the
trailing token as if it were a qualifier, and lands the mention on *Escherichia*.
A signal from an assay that could not distinguish two genera is then recorded as
evidence about one of them.

THE RULE. Refuse the qualifier trim when the token being discarded is ITSELF the
name of a genus or higher. "Escherichia" + discard "Shigella" is not a genus with
a qualifier; it is two genera. Same for "Lachnospiraceae_Eubacterium". This is the
same test `species_synonyms.py` uses to reject compound clade labels, applied one
level down, and it is deterministic: the discarded token either resolves to a
taxon or it does not.

Measured blast radius over all 1,090 distinct surface strings in the corpus: **2
strings**. It is a narrow rule, not a rewrite.

WHAT IT DOES NOT DECIDE. These mentions end up on a single joint node
("Escherichia-Shigella"), unresolved, with no taxid — deliberately. Whether a
joint 16S signal should instead be attributed to one genus, split across both, or
dropped is a modelling decision for a human; this only stops the graph from making
that decision *by accident*, differently, depending on which punctuation the paper
happened to use. The majority treatment already in the graph (20 of 27 mentions)
is exactly this joint node, so the rule follows the corpus rather than overriding
it.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "multi_taxon.json")
TAXON_RANKS = {"genus", "family", "order", "class", "phylum", "kingdom",
               "superkingdom"}


def norm(s):
    """Collapse every separator style to one space.

    Hyphen, en dash, em dash, slash and underscore are all used in this corpus for
    the same 'A and/or B' join, and treating them differently is what fragmented
    the concept across four nodes in the first place.
    """
    return " ".join(re.sub(r"[/_‐-―-]", " ", str(s or "")).split())


def load_table(path=OUT):
    try:
        with open(path) as f:
            return json.load(f).get("entries", {})
    except Exception:
        return {}


def main():
    import build_kg as BK
    from taxonomy_cache import CachedTaxonomy
    tax = CachedTaxonomy()
    tax.multi = {}                      # see the resolver: must observe the OLD fold

    seen, entries = set(), {}
    for r in json.load(open(os.path.join(HERE, "extractions_screened.json"))):
        for col in ("predicted_enriched", "predicted_depleted"):
            for raw in BK.parse_taxa(r.get(col)):
                if raw in seen:
                    continue
                seen.add(raw)
                tid, sci, _rank, _how = tax.resolve(raw)
                if not (tid and sci):
                    continue
                n = " ".join(str(raw).replace("_", " ").split())
                if n.lower() == sci.lower():
                    continue
                if not n.lower().startswith(sci.lower() + " "):
                    continue
                tail = n[len(sci):].strip()
                th = tax.resolve(tail)
                if th[0] and th[2] in TAXON_RANKS:
                    entries[norm(raw).lower()] = {
                        "surface": raw,
                        "was_folded_into": sci,
                        "discarded": tail,
                        "discarded_is": f"{th[1]} [{th[2]}]",
                        "joint_label": norm(raw),
                    }
    json.dump({"_note": "strings naming two taxa; see multi_taxon.py. Consulted by "
                        "the resolvers BEFORE any fold, so neither taxon absorbs "
                        "the other's evidence.",
               "entries": entries},
              open(OUT, "w"), indent=1, sort_keys=True)
    print(f"strings naming two taxa: {len(entries)}")
    for k, v in sorted(entries.items()):
        print(f"  {v['surface']!r:34} was folded into {v['was_folded_into']!r}, "
              f"discarding {v['discarded']!r} = {v['discarded_is']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
