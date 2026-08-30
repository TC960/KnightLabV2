#!/usr/bin/env python3
"""Resolve taxon strings to NCBI taxids, folding synonyms and renames.

Why this exists: the extractor emits whatever the paper printed, so the same
organism arrives under several names. "Bacteroidetes" (pre-2021) and
"Bacteroidota" (current) are the same phylum, taxid 976; likewise Firmicutes /
Bacillota. Without resolution they are separate graph nodes and their evidence
never pools.

Reads NCBI names.dmp / nodes.dmp directly rather than shelling out to taxonkit:
the taxdump IS the synonym table, and parsing it here avoids the linux-only
binaries the eval harness uses (this runs on the analyst's mac too).

Two things that need care:

1. **Ambiguity.** "Bacteroidetes" is a synonym for BOTH the phylum Bacteroidota
   (976) and the class Bacteroidia (200643). We prefer scientific-name matches
   over synonyms, then the higher rank -- in microbiome papers the bare phylum
   name is the intended reading.
2. **Cross-kingdom homonyms.** Genus names are reused across kingdoms, so a
   candidate is rejected unless its lineage sits under Bacteria, Archaea, Fungi
   or Viruses. Without this an animal homonym can silently win.

Names that do not resolve are NOT dropped -- they keep their surface string and
are marked unresolved, so nothing disappears from the graph silently.
"""
import os
import re
from functools import lru_cache

DATA = os.environ.get("TAX_DATA", os.path.expanduser("~/.ncbi-taxdump"))
NAMES = os.path.join(DATA, "names.dmp")
NODES = os.path.join(DATA, "nodes.dmp")

MICROBIAL_ROOTS = {"2", "2157", "4751", "10239"}   # Bacteria, Archaea, Fungi, Viruses
RANK_ORDER = ["superkingdom", "kingdom", "phylum", "class", "order",
              "family", "genus", "species", "strain", "no rank"]
_RANKPOS = {r: i for i, r in enumerate(RANK_ORDER)}

# The classical Linnaean ranks are preferred over NCBI's newer intermediate ranks
# when a name is ambiguous. This matters for the phylum renames, which are exactly
# the ambiguous cases we care about:
#   "Firmicutes"    -> synonym of BOTH Bacillati (kingdom) and Bacillota (phylum)
#   "Bacteroidetes" -> synonym of BOTH Bacteroidota (phylum) and Bacteroidia (class)
# Preferring classical ranks picks Bacillota; among classical ranks the higher one
# wins, which picks Bacteroidota. Ranking purely by height would get Firmicutes
# wrong (kingdom outranks phylum), which is what a first pass here actually did.
_CLASSICAL = {"phylum", "class", "order", "family", "genus", "species"}

# rank prefixes the pipelines emit: "o-Clostridia", "f__Rikenellaceae", "g_Blautia"
_PREFIX = re.compile(r"^([pcofgsdk])[_-]{1,2}", re.I)
_PREFIX_RANK = {"d": "superkingdom", "k": "kingdom", "p": "phylum", "c": "class",
                "o": "order", "f": "family", "g": "genus", "s": "species"}


class Taxonomy:
    def __init__(self, data_dir=DATA):
        self.ok = os.path.exists(os.path.join(data_dir, "names.dmp"))
        self.name2ids = {}      # lowercased name -> [(taxid, name_class)]
        self.sci = {}           # taxid -> scientific name
        self.rank = {}          # taxid -> rank
        self.parent = {}        # taxid -> parent taxid
        if not self.ok:
            return
        with open(os.path.join(data_dir, "names.dmp"), encoding="utf-8", errors="replace") as f:
            for line in f:
                p = [x.strip() for x in line.split("|")]
                if len(p) < 4:
                    continue
                tid, name, cls = p[0], p[1], p[3]
                if cls == "scientific name":
                    self.sci[tid] = name
                if cls in ("scientific name", "synonym", "equivalent name",
                           "genbank synonym", "genbank common name"):
                    self.name2ids.setdefault(name.lower(), []).append((tid, cls))
        with open(os.path.join(data_dir, "nodes.dmp"), encoding="utf-8", errors="replace") as f:
            for line in f:
                p = [x.strip() for x in line.split("|")]
                if len(p) < 3:
                    continue
                self.parent[p[0]] = p[1]
                self.rank[p[0]] = p[2]

    def lineage(self, tid, cap=60):
        out, seen = [], set()
        while tid and tid not in seen and len(out) < cap:
            out.append(tid)
            seen.add(tid)
            nxt = self.parent.get(tid)
            if nxt == tid:
                break
            tid = nxt
        return out

    def _microbial(self, tid):
        return bool(MICROBIAL_ROOTS & set(self.lineage(tid)))

    @lru_cache(maxsize=20000)
    def resolve(self, name):
        """-> (taxid, scientific_name, rank, matched_how) or (None, cleaned, hint, 'unresolved')"""
        raw = (name or "").strip()
        hint = None
        m = _PREFIX.match(raw)
        if m:
            hint = _PREFIX_RANK.get(m.group(1).lower())
            raw = raw[m.end():]
        raw = re.sub(r"[_]+", " ", raw).strip()
        raw = re.sub(r"\s+", " ", raw)
        if not self.ok or not raw:
            return (None, raw, hint, "unresolved")

        cands = self.name2ids.get(raw.lower(), [])
        # try trimming qualifier tails: "Clostridium sensu stricto 1" -> "Clostridium"
        if not cands and " " in raw:
            for stop in range(len(raw.split()) - 1, 0, -1):
                head = " ".join(raw.split()[:stop])
                if self.name2ids.get(head.lower()):
                    cands = self.name2ids[head.lower()]
                    raw = head
                    break
        cands = [(t, c) for t, c in cands if self._microbial(t)]
        if not cands:
            return (None, raw, hint, "unresolved")

        def key(tc):
            tid, cls = tc
            r = self.rank.get(tid, "no rank")
            return (0 if cls == "scientific name" else 1,   # real name beats synonym
                    0 if r in _CLASSICAL else 1,            # classical rank beats intermediate
                    _RANKPOS.get(r, 99))                    # then the higher rank
        tid, cls = sorted(cands, key=key)[0]
        return (tid, self.sci.get(tid, raw), self.rank.get(tid, hint or "no rank"),
                "scientific" if cls == "scientific name" else "synonym")


_shared = None


def shared():
    global _shared
    if _shared is None:
        _shared = Taxonomy()
    return _shared


if __name__ == "__main__":
    t = Taxonomy()
    print("taxdump loaded:", t.ok, "| names:", len(t.name2ids), "| nodes:", len(t.rank))
    for n in ["Bacteroidetes", "Bacteroidota", "Firmicutes", "Bacillota", "Akkermansia",
              "Faecalibacterium prausnitzii", "o-Clostridia", "Clostridium_sensu_stricto_1",
              "f__Rikenellaceae", "[Eubacterium] ventriosum group", "Not A Real Taxon"]:
        print(f"  {n:34} -> {t.resolve(n)}")
