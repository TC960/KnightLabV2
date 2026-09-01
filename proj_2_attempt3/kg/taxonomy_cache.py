#!/usr/bin/env python3
"""A drop-in stand-in for taxonomy.Taxonomy that replays a previous resolution.

WHY THIS EXISTS -- read before trusting it.

`taxonomy.py` resolves taxon strings against the NCBI taxdump (names.dmp /
nodes.dmp). In some environments the taxdump cannot be fetched: this analysis ran
where the network policy denies ftp.ncbi.nih.gov outright (CONNECT -> 403). Running
`build_kg.py` there silently degrades to string folding, which would drop taxid
resolution for 681 of 833 taxa AND all 625 containment links -- a large regression
disguised as a successful rebuild. That is exactly the class of failure this project
has been bitten by before, so it must not happen quietly.

The way out is that `graph.json` already records the taxdump's answers:
  - every taxon node carries `taxid`, its scientific-name `label`, its `rank`, and
    `aliases` -- the complete list of raw surface strings that folded into it;
  - `hierarchy` records, for each taxon, its nearest ancestor that is also a node.

So for any question restricted to strings already in the graph, this class returns
precisely what the taxdump returned.

THE SCOPE CONDITION -- this is a replay cache, not a taxonomy:

  Valid ONLY for rebuilds over a SUBSET of the papers that produced graph.json.

Filtering papers can only remove taxa, never introduce a string the cache has not
already seen, so every lookup hits. Add a new paper and it will miss, and a miss is
reported as unresolved -- indistinguishable from a genuine unresolvable name. That
would understate resolution without erroring. `resolve()` therefore counts misses,
and `report_misses()` prints them; callers MUST check it. For anything that adds
papers, get the real taxdump.

LINEAGE. graph.json stores only nearest-present-ancestor links, not full NCBI
lineages, so `lineage()` walks that chain rather than the true one. This is
sufficient -- and exactly correct -- for how build_kg.py uses it: it asks for the
nearest ancestor that survives in the current node set. Because the stored links
form a chain through the original graph (each child has at most one parent), the
first surviving node along that chain IS the correct nearest surviving ancestor
after any deletion. It is NOT a general lineage: intermediate NCBI ranks that were
never graph nodes are absent, so do not use this for taxonomic distance.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GRAPH = os.path.join(HERE, "graph.json")


class CachedTaxonomy:
    def __init__(self, graph_path=DEFAULT_GRAPH):
        self.ok = os.path.exists(graph_path)
        self.name2tid = {}     # lowercased surface string -> taxid
        self.sci = {}          # taxid -> scientific name
        self.rank = {}         # taxid -> rank
        self.parent = {}       # taxid -> nearest-ancestor taxid (graph-local)
        self.misses = set()
        self._source = graph_path
        if not self.ok:
            return

        g = json.load(open(graph_path))
        for n in g.get("nodes", []):
            if n.get("type") != "taxon":
                continue
            tid = n.get("taxid")
            if not tid or not n.get("resolved"):
                # Unresolved nodes are intentionally NOT cached: build_kg.py must
                # take its string-folding path for them, exactly as it did before,
                # so their keys and ranks come out identical.
                continue
            self.sci[tid] = n.get("label", "")
            self.rank[tid] = n.get("rank", "no rank")
            for a in n.get("aliases", []) or []:
                self.name2tid.setdefault(self._norm(a), tid)
            self.name2tid.setdefault(self._norm(n.get("label", "")), tid)

        for h in g.get("hierarchy", []):
            child = str(h.get("child", "")).replace("t:ncbi:", "")
            par = str(h.get("parent", "")).replace("t:ncbi:", "")
            if child and par:
                self.parent[child] = par

    @staticmethod
    def _norm(s):
        return " ".join((s or "").strip().split()).lower()

    def lineage(self, tid, cap=60):
        """[tid, nearest ancestor, its nearest ancestor, ...] within the graph."""
        out, seen = [], set()
        while tid and tid not in seen and len(out) < cap:
            out.append(tid)
            seen.add(tid)
            tid = self.parent.get(tid)
        return out

    def resolve(self, name):
        """Mirror taxonomy.Taxonomy.resolve's contract.

        -> (taxid, scientific_name, rank, how) or (None, cleaned, None, 'unresolved')
        """
        raw = (name or "").strip()
        tid = self.name2tid.get(self._norm(raw))
        if tid:
            return (tid, self.sci.get(tid, raw), self.rank.get(tid, "no rank"), "cached")
        self.misses.add(raw)
        return (None, raw, None, "unresolved")

    def report_misses(self):
        """Print names the cache had never seen. Non-empty => out of scope."""
        if self.misses:
            print(f"  cache MISSES ({len(self.misses)}) -- these were NOT in graph.json; "
                  f"they will be treated as unresolved:")
            for m in sorted(self.misses)[:40]:
                print(f"    - {m}")
        return len(self.misses)


def load_taxonomy(verbose=True):
    """Real taxdump if present, else the replay cache. Never silently string-folds."""
    try:
        from taxonomy import Taxonomy
        t = Taxonomy()
        if t.ok:
            if verbose:
                print(f"taxonomy: NCBI taxdump ({len(t.name2ids)} names)")
            return t
    except Exception:
        pass
    c = CachedTaxonomy()
    if verbose:
        print(f"taxonomy: NO taxdump -> replay cache from {os.path.basename(c._source)} "
              f"({len(c.name2tid)} names, {len(c.parent)} parent links). "
              f"VALID ONLY FOR SUBSETS of that graph.")
    return c


if __name__ == "__main__":
    c = CachedTaxonomy()
    print("cache loaded:", c.ok, "| names:", len(c.name2tid), "| parents:", len(c.parent))
    for n in ["Bacteroidetes", "Bacteroidota", "Firmicutes", "Bacillota", "Akkermansia",
              "Faecalibacterium prausnitzii", "Not A Real Taxon"]:
        print(f"  {n:32} -> {c.resolve(n)}")
    print("\nlineage of Akkermansia muciniphila-ish node chain:")
    tid = c.name2tid.get("akkermansia")
    print("  ", c.lineage(tid) if tid else "n/a")
