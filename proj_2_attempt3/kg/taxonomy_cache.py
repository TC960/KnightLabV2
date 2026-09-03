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
        self.ph2tid = {}       # rank-placeholder surface string -> PARENT taxid
        self.ph_unrecovered = set()
        self.misses = set()
        self._name2ids = None
        self._source = graph_path
        if not self.ok:
            return

        g = json.load(open(graph_path))
        placeholders = []
        for n in g.get("nodes", []):
            if n.get("type") != "taxon":
                continue
            if n.get("placeholder"):
                placeholders.append(n)
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

        # Only taxid->taxid links belong in the lineage chain; placeholder children
        # are keyed "t:ph:..." and would otherwise sit in here as junk keys.
        ph_parent_id = {}
        for h in g.get("hierarchy", []):
            child, par = str(h.get("child", "")), str(h.get("parent", ""))
            if not (child and par and par.startswith("t:ncbi:")):
                continue
            if child.startswith("t:ncbi:"):
                self.parent[child[7:]] = par[7:]
            elif child.startswith("t:ph:"):
                ph_parent_id[child] = par[7:]

        self._cache_placeholders(placeholders, ph_parent_id)

    def _cache_placeholders(self, nodes, ph_parent_id):
        """Make rank-placeholder strings resolve to their PARENT taxid again.

        A placeholder node ("Erysipelotrichaceae UCG-003") deliberately carries no
        taxid, so the loop above skips it -- and then every one of its surface
        strings is a cache MISS. build_kg.py's placeholder branch only fires when
        the string resolves to a taxid whose scientific name differs from it, so a
        miss silently sends the string down the string-folding path: the node keeps
        no `placeholder` flag, gets NO containment link to its parent, and splits
        across spelling variants ("X_NK4A214_group" vs "X NK4A214 group") that the
        real taxdump folds together. The build still prints success.

        That is the fix from 2026-09-01 quietly undoing itself on any rebuild here,
        so the parent must be recoverable from graph.json. Two ways, in order:
          1. the `hierarchy` link the placeholder node already has to its parent;
          2. failing that (the parent is not itself a node), replay the taxdump's
             own rule -- trim trailing qualifier tokens until the remainder is a
             name we know.
        Placeholder strings are kept in their OWN table, not `name2tid`, so this
        can never make an ordinary taxon name resolve to something it did not
        before.
        """
        for n in nodes:
            # 0. the parent recorded on the node itself (build_kg writes it), which
            #    is exact and works even when the parent is not a node in the graph.
            tid = n.get("parent_taxid") or ph_parent_id.get(str(n.get("id", "")))
            names = [a for a in (n.get("aliases") or [])] + [n.get("label", "")]
            if not tid:
                tid = next((t for t in (self._trim_to_parent(a) for a in names) if t), None)
            if not tid:
                self.ph_unrecovered.add(n.get("label", ""))
                continue
            for a in names:
                if a:
                    self.ph2tid.setdefault(self._norm(a), tid)

    # A phage is NOT a member of the genus it infects. build_kg.py's placeholder
    # regex over-matches strain-code-looking names, so "Escherichia virus JES2013"
    # and "Klebsiella virus KP36" arrive here looking like rank placeholders; naive
    # trimming would hang them under Escherichia and Klebsiella as containment
    # children, inventing two taxonomically false edges. Refuse any trim whose
    # discarded tail says the string names a different kind of organism.
    NOT_CONTAINED = ("virus", "phage", "bacteriophage")

    def _trim_to_parent(self, name):
        """Drop trailing tokens until what is left is a name the cache knows.

        Replays what taxonomy.py does with a qualifier tail, using only names the
        cache already holds. Brackets are stripped because 16S pipelines write
        "[Eubacterium] nodatum group" for what NCBI calls Eubacterium.
        """
        toks = self._norm(str(name).replace("_", " ").replace("[", "").replace("]", "")).split()
        for i in range(len(toks) - 1, 0, -1):
            if any(t in self.NOT_CONTAINED for t in toks[i:]):
                return None
            tid = self.name2tid.get(" ".join(toks[:i]))
            if tid:
                return tid
        return None

    @property
    def name2ids(self):
        """Mirror of Taxonomy.name2ids so this is a true drop-in.

        Callers use it two ways: as a membership test ("is this string a known
        name?") and as a candidate list. Both are satisfied by wrapping each cached
        taxid in the same (taxid, name_class) shape. Everything cached came from a
        node the taxdump had already resolved, so "scientific name" is the honest
        class -- the cache never stores a name it could not resolve.
        """
        if self._name2ids is None:
            self._name2ids = {k: [(v, "scientific name")] for k, v in self.name2tid.items()}
        return self._name2ids

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
        key = self._norm(raw)
        tid = self.name2tid.get(key)
        if tid:
            return (tid, self.sci.get(tid, raw), self.rank.get(tid, "no rank"), "cached")
        # A rank placeholder resolves to its parent, and the caller detects that by
        # the returned scientific name differing from the string it asked about --
        # which is exactly what makes build_kg.py split it off as a child node.
        tid = self.ph2tid.get(key) or self.ph2tid.get(self._norm(raw.replace("_", " ")))
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
              f"({len(c.name2tid)} names, {len(c.parent)} parent links, "
              f"{len(c.ph2tid)} placeholder strings). "
              f"VALID ONLY FOR SUBSETS of that graph.")
        if c.ph_unrecovered:
            print(f"  {len(c.ph_unrecovered)} placeholder node(s) have no recoverable "
                  f"parent taxid -> they will string-fold (no containment link): "
                  f"{', '.join(sorted(c.ph_unrecovered)[:5])}")
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
