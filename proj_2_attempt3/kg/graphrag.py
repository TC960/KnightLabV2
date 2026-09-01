#!/usr/bin/env python3
"""Graph retrieval over the KG: personalized PageRank instead of BM25.

`build_rag.py` retrieves with BM25 over one flattened sentence per edge, plus
entity matching. That is the wrong primitive here, and not because BM25 is a weak
ranker -- because the corpus it ranks has had the graph deleted from it. Each
document is an isolated claim; nothing in the index knows that Roseburia and
Faecalibacterium are both depleted in the same disease, or that Lachnospiraceae
contains Hungatella. "What else is connected to this" is unanswerable by keyword
scoring, and it is the entire reason the graph exists.

So retrieval here is traversal:

  1. **Entity-link the query.** The vocabulary is closed and known -- 832 taxon
     nodes with their aliases, 43 disease nodes -- so this is exact longest-match
     lookup, not guessing. Disease surface forms ("PD", "Parkinson's") fold
     through the same map build_kg.py uses.
  2. **Personalized PageRank** from the matched nodes. Damping 0.85, power
     iteration to convergence. ~900 nodes, so this is milliseconds and needs no
     library.
  3. **Return a connected subgraph** -- seeds, high-PPR neighbours, the edges
     between them, and the papers backing each edge -- rather than a flat list of
     documents.

TWO DESIGN DECISIONS THAT CARRY THE RESULT

*Containment links are traversable.* 625 of them exist and nothing consumed them
before. They are what lets a query about a genus reach its family. They are given
a modest fixed weight and are NOT treated as evidence: containment is a statement
about taxonomy, not about disease, so it should move probability without voting.

*Multi-entity queries are answered with a product, not a union.* "What links
Parkinson's and Alzheimer's?" seeded jointly into one PPR run returns taxa near
EITHER disease -- which is mostly just the taxa near the bigger one, and is not the
question. Instead each seed gets its own PPR run and nodes are ranked by the
GEOMETRIC MEAN of their scores. A node scores well only if it is close to ALL
seeds, which is what "links" means. This is the multi-hop payoff BM25 cannot
reach at any k, because no single document mentions both diseases.

    python graphrag.py --query "What links Parkinson's and Alzheimer's?"
    python graphrag.py --query "Akkermansia" -k 12
    python graphrag.py --compare        # side by side against the BM25 baseline
"""
import argparse
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")

DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-10

# Containment is a taxonomy fact, not evidence. Weight it enough to make a
# family reachable from its genus, low enough that it cannot outrank a
# well-replicated association.
CONTAINMENT_W = 1.0

# Disease surface forms -> canonical labels, mirroring build_kg.DISEASE_MAP so a
# query can say "PD" or "Alzheimer's" and land on the same node the builder made.
DISEASE_PATTERNS = [
    (r"\bparkinson'?s?\b|\bpd\b", "Parkinson's disease"),
    (r"\balzheimer'?s?\b|\bad\b", "Alzheimer's disease"),
    (r"multiple sclerosis|\bms\b", "Multiple sclerosis"),
    (r"amyotrophic lateral sclerosis|\bals\b", "Amyotrophic lateral sclerosis"),
    (r"mild cognitive impairment|\bmci\b", "Mild cognitive impairment"),
    (r"\bstroke\b|cerebral infarct", "Stroke"),
    (r"huntington'?s?\b|\bhd\b", "Huntington's disease"),
    (r"\bdementia\b", "Dementia"),
    (r"spinal muscular atrophy|\bsma\b", "Spinal muscular atrophy"),
    (r"epilep\w*", "Epilepsy"),
    (r"autism\w*|\basd\b", "Autism spectrum disorder"),
    (r"depress\w*|\bmdd\b", "Depressive disorder"),
    (r"schizophren\w*", "Schizophrenia"),
    (r"neuromyelitis", "Neuromyelitis optica"),
    (r"myasthenia", "Myasthenia gravis"),
    (r"migraine", "Migraine"),
]

STOP = {"what", "which", "the", "a", "an", "is", "are", "in", "of", "and", "or",
        "links", "link", "linked", "between", "with", "to", "for", "does", "do",
        "how", "why", "tell", "me", "about", "show", "find", "shared", "common",
        "both", "differ", "different", "connects", "connected", "related"}


class GraphRAG:
    def __init__(self, graph_path=GRAPH):
        self.G = json.load(open(graph_path))
        self.nodes = {n["id"]: n for n in self.G["nodes"]}

        # --- adjacency -----------------------------------------------------
        # Undirected: retrieval asks "what is near this", which has no direction.
        # Association weight combines how much evidence the edge has with how
        # consistent that evidence is, so a 16-paper unanimous edge pulls harder
        # than a 2-paper contested one -- but a contested edge still conducts,
        # because a contested pair is a legitimate thing to retrieve.
        self.adj = defaultdict(dict)
        for e in self.G["edges"]:
            w = e["n_papers"] * (0.5 + 0.5 * e.get("consistency", 1.0))
            s, t = e["source"], e["target"]
            self.adj[s][t] = self.adj[s].get(t, 0.0) + w
            self.adj[t][s] = self.adj[t].get(s, 0.0) + w
        for h in self.G.get("hierarchy", []):
            p, c = h["parent"], h["child"]
            if p in self.nodes and c in self.nodes:
                self.adj[p][c] = self.adj[p].get(c, 0.0) + CONTAINMENT_W
                self.adj[c][p] = self.adj[c].get(p, 0.0) + CONTAINMENT_W

        self.edge_by_pair = {}
        for e in self.G["edges"]:
            self.edge_by_pair[(e["source"], e["target"])] = e
        self.contains = {(h["parent"], h["child"]) for h in self.G.get("hierarchy", [])}
        self.papers = self.G.get("papers", [])

        # --- closed-vocabulary entity index --------------------------------
        self.vocab = {}
        for n in self.G["nodes"]:
            names = [n.get("label", "")] + list(n.get("aliases") or [])
            for nm in names:
                k = self._norm(nm)
                if k and k not in self.vocab:
                    self.vocab[k] = n["id"]
        self.max_ngram = max((len(k.split()) for k in self.vocab), default=1)

    @staticmethod
    def _norm(s):
        s = (s or "").lower().replace("_", " ")
        s = re.sub(r"^[a-z]__", "", s)
        s = re.sub(r"[^a-z0-9. \-\[\]]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    # ---------------------------------------------------------------- linking
    def link(self, query):
        """-> list of node ids the query names. Exact, longest-match-first."""
        seeds, low = [], self._norm(query)

        for pat, label in DISEASE_PATTERNS:
            if re.search(pat, low) and f"d:{label}" in self.nodes:
                seeds.append(f"d:{label}")

        words = low.split()
        used = [False] * len(words)
        for n in range(min(self.max_ngram, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                if any(used[i:i + n]):
                    continue
                cand = " ".join(words[i:i + n])
                if n == 1 and (cand in STOP or len(cand) < 4):
                    continue
                nid = self.vocab.get(cand)
                if nid:
                    seeds.append(nid)
                    for j in range(i, i + n):
                        used[j] = True
        out = []
        for s in seeds:
            if s not in out:
                out.append(s)
        return out

    # ------------------------------------------------------------------- ppr
    def ppr(self, seeds, damping=DAMPING):
        """Personalized PageRank restarting on `seeds`."""
        if not seeds:
            return {}
        ids = list(self.adj)
        for s in seeds:
            if s not in self.adj:
                ids.append(s)
        deg = {i: sum(self.adj[i].values()) for i in ids}
        restart = {s: 1.0 / len(seeds) for s in seeds}
        r = dict(restart)
        for _ in range(MAX_ITER):
            nxt = defaultdict(float)
            leak = 0.0
            for i, ri in r.items():
                if ri <= 0:
                    continue
                d = deg.get(i, 0.0)
                if d <= 0:
                    leak += ri            # dangling node: send it back to the seeds
                    continue
                share = damping * ri / d
                for j, w in self.adj[i].items():
                    nxt[j] += share * w
            for s, p in restart.items():
                nxt[s] += (1 - damping) * p + damping * leak * p
            diff = sum(abs(nxt.get(k, 0.0) - r.get(k, 0.0)) for k in set(nxt) | set(r))
            r = dict(nxt)
            if diff < TOL:
                break
        return r

    def rank(self, seeds, k=10, exclude_seeds=True):
        """Rank nodes. One seed -> plain PPR. Several -> geometric mean per seed.

        The geometric mean is the point: it scores a node by its WEAKEST link to
        any seed, so a taxon reported only in Parkinson's cannot rank highly on a
        Parkinson's-and-Alzheimer's query no matter how strong that one tie is.
        """
        if not seeds:
            return [], {}
        if len(seeds) == 1:
            scores = self.ppr(seeds)
        else:
            per = [self.ppr([s]) for s in seeds]
            common = set(per[0])
            for p in per[1:]:
                common &= set(p)
            scores = {}
            for n in common:
                prod = 1.0
                for p in per:
                    prod *= p.get(n, 0.0)
                if prod > 0:
                    scores[n] = prod ** (1.0 / len(per))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if exclude_seeds:
            ranked = [(n, s) for n, s in ranked if n not in seeds]
        return ranked[:k], scores

    @staticmethod
    def query_direction(q):
        """'depleted'/'enriched' if the query asks for one, else None."""
        low = (q or "").lower()
        if re.search(r"deplet|decreas|lower|reduc|less abundant|loss of", low):
            return "depleted"
        if re.search(r"enrich|increas|higher|elevat|more abundant|expand", low):
            return "enriched"
        return None

    # ------------------------------------------------------------- subgraph
    def subgraph(self, query, k=10, node_type=None):
        seeds = self.link(query)
        if not seeds:
            return {"query": query, "seeds": [], "error": "no entity in the query matched the graph"}
        want_dir = self.query_direction(query)
        # Disease nodes are hubs, so on a two-disease query they crowd out the taxa
        # that actually answer it. node_type="taxon" asks the question the user
        # usually means: which ORGANISMS link these diseases.
        wide = bool(node_type or want_dir)
        ranked, _ = self.rank(seeds, k=10 ** 6 if wide else k)
        if node_type:
            ranked = [(n, s) for n, s in ranked if self.nodes[n]["type"] == node_type]
        if want_dir:
            # PPR is direction-blind by construction -- proximity has no sign -- so
            # "what is DEPLETED in Parkinson's" would otherwise return everything
            # near Parkinson's, enriched taxa included. Measured against the BM25
            # baseline this was the one query type where the graph lost outright
            # (P@10 0.60 vs 1.00), so direction is applied as an explicit filter on
            # the seed disease, exactly as build_rag.py does. Contested edges are
            # kept: they do report the asked-for direction, in part.
            dis_seeds = [s for s in seeds if self.nodes[s]["type"] == "disease"]
            if dis_seeds:
                ok = set()
                for (src, tgt), e in self.edge_by_pair.items():
                    if tgt in dis_seeds and (e["direction"] == want_dir or e["contested"]):
                        ok.add(src)
                ranked = [(n, s) for n, s in ranked if n in ok]
        ranked = ranked[:k]
        keep = list(seeds) + [n for n, _ in ranked]
        keepset = set(keep)

        assoc, contain = [], []
        for (s, t), e in self.edge_by_pair.items():
            if s in keepset and t in keepset:
                assoc.append({
                    "taxon": e["taxon"], "disease": e["disease"],
                    "direction": e["direction"], "n_papers": e["n_papers"],
                    "n_up": e["n_up"], "n_down": e["n_down"],
                    "contested": e["contested"], "consistency": e.get("consistency"),
                    "papers": e.get("papers", [])[:6],
                })
        for p, c in self.contains:
            if p in keepset and c in keepset:
                contain.append({"parent": self.nodes[p]["label"], "child": self.nodes[c]["label"]})

        assoc.sort(key=lambda a: -a["n_papers"])
        return {
            "query": query,
            "seeds": [{"id": s, "label": self.nodes[s]["label"],
                       "type": self.nodes[s]["type"]} for s in seeds],
            "nodes": [{"id": n, "label": self.nodes[n]["label"],
                       "type": self.nodes[n]["type"],
                       "rank": self.nodes[n].get("rank"), "score": round(sc, 8)}
                      for n, sc in ranked],
            "edges": assoc,
            "containment": contain,
        }


def render(sg):
    out = [f"QUERY: {sg['query']}"]
    if sg.get("error"):
        out.append(f"  !! {sg['error']}")
        return "\n".join(out)
    out.append("  seeds: " + ", ".join(f"{s['label']} ({s['type']})" for s in sg["seeds"]))
    out.append(f"\n  top nodes by personalized PageRank:")
    for n in sg["nodes"]:
        out.append(f"    {n['score']:.6f}  {n['label'][:38]:39} {n['type']}"
                   f"{'/' + n['rank'] if n.get('rank') else ''}")
    out.append(f"\n  edges in the retrieved subgraph ({len(sg['edges'])}):")
    for e in sg["edges"][:18]:
        flag = " CONTESTED" if e["contested"] else ""
        out.append(f"    {e['taxon'][:26]:27} {e['direction']:9} in {e['disease'][:26]:27}"
                   f" {e['n_papers']:>2}p (up={e['n_up']} dn={e['n_down']}){flag}")
    if sg["containment"]:
        out.append(f"\n  containment links used ({len(sg['containment'])}):")
        for c in sg["containment"][:10]:
            out.append(f"    {c['parent']} contains {c['child']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--type", dest="node_type", choices=["taxon", "disease"],
                    help="restrict ranked nodes to one type")
    ap.add_argument("--compare", action="store_true",
                    help="run the standard query set against BM25 too")
    a = ap.parse_args()

    g = GraphRAG(a.graph)
    print(f"graph: {len(g.nodes)} nodes, {len(g.G['edges'])} edges, "
          f"{len(g.G.get('hierarchy', []))} containment links\n")

    if a.query:
        sg = g.subgraph(a.query, k=a.k, node_type=a.node_type)
        print(json.dumps(sg, indent=1) if a.json else render(sg))
        return

    if a.compare:
        from compare_retrieval import run_comparison
        run_comparison(g)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
