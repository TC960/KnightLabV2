#!/usr/bin/env python3
"""GraphRAG vs the BM25 baseline, on ground truth taken from the graph itself.

The hard part of evaluating retrieval here is that there is no relevance-labelled
query set, and inventing one by hand would just measure my own expectations. So
every query below has ground truth that is DEFINED, computable from graph.json,
and fixed before either retriever runs:

  bridge queries  "what links A and B"   -> truth = taxa holding an edge to BOTH
                                            A and B. Set-valued, objective.
  neighbour query "Akkermansia"          -> truth = diseases holding an edge to it.
  directional     "depleted in PD"       -> truth = taxa whose PD edge is depleted.

Both systems return a ranked list; both are scored with precision@k and recall@k
against that set. This is deliberately favourable to BM25 in one respect -- the
truth sets are all one hop from a named entity, which is the regime keyword
matching handles best. The point is not to stage a defeat but to find where a
graph is actually necessary, and where it is not.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def bridge_truth(G, dis_a, dis_b, min_papers=2):
    """Taxa linked to BOTH diseases with real evidence on each side.

    The first version of this used bare co-membership and was useless: 125 of the
    832 taxa are linked to both Parkinson's and Alzheimer's, so returning any ten
    well-connected hubs scored precision 1.00 and BM25 tied trivially. A metric
    that cannot separate the systems is not evidence that they are equal.

    Requiring >=2 papers on EACH side cuts that to 64, and to 8 for
    Alzheimer's/MCI -- now the ranking has to be about the pair rather than about
    which taxa are popular overall.
    """
    ea = {e["taxon"]: e for e in G["edges"] if e["disease"] == dis_a}
    eb = {e["taxon"]: e for e in G["edges"] if e["disease"] == dis_b}
    return {t for t in set(ea) & set(eb)
            if ea[t]["n_papers"] >= min_papers and eb[t]["n_papers"] >= min_papers}


def neighbour_truth(G, taxon):
    return {e["disease"] for e in G["edges"] if e["taxon"] == taxon}


def direction_truth(G, disease, direction):
    return {e["taxon"] for e in G["edges"]
            if e["disease"] == disease and e["direction"] == direction}


def bm25_taxa(ret, q, k):
    """Unique taxa from BM25's top documents, in rank order."""
    hits = ret.search(q, k=k * 4)
    out = []
    for i, _ in hits:
        t = ret.meta[i]["taxon"]
        if t not in out:
            out.append(t)
        if len(out) >= k:
            break
    return out


def bm25_diseases(ret, q, k):
    hits = ret.search(q, k=k * 4)
    out = []
    for i, _ in hits:
        d = ret.meta[i]["disease"]
        if d not in out:
            out.append(d)
        if len(out) >= k:
            break
    return out


def score(pred, truth, k):
    pred = pred[:k]
    hit = [p for p in pred if p in truth]
    prec = len(hit) / max(len(pred), 1)
    rec = len(hit) / max(len(truth), 1)
    return prec, rec, len(hit), len(pred)


def run_comparison(g, k=10):
    import build_rag
    G = g.G
    docs = build_rag.build(graph_path=os.path.join(HERE, "graph.json"),
                           out=os.path.join(HERE, "rag_corpus.jsonl"))
    ret = build_rag.Retriever(docs)

    cases = []

    # Includes two pairs where the answer is NOT the popular hubs
    # (Alzheimer's/MCI: 8 true bridges; MS/ALS: 12), because that is where a
    # ranking has to know about the pair rather than about global popularity.
    for a, b in [("Parkinson's disease", "Alzheimer's disease"),
                 ("Multiple sclerosis", "Stroke"),
                 ("Alzheimer's disease", "Mild cognitive impairment"),
                 ("Multiple sclerosis", "Amyotrophic lateral sclerosis")]:
        q = f"What links {a} and {b}?"
        cases.append({
            "query": q, "kind": "bridge (multi-hop)",
            "truth": bridge_truth(G, a, b),
            "graph": [n["label"] for n in g.subgraph(q, k=k, node_type="taxon")["nodes"]],
            "bm25": bm25_taxa(ret, q, k),
        })

    q = "Akkermansia"
    cases.append({
        "query": q, "kind": "neighbour (1 hop)",
        "truth": neighbour_truth(G, "Akkermansia"),
        "graph": [n["label"] for n in g.subgraph(q, k=k, node_type="disease")["nodes"]],
        "bm25": bm25_diseases(ret, q, k),
    })

    q = "What is depleted in Parkinson's disease?"
    cases.append({
        "query": q, "kind": "directional (1 hop)",
        "truth": direction_truth(G, "Parkinson's disease", "depleted"),
        "graph": [n["label"] for n in g.subgraph(q, k=k, node_type="taxon")["nodes"]],
        "bm25": bm25_taxa(ret, q, k),
    })

    print("=" * 92)
    print(f"{'query':52} {'|truth|':>7} {'GraphRAG P@k':>13} {'BM25 P@k':>10}")
    print("=" * 92)
    rows = []
    for c in cases:
        gp, gr, gh, gn = score(c["graph"], c["truth"], k)
        bp, br, bh, bn = score(c["bm25"], c["truth"], k)
        print(f"{c['query'][:52]:52} {len(c['truth']):>7} "
              f"{gp:>12.2f} {bp:>10.2f}")
        print(f"   {c['kind']:<24} recall@{k}: graph {gr:.2f} ({gh}/{len(c['truth'])})"
              f"   bm25 {br:.2f} ({bh}/{len(c['truth'])})")
        print(f"   graph: {', '.join(c['graph'][:6])}")
        print(f"   bm25 : {', '.join(c['bm25'][:6])}")
        print()
        rows.append({"query": c["query"], "kind": c["kind"], "n_truth": len(c["truth"]),
                     "graph_p": round(gp, 3), "graph_r": round(gr, 3),
                     "bm25_p": round(bp, 3), "bm25_r": round(br, 3),
                     "graph_top": c["graph"][:10], "bm25_top": c["bm25"][:10]})

    gpm = sum(r["graph_p"] for r in rows) / len(rows)
    bpm = sum(r["bm25_p"] for r in rows) / len(rows)
    print("=" * 92)
    print(f"mean precision@{k}:  GraphRAG {gpm:.3f}   BM25 {bpm:.3f}")
    json.dump(rows, open(os.path.join(HERE, "retrieval_comparison.json"), "w"), indent=1)
    print("wrote retrieval_comparison.json")
    return rows


if __name__ == "__main__":
    from graphrag import GraphRAG
    run_comparison(GraphRAG())
