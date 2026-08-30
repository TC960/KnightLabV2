#!/usr/bin/env python3
"""Turn the knowledge graph into a retrieval corpus, and retrieve over it.

Produces rag_corpus.jsonl -- one document per graph edge, written as a sentence a
retriever can match and an LLM can quote, with the structured fields kept alongside
as metadata. Drop-in for any vector store (Chroma, FAISS, pgvector, LanceDB): embed
`text`, keep `meta`, filter on `meta.*`.

Why one document per EDGE rather than per paper: the edge is the unit of claim.
A paper-level chunk buries "Akkermansia is enriched in Parkinson's" inside 50k
characters of methods; an edge-level chunk states it, says how many papers agree,
says how many disagree, and names them. That is what you want a model quoting.

Retrieval is BM25 over the claim text PLUS explicit entity/direction matching --
pure Python, no model, no service. Hybrid rather than pure lexical because pure
BM25 measurably failed here: disease names are common inside the corpus so their
idf is low, and an incidental query word can outrank them (see the STOP comment).
Since the entity vocabulary is known exactly -- every disease and taxon is in the
graph -- matching it directly is both more accurate and cheaper than embeddings.
Swap in dense retrieval over the same `text` field if you later want paraphrase
matching; the corpus format does not change.

    python build_rag.py                        # build the corpus
    python build_rag.py --query "what is depleted in parkinson's"
    python build_rag.py --query "contested findings for alzheimer's" -k 8
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
CORPUS = os.path.join(HERE, "rag_corpus.jsonl")
METADATA = os.path.join(HERE, "metadata.jsonl")


def edge_text(e, md=None):
    """One retrievable sentence per edge. Written so a quoted span is self-contained:
    the direction, the evidence weight, and the disagreement all appear in prose,
    because a model quoting only half of it should still say something true."""
    d = "contested" if e["contested"] else e["direction"]
    if e["contested"]:
        claim = (f"{e['taxon']} is reported with CONFLICTING direction in {e['disease']}: "
                 f"{e['n_up']} paper(s) report it enriched (higher in disease) and "
                 f"{e['n_down']} report it depleted (lower in disease).")
    else:
        higher = "higher" if e["direction"] == "enriched" else "lower"
        claim = (f"{e['taxon']} is {e['direction']} ({higher} in disease than in healthy "
                 f"controls) in {e['disease']}, reported by {e['n_papers']} paper(s) "
                 f"with no paper reporting the opposite direction.")
    bits = [claim]
    if e.get("rank"):
        bits.append(f"{e['taxon']} is a {e['rank']}.")
    bits.append(f"Evidence: {e['n_papers']} paper(s); directional consistency "
                f"{int(e['consistency']*100)}%.")
    if md:
        countries = sorted({m.get("country") for m in md if m.get("country")})
        seqs = sorted({m.get("sequencing") for m in md if m.get("sequencing")})
        sites = sorted({m.get("body_site") for m in md if m.get("body_site")})
        n = [m.get("n_cases", 0) + m.get("n_controls", 0) for m in md]
        n = [x for x in n if x]
        if countries:
            bits.append("Cohorts: " + ", ".join(countries[:6]) + ".")
        if seqs:
            bits.append("Methods: " + ", ".join(seqs[:4]) + ".")
        if sites:
            bits.append("Body site: " + ", ".join(sites[:4]) + ".")
        if n:
            bits.append(f"Cohort sizes: {min(n)}-{max(n)} subjects.")
    if e.get("papers"):
        bits.append("Sources: " + "; ".join(p[:90] for p in e["papers"][:4])
                    + (f" (+{len(e['papers'])-4} more)" if len(e["papers"]) > 4 else ""))
    return " ".join(bits)


_tok = re.compile(r"[a-z0-9']+")

# Paper titles are deliberately NOT indexed. Indexing them let a title's incidental
# vocabulary outrank the query's real subject: "Disturbance of Gut Bacteria and
# Metabolites Are Associated..." made `bacteria` (idf 3.26) and `are` (idf 3.02)
# score higher than `parkinson's` (idf 1.58, common across 289 docs), so a query
# about Parkinson's returned NMDAR encephalitis. Titles stay in the DISPLAY text.
STOP = set("a an the of in on for to and or is are was were be been being with by from "
           "as at it its this that these those what which who whom how why when where "
           "do does did can could should would may might will shall about into than then "
           "there here any some all more most other such no nor not only own same so too "
           "very s t just".split())


def toks(s, drop_stop=True):
    out = _tok.findall(s.lower())
    return [t for t in out if t not in STOP] if drop_stop else out


class Retriever:
    """Okapi BM25 over the claim text, plus explicit entity matching.

    BM25 alone is not enough here. Disease names are COMMON inside this corpus
    (289 docs mention Parkinson's) so their idf is low, while an incidental word
    from a query ("bacteria", "are") can be rare and score higher. Lexical
    frequency therefore ranks the wrong thing. Since we know the entity
    vocabulary exactly -- every disease and taxon in the graph -- we match it
    directly and boost, which is both more accurate and cheaper than embeddings."""

    def __init__(self, docs, k1=1.5, b=0.5):
        self.k1, self.b = k1, b
        self.meta = [d["meta"] for d in docs]
        self.diseases = sorted({m["disease"] for m in self.meta}, key=len, reverse=True)
        self.taxa = sorted({m["taxon"] for m in self.meta}, key=len, reverse=True)
        self.docs = [toks(d["index_text"]) for d in docs]
        self.N = len(self.docs)
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)

    def _entities(self, q):
        ql = " " + q.lower() + " "
        dis = [d for d in self.diseases if d.lower() in ql]
        tax = [t for t in self.taxa if len(t) > 4 and t.lower() in ql]
        return dis, tax

    def search(self, q, k=6):
        qt = toks(q)
        scores = defaultdict(float)
        for t in qt:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += idf * f * (self.k1 + 1) / denom

        dis, tax = self._entities(q)
        ql = q.lower()
        want_contested = any(w in ql for w in ("contested", "conflict", "disagree", "contradict"))
        # direction is an explicit filter, not a term to match: "depleted in X" must
        # not return enriched edges just because they mention the disease.
        want_dir = None
        if any(w in ql for w in ("deplet", "decreas", "lower", "reduc", "less")):
            want_dir = "depleted"
        elif any(w in ql for w in ("enrich", "increas", "higher", "elevat", "more abundant")):
            want_dir = "enriched"
        for i, m in enumerate(self.meta):
            if dis and m["disease"] in dis:
                scores[i] = scores.get(i, 0) + 12.0        # named disease dominates
            if tax and m["taxon"] in tax:
                scores[i] = scores.get(i, 0) + 12.0
            if want_contested and m["contested"]:
                scores[i] = scores.get(i, 0) + 6.0
            # among equals, prefer better-evidenced edges
            if scores.get(i):
                scores[i] += min(m["n_papers"], 12) * 0.9   # evidence weight, not a tiebreak
                # An edge whose MAJORITY direction is the one asked for is a fuller
                # answer than a contested edge that merely contains some of it.
                # Without this, "enriched in MS" put Prevotella (up=2, down=8) first.
                if want_dir:
                    if m["direction"] == want_dir:
                        scores[i] += 8.0
                    elif m["contested"]:
                        scores[i] += 2.0 * (m["n_up"] if want_dir == "enriched" else m["n_down"]) \
                                     / max(m["n_up"] + m["n_down"], 1)
        # a named entity in the query is a hard constraint, not a hint
        if dis or tax:
            scores = {i: s for i, s in scores.items()
                      if (not dis or self.meta[i]["disease"] in dis)
                      and (not tax or self.meta[i]["taxon"] in tax)}
        if want_contested:
            c = {i: s for i, s in scores.items() if self.meta[i]["contested"]}
            if c:
                scores = c
        elif want_dir:
            # keep contested edges: they DO report the asked-for direction, in part
            d = {i: s for i, s in scores.items()
                 if self.meta[i]["direction"] == want_dir or self.meta[i]["contested"]}
            if d:
                scores = d
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


def load_metadata():
    """paper title -> study metadata dict (empty if the pass has not run)."""
    if not os.path.exists(METADATA):
        return {}
    out = {}
    for line in open(METADATA):
        try:
            r = json.loads(line)
            if r.get("meta") and not r.get("parse_error"):
                out[r["title"]] = r["meta"]
        except Exception:
            continue
    return out


def build(graph_path=GRAPH, out=CORPUS):
    G = json.load(open(graph_path))
    meta_by_paper = load_metadata()
    docs = []
    for e in G["edges"]:
        md = [meta_by_paper[p] for p in e.get("papers", []) if p in meta_by_paper]
        d_full = edge_text(e, md or None)
        # what BM25 sees: the claim, without the Sources line (see STOP comment)
        d_index = d_full.split(" Sources:")[0]
        docs.append({
            "id": f"{e['taxon_key']}|{e['disease']}",
            "text": d_full,
            "index_text": d_index,
            "meta": {
                "taxon": e["taxon"], "taxon_key": e["taxon_key"], "rank": e.get("rank"),
                "resolved": e.get("resolved"), "disease": e["disease"], "mondo": e.get("mondo"),
                "direction": e["direction"], "contested": e["contested"],
                "n_papers": e["n_papers"], "n_up": e["n_up"], "n_down": e["n_down"],
                "consistency": e["consistency"], "papers": e.get("papers", [])[:25],
                "n_with_metadata": len(md),
            },
        })
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    enriched = sum(1 for d in docs if d["meta"]["n_with_metadata"])
    print(f"wrote {out}: {len(docs)} documents "
          f"({enriched} carry study metadata; {len(meta_by_paper)} papers had metadata)")
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--out", default=CORPUS)
    ap.add_argument("--query")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    if a.rebuild or not os.path.exists(a.out) or a.query is None:
        docs = build(a.graph, a.out)
    else:
        docs = [json.loads(l) for l in open(a.out)]

    if a.query:
        bm = Retriever(docs)
        hits = bm.search(a.query, a.k)
        print(f"\nquery: {a.query!r}\n" + "=" * 74)
        for i, (idx, sc) in enumerate(hits, 1):
            d = docs[idx]
            m = d["meta"]
            flag = " [CONTESTED]" if m["contested"] else ""
            print(f"\n{i}. score {sc:.2f}{flag}  {m['taxon']} / {m['disease']}  "
                  f"({m['n_papers']}p, up={m['n_up']} down={m['n_down']})")
            print("   " + d["text"][:300])


if __name__ == "__main__":
    main()
