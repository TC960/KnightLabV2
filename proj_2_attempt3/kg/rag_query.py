#!/usr/bin/env python3
"""Ask the corpus a question in plain words. An LLM fixes the query, embeddings find the text.

The gap this closes: `chunk_search.py` showed that querying a VARIABLE NAME fails
while querying a SENTENCE works.

    "hospital beds"                                     top 0.347, mostly junk
    "participants had not taken antibiotics in the
     previous three months"                             top 0.707, 4/4 correct

Nobody writes "hospital beds" in a methods section, so nearest-neighbour lands on
whatever else contains the word "hospital" -- which is overwhelmingly author
affiliations. The fix is not a better embedding model. It is to turn the keyword
into the sentences a paper would actually contain, which is exactly what an LLM
is good at and costs a few hundred tokens.

    keyword  ->  LLM writes N paraphrases in methods-section prose
             ->  retrieve against each, merge, keep each paper's best chunk
             ->  (optional) LLM answers, grounded ONLY in retrieved text

Usage
    python rag_query.py "hospital beds"
    python rag_query.py "were patients on antibiotics" --answer
    python rag_query.py "how were stool samples stored" -k 15 --answer

Expansions are cached in rag_cache/ so re-running a query is free.

The --answer mode is told to say when the corpus does not support a claim, and
every sentence must carry a [n] pointing at a retrieved chunk. Treat it as a
reading aid, not a result: it summarises what was retrieved, and retrieval at
this corpus size has a lift of ~2-6x over random (see variable_sweep.py), so it
can and will miss things.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
CACHE = os.path.join(HERE, "rag_cache")
ENV = os.path.join(HERE, "..", ".env")
MODEL_EMBED = "all-MiniLM-L6-v2"
MODEL_EXPAND = "claude-haiku-4-5-20251001"     # cheap; this is a rewriting task
MODEL_ANSWER = "claude-sonnet-5"
API = "https://api.anthropic.com/v1/messages"


def api_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    if os.path.exists(ENV):
        for line in open(ENV):
            if line.startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ANTHROPIC_API_KEY not found (checked env and ../.env)")


def claude(prompt, model, max_tokens=1024):
    body = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "content-type": "application/json",
        "x-api-key": api_key(),
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return "".join(b.get("text", "") for b in out.get("content", []))


EXPAND_PROMPT = """You are helping search a corpus of gut-microbiome case-control papers.

A user typed this search term:

    {q}

Write {n} sentences that a paper's Methods or Results section would PLAUSIBLY \
CONTAIN if that paper reported on this topic. Not questions. Not definitions. \
Not the term itself. Write them the way a paper writes them, including the \
typical numbers, units and hedging.

Example -- for "antibiotic washout" a good set includes:
  Participants who had received antibiotics within three months prior to sampling were excluded.
  None of the subjects had taken antibiotics or probiotics in the preceding 4 weeks.
  Antibiotic use in the past 6 months was an exclusion criterion for both groups.

Vary the phrasing so the set covers different ways authors express it. Output \
ONLY the sentences, one per line, no numbering, no commentary."""

ANSWER_PROMPT = """Below are passages retrieved from gut-microbiome papers, \
numbered [1]..[{n}].

QUESTION: {q}

Answer using ONLY these passages. Rules:
- Every claim must end with the bracket number(s) it came from, e.g. [3].
- If the passages do not answer the question, say so plainly. Do not speculate.
- If passages disagree, say so and cite both.
- Note explicitly if a passage is boilerplate (ethics statement, author \
contributions, funding) rather than a study fact.
- Be brief: a short paragraph, or bullets.

PASSAGES
{ctx}"""


def expand(q, n):
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, hashlib.sha1(f"{q}|{n}".encode()).hexdigest() + ".json")
    if os.path.exists(key):
        return json.load(open(key))
    txt = claude(EXPAND_PROMPT.format(q=q, n=n), MODEL_EXPAND)
    qs = [re.sub(r"^[-*\d.)\s]+", "", l).strip()
          for l in txt.splitlines() if len(l.strip()) > 25]
    qs = qs[:n] or [q]
    json.dump(qs, open(key, "w"))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="+")
    ap.add_argument("-k", type=int, default=10, help="papers to return")
    ap.add_argument("-n", type=int, default=6, help="expansions to generate")
    ap.add_argument("--answer", action="store_true", help="LLM synthesis (costs more)")
    ap.add_argument("--raw", action="store_true", help="skip expansion, query verbatim")
    a = ap.parse_args()
    q = " ".join(a.q)

    if not (os.path.exists(CHUNKS) and os.path.exists(VECS)):
        sys.exit("index missing -- run:  python chunk_search.py --build")
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS)

    queries = [q] if a.raw else expand(q, a.n)
    print(f'\nquery: "{q}"')
    if not a.raw:
        print("expanded to:")
        for s in queries:
            print(f"   - {s}")

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_EMBED)
    qv = m.encode(queries, normalize_embeddings=True)
    # max over expansions: a chunk matching ANY phrasing should score well.
    # Averaging would punish a chunk that matches one phrasing precisely.
    sims = (V @ qv.T).max(axis=1)

    best = {}
    for i, s in enumerate(sims):
        t = rows[i]["title"]
        if t not in best or s > best[t][0]:
            best[t] = (float(s), i)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:a.k]

    print(f"\ntop {len(ranked)} papers:\n")
    ctx = []
    for rank, (title, (s, i)) in enumerate(ranked, 1):
        body = re.sub(r"\s+", " ", rows[i]["text"])
        print(f"[{rank}] ({s:.3f}) {title[:88]}")
        print(f"     {body[:230]}\n")
        ctx.append(f"[{rank}] from \"{title[:80]}\"\n{body[:900]}")

    if a.answer:
        print("=" * 72)
        print(claude(ANSWER_PROMPT.format(q=q, n=len(ctx), ctx="\n\n".join(ctx)),
                     MODEL_ANSWER, max_tokens=1200).strip())


if __name__ == "__main__":
    main()
