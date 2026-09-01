#!/usr/bin/env python3
"""Push the chunk index into MongoDB Atlas, and query it with $vectorSearch.

The numpy array already answers plain nearest-neighbour queries in about a
millisecond, so Atlas is not here for speed. What it adds is:

  - persistence and access from somewhere other than this laptop
  - FILTERED vector search -- restrict a query to one disease or one paper.
    Doing that against the flat array means rescanning all 19,483 rows.
  - headroom if the corpus grows past what fits comfortably in memory

Credentials are read from MONGODB_URI (env, or ../.env). This script never
prints the URI, and never writes it anywhere.

    export MONGODB_URI='...'         # or put it in proj_2_attempt3/.env
    python atlas_ingest.py --push
    python atlas_ingest.py --query "how were stool samples stored"
    python atlas_ingest.py --query "antibiotic exclusion" --disease "Parkinson's disease (PD)"

BEFORE --push works you must create the vector index in Atlas (UI or CLI). It
is NOT created by inserting documents, and a dimension mismatch fails at QUERY
time with an empty result rather than at index time, which is a confusing way
to lose an afternoon:

    {
      "fields": [
        {"type": "vector", "path": "embedding",
         "numDimensions": 384, "similarity": "cosine"},
        {"type": "filter", "path": "title"},
        {"type": "filter", "path": "disease"}
      ]
    }

384 because that is all-MiniLM-L6-v2's output width. Our vectors are unit-norm,
so "cosine" and "dotProduct" rank identically; cosine is the safer default in
case a future model is not normalised.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
ENV = os.path.join(HERE, "..", ".env")
DB = "knightlab"
COLL = "chunks"
INDEX = "vector_index"
MODEL = "all-MiniLM-L6-v2"
BATCH = 500


def uri():
    u = os.environ.get("MONGODB_URI")
    if u:
        return u
    if os.path.exists(ENV):
        for line in open(ENV):
            if line.startswith("MONGODB_URI"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("MONGODB_URI not set (checked env and ../.env)")


def collection():
    try:
        from pymongo import MongoClient
    except ImportError:
        sys.exit("pip install 'pymongo[srv]'")
    c = MongoClient(uri(), appname="knightlab-kg")
    c.admin.command("ping")                 # fail fast on bad auth / IP allowlist
    return c[DB][COLL]


def push():
    if not (os.path.exists(CHUNKS) and os.path.exists(VECS)):
        sys.exit("index missing -- run:  python chunk_search.py --build")
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS)
    if len(rows) != V.shape[0]:
        sys.exit(f"{len(rows)} chunks vs {V.shape[0]} vectors -- rebuild first")

    col = collection()
    col.delete_many({})                     # full replace; the index survives
    print(f"pushing {len(rows)} chunks, dim {V.shape[1]}")
    for i in range(0, len(rows), BATCH):
        docs = [{
            "chunk_id": i + j,
            "title": r["title"],
            "disease": r.get("disease", ""),
            "text": r["text"],
            "embedding": V[i + j].tolist(),
        } for j, r in enumerate(rows[i:i + BATCH])]
        col.insert_many(docs)
        print(f"  {min(i+BATCH, len(rows))}/{len(rows)}", end="\r", flush=True)
    print(f"\ndone. documents: {col.count_documents({})}")
    print(f"index '{INDEX}' must exist with numDimensions={V.shape[1]}")


def query(q, k, disease):
    from sentence_transformers import SentenceTransformer
    qv = SentenceTransformer(MODEL).encode([q], normalize_embeddings=True)[0]

    stage = {
        "index": INDEX,
        "path": "embedding",
        "queryVector": qv.tolist(),
        # numCandidates is the ANN search width. Too low and recall suffers;
        # ~20x limit is the usual guidance.
        "numCandidates": max(200, k * 20),
        "limit": k,
    }
    if disease:
        stage["filter"] = {"disease": disease}

    col = collection()
    hits = list(col.aggregate([
        {"$vectorSearch": stage},
        {"$project": {"_id": 0, "title": 1, "disease": 1, "text": 1,
                      "score": {"$meta": "vectorSearchScore"}}},
    ]))
    if not hits:
        print("no results. usual causes: the vector index does not exist yet, "
              "its numDimensions is not 384, or the disease filter matched nothing.")
        return
    print(f'\nquery: "{q}"' + (f'   [disease={disease}]' if disease else "") + "\n")
    for rank, h in enumerate(hits, 1):
        print(f"{rank:>2}. [{h['score']:.3f}] {h['title'][:80]}")
        print(f"      {' '.join(h['text'].split())[:220]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--disease", default=None, help="restrict to one disease")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    if a.push:
        push()
    elif a.query:
        query(a.query, a.k, a.disease)
    else:
        ap.error("give --push or --query")


if __name__ == "__main__":
    main()
