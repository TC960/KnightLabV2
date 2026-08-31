#!/usr/bin/env python3
"""Semantic chunk index over the corpus: type a phrase, get the passages that discuss it.

This is the DIRECT version of the embedding idea -- you already suspect a variable
("hospital beds", "antibiotic washout"), so you query for it and read what comes
back. It is hypothesis TESTING, and it is the cheap thing to do first. Probing
(fit a classifier, read the axis) is hypothesis GENERATION and only earns its keep
once you have run out of things to guess.

Three decisions here are load-bearing:

1. FRONT MATTER AND REFERENCES ARE DROPPED.
   `audit_variable_coverage.py` showed the top "recruitment setting" hit in the
   corpus was "Department of General Surgery, Shanghai Tenth People's Hospital,
   Tongji ..." -- an author affiliation. Coverage fell 96.4% -> 34.7% once hits
   had to be real statements. Query "hospital" against raw text and you retrieve
   the authors, not the cohort. Same error already recorded for `country` in
   FINDINGS_task0_rescore.md.

2. CHUNKS BREAK ON SENTENCE BOUNDARIES, NEVER MID-WORD.
   Sentences are packed up to a character budget with a one-sentence overlap so a
   fact split across a boundary survives in at least one chunk.

3. EMBEDDINGS ARE CACHED.
   ~17k chunks on CPU is minutes, not seconds. Re-running a query must be instant,
   so vectors go to an .npy keyed by a hash of the chunk text.

Usage
    python chunk_search.py --build                     # build + embed (once)
    python chunk_search.py "hospital beds"             # query
    python chunk_search.py "antibiotic washout period" -k 15
    python chunk_search.py "recruited from the community" --by-paper

`--by-paper` collapses to one row per paper (max chunk score), which is the view
you want when the question is "how many papers discuss this".
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "new_papers.json"),
]
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
MODEL = "all-MiniLM-L6-v2"

FRONT_CHARS = 2500          # title + authors + affiliations
TARGET = 900                # chars per chunk; MiniLM truncates ~256 tokens anyway
OVERLAP_SENTS = 1

REF_HEAD = re.compile(
    r"\n\s*(?:References|REFERENCES|Bibliography|Literature Cited)\s*\n", re.M)
# Split on sentence enders followed by whitespace + a capital/digit. Protects the
# common biomedical abbreviations that would otherwise cut a sentence in half.
ABBREV = re.compile(r"\b(?:et al|e\.g|i\.e|vs|Fig|Dr|approx|no|No|cf|spp|sp)\.$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def body_of(text):
    m = list(REF_HEAD.finditer(text))
    end = m[-1].start() if m and m[-1].start() > len(text) * 0.4 else len(text)
    return text[FRONT_CHARS:end]


def sentences(text):
    out = []
    buf = ""
    for piece in SENT_SPLIT.split(text):
        buf = (buf + " " + piece).strip() if buf else piece.strip()
        # an abbreviation at the end means the split was spurious -- keep accruing
        if ABBREV.search(buf):
            continue
        if buf:
            out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def chunk_paper(text):
    """Pack whole sentences up to TARGET chars, with a one-sentence overlap."""
    sents = [s for s in sentences(text) if len(s) > 25]
    chunks, cur, n = [], [], 0
    for s in sents:
        if cur and n + len(s) > TARGET:
            chunks.append(" ".join(cur))
            cur = cur[-OVERLAP_SENTS:] if OVERLAP_SENTS else []
            n = sum(len(x) for x in cur)
        cur.append(s)
        n += len(s)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) > 120]


def s(v):
    """Fields came through pandas, so a blank is NaN (a float), not ''."""
    if v is None or isinstance(v, float):
        return ""
    return str(v).strip()


def load_papers():
    papers, seen = [], set()
    for src in SOURCES:
        if not os.path.exists(src):
            continue
        for p in json.load(open(src)):
            t = s(p.get("title"))
            if not t or t.lower() in seen or not p.get("text"):
                continue
            seen.add(t.lower())
            papers.append({"title": t, "disease": s(p.get("disease")),
                           "text": p["text"]})
    return papers


def build():
    papers = load_papers()
    print(f"{len(papers)} papers")
    rows = []
    for p in papers:
        for c in chunk_paper(body_of(p["text"])):
            rows.append({"title": p["title"], "disease": p["disease"], "text": c})
    print(f"{len(rows)} chunks  (mean {np.mean([len(r['text']) for r in rows]):.0f} chars)")

    with open(CHUNKS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from sentence_transformers import SentenceTransformer
    print(f"embedding with {MODEL} (CPU, this takes a few minutes)...")
    m = SentenceTransformer(MODEL)
    V = m.encode([r["text"] for r in rows], batch_size=64,
                 show_progress_bar=True, normalize_embeddings=True)
    np.save(VECS, V.astype(np.float32))
    print(f"wrote {CHUNKS} and {VECS}  shape={V.shape}")


def load_index():
    if not (os.path.exists(CHUNKS) and os.path.exists(VECS)):
        sys.exit("index missing -- run:  python chunk_search.py --build")
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS)
    if len(rows) != V.shape[0]:
        sys.exit(f"index corrupt: {len(rows)} chunks vs {V.shape[0]} vectors; rebuild")
    return rows, V


def query(q, k, by_paper):
    rows, V = load_index()
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    qv = m.encode([q], normalize_embeddings=True)[0]
    sims = V @ qv                      # both normalized -> cosine

    if by_paper:
        best = {}
        for i, s in enumerate(sims):
            t = rows[i]["title"]
            if t not in best or s > best[t][0]:
                best[t] = (float(s), i)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        print(f'\nquery: "{q}"   (top {k} PAPERS by best chunk)\n')
        for rank, (title, (s, i)) in enumerate(ranked, 1):
            print(f"{rank:>2}. [{s:.3f}] {title[:88]}")
            print(f"      {re.sub(chr(92)+'s+', ' ', rows[i]['text'])[:200]}\n")
    else:
        # The one-sentence overlap between adjacent chunks means neighbours are
        # near-identical; without this they occupy two slots in the top-k and
        # crowd out a different paper. Key on the opening of the text.
        seen, top = set(), []
        for i in np.argsort(-sims):
            key = (rows[i]["title"], re.sub(r"\W+", "", rows[i]["text"])[:90])
            if key in seen:
                continue
            seen.add(key)
            top.append(i)
            if len(top) == k:
                break
        print(f'\nquery: "{q}"   (top {k} CHUNKS)\n')
        for rank, i in enumerate(top, 1):
            r = rows[i]
            print(f"{rank:>2}. [{sims[i]:.3f}] {r['title'][:78]}")
            print(f"      {re.sub(chr(92)+'s+', ' ', r['text'])[:260]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="*", help="query phrase")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--by-paper", action="store_true")
    a = ap.parse_args()

    if a.build:
        build()
        return
    if not a.q:
        sys.exit('give a query, e.g.  python chunk_search.py "hospital beds"')
    query(" ".join(a.q), a.k, a.by_paper)


if __name__ == "__main__":
    main()
