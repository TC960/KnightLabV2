#!/usr/bin/env python3
"""Do papers that AGREE on a contested edge use more similar methods than those that disagree?

THE IDEA THIS TESTS. Rather than hoping some dimension of a generic embedding
means "study design", build a SEPARATE embedding from only the methods text. Then
cosine similarity between two papers' methods-vectors IS "do these studies use
similar protocols" -- interpretable by construction, no disentanglement required,
no need to discover what any dimension means.

WHY THIS FRAMING AND NOT THE EARLIER ONE. A previous analysis asked "what property
of a study makes it report UP?" -- malformed, because a study reports a list of
taxa, some up and some down. That question is retired. This one is well posed:
inside a single contested edge every paper takes exactly one side, so the
comparison is between two papers that genuinely disagree about the same claim.

THE TEST.
    for each contested edge with >= 2 papers per side
        AGREE pairs    = both papers on the same side
        DISAGREE pairs = one paper from each side
    compare mean methods-similarity of AGREE vs DISAGREE pairs

If protocol drives disagreement, agreeing papers should share methods. Null is
built by shuffling side labels WITHIN each edge, which preserves each edge's
paper composition and its up/down counts -- the same scheme used elsewhere in this
project, because observations are not independent across edges.

Reported against the generic all-chunks embedding as a control: if methods-only
does no better than generic text, the section split is not buying anything.

    python methods_similarity.py
"""
import collections
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = os.path.join(HERE, "chunks.jsonl")
VECS = os.path.join(HERE, "chunk_vecs.npy")
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "methods_similarity.json")
N_PERM = 5000

# Cues for text that describes HOW the study was run. Deliberately protocol-only:
# no taxon names, no direction words, nothing about findings -- otherwise the
# "methods" vector would smuggle in the result it is supposed to be independent of.
METHODS = re.compile(
    r"\b(DNA was extracted|extraction kit|QIAamp|PowerSoil|MoBio|sequenc\w+ (was|were) "
    r"performed|PCR|primers|V3-V4|V4 region|16S rRNA gene|shotgun metagenomic|"
    r"QIIME|DADA2|SILVA|OTU|ASV|Illumina|MiSeq|HiSeq|were recruited|inclusion criteri\w+|"
    r"exclusion criteri\w+|institutional review|informed consent|stored at -80|"
    r"LEfSe|DESeq2|Wilcoxon|Mann-Whitney|linear discriminant)\b", re.I)


def paper_vectors(rows, V, mask_fn):
    """Mean of the chunks a paper contributes that pass mask_fn, L2-normalised."""
    acc, cnt = collections.defaultdict(lambda: np.zeros(V.shape[1])), collections.Counter()
    for i, r in enumerate(rows):
        if mask_fn(r["text"]):
            acc[r["title"]] += V[i]
            cnt[r["title"]] += 1
    out = {}
    for t, v in acc.items():
        v = v / cnt[t]
        n = np.linalg.norm(v)
        if n > 0:
            out[t] = v / n
    return out, cnt


def edges_with_sides(pidx):
    g = json.load(open(GRAPH))
    ptbl = g.get("papers", [])

    def title_at(i):
        if not isinstance(i, int) or i >= len(ptbl):
            return None
        rec = ptbl[i]
        return (rec["title"] if isinstance(rec, dict) else rec).strip()

    out = []
    for e in g["edges"]:
        if not e.get("contested"):
            continue
        up, dn = set(), set()
        for ev in (e.get("ev") or []):
            t = title_at(ev.get("i"))
            if t and t in pidx:
                (up if ev.get("d") == "e" else dn).add(t)
        if len(up) >= 2 and len(dn) >= 2:
            out.append((sorted(up), sorted(dn)))
    return out


def statistic(edges, vec):
    """mean(similarity | agree) - mean(similarity | disagree), pooled over edges."""
    agree, disagree = [], []
    for up, dn in edges:
        for grp in (up, dn):
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    agree.append(float(vec[grp[a]] @ vec[grp[b]]))
        for a in up:
            for b in dn:
                disagree.append(float(vec[a] @ vec[b]))
    if not agree or not disagree:
        return None, 0, 0
    return float(np.mean(agree) - np.mean(disagree)), len(agree), len(disagree)


def run(name, vec, edges, rng):
    obs, na, nd = statistic(edges, vec)
    if obs is None:
        print(f"{name}: not enough pairs")
        return None
    null = []
    for _ in range(N_PERM):
        shuf = []
        for up, dn in edges:
            pool = up + dn
            p = list(rng.permutation(pool))
            shuf.append((p[:len(up)], p[len(up):]))
        s, _, _ = statistic(shuf, vec)
        null.append(s)
    null = np.array(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    lo, hi = np.percentile(null, [2.5, 97.5])
    print(f"  {name:<22} agree-pairs {na:>4}  disagree-pairs {nd:>4}")
    print(f"  {'':<22} observed gap {obs:+.5f}   null 95% [{lo:+.5f}, {hi:+.5f}]   p = {p:.4f}")
    return {"statistic": obs, "p": float(p), "n_agree": na, "n_disagree": nd,
            "null_lo": float(lo), "null_hi": float(hi)}


def main():
    rows = [json.loads(l) for l in open(CHUNKS)]
    V = np.load(VECS).astype(np.float64)
    if len(rows) != V.shape[0]:
        raise SystemExit(f"{len(rows)} chunks vs {V.shape[0]} vectors -- rebuild")

    meth, mcnt = paper_vectors(rows, V, lambda t: bool(METHODS.search(t)))
    gen, gcnt = paper_vectors(rows, V, lambda t: True)
    print(f"papers with methods chunks : {len(meth)}  "
          f"(median {int(np.median(list(mcnt.values())))} chunks each)")
    print(f"papers with any chunks     : {len(gen)}")

    edges = edges_with_sides(meth)
    print(f"contested edges usable     : {len(edges)}\n")

    rng = np.random.default_rng(0)
    res = {}
    print("Does methods-similarity separate agreeing from disagreeing papers?")
    res["methods_only"] = run("methods-only", meth, edges, rng)
    print()
    print("Control -- the same test on generic all-text embeddings:")
    res["generic_text"] = run("generic all-chunks", gen, edges, rng)

    m, g = res["methods_only"], res["generic_text"]
    if m and g:
        print()
        if m["p"] < 0.05:
            print("  -> methods similarity DOES track agreement. The section split buys something.")
        else:
            print("  -> NULL. Papers that agree on an edge do not use measurably more")
            print("     similar methods than papers that disagree.")
            sd = (m["null_hi"] - m["null_lo"]) / (2 * 1.96)
            print(f"     Smallest gap this design could detect: ~{1.96*sd:.4f}; "
                  f"observed {abs(m['statistic']):.4f}.")

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
