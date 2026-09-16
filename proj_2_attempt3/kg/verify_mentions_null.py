#!/usr/bin/env python3
"""Negative control for verify_taxon_mentions.py.

A 99.6% mention rate proves nothing on its own. Microbiome papers share a small
core vocabulary -- `Bacteroides`, `Firmicutes`, `Faecalibacterium` appear in
almost every one of them -- so a taxon list checked against a RANDOM paper from
the same corpus might also "match" at 90%+. The quantity that carries
information is the GAP between the true pairing and that null.

Design, per the repo rule that observations are not independent: the shuffle is
at the PAPER level. Each paper keeps its own taxon list intact and is re-paired
with a different paper's full text. Shuffling individual claims would break the
within-paper correlation and inflate the gap.

Reported: the true rate, the null distribution over N permutations, the gap, and
an empirical p-value. Also a per-taxon breakdown of how discriminating each
claim is, because the aggregate hides that a handful of ubiquitous genera carry
no evidential weight at all.
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

from verify_taxon_mentions import (EXTRACTIONS, PAPERS, classify, norm_taxon,
                                   norm_text, squash)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "taxon_mentions_null.json")
N_PERM = 200
SEED = 42


def load():
    extractions = json.load(open(EXTRACTIONS))
    papers = json.load(open(PAPERS))
    by_title = {}
    for p in papers:
        key = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())
        if key and len(p.get("text") or "") > 500:
            t = norm_text(p["text"])
            by_title[key] = (t, squash(t))

    # paper -> list of claimed taxon surface strings (deduped within paper)
    claims = defaultdict(list)
    for rec in extractions:
        key = re.sub(r"[^a-z0-9]", "", (rec.get("title") or "").lower())
        if key not in by_title:
            continue
        seen = set()
        for field in ("predicted_enriched", "predicted_depleted"):
            raw = rec.get(field) or ""
            if not isinstance(raw, str):
                continue
            for chunk in re.split(r"[,;]", raw):
                t = chunk.strip()
                n = norm_taxon(t)
                if not n or len(n) < 3 or n in seen:
                    continue
                seen.add(n)
                claims[key].append(t)
    claims = {k: v for k, v in claims.items() if v}
    return claims, by_title


def rate(claims, by_title, pairing, strict=True):
    """Fraction of claims found in the paired text.
    strict=True excludes the weak `head` (genus-only) tier."""
    hit = tot = 0
    for paper_key, taxa in claims.items():
        text, text_sq = by_title[pairing[paper_key]]
        for t in taxa:
            tier, _ = classify(t, text, text_sq)
            if tier == "unscoreable":
                continue
            tot += 1
            if tier in ("exact", "variant", "abbrev") or (not strict and tier == "head"):
                hit += 1
    return hit / tot if tot else 0.0, tot


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM
    claims, by_title = load()
    keys = sorted(claims)
    print(f"{len(keys)} papers, {sum(len(v) for v in claims.values())} claims", flush=True)

    identity = {k: k for k in keys}
    true_strict, tot = rate(claims, by_title, identity, strict=True)
    true_any, _ = rate(claims, by_title, identity, strict=False)
    print(f"TRUE  strict={true_strict:.4f}  any={true_any:.4f}  (n={tot})", flush=True)

    rng = random.Random(SEED)
    null = []
    for i in range(n_perm):
        # derangement-ish: reshuffle until no paper keeps its own text
        shuf = keys[:]
        for _ in range(50):
            rng.shuffle(shuf)
            if all(a != b for a, b in zip(keys, shuf)):
                break
        r, _ = rate(claims, by_title, dict(zip(keys, shuf)), strict=True)
        null.append(r)
        if (i + 1) % 25 == 0:
            print(f"  perm {i+1}/{n_perm}  mean null={sum(null)/len(null):.4f}", flush=True)

    null_mean = sum(null) / len(null)
    null_sd = (sum((x - null_mean) ** 2 for x in null) / max(1, len(null) - 1)) ** 0.5
    n_ge = sum(1 for x in null if x >= true_strict)
    p = (n_ge + 1) / (n_perm + 1)

    # Per-taxon discriminativeness: how many of the corpus's papers contain this
    # string at all? A taxon in 90% of papers is not evidence of anything.
    texts = [v[0] for v in by_title.values()]
    ubiquity = {}
    all_taxa = sorted({norm_taxon(t) for v in claims.values() for t in v})
    for t in all_taxa:
        if len(t) < 3:
            continue
        ubiquity[t] = sum(1 for tx in texts if t in tx) / len(texts)

    ub_sorted = sorted(ubiquity.items(), key=lambda kv: -kv[1])
    # weight each claim by (1 - ubiquity): evidential mass, not claim count
    mass_hit = mass_tot = 0.0
    for paper_key, taxa in claims.items():
        text, text_sq = by_title[paper_key]
        for t in taxa:
            n = norm_taxon(t)
            if n not in ubiquity:
                continue
            w = 1.0 - ubiquity[n]
            mass_tot += w
            tier, _ = classify(t, text, text_sq)
            if tier in ("exact", "variant", "abbrev"):
                mass_hit += w

    out = {
        "n_papers": len(keys),
        "n_claims": tot,
        "true_strict": round(true_strict, 4),
        "true_any_incl_head": round(true_any, 4),
        "n_permutations": n_perm,
        "null_mean_strict": round(null_mean, 4),
        "null_sd": round(null_sd, 4),
        "null_min": round(min(null), 4),
        "null_max": round(max(null), 4),
        "gap_points": round((true_strict - null_mean) * 100, 2),
        "p_empirical": p,
        "rarity_weighted_rate": round(mass_hit / mass_tot, 4) if mass_tot else None,
        "most_ubiquitous_20": [[t, round(u, 3)] for t, u in ub_sorted[:20]],
        "frac_claims_in_over_half_of_corpus": round(
            sum(1 for v in claims.values() for t in v
                if ubiquity.get(norm_taxon(t), 0) > 0.5) / tot, 4),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
