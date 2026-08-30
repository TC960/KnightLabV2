#!/usr/bin/env python3
"""Adjudicate the taxa the Opus 4.8 re-annotation adds over the human gold.

Answers two questions the viewer cannot:

1. **How many extras are real?** The viewer paints a taxon yellow on an exact
   lowercase string diff, so near-misses ("Bacteroidetes" vs "Bacteroidota",
   rank prefixes, spelling) count as additions when the scoring metric would
   match them. This re-runs the diff through `taxonomy_match.match_taxa_lca`,
   the same matcher the leaderboard uses, to get the number that matters.

2. **Are the extras defensible?** For each genuine extra, look at every sentence
   naming it and bucket by the strongest evidence present:

       strong        a hard statistical cue (p-value, LEfSe, LDA, FDR, "significantly")
       weak          only soft language (enriched / abundant / increased / reduced)
       mention_only  named in the paper, but no significance language anywhere
       absent        name never appears in the text -> hallucination or figure-only

CAVEAT: bucketing is sentence-level co-occurrence, not attribution. A sentence
listing six taxa alongside one p-value puts all six in `strong`, which is right
when the cue distributes over the list and wrong when it doesn't. Treat `strong`
as "worth a human's time", not "confirmed".

Usage:  python adjudicate_extras.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EVAL))
sys.path.insert(0, EVAL)

from taxonomy_match import match_taxa_lca  # noqa: E402

GOLD = os.path.join(ROOT, "EmilySong_GoldStandardPaper", "test_set_v2.json")
OPUS = os.path.join(EVAL, "results", "opus-4.8-gold.json")

STRONG = re.compile(
    r"(p\s*[<=>]\s*0?\.\d|p-value|LEfSe|LDA\s*score|FDR|q\s*[<=]|significantl)", re.I
)
SOFT = re.compile(
    r"(significant|enrich|deplet|abundan|elevated|reduced|decreas|increas)", re.I
)


def split_taxa(s):
    return [t.strip() for t in re.split(r"[;,]", s or "") if t.strip()]


def sentences(t):
    return re.split(r"(?<=[.!?])\s+", t)


def main():
    gold = json.load(open(GOLD))
    opus = json.load(open(OPUS))
    gby = {g["link"]: g for g in gold}

    n_gold = n_opus = n_naive = n_real = 0
    buckets = {"strong": 0, "weak": 0, "mention_only": 0, "absent": 0}
    absent = []

    for f in opus:
        g = gby.get(f["link"], {})
        gset = {t.lower() for t in split_taxa(g.get("taxa_enriched")) + split_taxa(g.get("taxa_depleted"))}
        extras_cased = [t for t in split_taxa(f.get("taxa_enriched")) + split_taxa(f.get("taxa_depleted"))
                        if t.lower() not in gset]
        fset = {t.lower() for t in split_taxa(f.get("taxa_enriched")) + split_taxa(f.get("taxa_depleted"))}

        n_gold += len(gset)
        n_opus += len(fset)
        n_naive += len(extras_cased)

        pred, exp = sorted(fset), sorted(gset)
        if pred and exp:
            _, fp, _ = match_taxa_lca(pred, exp)
        else:
            fp = len(pred)
        n_real += fp

        sents = sentences(f["text"])
        for t in extras_cased:
            head = t.split()[0] if t.split() else t
            pat = re.compile(r"\b" + re.escape(head) + r"\w*", re.I)
            hits = [s for s in sents if pat.search(s)]
            if not hits:
                buckets["absent"] += 1
                absent.append((t, f["title"][:40]))
            elif any(STRONG.search(s) for s in hits):
                buckets["strong"] += 1
            elif any(SOFT.search(s) for s in hits):
                buckets["weak"] += 1
            else:
                buckets["mention_only"] += 1

    print(f"human gold taxa                : {n_gold}")
    print(f"Opus 4.8 re-annotation taxa    : {n_opus}")
    print()
    print(f"extras, exact-string diff      : {n_naive}   <- what build_viewer paints yellow")
    print(f"extras, taxonomy-aware matcher : {n_real}   <- the real disagreement (the documented '~70')")
    print(f"  absorbed as near-matches     : {n_naive - n_real}")
    print()
    total = sum(buckets.values())
    print(f"evidence for the {total} string-diff extras:")
    for k, label in [
        ("strong", "hard stat cue in a sentence naming it (p/LEfSe/LDA/FDR)"),
        ("weak", "soft language only (enriched/abundant/increased)"),
        ("mention_only", "named, but no significance language at all"),
        ("absent", "name absent from the paper text entirely"),
    ]:
        print(f"  {buckets[k]:4}  {100 * buckets[k] / total:5.1f}%  {label}")
    if absent:
        print("\nabsent-from-text (check these first):")
        for t, title in absent:
            print(f"  [{t}]  <- {title}")


if __name__ == "__main__":
    main()
