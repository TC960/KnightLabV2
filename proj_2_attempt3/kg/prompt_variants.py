#!/usr/bin/env python3
"""Prompt variants for the gate experiment, plus the stratified paper subset.

Design and pre-registered decision rule: PROMPT_EXPERIMENT.md.

The baseline is `samgated-v1` exactly as it ships in eval-v2/run_eval.py. Each
variant changes ONE gate so a difference in F1 is attributable. Everything else --
model, quantisation, temperature, grammar, examples -- is held fixed.

    python prompt_variants.py --make-subset      # write the 80-paper subset
    python prompt_variants.py --show B           # print a variant to eyeball it
"""
import argparse
import collections
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "run_eval.py")
SUBSET = os.path.join(HERE, "prompt_exp_subset.json")
N_SUBSET = 80
SEED = 17


def baseline():
    """The shipped template, read from source so it cannot drift out of sync."""
    src = open(EVAL).read()
    m = re.search(r'PROMPT_TEMPLATE = """(.*?)"""', src, re.S)
    if not m:
        raise SystemExit("could not find PROMPT_TEMPLATE in run_eval.py")
    return m.group(1)


SIG_ORIGINAL = """- SIGNIFICANCE: include a taxon ONLY if the paper reports it as statistically significant
  (p / FDR / q < 0.05, or a significant LEfSe/LDA or differential-abundance result). If significance
  is unclear or unreported for a taxon, omit it."""

# B: the gate that the local measurement says is costing recall. 159 of the 653
# missed gold taxa sit in a sentence that DOES carry a significance cue, so the
# model is already seeing evidence it is declining to use -- most likely because
# the statistic is reported elsewhere (a table, an earlier sentence) than the
# sentence naming the taxon. This wording keeps the requirement that the finding
# BE significant while allowing the evidence to live nearby rather than inline.
# It deliberately still rejects the two things the audit found the model was right
# to omit: purely descriptive abundance statements, and correlation-with-symptom
# findings.
SIG_SOFT = """- SIGNIFICANCE: include a taxon if the paper presents it as a significant difference between
  the disease and control groups. The supporting statistic (p / FDR / q value, LEfSe/LDA score, or
  differential-abundance result) does NOT have to appear in the same sentence as the taxon name --
  it may be reported elsewhere in the paper, including in a table the text refers to. Do NOT include
  a taxon that is only described as abundant, dominant or present, and do NOT include one whose only
  reported relationship is a correlation with a symptom, score or severity measure."""

TEXT_ORIGINAL = """- MAIN TEXT ONLY: only include taxa named in the running text (Abstract, Results, Discussion).
  Ignore taxa that appear only in tables, figures, figure/table captions, or supplementary material."""

# C: a control. The local measurement says this should do NOTHING -- across 50
# papers, zero gold taxa appeared in a table but not in the body text. If C moves
# F1, that measurement was wrong and we need to know.
TEXT_OPEN = """- SOURCES: include taxa named anywhere in the paper, including tables, figure and table captions,
  and supplementary material, as well as the running text."""


def variant(vid):
    p = baseline()
    if vid == "A":
        return p
    if vid == "B":
        assert SIG_ORIGINAL in p, "significance gate text not found -- prompt changed?"
        return p.replace(SIG_ORIGINAL, SIG_SOFT)
    if vid == "C":
        assert TEXT_ORIGINAL in p, "main-text gate not found -- prompt changed?"
        return p.replace(TEXT_ORIGINAL, TEXT_OPEN)
    raise SystemExit(f"unknown variant {vid}")


def make_subset():
    """80 papers stratified by disease, drawn only from the scoreable gold set."""
    import csv

    def doi_of(s):
        m = re.search(r"10\.\d{4,9}/[^\s\"<>,;\]]+", s or "")
        return m.group(0).rstrip(".").lower() if m else None

    gold = {}
    for x in csv.DictReader(open("/Users/mohak/Downloads/high_confidence - "
                                 "final_constrained_override.csv",
                                 encoding="utf-8-sig")):
        d = x["DOI"].strip().lower()
        gold.setdefault(d, {"taxa": "", "disorder": x["disorder"].strip()})
        gold[d]["taxa"] += ";" + x["high_confidence_taxa"]

    sheet = os.path.join(HERE, "Microbiota Signatures Neurological Disorders "
                               "Sheet 2 - Main Datasheet.csv")
    d2t = {}
    for x in csv.DictReader(open(sheet, encoding="utf-8-sig")):
        d = doi_of((x.get("DOI") or "").strip().lower()) or ""
        t = (x.get("Title") or "").strip()
        if d and t:
            d2t[d] = t

    texts = {}
    for src in ["../EmilySong_GoldStandardPaper/all_usable_papers.json",
                "new_papers.json"]:
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            t = r.get("title")
            if isinstance(t, str) and r.get("text"):
                texts[t.strip()] = r["text"]

    pool = []
    for d, g in gold.items():
        if not g["taxa"].strip(";").strip():
            continue                      # blank gold cannot be scored
        t = d2t.get(d)
        if not t or t not in texts:
            continue
        pool.append({"doi": d, "title": t, "disease": g["disorder"],
                     "text": texts[t], "char_len": len(texts[t])})

    by_dis = collections.defaultdict(list)
    for p in pool:
        by_dis[p["disease"]].append(p)

    rng = random.Random(SEED)
    # Proportional allocation, then top up at random to reach exactly N_SUBSET.
    # The top-up matters: without it, rounding down across many small strata left
    # the subset at 51 of the intended 80.
    out = []
    for dis, papers in by_dis.items():
        share = max(1, int(N_SUBSET * len(papers) / len(pool)))
        out += rng.sample(papers, min(share, len(papers)))
    if len(out) > N_SUBSET:
        out = rng.sample(out, N_SUBSET)
    else:
        chosen = {p["doi"] for p in out}
        rest = [p for p in pool if p["doi"] not in chosen]
        rng.shuffle(rest)
        out += rest[:N_SUBSET - len(out)]

    json.dump(out, open(SUBSET, "w"))
    print(f"pool (scoreable, with text): {len(pool)} papers")
    print(f"subset written              : {len(out)} papers -> {SUBSET}")
    c = collections.Counter(p["disease"] for p in out)
    for k, v in c.most_common(8):
        print(f"    {v:>3}  {k[:52]}")
    tot = sum(p["char_len"] for p in out)
    mins = tot / 1e4 * 3.33 / 60     # 3.33 s per 10k chars, measured on the prior run
    print(f"\ntotal chars {tot/1e6:.1f}M -> ~{mins:.0f} min per variant, "
          f"~{3*mins:.0f} min for all three")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-subset", action="store_true")
    ap.add_argument("--show")
    a = ap.parse_args()
    if a.make_subset:
        make_subset()
    elif a.show:
        print(variant(a.show.upper()))
    else:
        for v in "ABC":
            p = variant(v)
            print(f"variant {v}: {len(p)} chars, "
                  f"{'BASELINE' if v == 'A' else 'modified'}")


if __name__ == "__main__":
    main()
