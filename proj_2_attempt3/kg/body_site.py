#!/usr/bin/env python3
"""Assign a sampled body site to every paper, and measure the assignment.

WHY. `metadata.jsonl` carries an LLM-extracted `body_site`, but only for the
original 250 papers -- 70 of the 281 papers now contributing edges have none.
Body site was flagged as the top lever out of the adjudication (Rothia/Parkinson's
is not a contradiction of the curated databases, it is a saliva study colliding
with gut records on one node), and that question cannot be asked while a quarter
of the corpus is unlabelled.

WHY KEYWORDS RATHER THAN AN LLM PASS. The sampled site is stated almost verbatim
in Methods ("faecal samples were collected", "saliva was collected"), so this is
a lookup, not a judgement -- and unlike an LLM pass it is free, deterministic, and
auditable. It is also *measurable*: 250 papers already carry an independent LLM
label, so the scanner can be scored against them before being trusted on the 70.

HOW. Cue terms are counted only where they sit near sampling language (sample,
specimen, swab, collected, extracted...), because papers routinely mention blood
for an unrelated assay while the microbiome itself is stool -- raw term frequency
mislabels those. Ties and empty scans are reported as `unknown` rather than
guessed; a wrong confident label is worse here than an honest gap.

    python body_site.py                 # score vs the LLM labels, then label the rest
    python body_site.py --out body_site.json
"""
import argparse
import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# Site vocabulary. Grouped to the granularity the question needs: what the curated
# databases are weighted toward (gut) versus what would be scored against them
# wrongly (everything else). Gut lumen and gut mucosa are kept apart because a
# biopsy and a stool sample are not interchangeable measurements.
CUES = {
    "stool": [r"\bstool\b", r"\bfa?ec(?:al|es)\b", r"\bfaeces\b", r"\brectal swab"],
    "gut biopsy": [r"\b(?:colonic|intestinal|mucosal|sigmoid|duodenal|ileal|colon)\s+biops",
                   r"\bbiops\w+\s+(?:of|from)\s+the\s+(?:colon|ileum|duodenum)",
                   r"\bmucosal?\s+(?:sample|specimen|tissue)"],
    "oral": [r"\bsaliva\w*\b", r"\boral\s+(?:sample|specimen|swab|rinse|cavity|microbiom)",
             r"\bbuccal\b", r"\bdental\s+plaque\b", r"\bsubgingival\b", r"\bsupragingival\b",
             r"\btongue\s+coat", r"\bperiodontal\s+pocket", r"\bmouth\s+rinse"],
    "blood": [r"\bblood\s+(?:sample|specimen|was|were|drawn)", r"\bserum\b", r"\bplasma\b",
              r"\bvenipuncture\b", r"\bwhole\s+blood\b"],
    "nasal": [r"\bnasal\s+(?:sample|specimen|swab|lavage)", r"\bnasopharyngeal\b",
              r"\bnares\b"],
    "skin": [r"\bskin\s+(?:sample|specimen|swab)", r"\bcutaneous\s+swab"],
    "urine": [r"\burine\s+(?:sample|specimen)", r"\burinary\s+(?:sample|specimen)"],
    "csf": [r"\bcerebrospinal\s+fluid\b", r"\bCSF\s+(?:sample|specimen)"],
    "vaginal": [r"\bvaginal\s+(?:sample|specimen|swab)"],
    "sputum": [r"\bsputum\b", r"\bbronchoalveolar\b"],
}

# Sampling context. A cue only counts when one of these sits within WINDOW
# characters of it, so "serum cholesterol was also measured" in a stool study
# does not vote for blood.
SAMPLING = re.compile(
    r"\b(sample|samples|sampling|specimen|specimens|swab|swabs|collect\w*|obtain\w*|"
    r"extract\w*|DNA|sequenc\w*|microbiot\w*|microbiom\w*|16S|donor\w*|provided)\b", re.I)
WINDOW = 120


def scan(text):
    """-> (site, scores, n_hits). `unknown` when nothing scores or the top ties.

    DECISION RULE: any stool cue wins outright, rather than argmax over sites.
    Argmax scored 84.3% against the independent LLM labels and its dominant error
    was stool studies read as blood -- not context noise but CO-SAMPLING: papers
    that take stool for the microbiome and draw serum or plasma for metabolomics
    talk about blood constantly. Tightening the context window did not fix it (it
    turned those errors into `unknown`, 83.5% / 79.9%); giving stool priority did,
    at 92.4%. That is also the rule the question needs -- Disbiome and Peryton are
    gut-weighted, so what matters is whether the microbiome was measured in stool,
    not which site the paper mentions most.

    Known residual failure modes, measured on the 249 papers with both labels:
      - dual-site papers ("oral AND gut microbiota") collapse to stool: 3 cases;
      - a genuine blood-microbiome study is outvoted by its stool arm: 1 case;
      - 10 papers name no site at all near sampling language and return `unknown`.
    """
    t = text or ""
    scores = Counter()
    for site, pats in CUES.items():
        for p in pats:
            for m in re.finditer(p, t, flags=re.I):
                lo = max(0, m.start() - WINDOW)
                if SAMPLING.search(t[lo:m.end() + WINDOW]):
                    scores[site] += 1
    if scores.get("stool", 0) > 0:
        return "stool", dict(scores), sum(scores.values())
    if not scores:
        return "unknown", dict(scores), 0
    top = scores.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unknown", dict(scores), sum(scores.values())
    return top[0][0], dict(scores), sum(scores.values())


def load_llm_labels():
    """title -> body_site from the LLM metadata pass (the 250)."""
    out = {}
    path = os.path.join(HERE, "metadata.jsonl")
    for line in open(path):
        r = json.loads(line)
        if r.get("meta") and not r.get("parse_error"):
            out[r["title"]] = (r["meta"].get("body_site") or "").strip().lower()
    return out


def load_texts():
    """title -> full text, from every corpus file that carries one."""
    texts = {}
    for rel in ("../EmilySong_GoldStandardPaper/all_usable_papers.json",
                "extract_input.json", "new_papers.json"):
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            if r.get("text") and r.get("title"):
                texts.setdefault(r["title"], r["text"])
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(HERE, "graph.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "body_site.json"))
    a = ap.parse_args()

    texts, llm = load_texts(), load_llm_labels()
    G = json.load(open(a.graph))
    contributing = sorted({p["title"] for p in G["papers"]})
    print(f"texts available: {len(texts)}   LLM-labelled: {len(llm)}   "
          f"contributing papers: {len(contributing)}")

    # --- 1. score the scanner against the independent LLM labels --------------
    both = [t for t in llm if t in texts and llm[t]]
    agree, conf, disagreements = 0, Counter(), []
    for t in both:
        pred, sc, n = scan(texts[t])
        gold = llm[t]
        conf[(gold, pred)] += 1
        if pred == gold:
            agree += 1
        else:
            disagreements.append({"title": t, "llm": gold, "scanner": pred, "scores": sc})
    print(f"\nscanner vs LLM label on {len(both)} papers with both: "
          f"{agree} agree ({100*agree/max(len(both),1):.1f}%)")
    print("  confusion (llm -> scanner), non-diagonal:")
    for (g, p), c in conf.most_common():
        if g != p:
            print(f"    {g:12} -> {p:12} {c}")

    # --- 2. label everything --------------------------------------------------
    # The LLM label wins wherever it exists. It read the co-sampling cases the
    # scanner gets wrong (stool microbiome + plasma metabolomics; a real blood
    # study with a stool arm), and it was produced by reading the paper rather
    # than counting terms. The scanner's job is the 70 papers that have no LLM
    # label at all -- its value is the 92.4% agreement that licenses using it there.
    out, src = {}, Counter()
    for t in contributing:
        pred, sc, _ = scan(texts[t]) if t in texts else ("unknown", {}, 0)
        if llm.get(t):
            out[t] = {"site": llm[t], "source": "llm", "scores": sc, "scanner": pred,
                      "agrees": pred == llm[t]}
            src["llm"] += 1
        elif t in texts:
            out[t] = {"site": pred, "source": "scanner", "scores": sc, "scanner": pred,
                      "agrees": None}
            src["scanner"] += 1
        else:
            out[t] = {"site": "unknown", "source": "none", "scores": {},
                      "scanner": "unknown", "agrees": None}
            src["none"] += 1
    print(f"\nlabel source: {dict(src)}")
    print("final site distribution over contributing papers:")
    for s, c in Counter(v["site"] for v in out.values()).most_common():
        print(f"    {s:14} {c}")

    json.dump({"scanner_vs_llm": {"n": len(both), "agree": agree,
                                  "confusion": {f"{g}->{p}": c for (g, p), c in conf.items()},
                                  "disagreements": disagreements},
               "papers": out}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
