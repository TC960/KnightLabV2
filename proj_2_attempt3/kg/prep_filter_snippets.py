#!/usr/bin/env python3
"""Build compact study-design snippets for the 45 MAIN_DATA papers.

Why: those 45 entered the graph by TITLE KEYWORD match against MAIN_DATA.json and
were never screened for design. Agreement with Disbiome and Peryton fell ~4 points
when they landed, so the hypothesis is that reviews, animal studies and
single-arm interventions are contaminating the edge set.

Deciding that needs the papers' own words, not their titles -- "Microbiota from
Alzheimer's patients induce deficits in cognition" is a rat experiment and
"Gut microbiome is associated with multiple sclerosis activity in children" is a
human cohort, and no title rule separates those. But the full texts run ~71k chars
each; feeding 45 of those to a classifier is wasteful and buries the evidence.

So: pull the abstract plus windows around the design vocabulary (enrolled,
recruited, healthy control, mice, review, ...) and cap each paper at ~7k chars.
That keeps the sentence that decides the call while cutting ~90% of the text.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# Vocabulary that actually discriminates study design. Deliberately includes the
# terms for the categories we want to DROP (mice, review, case report) as well as
# the ones we want to KEEP (healthy controls, enrolled, case-control) -- a snippet
# that only showed keep-evidence would bias the classifier toward keeping.
CUES = [
    r"healthy control", r"healthy subject", r"healthy volunteer", r"healthy individual",
    r"case[- ]control", r"cross[- ]sectional", r"cohort stud", r"were recruited",
    r"were enrolled", r"we enrolled", r"we recruited", r"participants",
    r"inclusion criteri", r"exclusion criteri", r"written informed consent",
    r"ethics committee", r"institutional review board",
    r"\bmice\b", r"\bmouse\b", r"\brats?\b", r"\bmurine\b", r"C57BL", r"animal model",
    r"transgenic", r"APP/PS1", r"5xFAD", r"germ[- ]free", r"gnotobiotic",
    r"\breview\b", r"systematic review", r"meta[- ]analys", r"narrative review",
    r"case report", r"\bwe searched\b", r"literature search",
    r"randomi[sz]ed", r"placebo", r"open[- ]label", r"intervention", r"supplementation",
    r"before and after", r"pre[- ]treatment", r"post[- ]treatment",
    r"fecal microbiota transplant", r"faecal microbiota transplant", r"\bFMT\b",
]
CUE_RE = re.compile("|".join(CUES), re.I)

HEAD_CHARS = 2600      # title + abstract + opening of intro
WIN = 240              # chars either side of a cue hit
BUDGET = 7000          # per-paper cap on cue windows


def snippet(text):
    text = re.sub(r"[ \t]+", " ", text or "")
    head = text[:HEAD_CHARS]
    body = text[HEAD_CHARS:]

    # Collect cue windows, merging overlaps so we don't repeat text.
    spans = []
    for m in CUE_RE.finditer(body):
        a, b = max(0, m.start() - WIN), min(len(body), m.end() + WIN)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])

    out, used = [], 0
    for a, b in spans:
        chunk = body[a:b]
        if used + len(chunk) > BUDGET:
            chunk = chunk[: max(0, BUDGET - used)]
        if not chunk:
            break
        out.append(chunk)
        used += len(chunk)
        if used >= BUDGET:
            break
    return head, " […] ".join(out)


def main():
    titles = set(json.load(open(os.path.join(HERE, "_main_data_titles.json"))))
    recs = json.load(open(os.path.join(HERE, "extract_input.json")))
    picked = [r for r in recs if (r.get("title") or "").strip() in titles]
    assert len(picked) == len(titles), f"{len(picked)} != {len(titles)}"

    picked.sort(key=lambda r: r["title"])
    os.makedirs(os.path.join(HERE, "_filter_batches"), exist_ok=True)

    per_batch = 9
    nb = 0
    for i in range(0, len(picked), per_batch):
        nb += 1
        batch = picked[i:i + per_batch]
        lines = []
        for r in batch:
            head, cues = snippet(r.get("text", ""))
            lines.append(
                f"### PAPER_ID: {r['title'][:150]}\n"
                f"LINK: {r.get('link','')}\n"
                f"STATED_DISEASE: {r.get('disease','')}\n"
                f"FULL_TEXT_CHARS: {len(r.get('text',''))}\n\n"
                f"--- OPENING (title/abstract) ---\n{head}\n\n"
                f"--- STUDY-DESIGN EXCERPTS ---\n{cues}\n"
            )
        p = os.path.join(HERE, "_filter_batches", f"batch{nb}.md")
        open(p, "w").write("\n\n" + ("=" * 90 + "\n\n").join(lines))
        print(f"batch{nb}: {len(batch)} papers, {os.path.getsize(p)/1000:.0f}kB -> {p}")
    print(f"\n{len(picked)} papers in {nb} batches")


if __name__ == "__main__":
    main()
