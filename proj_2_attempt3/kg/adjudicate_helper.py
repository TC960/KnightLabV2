#!/usr/bin/env python3
"""Read-only helper for Task 3 adjudication.

Pulls the verbatim sentences that mention a taxon out of a paper's full text in
../EmilySong_GoldStandardPaper/all_usable_papers.json, so every verdict in
FINDINGS_task3_adjudication.md can be backed by a quote.

    python adjudicate_helper.py --paper "Analysis of the Gut Microflora" --taxon Bulleidia
    python adjudicate_helper.py --list
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")


def load():
    return json.load(open(PAPERS))


def find_paper(recs, needle):
    n = needle.lower()
    hits = [r for r in recs if n in (r.get("title") or "").lower()]
    return hits


def sentences(text):
    # papers are scraped HTML->text; split on sentence enders and newlines
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", help="substring of the title")
    ap.add_argument("--taxon", help="regex to look for")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--ctx", type=int, default=0, help="+/- N sentences of context")
    a = ap.parse_args()

    recs = load()
    if a.list:
        for r in recs:
            print(f"{r.get('disease','?'):40} {r.get('title','')[:110]}")
        return

    hits = find_paper(recs, a.paper)
    if not hits:
        print("NO PAPER MATCH")
        return
    for r in hits:
        print("=" * 100)
        print("TITLE   :", r.get("title"))
        print("DISEASE :", r.get("disease"))
        print("LINK    :", r.get("link"))
        print("GOLD UP :", r.get("taxa_enriched"))
        print("GOLD DN :", r.get("taxa_depleted"))
        print("CHARS   :", len(r.get("text") or ""))
        if not a.taxon:
            continue
        pat = re.compile(a.taxon, re.I)
        sents = sentences(r.get("text") or "")
        for i, s in enumerate(sents):
            if pat.search(s):
                lo, hi = max(0, i - a.ctx), min(len(sents), i + a.ctx + 1)
                print("-" * 90)
                for j in range(lo, hi):
                    mark = ">>" if j == i else "  "
                    print(f"[{j}]{mark} {sents[j][:a.width]}")


if __name__ == "__main__":
    main()
