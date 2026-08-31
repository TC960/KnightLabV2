#!/usr/bin/env python3
"""Fold the GPU run's new extractions into the corpus, then rebuild the graph.

Inputs
  gpu_results/extract_out/*.checkpoint.jsonl   the new run (98 papers)
  ../dsmlp_model_prompting/eval-v2/results/...all250.json   the original 250

Output
  extractions_merged.json    combined, deduped on title
  then: python build_kg.py && python build_viz.py

The two halves are NOT equivalent and the merge records which is which:
  - 53 papers came from the datasheet and carry gold taxa -> they extend the
    EVALUATION set (the only papers where accuracy is measurable).
  - 45 came from MAIN_DATA by title keyword and have no gold, and may include
    reviews or animal studies -> graph coverage only. `source` marks them so a
    later filter can drop them without unpicking the merge.

Run with --dry-run first; it reports what would change without writing.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                   "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEWDIR = os.path.join(HERE, "gpu_results", "extract_out")
SHEET_PAPERS = os.path.join(HERE, "new_papers.json")
OUT = os.path.join(HERE, "extractions_merged.json")


def load_new():
    rows = []
    for f in glob.glob(os.path.join(NEWDIR, "*.checkpoint.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # de-dupe within the new run (a resumed run can repeat a row)
    seen, out = set(), []
    for r in rows:
        t = (r.get("title") or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    old = json.load(open(OLD))
    new = load_new()
    if not new:
        print(f"No new checkpoint rows under {NEWDIR} — has the GPU run finished?")
        return

    with_gold = {(p["title"] or "").strip() for p in json.load(open(SHEET_PAPERS))} \
        if os.path.exists(SHEET_PAPERS) else set()

    for r in old:
        r.setdefault("source", "original250")
    kept, dupes = [], 0
    have = {(r.get("title") or "").strip() for r in old}
    for r in new:
        t = (r.get("title") or "").strip()
        if t in have:
            dupes += 1
            continue
        r["source"] = "sheet_new" if t in with_gold else "main_data_neuro"
        have.add(t)
        kept.append(r)

    merged = old + kept
    n_gold = sum(1 for r in merged
                 if (r.get("expected_enriched") or "").strip()
                 or (r.get("expected_depleted") or "").strip())
    perr = sum(1 for r in kept if r.get("parse_error"))

    print(f"original            : {len(old)}")
    print(f"new run             : {len(new)}  (dupes skipped: {dupes}, parse errors: {perr})")
    print(f"  with gold taxa    : {sum(1 for r in kept if r['source']=='sheet_new')}")
    print(f"  graph-coverage only: {sum(1 for r in kept if r['source']=='main_data_neuro')}")
    print(f"MERGED              : {len(merged)}")
    print(f"  scoreable (has gold): {n_gold}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    json.dump(merged, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print("next: python build_kg.py --input extractions_merged.json && python build_viz.py")


if __name__ == "__main__":
    main()
