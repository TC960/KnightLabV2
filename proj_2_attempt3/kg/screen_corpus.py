#!/usr/bin/env python3
"""Screen every contributing paper for study design -- not just the 45.

WHY
---
`maindata_screen.json` adjudicated 45 papers: the title-matched MAIN_DATA
additions, which entered by keyword-matching titles and were known to be
unvetted. 22 of the 45 were not human case-control studies at all.

The other ~250 papers came from the annotation datasheet and were NEVER put
through that screen. `filter_maindata.py`'s docstring asserts they were
screened "unlike" the 45 -- that contrast was assumed, never established.
As of 2026-09-11 only **23 of 271 contributing papers** have ever been checked.

That matters for the same reason it mattered for the 45: the extractor was
never asked "is this a human case-control study?", it was asked which taxa go
up and down. Point it at a paper where 3xTgAD mice differ from wild-type
littermates and it faithfully returns that contrast as a disease-vs-healthy
human finding. Those edges are not extraction errors; they are correct readings
of papers that do not belong in a human microbe-disease graph.

HOW
---
Two commands, deliberately separated so the expensive step is reproducible:

    python3 screen_corpus.py --prepare    # deterministic batches -> <workdir>
    python3 screen_corpus.py --score      # validate + score the returned JSON

`--prepare` writes batches of abstracts with **24 blinded controls** drawn from
the 45 already-adjudicated papers, shuffled in under the same id scheme so a
reader cannot tell them apart. `--score` recovers them and reports agreement.
That is the only available measure of whether the screen can be trusted, and it
is why the controls exist: an unvalidated 249-paper classification is not a
result, it is a pile of opinions.

The seed is fixed (11) so --prepare reproduces the same batches and the same
control assignment on a later run.

WHAT COUNTS
-----------
KEEP means a human case-vs-control microbiome comparison is reported. A paper
that ALSO runs mice still counts -- the human contrast is what the extractor
read. The four drop reasons mirror `maindata_screen.json`'s own vocabulary so
the two screens can be merged.

NOTE ON RECALL, because the previous animal-only sweep had a measurable one and
this does not: the deterministic animal prefilter used on 2026-09-11 was
validated at 15/15 against these same 45 papers. This screen covers the three
failure modes that prefilter CANNOT see (no healthy control, case report,
review), and for those there is no deterministic detector -- so the blinded
controls are the whole of the validation. Report their agreement, and treat
any category where the controls are thin as unvalidated.
"""
import argparse
import collections
import json
import os
import random
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
SCREEN = os.path.join(HERE, "maindata_screen.json")
TEXTS = [
    os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json"),
    os.path.join(HERE, "extract_input.json"),
    os.path.join(HERE, "new_papers.json"),
]
SEED = 11
BATCH = 34
N_CONTROLS = 24

CATEGORIES = {"KEEP", "DROP_ANIMAL", "DROP_NO_HEALTHY_CONTROL",
              "DROP_CASE_REPORT", "DROP_REVIEW", "UNCLEAR"}


def tkey(t):
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(t)).lower())


def load_texts():
    T = {}
    for path in TEXTS:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        recs = d if isinstance(d, list) else list(d.values())
        for r in recs:
            if not isinstance(r, dict):
                continue
            t = r.get("title") or r.get("name")
            b = r.get("text") or r.get("full_text") or r.get("body")
            if b is None and isinstance(r.get("chunks"), list):
                b = "\n".join(r["chunks"])
            if t and b:
                T.setdefault(tkey(t), b)
    return T


ABSTRACT_CUE = re.compile(
    r"\b(Background|Objectives?|Methods?|Results?|Conclusions?|Aims?|"
    r"We (?:recruited|enrolled|analy[sz]ed|investigated|compared)|"
    r"patients? (?:and|vs)|healthy (?:controls?|subjects?|individuals?))\b", re.I)


def abstract(body, n=3000):
    """The abstract, approximately -- found by scoring, not by anchoring.

    The first version anchored on the first "Abstract" marker in the first half
    of the document. It failed on **12.9% of the corpus (35 of 271 papers)**,
    returning a window containing no abstract at all: these PMC / J-STAGE dumps
    carry long front matter (journal navigation, repeated titles, full author
    lists and affiliations), and either never use the literal word early or
    push the real abstract past the 3000-character window.

    That was not a cosmetic bug. Those papers came back UNCLEAR from the
    screen, and the cause was the input, not the reader -- one screening agent
    diagnosed it unprompted ("unclear abstracts due to heavy metadata in the
    source text"). A screen is only as good as the span it is shown.

    So: slide a window over the first 70% of the document and keep the one with
    the most abstract-like cues. Zero-cue spans fall from 35 papers to 5 (1.8%).
    The remaining 5 have no recoverable abstract in the stored text and should
    be screened from full text or by a human.
    """
    limit = max(int(len(body) * 0.7), n)
    best_score, best_start = -1, 0
    for start in range(0, max(limit - n, 1), 500):
        s = len(ABSTRACT_CUE.findall(body[start:start + n]))
        if s > best_score:
            best_score, best_start = s, start
    return re.sub(r"\s+", " ", body[best_start:best_start + n]).strip()


def prepare(workdir):
    os.makedirs(workdir, exist_ok=True)
    T = load_texts()
    papers = [p["title"] for p in json.load(open(GRAPH))["papers"]]
    screen = json.load(open(SCREEN))
    screened = {tkey(k) for k in screen}

    unscreened = [t for t in papers if tkey(t) not in screened]
    gold = [(k, v) for k, v in screen.items() if tkey(k) in T]
    print(f"contributing {len(papers)}, never screened {len(unscreened)}, "
          f"gold with text {len(gold)}")

    rnd = random.Random(SEED)
    controls = rnd.sample(gold, N_CONTROLS)
    items = ([{"title": t, "gold": None} for t in unscreened]
             + [{"title": k, "gold": v["category"]} for k, v in controls])
    rnd.shuffle(items)

    key = {}
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    for bi, b in enumerate(batches, 1):
        out = []
        for j, it in enumerate(b, 1):
            pid = f"b{bi:02d}_{j:02d}"
            key[pid] = {"title": it["title"], "gold": it["gold"]}
            out.append(f"=== {pid} ===\nTITLE: {it['title']}\n"
                       f"{abstract(T[tkey(it['title'])])}\n")
        open(os.path.join(workdir, f"batch{bi:02d}.txt"), "w").write("\n".join(out))
    json.dump(key, open(os.path.join(workdir, "key.json"), "w"), indent=1)
    print(f"wrote {len(batches)} batches ({[len(b) for b in batches]}) to {workdir}")
    print(f"{N_CONTROLS} blinded controls mixed in, seed {SEED}")


def load_records(path):
    """Parse one agent output, tolerating invalid JSON.

    An `evidence` value is a verbatim span copied out of a paper, so it can
    contain quote characters -- and one batch came back with
    `"evidence": "..." and "..."`, two quoted spans joined by a bare `and`,
    which is not JSON. All 34 records were present and correct; only the
    serialisation was broken.

    Rather than lose a batch to that, fall back to a field-wise recovery. It
    reads only id / category / confidence, which is everything the verdict
    depends on, and drops `evidence` for the recovered records -- so a
    recovered batch is scored on its verdicts and contributes nothing to the
    verbatim-span rate. Returns (records, recovered?).
    """
    raw = open(path).read()
    try:
        return json.load(open(path)), False
    except json.JSONDecodeError:
        pass
    out = []
    for block in re.split(r"\}\s*,?\s*\{", raw):
        pid = re.search(r'"id"\s*:\s*"([^"]+)"', block)
        cat = re.search(r'"category"\s*:\s*"([^"]+)"', block)
        con = re.search(r'"confidence"\s*:\s*"([^"]+)"', block)
        if pid and cat:
            out.append({"id": pid.group(1), "category": cat.group(1),
                        "confidence": con.group(1) if con else None,
                        "evidence": None, "_recovered": True})
    return out, True


def score(workdir, out_json):
    key = json.load(open(os.path.join(workdir, "key.json")))
    verdicts = {}
    dupes = []
    recovered = []
    for fn in sorted(os.listdir(workdir)):
        if not re.fullmatch(r"out\d+\.json", fn):
            continue
        recs, was_recovered = load_records(os.path.join(workdir, fn))
        if was_recovered:
            recovered.append(f"{fn} ({len(recs)} records)")
        for r in recs:
            pid = r.get("id")
            if pid in verdicts:
                dupes.append(pid)
            verdicts[pid] = r
    if recovered:
        print(f"recovered from invalid JSON: {', '.join(recovered)}")

    missing = sorted(set(key) - set(verdicts))
    extra = sorted(set(verdicts) - set(key))
    print(f"records expected {len(key)}, returned {len(verdicts)}")
    if missing:
        print(f"  MISSING {len(missing)}: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if extra:
        print(f"  UNKNOWN IDS {len(extra)}: {extra[:12]}")
    if dupes:
        print(f"  DUPLICATE IDS {len(dupes)}: {dupes[:12]}")

    badcat = [p for p, r in verdicts.items() if r.get("category") not in CATEGORIES]
    if badcat:
        print(f"  INVALID CATEGORY {len(badcat)}: {[(p, verdicts[p].get('category')) for p in badcat[:8]]}")

    # -- evidence spans must be verbatim ------------------------------------
    # Not a correctness test on the VERDICT: an agent that paraphrases a real
    # cohort sentence produces a correct verdict and a failing span. Measured
    # anyway, because a span that appears nowhere is the one cheap signal of a
    # fabricated cohort, and because the rate is worth knowing.
    text = {}
    for fn in sorted(os.listdir(workdir)):
        if not re.fullmatch(r"batch\d+\.txt", fn):
            continue
        for chunk in open(os.path.join(workdir, fn)).read().split("=== "):
            m = re.match(r"(b\d+_\d+) ===\n(.*)", chunk, re.S)
            if m:
                text[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).lower()
    verb = miss = 0
    for pid, r in verdicts.items():
        ev = r.get("evidence")
        if not ev:
            continue
        if re.sub(r"\s+", " ", ev).strip().lower() in text.get(pid, ""):
            verb += 1
        else:
            miss += 1
    print(f"evidence spans: {verb} verbatim, {miss} not found "
          f"({verb / max(verb + miss, 1):.1%} verbatim)")

    # -- blinded controls ---------------------------------------------------
    ctrl = [(p, key[p]["gold"], verdicts[p]["category"])
            for p in key if key[p]["gold"] and p in verdicts]
    if ctrl:
        exact = sum(1 for _, g, v in ctrl if g == v)
        binary = sum(1 for _, g, v in ctrl
                     if (g == "KEEP") == (v == "KEEP"))
        print(f"\nBLINDED CONTROLS: {len(ctrl)}")
        print(f"  exact category agreement : {exact}/{len(ctrl)} = {exact/len(ctrl):.3f}")
        print(f"  keep-vs-drop agreement   : {binary}/{len(ctrl)} = {binary/len(ctrl):.3f}")
        conf = collections.Counter((g, v) for _, g, v in ctrl)
        print("  disagreements (gold -> predicted):")
        for (g, v), n in sorted(conf.items()):
            if g != v:
                ids = [p for p, gg, vv in ctrl if gg == g and vv == v]
                print(f"    {g} -> {v}  x{n}   {ids}")

    # -- the actual result --------------------------------------------------
    new = {p: r for p, r in verdicts.items() if not key.get(p, {}).get("gold")}
    cats = collections.Counter(r["category"] for r in new.values())
    print(f"\nNEWLY SCREENED PAPERS: {len(new)}")
    for c, n in cats.most_common():
        print(f"  {c:26} {n}")
    drops = {key[p]["title"]: {"category": r["category"],
                               "confidence": r.get("confidence"),
                               "evidence": r.get("evidence")}
             for p, r in new.items() if r["category"].startswith("DROP")}
    unclear = {key[p]["title"]: r.get("evidence")
               for p, r in new.items() if r["category"] == "UNCLEAR"}
    print(f"\nproposed drops: {len(drops)}")
    for t, d in sorted(drops.items()):
        print(f"  [{d['category']:24}] ({d['confidence']}) {t[:78]}")
    if unclear:
        print(f"\nUNCLEAR ({len(unclear)}) -- need a human or full text:")
        for t in sorted(unclear):
            print(f"  {t[:90]}")

    if out_json:
        json.dump({"n_expected": len(key), "n_returned": len(verdicts),
                   "n_missing": len(missing), "missing": missing,
                   "controls": [{"id": p, "gold": g, "pred": v} for p, g, v in ctrl],
                   "verdicts": {key[p]["title"]: r["category"]
                                for p, r in new.items()},
                   "drops": drops, "unclear": list(unclear)},
                  open(out_json, "w"), indent=1, sort_keys=True)
        print(f"\nwrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(HERE, "_screen248"))
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "corpus_screen.json"))
    a = ap.parse_args()
    if a.prepare:
        prepare(a.workdir)
    elif a.score:
        score(a.workdir, a.out)
    else:
        ap.error("pass --prepare or --score")


if __name__ == "__main__":
    main()
