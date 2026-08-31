#!/usr/bin/env python3
"""TASK 3b -- scale the hand-classified FP/FN categories to the full error pools.

Reading 22 FPs and 22 FNs by hand showed two mechanically-detectable classes that
dominate the FN side and are invisible in the headline P/R/F1:

  (i)  gold cells that are not taxon names at all ('fdr < 0.05', 'none',
       'human fecal', disease-prefixed entries, two taxa joined by 'and' that
       parse_taxa never splits because it only splits on [,;])
  (ii) gold taxa that do not occur anywhere in the text the model was shown --
       either the scrape is abstract-only, or the taxon lives in a figure/table/
       supplement. Neither is a recall failure the extractor could have avoided.

This measures both over all 957 FPs / 815 FNs, and tests whether (ii) is
explained by truncated scrapes. Also recomputes P/R/F1 after removing the
non-taxon gold entries, to show what the metric would say on a clean gold.

Writes quantify_errors.json.
"""
import csv, json, os, re, sys, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa, match_taxa  # noqa: E402

RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
PAPERS = os.path.join(P3, "EmilySong_GoldStandardPaper", "all_usable_papers.json")
ENR = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
DEP = "KeyTaxa_Depleted ↓"
SRC = "TaxaExtractionSource (Should be Main Text unless there is nothing listed)"
NPERM = 10000

# A gold string that cannot be a single taxon name.
STATS = re.compile(r"(fdr|p\s*[<>=]|q\s*[<>=]|\bp[- ]?value|lda|lefse|adjusted|%|"
                   r"\bci\b|significan|^n/?a$|^none$|^nan$|^no\b|^not\b)", re.I)
NONTAXON_WORDS = re.compile(r"(human fecal|fecal|stool|saliva|plasma|serum|blood|"
                            r"disease|patients?|controls?|group|cohort|sample)", re.I)
DISEASE_PREFIX = re.compile(r":")            # 'alzheimer's disease : bacteroidetes'
UNSPLIT_AND = re.compile(r"\band\b", re.I)   # parse_taxa splits only on [,;]


def classify_gold_string(s):
    t = s.strip()
    if STATS.search(t):
        return "stats_or_placeholder"
    if DISEASE_PREFIX.search(t):
        return "disease_prefixed"
    if UNSPLIT_AND.search(t):
        return "unsplit_and"
    if NONTAXON_WORDS.search(t) and not re.search(r"(bacter|coccus|monas|ella|aceae|ales|"
                                                  r"phyl|firmicut|proteo|actino|clostrid)", t, re.I):
        return "not_a_taxon"
    return "ok"


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def present(taxon, text_l):
    toks = [w for w in re.split(r"[^a-z0-9\[\]_]+", taxon.lower()) if len(w) > 3]
    toks = [w for w in toks if w not in ("group", "genus", "species", "family", "order",
                                         "class", "phylum", "unclassified", "uncultured",
                                         "spp", "bacterium", "bacteria")]
    return bool(toks) and max(toks, key=len) in text_l


def prf(TP, FP, FN):
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot_ci(flags, seed=0, iters=20000):
    rng = random.Random(seed); n = len(flags)
    b = sorted(st.mean([flags[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return [round(b[int(.025 * iters)], 4), round(b[int(.975 * iters)], 4)]


def main():
    res = json.load(open(RESULTS))
    sheet = {norm_title(r["Title"]): r for r in csv.DictReader(open(NEW_CSV))}
    papers = {norm_title(p["title"]): p for p in json.load(open(PAPERS))}
    out = {}

    # ---------- 1. how much of the GOLD is not a taxon at all? --------------
    cls = {}
    gold_rows = []
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue
        ge, gd = parse_taxa(g[ENR]), parse_taxa(g[DEP])
        if not (ge or gd):
            continue
        for t in ge + gd:
            c = classify_gold_string(t)
            cls[c] = cls.get(c, 0) + 1
            gold_rows.append((k, t, c))
    tot = len(gold_rows)
    print(f"=== gold taxon strings: {tot} across {len(set(x[0] for x in gold_rows))} papers ===")
    for c, n in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"   {c:22s} {n:5d}  {n/tot:6.2%}")
    bad = tot - cls.get("ok", 0)
    out["gold_string_quality"] = {"total": tot, "classes": cls, "n_not_clean": bad,
                                  "pct_not_clean": round(bad / tot, 4)}
    out["gold_bad_examples"] = [t for _, t, c in gold_rows if c != "ok"][:40]

    # ---------- 2. FN pool: is the taxon even in the text? ------------------
    print("\n=== FN pool: is the gold taxon present in the text the model saw? ===")
    fn_rows = []
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue
        ge, gd = parse_taxa(g[ENR]), parse_taxa(g[DEP])
        if not (ge or gd):
            continue
        p = papers.get(k)
        tl = re.sub(r"\s+", " ", p["text"]).lower() if p else ""
        pe, pd = parse_taxa(r["predicted_enriched"]), parse_taxa(r["predicted_depleted"])
        for direction, pred, gold in [("enriched", pe, ge), ("depleted", pd, gd)]:
            if not gold:
                continue
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            if pred:
                tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(pred + gold)
                sim = cosine_similarity(tf[:len(pred)], tf[len(pred):])
                matched = {int(sim[i].argmax()) for i in range(len(pred))
                           if float(sim[i][sim[i].argmax()]) >= 0.5}
            else:
                matched = set()
            for j, t in enumerate(gold):
                if j in matched:
                    continue
                fn_rows.append({"key": k, "taxon": t, "direction": direction,
                                "clean": classify_gold_string(t) == "ok",
                                "in_text": present(t, tl), "chars": len(tl),
                                "src": g[SRC].strip()})
    n = len(fn_rows)
    clean = [x for x in fn_rows if x["clean"]]
    print(f"   total FN: {n}; after dropping non-taxon gold strings: {len(clean)} "
          f"({(n-len(clean))/n:.1%} of FNs were never real taxa)")
    it = [1.0 if x["in_text"] else 0.0 for x in clean]
    print(f"   of the {len(clean)} real FN taxa, {sum(it):.0f} ({st.mean(it):.1%}) occur in the text; "
          f"{len(clean)-sum(it):.0f} ({1-st.mean(it):.1%}) DO NOT")
    out["fn_pool"] = {
        "n_fn": n, "n_nontaxon_gold": n - len(clean),
        "pct_fn_that_are_gold_artifacts": round((n - len(clean)) / n, 4),
        "n_real_fn": len(clean),
        "pct_real_fn_present_in_text": round(st.mean(it), 4),
        "pct_real_fn_present_ci95": boot_ci(it, seed=21),
    }

    # is absence explained by truncated (abstract-only) scrapes?
    SHORT = 15000
    sh = [x for x in clean if x["chars"] < SHORT]
    lo = [x for x in clean if x["chars"] >= SHORT]
    a = [1.0 if x["in_text"] else 0.0 for x in sh]
    b = [1.0 if x["in_text"] else 0.0 for x in lo]
    obs = st.mean(b) - st.mean(a)
    pool = a + b; rng = random.Random(22); ge_ = 0
    for _ in range(NPERM):
        rng.shuffle(pool)
        if abs(st.mean(pool[len(a):]) - st.mean(pool[:len(a)])) >= abs(obs) - 1e-12:
            ge_ += 1
    print(f"   present-in-text | scrape <{SHORT} chars: {st.mean(a):.1%} (n={len(a)});  "
          f">= {SHORT}: {st.mean(b):.1%} (n={len(b)});  diff={obs:+.1%} p={(ge_+1)/(NPERM+1):.4f}")
    print("   NOTE: this permutation shuffles at the TAXON level within a fixed paper split; "
          "papers are the true unit, so treat p as anti-conservative -- see the paper-level test below.")
    out["fn_absence_vs_scrape_length"] = {
        "short_scrape_present_rate": round(st.mean(a), 4), "n_short": len(a),
        "long_scrape_present_rate": round(st.mean(b), 4), "n_long": len(b),
        "diff": round(obs, 4), "p_two_sided_taxon_level": round((ge_ + 1) / (NPERM + 1), 4)}

    # paper-level version: one observation per paper = its fraction present
    bypap = {}
    for x in clean:
        bypap.setdefault(x["key"], []).append(x)
    pa = [st.mean([1.0 if y["in_text"] else 0.0 for y in v])
          for k, v in bypap.items() if v[0]["chars"] < SHORT]
    pb = [st.mean([1.0 if y["in_text"] else 0.0 for y in v])
          for k, v in bypap.items() if v[0]["chars"] >= SHORT]
    obs2 = st.mean(pb) - st.mean(pa)
    pool2 = pa + pb; rng = random.Random(23); ge2 = 0
    for _ in range(NPERM):
        rng.shuffle(pool2)
        if abs(st.mean(pool2[len(pa):]) - st.mean(pool2[:len(pa)])) >= abs(obs2) - 1e-12:
            ge2 += 1
    print(f"   PAPER-level: short {st.mean(pa):.1%} (n={len(pa)} papers) vs long {st.mean(pb):.1%} "
          f"(n={len(pb)}); diff={obs2:+.1%} p={(ge2+1)/(NPERM+1):.4f}")
    out["fn_absence_vs_scrape_length_paperlevel"] = {
        "short_rate": round(st.mean(pa), 4), "n_short_papers": len(pa),
        "long_rate": round(st.mean(pb), 4), "n_long_papers": len(pb),
        "diff": round(obs2, 4), "p_two_sided": round((ge2 + 1) / (NPERM + 1), 4)}

    # ---------- 3. FP pool: presence in text (hallucination rate) -----------
    fp_rows = []
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue
        ge, gd = parse_taxa(g[ENR]), parse_taxa(g[DEP])
        if not (ge or gd):
            continue
        p = papers.get(k)
        tl = re.sub(r"\s+", " ", p["text"]).lower() if p else ""
        for direction, pred, gold in [("enriched", parse_taxa(r["predicted_enriched"]), ge),
                                      ("depleted", parse_taxa(r["predicted_depleted"]), gd)]:
            if not pred:
                continue
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            if gold:
                tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(pred + gold)
                sim = cosine_similarity(tf[:len(pred)], tf[len(pred):])
                fpi = [i for i in range(len(pred)) if float(sim[i][sim[i].argmax()]) < 0.5]
            else:
                fpi = list(range(len(pred)))
            for i in fpi:
                fp_rows.append({"key": k, "taxon": pred[i], "in_text": present(pred[i], tl)})
    fi = [1.0 if x["in_text"] else 0.0 for x in fp_rows]
    print(f"\n=== FP pool: {len(fp_rows)} FP taxa; {sum(fi):.0f} ({st.mean(fi):.1%}) occur in the "
          f"paper text -> hallucination rate {1-st.mean(fi):.1%}")
    out["fp_pool"] = {"n_fp": len(fp_rows), "pct_present_in_text": round(st.mean(fi), 4),
                      "hallucination_rate": round(1 - st.mean(fi), 4),
                      "hallucination_ci95": [round(1 - boot_ci(fi, seed=24)[1], 4),
                                             round(1 - boot_ci(fi, seed=24)[0], 4)],
                      "absent_examples": [x["taxon"] for x in fp_rows if not x["in_text"]][:25]}

    # ---------- 4. rescore on a cleaned gold -------------------------------
    print("\n=== rescoring after dropping non-taxon gold strings ===")
    TP = FP = FN = 0
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue
        ge = [t for t in parse_taxa(g[ENR]) if classify_gold_string(t) == "ok"]
        gd = [t for t in parse_taxa(g[DEP]) if classify_gold_string(t) == "ok"]
        if not (ge or gd):
            continue
        a = match_taxa(parse_taxa(r["predicted_enriched"]), ge)
        b = match_taxa(parse_taxa(r["predicted_depleted"]), gd)
        TP += a[0] + b[0]; FP += a[1] + b[1]; FN += a[2] + b[2]
    p, r_, f = prf(TP, FP, FN)
    print(f"   cleaned gold: P={p:.4f} R={r_:.4f} F1={f:.4f}  (TP={TP} FP={FP} FN={FN})")
    print(f"   baseline    : P=0.5771 R=0.6157 F1=0.5958")
    out["rescored_on_cleaned_gold"] = {"TP": TP, "FP": FP, "FN": FN,
                                       "precision": round(p, 4), "recall": round(r_, 4),
                                       "f1": round(f, 4),
                                       "baseline_f1": 0.5958, "delta_f1": round(f - 0.5958, 4)}

    # ---------- 5. TaxaExtractionSource cross-check ------------------------
    src = {}
    for x in clean:
        s = x["src"] or "(blank)"
        src.setdefault(s, []).append(1.0 if x["in_text"] else 0.0)
    print("\n=== FN present-in-text by the sheet's own TaxaExtractionSource column ===")
    out["fn_by_extraction_source"] = {}
    for s, v in sorted(src.items(), key=lambda kv: -len(kv[1])):
        print(f"   {s[:46]:48s} n={len(v):4d}  present {st.mean(v):.1%}")
        out["fn_by_extraction_source"][s] = {"n": len(v), "present_rate": round(st.mean(v), 4)}

    json.dump(out, open(os.path.join(HERE, "quantify_errors.json"), "w"), indent=2)
    print("\nwrote quantify_errors.json")


if __name__ == "__main__":
    main()
