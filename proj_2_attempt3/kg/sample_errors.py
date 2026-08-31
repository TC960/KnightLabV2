#!/usr/bin/env python3
"""TASK 3 -- sample the extractor's false positives and false negatives and pull
the evidence needed to classify each one by hand.

Reproduces the leaderboard matcher exactly (char-ngram cosine >= 0.5, greedy),
then for a random sample of FP and FN taxa emits:
  * whether the taxon name occurs in the paper at all (hallucination test)
  * whether it occurs in the OPPOSITE direction of the gold (direction flip)
  * every sentence naming it, tagged with the significance cues present
  * for FNs, whether the model predicted it in the other direction

Sampling is at the TAXON level but stratified so no single paper dominates;
the paper id is carried through so the write-up can cluster by paper.

Writes sample_errors.json (evidence dossiers) + error_pool.json (full pools).
"""
import csv, json, os, re, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
from run_eval import parse_taxa  # noqa: E402

RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results",
                       "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
NEW_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
PAPERS = os.path.join(P3, "EmilySong_GoldStandardPaper", "all_usable_papers.json")
ENR = "KeyTaxa_Enriched ↑ (Taxa1, Taxa2, etc.)"
DEP = "KeyTaxa_Depleted ↓"

N_SAMPLE = 22          # per class; task asked for ~20, a couple spare for unusable text

SIG_STRONG = re.compile(
    r"(p\s*[<>=]\s*0?\.\d+|p\s*[-–]?\s*value|LDA|LEfSe|FDR|q\s*[<>=]\s*0?\.\d+|"
    r"adjusted p|Benjamini|significan\w*|CI\s*[:=]|95%)", re.I)
SIG_SOFT = re.compile(r"(increas\w*|decreas\w*|enrich\w*|deplet\w*|higher|lower|elevated|"
                      r"reduced|abundan\w*|greater|less)", re.I)
COMPARISON = re.compile(
    r"(versus|vs\.?|compared (?:with|to)|relative to|than (?:in )?(?:the )?"
    r"(?:healthy|control|HC\b)|healthy control|control group)", re.I)
SUBGROUP = re.compile(
    r"(severity|severe|mild|moderate|stage|H&Y|Hoehn|UPDRS|MMSE|MoCA|subtype|subgroup|"
    r"progress\w*|duration|responder|treated|untreated|medicat\w*|constipat\w*|"
    r"correlat\w*|associat\w* with (?:the )?(?:score|scale|severity))", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def match_pairs(predicted, expected):
    """Same greedy char-ngram matcher as run_eval.match_taxa, but returns WHICH
    predicted are FPs and WHICH expected are unmatched (FNs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not predicted and not expected:
        return [], [], []
    if not predicted:
        return [], [], list(range(len(expected)))
    if not expected:
        return [], list(range(len(predicted))), []
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(predicted + expected)
    sim = cosine_similarity(tf[:len(predicted)], tf[len(predicted):])
    matched, tp_idx, fp_idx = set(), [], []
    for i in range(len(predicted)):
        j = int(sim[i].argmax())
        if float(sim[i][j]) >= 0.5:
            tp_idx.append(i); matched.add(j)
        else:
            fp_idx.append(i)
    fn_idx = [j for j in range(len(expected)) if j not in matched]
    return tp_idx, fp_idx, fn_idx


def sentences_for(taxon, text):
    """Sentences naming the taxon. Matches on the most distinctive word of the
    name (genus/species token) so rank prefixes and 'g__' forms still hit."""
    toks = [w for w in re.split(r"[^a-z0-9\[\]_]+", taxon.lower()) if len(w) > 3]
    toks = [w for w in toks if w not in
            ("group", "genus", "species", "family", "order", "class", "phylum",
             "unclassified", "uncultured", "spp", "bacterium", "bacteria")]
    if not toks:
        return [], False
    key = max(toks, key=len)
    if key not in text.lower():
        return [], False
    sents = re.split(r"(?<=[.!?])\s+", text)
    hits = [s.strip() for s in sents if key in s.lower()]
    return hits[:8], True


def tag(sents):
    return {
        "n_sentences": len(sents),
        "has_strong_sig": any(SIG_STRONG.search(s) for s in sents),
        "has_soft_sig": any(SIG_SOFT.search(s) for s in sents),
        "has_case_control_comparison": any(COMPARISON.search(s) for s in sents),
        "has_subgroup_language": any(SUBGROUP.search(s) for s in sents),
    }


def main():
    res = json.load(open(RESULTS))
    sheet = {norm_title(r["Title"]): r for r in csv.DictReader(open(NEW_CSV))}
    papers = {norm_title(p["title"]): p for p in json.load(open(PAPERS))}

    fps, fns = [], []
    for r in res:
        k = norm_title(r["title"])
        g = sheet.get(k)
        if not g:
            continue
        ge, gd = parse_taxa(g[ENR]), parse_taxa(g[DEP])
        if not (ge or gd):
            continue
        pe, pd = parse_taxa(r["predicted_enriched"]), parse_taxa(r["predicted_depleted"])
        for direction, pred, gold, gold_other, pred_other in [
                ("enriched", pe, ge, gd, pd), ("depleted", pd, gd, ge, pe)]:
            _, fp_i, fn_i = match_pairs(pred, gold)
            for i in fp_i:
                fps.append({"key": k, "title": r["title"], "direction": direction,
                            "taxon": pred[i], "gold_same_dir": gold, "gold_other_dir": gold_other})
            for j in fn_i:
                fns.append({"key": k, "title": r["title"], "direction": direction,
                            "taxon": gold[j], "pred_same_dir": pred, "pred_other_dir": pred_other})

    print(f"pool: {len(fps)} FP taxa, {len(fns)} FN taxa "
          f"(across {len(set(x['key'] for x in fps))} / {len(set(x['key'] for x in fns))} papers)")
    json.dump({"n_fp": len(fps), "n_fn": len(fns),
               "fp": [{kk: v for kk, v in x.items() if kk != 'gold_same_dir'} for x in fps],
               "fn": [{kk: v for kk, v in x.items() if kk != 'pred_same_dir'} for x in fns]},
              open(os.path.join(HERE, "error_pool.json"), "w"), indent=2)

    def enrich(sample, kind):
        out = []
        for x in sample:
            p = papers.get(x["key"])
            text = re.sub(r"\s+", " ", p["text"]) if p else ""
            sents, present = sentences_for(x["taxon"], text)
            d = {"kind": kind, "title": x["title"][:110], "direction": x["direction"],
                 "taxon": x["taxon"], "present_in_text": present,
                 "char_len": len(text), "evidence": tag(sents), "sentences": sents}
            if kind == "FP":
                # is it in the gold under the OTHER direction? -> direction flip
                _, fp2, _ = match_pairs([x["taxon"]], x["gold_other_dir"]) if x["gold_other_dir"] \
                    else ([], [0], [])
                d["in_gold_opposite_direction"] = (len(fp2) == 0)
            else:
                _, fp2, _ = match_pairs([x["taxon"]], x["pred_other_dir"]) if x["pred_other_dir"] \
                    else ([], [0], [])
                d["predicted_in_opposite_direction"] = (len(fp2) == 0)
            out.append(d)
        return out

    # stratified sample: shuffle papers, then take taxa round-robin so one
    # verbose paper cannot supply the whole sample
    def strat(pool, n, seed):
        rng = random.Random(seed)
        by = {}
        for x in pool:
            by.setdefault(x["key"], []).append(x)
        keys = list(by)
        rng.shuffle(keys)
        for k in keys:
            rng.shuffle(by[k])
        out, rnd = [], 0
        while len(out) < n:
            added = False
            for k in keys:
                if rnd < len(by[k]):
                    out.append(by[k][rnd]); added = True
                    if len(out) >= n:
                        break
            if not added:
                break
            rnd += 1
        return out

    fp_s = enrich(strat(fps, N_SAMPLE, 7), "FP")
    fn_s = enrich(strat(fns, N_SAMPLE, 8), "FN")
    json.dump({"fp_sample": fp_s, "fn_sample": fn_s},
              open(os.path.join(HERE, "sample_errors.json"), "w"), indent=2)

    # quick automatic tallies over the FULL pools (cheap signal to complement
    # the hand read of the samples)
    print("\nautomatic scan over the FULL pools (presence in text only):")
    for kind, pool in [("FP", fps), ("FN", fns)]:
        pres = 0
        for x in pool:
            p = papers.get(x["key"])
            if not p:
                continue
            _, ok = sentences_for(x["taxon"], re.sub(r"\s+", " ", p["text"]))
            pres += ok
        print(f"   {kind}: {pres}/{len(pool)} = {pres/len(pool):.1%} of taxa appear in the paper text")

    print(f"\nwrote sample_errors.json ({len(fp_s)} FP + {len(fn_s)} FN dossiers) and error_pool.json")


if __name__ == "__main__":
    main()
