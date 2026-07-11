#!/usr/bin/env python3
"""Shared helpers for the extraction-experiment suite.

Each run_*.py produces results/<method>.json (per-paper predictions) and appends a row to
results/experiments_leaderboard.csv, scored with the taxonomy-aware metric (taxonomy_match).

LLM backend: scripts read model outputs from cache/<method>.json (generated once, e.g. via
Fable subagents for the pilot, or by a GGUF batch on DSMLP). Deterministic stages (retrieval,
grounding checks, normalization, scoring) run in pure Python here.
"""
import json, os, re, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL); sys.path.insert(0, "/tmp/pylibs")
PILOT = os.path.join(HERE, "papers_pilot.json")
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
LEDGER = os.path.join(RESULTS, "experiments_leaderboard.csv")

# reference gold to score against: "fable" (the thorough benchmark) or "orig"
GOLD = os.environ.get("EXP_GOLD", "fable")


def load_papers():
    return json.load(open(PILOT))


def parse_taxa(val):
    if val is None or str(val).strip().lower() in ("", "nan", "none"):
        return []
    out = []
    for t in re.split(r"[,;]", str(val)):
        t = re.sub(r"\(.*?\)", "", t).strip().lower()
        t = re.sub(r"p\s*[<>=]\s*[\d.]+", "", t).strip().strip(".) ")
        if t and t != "nan" and len(t) > 2:
            out.append(t)
    return out


def gold_for(paper):
    if GOLD == "orig":
        return parse_taxa(paper["orig_enriched"]), parse_taxa(paper["orig_depleted"])
    return parse_taxa(paper["fable_enriched"]), parse_taxa(paper["fable_depleted"])


_resolver = None
def _get_resolver(names):
    global _resolver
    from taxonomy_match import TaxResolver
    if _resolver is None:
        _resolver = TaxResolver()
    _resolver.warm(names)
    return _resolver


def score(predictions):
    """predictions: list of {idx, pred_enriched:[..], pred_depleted:[..]} -> (P,R,F1,TP,FP,FN)."""
    from taxonomy_match import match_taxa_lca
    papers = {p["idx"]: p for p in load_papers()}
    names = []
    for pr in predictions:
        names += [x.lower() for x in pr["pred_enriched"] + pr["pred_depleted"]]
    for p in papers.values():
        ge, gd = gold_for(p); names += ge + gd
    R = _get_resolver(names)
    TP = FP = FN = 0
    for pr in predictions:
        p = papers[pr["idx"]]; ge, gd = gold_for(p)
        for pred, exp in ((pr["pred_enriched"], ge), (pr["pred_depleted"], gd)):
            tp, fp, fn = match_taxa_lca([x.lower() for x in pred], exp, R)
            TP += tp; FP += fp; FN += fn
    P = TP / (TP + FP) if TP + FP else 0
    Rc = TP / (TP + FN) if TP + FN else 0
    F = 2 * P * Rc / (P + Rc) if P + Rc else 0
    return P, Rc, F, TP, FP, FN


def save_and_report(method, predictions, note=""):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(predictions, open(os.path.join(RESULTS, f"{method}.json"), "w"), indent=2)
    P, Rc, F, TP, FP, FN = score(predictions)
    exists = os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["method", "gold", "n", "precision", "recall", "f1", "TP", "FP", "FN", "note"])
        w.writerow([method, GOLD, len(predictions), round(P, 3), round(Rc, 3), round(F, 3), TP, FP, FN, note])
    print(f"[{method}] gold={GOLD}  P={P:.3f} R={Rc:.3f} F1={F:.3f}  (TP={TP} FP={FP} FN={FN})  {note}")
    return F


def read_cache(method):
    path = os.path.join(CACHE, f"{method}.json")
    if not os.path.exists(path):
        print(f"!! missing cache/{method}.json — generate LLM outputs first "
              f"(Fable subagents for pilot, or GGUF batch on DSMLP). Skipping.")
        return None
    return json.load(open(path))


# ---- RELATE stage-1: deterministic candidate retrieval ----
CUE = re.compile(r"(lefse|lda|significant|enrich|deplet|abundan|higher|lower|elevat|reduc|"
                 r"increas|decreas|over-?repres|under-?repres|p\s*[<>=]|q\s*[<>=]|fdr|differ)", re.I)
# a taxon-looking token: Capitalized word, optionally binomial / with SILVA underscores/brackets
TAXON = re.compile(r"\[?[A-Z][A-Za-z]+(?:[_\- ][A-Za-z0-9]+){0,2}\]?")
STOP = {"The","This","We","Our","In","Results","Discussion","Table","Figure","Fig","However",
        "These","Group","Groups","Control","Controls","Patients","Disease","Abstract","Methods",
        "Study","Data","Analysis","Among","Both","Although","Here","There","Significant"}


def retrieve_candidates(text, max_sents=60):
    """Return (candidate_taxa, evidence_sentences): sentences near a significance cue + the
    Capitalized taxon-like tokens in them. This is RELATE stage 1 (recall-biased retrieval)."""
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    ev, cands = [], {}
    for s in sents:
        if CUE.search(s):
            ev.append(s.strip())
            for m in TAXON.findall(s):
                tok = m.strip()
                head = tok.split()[0].strip("[]")
                if head in STOP or len(tok) < 4:
                    continue
                cands.setdefault(tok.lower(), tok)
        if len(ev) >= max_sents:
            break
    return sorted(set(cands.values())), ev
