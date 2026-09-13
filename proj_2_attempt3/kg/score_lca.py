#!/usr/bin/env python3
"""Taxonomy-aware (LCA / nested-lineage) scoring, on macOS, with no external binaries.

WHY THIS FILE EXISTS
--------------------
`dsmlp_model_prompting/eval-v2/taxonomy_match.py` implements the taxonomy-aware
metric that lifted Qwopus3.5 from .642 -> .751 on testv2. It cannot run here: it
shells out to `taxonkit` and `gnparser`, which are linux-amd64 binaries. The
NCBI taxdump is just two pipe-delimited text files, and `kg/taxonomy.py` already
parses them directly (plus it folds synonyms, strips rank prefixes like
`f__Rikenellaceae`, and rejects cross-kingdom homonyms -- none of which the
taxonkit path does). So the LCA rule is re-implemented on top of taxonomy.py.

THE RULE (identical in substance to taxonomy_match.match_taxa_lca)
------------------------------------------------------------------
Greedy, one pass over predictions. A predicted taxon matches an expected one if

    (a) TF-IDF char_wb n-gram(2,4) cosine >= 0.5   -- the original fuzzy metric, OR
    (b) their NCBI lineages are NESTED: one taxid is an ancestor of the other.

(b) only fires when (a) has already failed, so LCA is a pure *superset* of char:
TP can only go up, FP only down. Any F1 delta is the metric forgiving, never
punishing.

GUARD. Nesting with a very high node (Bacteria, or a phylum) would match almost
anything. `--min-lca-rank` drops nested pairs whose shallower member sits above a
given rank; the default keeps everything (faithful to eval-v2) but the driver
reports the sensitivity so the forgiveness can be audited.

Import surface (this is meant to be reused):
    LCA()                         -> resolver, .ok tells you if the taxdump loaded
    match_taxa_lca(pred, exp, r)  -> (tp, fp, fn)
    align_lca(pred, exp, r)       -> per-pair detail incl. WHICH rule fired
    score_rows(rows, matcher)     -> P/R/F1 per direction + combined
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "dsmlp_model_prompting", "eval-v2"))
sys.path.insert(0, HERE)

from run_eval import parse_taxa                      # noqa: E402  (shared tokenizer)
import taxonomy                                       # noqa: E402

GOLD_CSV = "/Users/mohak/Downloads/high_confidence - final_constrained_override.csv"
SHEET_CSV = os.path.join(HERE, "Microbiota Signatures Neurological Disorders Sheet 2 - Main Datasheet.csv")
EXTRACTIONS = os.path.join(HERE, "extractions_corrected.json")
EVAL_RESULTS = os.path.join(P3, "dsmlp_model_prompting", "eval-v2", "results")
TESTV2 = os.path.join(P3, "EmilySong_GoldStandardPaper", "test_set_v2.json")

# NOTE "domain" not "superkingdom": the 2024+ taxdump renamed that rank, and
# taxonomy.py passes NCBI's string through verbatim. Keying only on the old name
# silently gave Bacteria depth 99 (i.e. "very specific"), inverting the guard.
RANK_DEPTH = {"domain": 0, "superkingdom": 0, "kingdom": 1, "phylum": 2, "class": 3,
              "order": 4, "family": 5, "genus": 6, "species": 7,
              "subspecies": 8, "strain": 8}


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


# ------------------------------------------------------------------ resolver

class LCA:
    """Name -> (taxid, rank, ancestor-set). Everything memoised; the taxdump load
    is ~15 s and ~1 GB so exactly one instance should exist per process."""

    def __init__(self, min_rank=None):
        self.tax = taxonomy.shared()
        self.ok = self.tax.ok
        self._info = {}
        self._lin = {}
        # nested pairs are rejected if the SHALLOWER member is above this rank
        self.min_depth = RANK_DEPTH.get(min_rank) if min_rank else None

    def info(self, name):
        """-> (taxid, scientific_name, rank). taxid is '' when unresolved."""
        k = (name or "").lower()
        if k not in self._info:
            if not self.ok:
                self._info[k] = ("", name, None)
            else:
                tid, sci, rank, how = self.tax.resolve(name)
                self._info[k] = (tid or "", sci, rank)
        return self._info[k]

    def lineage(self, tid):
        if tid not in self._lin:
            self._lin[tid] = set(self.tax.lineage(tid))
        return self._lin[tid]

    def nested(self, a, b):
        """True iff a and b resolve and one lineage contains the other."""
        ta, _, ra = self.info(a)
        tb, _, rb = self.info(b)
        if not ta or not tb or ta == tb:
            # identical taxids count as nested (trivially) -- char usually caught
            # these already, but synonym pairs like Firmicutes/Bacillota do not
            # share characters and DO share a taxid.
            return bool(ta and tb and ta == tb)
        if not (ta in self.lineage(tb) or tb in self.lineage(ta)):
            return False
        if self.min_depth is not None:
            shallow = min(RANK_DEPTH.get(ra, 99), RANK_DEPTH.get(rb, 99))
            if shallow < self.min_depth:
                return False
        return True

    def warm(self, names):
        for n in names:
            tid, _, _ = self.info(n)
            if tid:
                self.lineage(tid)


# ------------------------------------------------------------------- matchers

def _char_sim(pred, exp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(pred + exp)
    return cosine_similarity(tf[:len(pred)], tf[len(pred):])


def match_taxa_char(predicted, expected, resolver=None):
    """Verbatim reimplementation of eval-v2/run_eval.match_taxa -- greedy, >=0.5."""
    if not predicted and not expected:
        return 0, 0, 0
    if not predicted:
        return 0, 0, len(expected)
    if not expected:
        return 0, len(predicted), 0
    sim = _char_sim(predicted, expected)
    matched, tp, fp = set(), 0, 0
    for i in range(len(predicted)):
        j = int(sim[i].argmax())
        if float(sim[i][j]) >= 0.5:
            tp += 1
            matched.add(j)
        else:
            fp += 1
    return tp, fp, len(expected) - len(matched)


def align_lca(predicted, expected, resolver, deepest=False):
    """Greedy char-then-LCA. Returns (pairs, fp, fn) where each pair is
    (pred, exp, rule, sim) and rule is 'char' or 'lca'."""
    if not predicted and not expected:
        return [], [], []
    if not predicted:
        return [], [], list(expected)
    if not expected:
        return [], list(predicted), []
    sim = _char_sim(predicted, expected)
    pairs, fp, matched = [], [], set()
    for i in range(len(predicted)):
        j = int(sim[i].argmax())
        s = float(sim[i][j])
        if s >= 0.5:
            pairs.append((predicted[i], expected[j], "char", round(s, 3)))
            matched.add(j)
            continue
        hit = -1
        if resolver is not None and resolver.ok:
            cands = [k for k in range(len(expected)) if resolver.nested(predicted[i], expected[k])]
            if cands:
                if deepest:
                    # tie-break on the most specific expected taxon rather than
                    # document order -- used as a sensitivity check
                    hit = max(cands, key=lambda k: RANK_DEPTH.get(resolver.info(expected[k])[2], -1))
                else:
                    hit = cands[0]
        if hit >= 0:
            pairs.append((predicted[i], expected[hit], "lca", round(s, 3)))
            matched.add(hit)
        else:
            fp.append(predicted[i])
    fn = [expected[j] for j in range(len(expected)) if j not in matched]
    return pairs, fp, fn


def match_taxa_lca(predicted, expected, resolver, deepest=False):
    pairs, fp, fn = align_lca(predicted, expected, resolver, deepest)
    return len(pairs), len(fp), len(fn)


# --------------------------------------------------------------------- scoring

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def score_rows(rows, matcher):
    """rows: [(pred_enr, pred_dep, gold_enr, gold_dep)] of RAW strings."""
    acc = {"enriched": [0, 0, 0], "depleted": [0, 0, 0]}
    for pe, pd_, ge, gd in rows:
        for key, p, g in (("enriched", pe, ge), ("depleted", pd_, gd)):
            tp, fp, fn = matcher(parse_taxa(p), parse_taxa(g))
            acc[key][0] += tp
            acc[key][1] += fp
            acc[key][2] += fn
    out = {}
    for k, (tp, fp, fn) in acc.items():
        p, r, f = prf(tp, fp, fn)
        out[k] = {"n": len(rows), "TP": tp, "FP": fp, "FN": fn,
                  "precision": p, "recall": r, "f1": f}
    tp = sum(acc[k][0] for k in acc)
    fp = sum(acc[k][1] for k in acc)
    fn = sum(acc[k][2] for k in acc)
    p, r, f = prf(tp, fp, fn)
    out["combined"] = {"n": len(rows), "TP": tp, "FP": fp, "FN": fn,
                       "precision": p, "recall": r, "f1": f}
    return out


def f1_only(rows, matcher):
    tp = fp = fn = 0
    for pe, pd_, ge, gd in rows:
        for p, g in ((pe, ge), (pd_, gd)):
            a, b, c = matcher(parse_taxa(p), parse_taxa(g))
            tp += a
            fp += b
            fn += c
    return prf(tp, fp, fn)[2]


# ------------------------------------------------------------------ gold loading

def load_new_gold():
    gold = defaultdict(dict)
    disorder = {}
    for r in csv.DictReader(open(GOLD_CSV)):
        doi = r["DOI"].strip().lower()
        gold[doi][r["field"].strip()] = r["high_confidence_taxa"].strip()
        disorder[doi] = r["disorder"].strip()
    return dict(gold), disorder


def load_sheet():
    """-> (doi -> row, normalised-title -> doi)"""
    doi2row, title2doi = {}, {}
    for r in csv.DictReader(open(SHEET_CSV)):
        d = r["DOI"].strip().lower()
        if not d:
            continue
        doi2row.setdefault(d, r)
        title2doi.setdefault(norm_title(r["Title"]), d)
    return doi2row, title2doi


def gold_is_blank(cells):
    return not (parse_taxa(cells.get("Enriched", "")) or parse_taxa(cells.get("Depleted", "")))


def dedup_first(records, key="title"):
    seen, keep = {}, []
    for e in records:
        k = norm_title(e[key])
        if k in seen:
            continue
        seen[k] = e
        keep.append(e)
    return seen, keep
