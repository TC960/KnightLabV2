#!/usr/bin/env python3
"""
Taxonomy-aware re-scorer for the saved eval outputs (NO GPU / NO re-inference).

Re-scores the per-model result JSONs in results/ under a ladder of metrics so we can
see exactly how much each idea cuts false positives:

  V0  char-greedy        original metric (TF-IDF char-ngram cosine >=0.5, greedy)  -> reproduces leaderboard
  V1  +dedup             same, but duplicate predicted taxa collapsed first
  V2  +hungarian         dedup + optimal 1:1 assignment (scipy) instead of greedy
  V3  +taxonomy(exact)   dedup + hungarian + NCBI-taxid match (exact taxid=1.0,
                         ancestor/descendant=0.85, both-resolved-but-unrelated=0.0);
                         char-cosine fallback only when a name can't be resolved
  V4  +taxonomy(fuzzy)   V3 but names that miss exact NCBI lookup are fuzzy-resolved
                         via gnverifier (typos/synonyms) before matching

Pipeline per unique taxon string:  clean -> gnparser (canonical, strips authorship/spp.)
-> taxonkit name2taxid (exact) -> [gnverifier fuzzy] -> taxonkit lineage -t (ancestor taxids).

Direction-aware: enriched vs enriched, depleted vs depleted, summed (mirrors run_eval.py).
"""
import json, os, re, glob, subprocess, csv, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
BIN     = os.environ.get("TAX_BIN", "/tmp/bin")
DUMP    = os.environ.get("TAX_DUMP", "/tmp/taxdump")

THRESH     = 0.50   # score >= THRESH counts as a match (same cut as the char metric)
LINEAGE_SC = 0.85   # ancestor/descendant (rank mismatch, same clade) -> treated as a hit

# 3 keep-models first, then the 2 dropped MoEs (shown for completeness).
KEEP = ["qwen2.5-32b-instruct", "qwythos-9b", "qwopus3.5-27b-v3"]
DROP = ["qwen3.6-35b-a3b", "qwopus3.6-35b-a3b-mtp"]


# ---------------------------------------------------------------- parsing
def split_clean(val):
    """Raw '<taxon; taxon, ...>' string -> list of cleaned names (case preserved)."""
    if val is None or str(val).strip().lower() in ("", "nan", "none"):
        return []
    out = []
    for t in re.split(r"[,;]", str(val)):
        t = re.sub(r"\(.*?\)", "", t)                       # drop parentheticals
        t = re.sub(r"p\s*[<>=]\s*[\d.]+", "", t, flags=re.I)  # drop p-values
        t = t.strip().strip(".,) ").strip()
        t = re.sub(r"\s+", " ", t)
        if t and t.lower() != "nan" and len(t) > 2:
            out.append(t)
    return out


# ---------------------------------------------------------------- external tools
def run(cmd, stdin_text):
    return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True).stdout


def gnparser_canonical(names):
    """name -> CanonicalSimple (strips authorship, 'spp.', normalizes)."""
    if not names:
        return {}
    out = run([f"{BIN}/gnparser", "-f", "tsv"], "\n".join(names) + "\n")
    m = {}
    for line in out.splitlines()[1:]:                # skip header
        c = line.split("\t")
        if len(c) >= 5 and c[1]:
            m[c[1]] = c[4] or c[1]                    # Verbatim -> CanonicalSimple
    return m


def taxonkit_name2taxid(canon_names):
    """canonical -> first NCBI taxid (exact match against names.dmp)."""
    if not canon_names:
        return {}
    out = run([f"{BIN}/taxonkit", "name2taxid", "--data-dir", DUMP],
              "\n".join(canon_names) + "\n")
    m = {}
    for line in out.splitlines():
        c = line.split("\t")
        if len(c) >= 2 and c[1] and c[0] not in m:
            m[c[0]] = c[1]
    return m


def gnverifier_fuzzy(canon_names):
    """canonical -> NCBI taxid via fuzzy match (typos/synonyms). Needs internet."""
    if not canon_names:
        return {}
    out = run([f"{BIN}/gnverifier", "-s", "4", "-f", "compact"],
              "\n".join(canon_names) + "\n")
    m = {}
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        best = j.get("bestResult")
        if best and best.get("dataSourceId") == 4 and best.get("recordId"):
            m[j.get("name", "")] = str(best["recordId"])
    return m


def taxonkit_lineage(taxids):
    """taxid -> set(ancestor taxids incl. self) from 'taxonkit lineage -t'."""
    ids = sorted({t for t in taxids if t})
    if not ids:
        return {}
    out = run([f"{BIN}/taxonkit", "lineage", "-t", "--data-dir", DUMP],
              "\n".join(ids) + "\n")
    m = {}
    for line in out.splitlines():
        c = line.split("\t")
        if len(c) >= 3 and c[2]:
            m[c[0]] = set(c[2].split(";"))
    return m


# ---------------------------------------------------------------- resolver
class Resolver:
    """cleaned name -> {'exact': taxid|None, 'fuzzy': taxid|None} + lineage sets."""
    def __init__(self, names):
        uniq = sorted(set(names))
        self.canon   = gnparser_canonical(uniq)                       # name -> canonical
        canon_vals   = sorted({self.canon.get(n, n) for n in uniq})
        self.exact   = taxonkit_name2taxid(canon_vals)                # canonical -> taxid
        missed       = [c for c in canon_vals if c not in self.exact]
        self.fuzzy   = gnverifier_fuzzy(missed)                       # canonical -> taxid
        all_taxids   = set(self.exact.values()) | set(self.fuzzy.values())
        self.lineage = taxonkit_lineage(all_taxids)                   # taxid -> ancestor set

    def taxid(self, name, use_fuzzy):
        c = self.canon.get(name, name)
        if c in self.exact:
            return self.exact[c]
        if use_fuzzy and c in self.fuzzy:
            return self.fuzzy[c]
        return None

    def related(self, ta, tb):
        """1.0 same taxid; LINEAGE_SC ancestor/descendant; 0.0 both resolved & unrelated."""
        if ta == tb:
            return 1.0
        la, lb = self.lineage.get(ta, {ta}), self.lineage.get(tb, {tb})
        if ta in lb or tb in la:
            return LINEAGE_SC
        return 0.0

    def coverage(self):
        canon_vals = set(self.canon.values())
        ex = len(self.exact); fz = len(self.fuzzy)
        tot = len(canon_vals)
        return tot, ex, fz, tot - ex - fz


# ---------------------------------------------------------------- matching
def char_sim(pred, gold):
    """char-ngram cosine matrix (lowercased), as in the original metric."""
    p = [x.lower() for x in pred]; g = [x.lower() for x in gold]
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(p + g)
    return cosine_similarity(tf[:len(p)], tf[len(p):])


def greedy_char(pred, gold):
    """V0: original greedy char metric (no dedup)."""
    if not pred and not gold: return 0, 0, 0
    if not pred: return 0, 0, len(gold)
    if not gold: return 0, len(pred), 0
    sim = char_sim(pred, gold)
    matched, tp, fp = set(), 0, 0
    for i in range(len(pred)):
        j = int(sim[i].argmax())
        if float(sim[i][j]) >= THRESH:
            tp += 1; matched.add(j)
        else:
            fp += 1
    return tp, fp, len(gold) - len(matched)


def hungarian(pred, gold, score_matrix):
    if not pred and not gold: return 0, 0, 0
    if not pred: return 0, 0, len(gold)
    if not gold: return 0, len(pred), 0
    S = score_matrix
    ri, ci = linear_sum_assignment(-S)
    tp, matched_gold = 0, set()
    for i, j in zip(ri, ci):
        if S[i, j] >= THRESH:
            tp += 1; matched_gold.add(int(j))
    return tp, len(pred) - tp, len(gold) - len(matched_gold)


def build_matrix(pred, gold, res, use_fuzzy):
    """Taxonomy score where both names resolve; char-cosine fallback otherwise."""
    S = char_sim(pred, gold).astype(float)
    pt = [res.taxid(x, use_fuzzy) for x in pred]
    gt = [res.taxid(x, use_fuzzy) for x in gold]
    for i, a in enumerate(pt):
        for j, b in enumerate(gt):
            if a and b:
                S[i, j] = res.related(a, b)          # override with taxonomy
    return S


def dedup(names, res=None, use_fuzzy=False):
    """Collapse duplicates by resolved taxid (if any) else lowercased name."""
    seen, out = set(), []
    for n in names:
        key = None
        if res is not None:
            key = res.taxid(n, use_fuzzy)
        key = key or n.lower()
        if key not in seen:
            seen.add(key); out.append(n)
    return out


# ---------------------------------------------------------------- scoring a model
VARIANTS = ["V0_char_greedy", "V1_dedup", "V2_hungarian",
            "V3_tax_exact", "V4_tax_fuzzy"]


def score_model(records, res):
    agg = {v: [0, 0, 0] for v in VARIANTS}   # TP, FP, FN
    for r in records:
        for pred_key, gold_key in (("predicted_enriched", "expected_enriched"),
                                   ("predicted_depleted", "expected_depleted")):
            pred = split_clean(r[pred_key])
            gold = split_clean(r[gold_key])

            # V0 greedy char, no dedup
            add(agg["V0_char_greedy"], greedy_char(pred, gold))

            # V1 dedup + greedy char
            pd = dedup(pred)
            add(agg["V1_dedup"], greedy_char(pd, gold))

            # V2 dedup + hungarian (char)
            if pd and gold:
                add(agg["V2_hungarian"], hungarian(pd, gold, char_sim(pd, gold)))
            else:
                add(agg["V2_hungarian"], greedy_char(pd, gold))

            # V3 taxonomy exact  (dedup by taxid)
            pde = dedup(pred, res, use_fuzzy=False)
            gde = dedup(gold, res, use_fuzzy=False)
            if pde and gde:
                add(agg["V3_tax_exact"], hungarian(pde, gde, build_matrix(pde, gde, res, False)))
            else:
                add(agg["V3_tax_exact"], greedy_char(pde, gde))

            # V4 taxonomy fuzzy
            pdf = dedup(pred, res, use_fuzzy=True)
            gdf = dedup(gold, res, use_fuzzy=True)
            if pdf and gdf:
                add(agg["V4_tax_fuzzy"], hungarian(pdf, gdf, build_matrix(pdf, gdf, res, True)))
            else:
                add(agg["V4_tax_fuzzy"], greedy_char(pdf, gdf))
    return agg


def add(acc, tpl):
    acc[0] += tpl[0]; acc[1] += tpl[1]; acc[2] += tpl[2]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------- main
def main():
    order = KEEP + DROP
    files = {}
    for path in glob.glob(os.path.join(RESULTS, "*__testv2.json")):
        key = os.path.basename(path).split("__")[0]
        files[key] = path

    # one resolver over ALL taxa in ALL files (batched external calls)
    all_names = []
    for path in files.values():
        for r in json.load(open(path)):
            for k in ("predicted_enriched", "expected_enriched",
                      "predicted_depleted", "expected_depleted"):
                all_names += split_clean(r[k])
    print(f"resolving {len(set(all_names))} unique taxa via gnparser/taxonkit/gnverifier ...",
          flush=True)
    res = Resolver(all_names)
    tot, ex, fz, un = res.coverage()
    print(f"  canonical names: {tot} | exact NCBI: {ex} | fuzzy: {fz} | unresolved: {un}\n",
          flush=True)

    rows = []
    for key in order:
        if key not in files:
            continue
        recs = json.load(open(files[key]))
        agg = score_model(recs, res)
        tag = "keep" if key in KEEP else "drop"
        print(f"### {key}  ({tag})")
        print(f"    {'variant':<16} {'P':>6} {'R':>6} {'F1':>6}   TP/FP/FN")
        for v in VARIANTS:
            tp, fp, fn = agg[v]
            p, r, f = prf(tp, fp, fn)
            print(f"    {v:<16} {p:6.3f} {r:6.3f} {f:6.3f}   {tp}/{fp}/{fn}")
            rows.append({"model": key, "tag": tag, "variant": v,
                         "TP": tp, "FP": fp, "FN": fn,
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
        print()

    out = os.path.join(RESULTS, "leaderboard_normalized.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "tag", "variant", "TP", "FP", "FN",
                                          "precision", "recall", "f1"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
