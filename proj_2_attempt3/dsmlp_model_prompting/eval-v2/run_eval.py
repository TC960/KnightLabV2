"""
run_eval.py — single-model microbe-taxa extraction eval on test_set_v2.

One model per invocation (single GPU -> run sequentially, not in parallel).
    python run_eval.py --model qwopus3.5-27b-v3
    python run_eval.py --model qwythos-9b --limit 2      # smoke test

Prompt: samgated-v1 (encodes Sam's 5 gates, Source-Of-Truth.md). Output is
grammar-constrained JSON {disease, taxa_enriched, taxa_depleted}, so every model
is scored on the same schema with no CoT (matches the earlier finding that CoT
hurt Qwopus).

Metric: greedy taxa matching, TF-IDF char-ngram cosine >= 0.5 (same as main_eval.ipynb).
NOTE: greedy, NOT Hungarian, and NOT taxonomy-normalized — see NAMING/METRIC notes in chat.

Results -> results/<model>__<quant>__samgated-v1__testv2.{json,csv}
Summary appended to results/leaderboard.csv
"""
import argparse, json, time, os, csv, re, sys, shutil

HERE      = os.path.dirname(os.path.abspath(__file__))
_EMILY    = os.path.join(HERE, "..", "..", "EmilySong_GoldStandardPaper")
DATA_PATH = os.path.join(_EMILY, "test_set_v2.json")
RESULTS   = os.environ.get("EVAL_OUT_DIR") or os.path.join(HERE, "results")
HF_CACHE  = os.path.join(os.environ.get("HF_HOME", "/tmp/hf"), "hub")

# ---- dataset registry: label -> path. "testv2" is the 15-paper benchmark the
# leaderboard was built on; "all250" is every usable Emily paper (corpus run). ----
DATASETS = {
    "testv2": os.path.join(_EMILY, "test_set_v2.json"),
    "all250": os.path.join(_EMILY, "all_usable_papers.json"),
}

PROMPT_VERSION = "samgated-v1"
DATASET        = "testv2"
N_CTX, MAX_TOKENS, TEMP = 24576, 1500, 0.0

# ---- model registry: key -> (repo_id, gguf filename, quant label) ----------
MODELS = {
    "qwen2.5-32b-instruct":   ("bartowski/Qwen2.5-32B-Instruct-GGUF",              "Qwen2.5-32B-Instruct-Q4_K_M.gguf",          "q4km"),
    "qwopus3.5-27b-v3":       ("Jackrong/Qwopus3.5-27B-v3-GGUF",                   "Qwopus3.5-27B-v3-Q4_K_M.gguf",              "q4km"),
    "qwen3.6-35b-a3b":        ("unsloth/Qwen3.6-35B-A3B-GGUF",                     "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",            "q4km"),
    "qwythos-9b":             ("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",     "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf",   "q8"),
    "qwopus3.6-35b-a3b-mtp":  ("Jackrong/Qwopus3.6-35B-A3B-v1-MTP-GGUF",           "Qwopus3.6-35B-A3B-v1-MTP-Q4_K_M.gguf",      "q4km"),
}

PROMPT_TEMPLATE = """<task>
You are extracting data for a systematic review of gut/oral microbiome differences in
neurocognitive disease. From the paper below, list the microbial taxa that are significantly
ENRICHED (higher in disease) and significantly DEPLETED (lower in disease) in the DISEASE group
versus a HEALTHY CONTROL group. Output exactly one JSON object and nothing else.
</task>

<inclusion_rules>
- SIGNIFICANCE: include a taxon ONLY if the paper reports it as statistically significant
  (p / FDR / q < 0.05, or a significant LEfSe/LDA or differential-abundance result). If significance
  is unclear or unreported for a taxon, omit it.
- DISEASE vs HEALTHY CONTROL ONLY: the comparison must be the disease group vs. a healthy control
  group. Ignore disease-vs-disease comparisons, disease-severity gradients, subgroup-vs-subgroup
  comparisons, and symptom-correlation findings (e.g. "associated with worse cognition"). If the
  study has NO healthy control group, return empty taxa lists.
- MAIN TEXT ONLY: only include taxa named in the running text (Abstract, Results, Discussion).
  Ignore taxa that appear only in tables, figures, figure/table captions, or supplementary material.
- HUMAN HOSTS ONLY: ignore results from non-human hosts / animal models.
- Copy taxon names verbatim (keep the exact spelling, brackets, and underscores as written).
- "disease": the primary disease studied.
</inclusion_rules>

<examples>
<example>
{{"disease": "Parkinson's disease", "taxa_enriched": ["Akkermansia", "Bifidobacterium"], "taxa_depleted": ["Faecalibacterium", "Roseburia"]}}
</example>
<example>
{{"disease": "Alzheimer's disease", "taxa_enriched": ["Bacteroides"], "taxa_depleted": ["Bifidobacterium"]}}
</example>
<example>
{{"disease": "Multiple sclerosis", "taxa_enriched": ["Akkermansia muciniphila"], "taxa_depleted": []}}
</example>
</examples>

<paper>
{text}
</paper>

<output>
"""

GRAMMAR_STR = r'''
root ::= "{" ws "\"disease\":" ws string "," ws "\"taxa_enriched\":" ws array "," ws "\"taxa_depleted\":" ws array ws "}"
array ::= "[" ws (string ("," ws string)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
'''


def smart_truncate(text, n_ctx=None):
    """Drop the reference list, then hard-cap to what n_ctx can actually hold.

    The reference-strip alone does NOT bound length: on the 250-paper set 12
    papers still exceed a 24576-token window after stripping (max 105k chars
    ~= 26k tokens), which would silently overflow the context. The cap reserves
    room for the prompt scaffold + MAX_TOKENS of output and is a no-op whenever
    the paper already fits (true for all 15 testv2 papers at either n_ctx, so
    existing leaderboard numbers are unaffected).
    """
    for marker in ["References\n", "REFERENCES\n", "Bibliography\n", "REFERENCE\n"]:
        idx = text.rfind(marker)
        if idx > 0:
            text = text[:idx]
            break
    if n_ctx:
        budget = (n_ctx - MAX_TOKENS - 400) * 4   # ~4 chars/token, 400 for scaffold
        if len(text) > budget:
            text = text[:budget]
    return text


def parse_output(raw):
    clean = raw.strip()
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()
    if not clean.startswith("{"):
        clean = clean[clean.find("{"):]
    return json.loads(clean)


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


def match_taxa(predicted, expected):
    """Greedy: each predicted taxon -> its best expected by char-ngram cosine; hit if >=0.5."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not predicted and not expected:
        return 0, 0, 0
    if not predicted:
        return 0, 0, len(expected)
    if not expected:
        return 0, len(predicted), 0
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(predicted + expected)
    sim = cosine_similarity(tf[:len(predicted)], tf[len(predicted):])
    matched_exp, tp, fp = set(), 0, 0
    for i in range(len(predicted)):
        j = sim[i].argmax()
        if float(sim[i][j]) >= 0.5:
            tp += 1; matched_exp.add(j)
        else:
            fp += 1
    return tp, fp, len(expected) - len(matched_exp)


def append_leaderboard(row):
    path = os.path.join(RESULTS, "leaderboard.csv")
    cols = ["model_key", "repo", "quant", "prompt", "dataset", "n",
            "TP", "FP", "FN", "precision", "recall", "f1", "avg_sec", "load_ok", "notes"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def cleanup_hf(repo_id):
    slug = "models--" + repo_id.replace("/", "--")
    d = os.path.join(HF_CACHE, slug)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        print(f"cleaned cache: {d}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep model in HF cache after run")
    ap.add_argument("--metric", choices=["taxonomy", "char"], default="taxonomy",
                    help="taxonomy = NCBI-lineage (LCA) matching w/ char fallback; char = original fuzzy only")
    ap.add_argument("--dataset", choices=list(DATASETS), default="testv2",
                    help="testv2 = 15-paper benchmark (default, leaderboard); all250 = full Emily corpus")
    ap.add_argument("--n-ctx", type=int, default=N_CTX,
                    help=f"context window (default {N_CTX}; use 32768 for --dataset all250)")
    ap.add_argument("--resume", action="store_true",
                    help="skip papers already present in the checkpoint file")
    args = ap.parse_args()

    key = args.model
    repo, fname, quant = MODELS[key]
    dataset = args.dataset
    n_ctx = args.n_ctx
    tag = f"{key}__{quant}__{PROMPT_VERSION}__{dataset}"
    os.makedirs(RESULTS, exist_ok=True)
    ckpt_path = os.path.join(RESULTS, f"{tag}.checkpoint.jsonl")

    with open(DATASETS[dataset]) as f:
        papers = json.load(f)
    if args.limit:
        papers = papers[:args.limit]

    # ---- resume: replay whatever the checkpoint already holds ----
    done = {}
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["title"]] = r
                except Exception:
                    continue
        print(f"resume: {len(done)} papers already done in {os.path.basename(ckpt_path)}", flush=True)

    print(f"=== {key} | {repo} / {fname} ===", flush=True)
    print(f"{len(papers)} papers | prompt={PROMPT_VERSION} | dataset={dataset} | n_ctx={n_ctx}", flush=True)

    # ---- load model (robust: a load failure is logged, not fatal) ----
    from llama_cpp import Llama, LlamaGrammar
    try:
        llm = Llama.from_pretrained(repo_id=repo, filename=fname,
                                    n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        grammar = LlamaGrammar.from_string(GRAMMAR_STR)
        print("model loaded\n", flush=True)
    except Exception as e:
        note = f"LOAD FAILED: {type(e).__name__}: {str(e)[:160]}"
        print(note, flush=True)
        append_leaderboard({"model_key": key, "repo": repo, "quant": quant,
                            "prompt": PROMPT_VERSION, "dataset": dataset,
                            "n": len(papers), "load_ok": False, "notes": note})
        if not args.keep:
            cleanup_hf(repo)
        return

    results, times = [], []
    ckpt = open(ckpt_path, "a")
    for i, paper in enumerate(papers):
        if paper["title"] in done:
            results.append(done[paper["title"]])
            print(f"[{i+1}/{len(papers)}] (cached) {paper['title'][:56]}...", flush=True)
            continue
        print(f"[{i+1}/{len(papers)}] {paper['title'][:68]}...", flush=True)
        text = smart_truncate(paper["text"], n_ctx=n_ctx)
        t0 = time.time()
        perr = False
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
                temperature=TEMP, max_tokens=MAX_TOKENS, grammar=grammar)
            raw = out["choices"][0]["message"]["content"]
            p = parse_output(raw)
        except Exception as e:
            print(f"    error: {type(e).__name__}: {str(e)[:120]}", flush=True)
            p = {"disease": paper.get("disease", ""), "taxa_enriched": [], "taxa_depleted": []}
            perr = True
        dt = time.time() - t0
        times.append(dt)
        row = {
            "title": paper["title"], "disease": paper.get("disease", ""),
            "link": paper.get("link", ""),
            "in_gold_standard": paper.get("in_gold_standard", ""),
            "expected_enriched": paper.get("taxa_enriched", ""),
            "expected_depleted": paper.get("taxa_depleted", ""),
            "predicted_enriched": ", ".join(p.get("taxa_enriched", [])),
            "predicted_depleted": ", ".join(p.get("taxa_depleted", [])),
            "predicted_disease": p.get("disease", ""),
            "time_seconds": round(dt, 2), "parse_error": perr}
        results.append(row)
        ckpt.write(json.dumps(row) + "\n"); ckpt.flush()   # crash-safe: 250 papers is hours
        print(f"    enr={p.get('taxa_enriched', [])}", flush=True)
        print(f"    dep={p.get('taxa_depleted', [])}  ({dt:.1f}s)", flush=True)
    ckpt.close()

    # ---- save per-model results ----
    with open(os.path.join(RESULTS, f"{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(RESULTS, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)

    # ---- score (taxonomy-aware by default; falls back to char if tools absent) ----
    metric_note = "char"
    resolver = None
    if args.metric == "taxonomy":
        from taxonomy_match import TaxResolver, match_taxa_lca
        resolver = TaxResolver()
        if resolver.ok:
            names = []
            for r in results:
                for col in ("predicted_enriched", "predicted_depleted",
                            "expected_enriched", "expected_depleted"):
                    names += parse_taxa(r[col])
            resolver.warm(names)
            metric_note = "taxonomy-lca"
        else:
            print("!! taxonomy tools missing (run: python taxonomy_match.py --setup) "
                  "-- falling back to char metric", flush=True)

    def score_pair(pred, exp):
        if resolver is not None and resolver.ok:
            return match_taxa_lca(pred, exp, resolver)
        return match_taxa(pred, exp)

    TP = FP = FN = 0
    for r in results:
        a = score_pair(parse_taxa(r["predicted_enriched"]), parse_taxa(r["expected_enriched"]))
        b = score_pair(parse_taxa(r["predicted_depleted"]), parse_taxa(r["expected_depleted"]))
        TP += a[0] + b[0]; FP += a[1] + b[1]; FN += a[2] + b[2]
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec  = TP / (TP + FN) if (TP + FN) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    avg  = sum(times) / len(times) if times else 0

    print("\n" + "=" * 60, flush=True)
    print(f"{key}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (TP={TP} FP={FP} FN={FN}, "
          f"{avg:.1f}s/paper, metric={metric_note})", flush=True)

    append_leaderboard({"model_key": key, "repo": repo, "quant": quant,
                        "prompt": PROMPT_VERSION, "dataset": dataset, "n": len(results),
                        "TP": TP, "FP": FP, "FN": FN,
                        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                        "avg_sec": round(avg, 1), "load_ok": True, "notes": f"metric={metric_note}"})

    if not args.keep:
        cleanup_hf(repo)


if __name__ == "__main__":
    main()
