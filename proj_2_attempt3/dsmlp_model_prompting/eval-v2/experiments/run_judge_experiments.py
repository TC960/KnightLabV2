#!/usr/bin/env python3
"""Gold-free LLM-as-judge verification pass. Each model re-reads the paper and verifies its OWN
single-shot extractions: for a batch of claims ("Taxon X is enriched/depleted in disease vs HC"),
answer yes/no per claim; drop the no's. No gold is used -> scalable to unlabelled papers.

Runs two batch sizes (1 claim/call and 3 claims/call) to measure the latency tradeoff: each call
re-processes the whole paper, so batching amortises that cost. Logs per-paper + total time and #calls.

    /tmp/venv/bin/python run_judge_experiments.py --model qwopus3.5-27b-v3
"""
import argparse, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EVAL)

MODELS = {
    "qwen2.5-32b-instruct": ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
                             "qwen2.5-32b-instruct__q4km__samgated-v1__testv2.json"),
    "qwopus3.5-27b-v3":     ("Jackrong/Qwopus3.5-27B-v3-GGUF", "Qwopus3.5-27B-v3-Q4_K_M.gguf",
                             "qwopus3.5-27b-v3__q4km__samgated-v1__testv2.json"),
    "qwythos-9b":           ("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF", "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf",
                             "qwythos-9b__q8__samgated-v1__testv2.json"),
}
N_CTX = 24576
TS = json.load(open(os.path.join(EVAL, "..", "..", "EmilySong_GoldStandardPaper", "test_set_v2.json")))

JUDGE_YN = r'''
root ::= "[" ws (yn (ws "," ws yn)*)? ws "]"
yn ::= "\"yes\"" | "\"no\""
ws ::= [ \t\n]*
'''
PROMPT = """<task>
Verify claims against the paper. For EACH numbered claim, does the paper report that taxon as a
STATISTICALLY SIGNIFICANT difference (enriched=higher / depleted=lower) in the DISEASE vs HEALTHY
CONTROL comparison? Answer "yes" only if the paper's main text supports it; otherwise "no".
Output ONLY a JSON array of "yes"/"no" in the same order as the claims, nothing else.
</task>
<claims>
{claims}
</claims>
<paper>
{text}
</paper>
<output>
"""


def parse_taxa_dir(val, direction):
    if val is None or str(val).strip().lower() in ("", "nan", "none"):
        return []
    out = []
    for t in re.split(r"[,;]", str(val)):
        t = re.sub(r"\(.*?\)", "", t).strip()
        t = re.sub(r"p\s*[<>=]\s*[\d.]+", "", t, flags=re.I).strip().strip(".) ")
        if t and t.lower() != "nan" and len(t) > 2:
            out.append((t, direction))
    return out


def parse_arr(raw):
    c = raw.strip()
    if "</think>" in c: c = c.split("</think>")[-1]
    if "```" in c: c = c.split("```")[-2] if c.count("```") >= 2 else c
    i, j = c.find("["), c.rfind("]")
    try:
        return json.loads(c[i:j+1]) if i >= 0 and j > i else []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    key = args.model; repo, fname, resfile = MODELS[key]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, "logs", f"judge_{key}.log"), "w")
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True); logf.write(line+"\n"); logf.flush()

    # original single-shot extractions, aligned to test_set order
    res = {r["title"]: r for r in json.load(open(os.path.join(EVAL, "results", resfile)))}
    claims_by_idx = {}
    for i, p in enumerate(TS):
        r = res.get(p["title"], {})
        claims = parse_taxa_dir(r.get("predicted_enriched"), "enriched") + \
                 parse_taxa_dir(r.get("predicted_depleted"), "depleted")
        claims_by_idx[i] = claims

    log(f"MODEL={key}  judging its own single-shot extractions ({sum(len(c) for c in claims_by_idx.values())} claims over {len(TS)} papers)")
    from llama_cpp import Llama, LlamaGrammar
    llm = Llama.from_pretrained(repo_id=repo, filename=fname, n_ctx=N_CTX, n_gpu_layers=-1, verbose=False)
    grammar = LlamaGrammar.from_string(JUDGE_YN)
    log(f"model loaded | metadata name={ (getattr(llm,'metadata',{}) or {}).get('general.name','?') }")

    for batch in (1, 3):
        judged = {}; total_calls = 0; t_start = time.time()
        for i, p in enumerate(TS):
            claims = claims_by_idx[i]
            keep = []
            t0 = time.time()
            for s in range(0, len(claims), batch):
                grp = claims[s:s+batch]
                cl = "\n".join(f"{n+1}. {tx} - {d} in disease" for n, (tx, d) in enumerate(grp))
                try:
                    out = llm.create_chat_completion(
                        messages=[{"role": "user", "content":
                            PROMPT.replace("{claims}", cl).replace("{text}", p["text"])}],
                        temperature=0.0, max_tokens=64, grammar=grammar)
                    verdicts = parse_arr(out["choices"][0]["message"]["content"])
                except Exception:
                    verdicts = ["yes"] * len(grp)   # fail-open (don't drop on parse error)
                total_calls += 1
                for k, (tx, d) in enumerate(grp):
                    v = verdicts[k] if k < len(verdicts) else "yes"
                    if str(v).lower().startswith("y"):
                        keep.append((tx, d))
            dt = time.time() - t0
            e = [t for t, d in keep if d == "enriched"]; dd = [t for t, d in keep if d == "depleted"]
            judged[str(i)] = {"taxa_enriched": e, "taxa_depleted": dd}
            log(f"  b{batch} [{i}] {len(claims)}claims -> kept {len(keep)} ({dt:.1f}s)")
        total = time.time() - t_start
        json.dump(judged, open(os.path.join(HERE, "cache", f"judged_b{batch}__{key}.json"), "w"), indent=2)
        log(f"BATCH={batch} DONE: {total_calls} calls, {total:.0f}s total ({total/len(TS):.1f}s/paper), "
            f"wrote cache/judged_b{batch}__{key}.json")


if __name__ == "__main__":
    main()
