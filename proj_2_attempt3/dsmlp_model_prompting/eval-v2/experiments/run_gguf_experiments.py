#!/usr/bin/env python3
"""Run RELATE (rerank) + grounded extraction with a REAL GGUF model over all 15 papers.
Emits cache/relate__<model>.json and cache/grounded__<model>.json for the scoring suite.
Heavily logged (model metadata, CUDA offload, GPU snapshot, per-paper timing) to prove the
actual model ran — see logs/<model>.log.

    /tmp/venv/bin/python run_gguf_experiments.py --model qwopus3.5-27b-v3
"""
import argparse, json, os, re, sys, time, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); EVAL = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EVAL)
import common

MODELS = {
    "qwen2.5-32b-instruct": ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5-32B-Instruct-Q4_K_M.gguf"),
    "qwopus3.5-27b-v3":     ("Jackrong/Qwopus3.5-27B-v3-GGUF", "Qwopus3.5-27B-v3-Q4_K_M.gguf"),
    "qwythos-9b":           ("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF", "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf"),
}
N_CTX = 24576
TS = os.path.join(EVAL, "..", "..", "EmilySong_GoldStandardPaper", "test_set_v2.json")

GROUNDED_PROMPT = """<task>
Source-grounded extraction. From the paper below, list microbial taxa reported as SIGNIFICANTLY
ENRICHED (higher in disease) or DEPLETED (lower in disease) in the DISEASE vs HEALTHY CONTROL
comparison. Include a taxon ONLY if a VERBATIM sentence in the paper states its significance AND its
direction -- if you cannot point to such a sentence in the text, do NOT include the taxon. Disease vs
healthy control only; if there is no healthy-control arm, return empty arrays. Copy taxon names
verbatim. Output exactly one JSON object and nothing else.
</task>
<paper>
{text}
</paper>
<output>
"""

RELATE_PROMPT = """<task>
You are re-ranking retrieved candidate taxa with a REJECT option. Below are candidates (many are NOISE:
non-taxa, method terms, disease names) pulled from sentences near significance cues in the paper.
KEEP a candidate ONLY if the paper reports it as a microbial taxon SIGNIFICANTLY ENRICHED (higher in
disease) or DEPLETED (lower in disease) in the DISEASE vs HEALTHY CONTROL comparison. REJECT everything
else (non-taxa, not significant, wrong comparison, no control arm). Do not invent names. If none qualify,
return empty arrays. Output exactly one JSON object and nothing else.
</task>
<candidates>{candidates}</candidates>
<paper>
{text}
</paper>
<output>
"""

GRAMMAR = r'''
root ::= "{" ws "\"taxa_enriched\":" ws array "," ws "\"taxa_depleted\":" ws array ws "}"
array ::= "[" ws (string ("," ws string)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws ::= [ \t\n]*
'''


def log(logf, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); logf.write(line + "\n"); logf.flush()


def gpu_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"


def parse_obj(raw):
    c = raw.strip()
    if "</think>" in c: c = c.split("</think>")[-1]
    if "```json" in c: c = c.split("```json")[1].split("```")[0]
    elif "```" in c: c = c.split("```")[1].split("```")[0]
    c = c[c.find("{"):] if "{" in c else c
    return json.loads(c)


def parse_arr(raw):
    c = raw.strip()
    if "</think>" in c: c = c.split("</think>")[-1]
    if "```json" in c: c = c.split("```json")[1].split("```")[0]
    elif "```" in c: c = c.split("```")[1].split("```")[0]
    i, j = c.find("["), c.rfind("]")
    return json.loads(c[i:j+1]) if i >= 0 and j > i else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = args.model; repo, fname = MODELS[key]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, "logs", f"{key}.log"), "w")

    papers = json.load(open(TS))
    if args.limit: papers = papers[:args.limit]

    log(logf, f"MODEL={key}  repo={repo}  file={fname}")
    log(logf, f"GPU before load: {gpu_snapshot()}")
    from llama_cpp import Llama, LlamaGrammar
    t0 = time.time()
    llm = Llama.from_pretrained(repo_id=repo, filename=fname, n_ctx=N_CTX,
                                n_gpu_layers=-1, verbose=True)   # verbose -> CUDA offload log
    grammar = LlamaGrammar.from_string(GRAMMAR)
    log(logf, f"model loaded in {time.time()-t0:.1f}s | n_ctx={N_CTX} | GPU after load: {gpu_snapshot()}")
    md = getattr(llm, "metadata", {}) or {}
    log(logf, f"model metadata name={md.get('general.name','?')} arch={md.get('general.architecture','?')}")

    grounded_out, relate_out = {}, {}
    for i, p in enumerate(papers):
        idx = i if args.limit else i  # test_set order == idx
        text = p["text"]
        # ---- grounded (grammar-constrained object schema; grounding enforced by instruction
        #      + text-presence check at scoring time) ----
        t = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": GROUNDED_PROMPT.replace("{text}", text)}],
                temperature=0.0, max_tokens=1500, grammar=grammar)
            gobj = parse_obj(out["choices"][0]["message"]["content"])
        except Exception as e:
            log(logf, f"  [{idx}] grounded ERROR {type(e).__name__}: {str(e)[:80]}")
            gobj = {"taxa_enriched": [], "taxa_depleted": []}
        ge = gobj.get("taxa_enriched", []); gd = gobj.get("taxa_depleted", [])
        grounded_out[str(idx)] = {"taxa_enriched": ge, "taxa_depleted": gd}
        dt_g = time.time() - t
        # ---- relate ----
        cands, _ev = common.retrieve_candidates(text)
        t = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content":
                    RELATE_PROMPT.replace("{candidates}", json.dumps(cands)).replace("{text}", text)}],
                temperature=0.0, max_tokens=1500, grammar=grammar)
            obj = parse_obj(out["choices"][0]["message"]["content"])
        except Exception as e:
            log(logf, f"  [{idx}] relate ERROR {type(e).__name__}: {str(e)[:80]}")
            obj = {"taxa_enriched": [], "taxa_depleted": []}
        relate_out[str(idx)] = {"taxa_enriched": obj.get("taxa_enriched", []),
                                "taxa_depleted": obj.get("taxa_depleted", [])}
        dt_r = time.time() - t
        log(logf, f"  [{idx}] grounded {len(ge)+len(gd)} taxa ({dt_g:.1f}s) | "
                  f"relate {len(cands)} cand->{len(relate_out[str(idx)]['taxa_enriched'])+len(relate_out[str(idx)]['taxa_depleted'])} kept ({dt_r:.1f}s) "
                  f"| GPU {gpu_snapshot()}")

    json.dump(grounded_out, open(os.path.join(HERE, "cache", f"grounded__{key}.json"), "w"), indent=2)
    json.dump(relate_out, open(os.path.join(HERE, "cache", f"relate__{key}.json"), "w"), indent=2)
    log(logf, f"DONE {key}: wrote cache/grounded__{key}.json + cache/relate__{key}.json")


if __name__ == "__main__":
    main()
