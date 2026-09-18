#!/usr/bin/env python3
"""Remote job: run all three prompt variants over the same 80 papers.

Runs ON the GPU box. One instance, one provisioning cost, three variants, so the
comparison is paired by construction -- every variant sees identical input in
identical conditions.

Writes, per variant, a checkpoint JSONL after EVERY paper so the watchdog's
60-second pull always has complete partial results. A crash at paper 60 loses one
paper, not the run.

Signals its state in ./STATUS, which the watchdog reads:
    RUNNING <variant> <n>/<total>
    DONE
    FAILED <reason>

    python run_prompt_exp.py --model-path <gguf> --subset prompt_exp_subset.json
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(HERE, "STATUS")
# MUST match what ~/.brev-watchdog/watchdog.sh pulls down:
#     scp -r "$INST:$REMOTE_DIR/extract_out" "$DEST/"
# The first run of this experiment wrote to exp_out/, the watchdog pulled
# extract_out/, and when the job reported DONE the watchdog deleted the instance
# with the results still on it. Three variants, 1h48m and ~$1.08 lost. If this
# directory name changes, change the watchdog too.
OUTDIR = os.path.join(HERE, "extract_out")

# GBNF: every rule MUST be on ONE line. LlamaGrammar.from_string() does NOT
# validate, so a multi-line `root` reports success and then SEGFAULTS during
# sampling (rc=139, no traceback). This has cost this project a run before.
GRAMMAR = r'''root ::= "{" ws "\"disease\"" ws ":" ws string ws "," ws "\"taxa_enriched\"" ws ":" ws arr ws "," ws "\"taxa_depleted\"" ws ":" ws arr ws "}"
arr ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" ["\\/bfnrt]
ws ::= [ \t\n]*'''


def mark(s):
    with open(STATUS, "w") as f:
        f.write(s + "\n")


def smart_truncate(text, budget):
    """Cut at References, then hard-cap. The cap is load-bearing: without it 12
    of 250 papers overflowed the context window on an earlier run."""
    for marker in ("\nReferences\n", "\nREFERENCES\n", "\nBibliography\n"):
        i = text.rfind(marker)
        if i > len(text) * 0.4:
            text = text[:i]
            break
    return text[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--subset", default=os.path.join(HERE, "prompt_exp_subset.json"))
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--char-budget", type=int, default=90000)
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    try:
        sys.path.insert(0, HERE)
        from prompt_variants import variant
        from llama_cpp import Llama, LlamaGrammar

        papers = json.load(open(a.subset))
        mark(f"RUNNING load 0/{len(papers)}")

        grammar = LlamaGrammar.from_string(GRAMMAR)
        llm = Llama(model_path=a.model_path, n_ctx=a.n_ctx, n_gpu_layers=-1,
                    verbose=False)

        for vid in a.variants.split(","):
            vid = vid.strip()
            tmpl = variant(vid)
            out = os.path.join(OUTDIR, f"variant_{vid}.checkpoint.jsonl")
            done = set()
            if os.path.exists(out):                      # resume-safe
                for line in open(out):
                    try:
                        done.add(json.loads(line)["title"])
                    except Exception:
                        pass
            print(f"=== variant {vid}: {len(papers)-len(done)} to do ===", flush=True)

            for i, p in enumerate(papers, 1):
                if p["title"] in done:
                    continue
                mark(f"RUNNING {vid} {i}/{len(papers)}")
                t0 = time.time()
                rec = {"variant": vid, "title": p["title"], "doi": p.get("doi", ""),
                       "disease_gold": p.get("disease", "")}
                try:
                    text = smart_truncate(p["text"], a.char_budget)
                    r = llm.create_chat_completion(
                        messages=[{"role": "user",
                                   "content": tmpl.format(text=text)}],
                        temperature=0, max_tokens=8192, grammar=grammar)
                    raw = r["choices"][0]["message"]["content"]
                    j = json.loads(raw)
                    rec.update({
                        "predicted_disease": j.get("disease", ""),
                        "predicted_enriched": "; ".join(j.get("taxa_enriched", [])),
                        "predicted_depleted": "; ".join(j.get("taxa_depleted", [])),
                        "parse_error": False,
                    })
                except Exception as e:
                    rec.update({"predicted_enriched": "", "predicted_depleted": "",
                                "parse_error": True, "error": f"{type(e).__name__}: {e}"})
                rec["time_seconds"] = round(time.time() - t0, 2)
                with open(out, "a") as f:                # append, then flush --
                    f.write(json.dumps(rec) + "\n")      # the watchdog pulls every 60s
                    f.flush()
                    os.fsync(f.fileno())
                if i % 10 == 0:
                    print(f"  {vid} {i}/{len(papers)}  {rec['time_seconds']}s",
                          flush=True)

        mark("DONE")
        print("ALL VARIANTS COMPLETE", flush=True)
    except Exception:
        traceback.print_exc()
        mark("FAILED " + traceback.format_exc().splitlines()[-1][:120])
        sys.exit(1)


if __name__ == "__main__":
    main()
