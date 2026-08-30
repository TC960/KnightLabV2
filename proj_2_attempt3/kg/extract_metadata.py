#!/usr/bin/env python3
"""Second extraction pass: study-level metadata, for stratifying contested edges.

Motivation. The graph has 151 taxon-disease pairs where papers disagree on
direction (e.g. Bacteroides in Alzheimer's: 5 papers up, 7 down). Direction alone
cannot say why. The microbiome literature attributes most such disagreement to
study design rather than biology -- cohort geography and diet, body site,
16S vs shotgun sequencing, 16S region, sample size, age, medication. None of that
is in the association extraction, so it cannot currently be tested.

This pass pulls those fields per paper so a contested edge can be split by cohort
and asked: do the "up" papers differ systematically from the "down" papers?

Same model, same harness, same grammar-constrained JSON as the association pass.
Resumable via a JSONL checkpoint.

    python extract_metadata.py --model qwopus3.5-27b-v3 --n-ctx 32768 --resume
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2")
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
sys.path.insert(0, EVAL)

MODELS = {
    "qwopus3.5-27b-v3": ("Jackrong/Qwopus3.5-27B-v3-GGUF", "Qwopus3.5-27B-v3-Q4_K_M.gguf"),
}
MAX_TOKENS, TEMP = 700, 0.0

PROMPT = """<task>
Extract the STUDY DESIGN metadata of the paper below, for a meta-analysis of microbiome
studies. Report only what the paper states. Output exactly one JSON object and nothing else.
</task>

<fields>
- "country": country or countries the cohort was recruited in, as written. "" if not stated.
- "n_cases": number of DISEASE subjects analysed (integer, 0 if not stated).
- "n_controls": number of HEALTHY CONTROL subjects analysed (integer, 0 if not stated).
- "mean_age": approximate mean age of the disease group in years (integer, 0 if not stated).
- "pct_female": percentage female across the cohort (integer 0-100, -1 if not stated).
- "body_site": one of "stool", "oral", "gut biopsy", "blood", "nasal", "skin", "other", "" .
- "sequencing": one of "16S", "shotgun metagenomics", "qPCR", "culture", "other", "".
- "region_16S": the 16S hypervariable region if stated, e.g. "V3-V4". "" otherwise.
- "medication_controlled": true if the paper says it excluded or adjusted for antibiotic
  or other medication use; false otherwise.
- "diet_controlled": true if the paper says it controlled for or recorded diet; false otherwise.
</fields>

<rules>
- Do NOT guess. If the paper does not state a field, use the empty/zero value.
- Counts are the number ANALYSED, not recruited, when both are given.
- Human studies only. If the cohort is animal, set n_cases and n_controls to 0.
</rules>

<examples>
<example>
{{"country":"China","n_cases":45,"n_controls":45,"mean_age":66,"pct_female":48,"body_site":"stool","sequencing":"16S","region_16S":"V3-V4","medication_controlled":true,"diet_controlled":false}}
</example>
<example>
{{"country":"Germany","n_cases":34,"n_controls":30,"mean_age":0,"pct_female":-1,"body_site":"stool","sequencing":"shotgun metagenomics","region_16S":"","medication_controlled":false,"diet_controlled":false}}
</example>
</examples>

<paper>
{text}
</paper>
<output>
"""

GRAMMAR = r'''
root ::= "{" ws
  "\"country\":" ws string "," ws
  "\"n_cases\":" ws int "," ws
  "\"n_controls\":" ws int "," ws
  "\"mean_age\":" ws int "," ws
  "\"pct_female\":" ws snum "," ws
  "\"body_site\":" ws string "," ws
  "\"sequencing\":" ws string "," ws
  "\"region_16S\":" ws string "," ws
  "\"medication_controlled\":" ws bool "," ws
  "\"diet_controlled\":" ws bool ws "}"
string ::= "\"" ([^"\\] | "\\" .)* "\""
int ::= [0-9]+
snum ::= "-"? [0-9]+
bool ::= "true" | "false"
ws ::= [ \t\n]*
'''


def smart_truncate(text, n_ctx):
    for m in ["References\n", "REFERENCES\n", "Bibliography\n", "REFERENCE\n"]:
        i = text.rfind(m)
        if i > 0:
            text = text[:i]
            break
    budget = (n_ctx - MAX_TOKENS - 400) * 4
    return text[:budget] if len(text) > budget else text


def parse(raw):
    c = raw.strip()
    if "</think>" in c:
        c = c.split("</think>")[-1]
    if "```json" in c:
        c = c.split("```json")[1].split("```")[0]
    elif "```" in c:
        c = c.split("```")[1].split("```")[0]
    i = c.find("{")
    return json.loads(c[i:]) if i >= 0 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwopus3.5-27b-v3", choices=list(MODELS))
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "metadata.jsonl"))
    a = ap.parse_args()

    papers = json.load(open(PAPERS))
    if a.limit:
        papers = papers[:a.limit]

    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["title"])
            except Exception:
                pass
        print(f"resume: {len(done)} already done", flush=True)

    repo, fname = MODELS[a.model]
    from llama_cpp import Llama, LlamaGrammar
    llm = Llama.from_pretrained(repo_id=repo, filename=fname, n_ctx=a.n_ctx,
                                n_gpu_layers=-1, verbose=False)
    grammar = LlamaGrammar.from_string(GRAMMAR)
    print(f"model loaded | {len(papers)} papers | n_ctx={a.n_ctx}", flush=True)

    out = open(a.out, "a")
    for i, p in enumerate(papers):
        if p["title"] in done:
            continue
        t0 = time.time()
        try:
            r = llm.create_chat_completion(
                messages=[{"role": "user",
                           "content": PROMPT.format(text=smart_truncate(p["text"], a.n_ctx))}],
                temperature=TEMP, max_tokens=MAX_TOKENS, grammar=grammar)
            md = parse(r["choices"][0]["message"]["content"])
            err = False
        except Exception as e:
            print(f"  [{i}] ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
            md, err = {}, True
        row = {"title": p["title"], "link": p.get("link", ""),
               "disease": p.get("disease", ""), "meta": md,
               "parse_error": err, "seconds": round(time.time() - t0, 1)}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"[{i+1}/{len(papers)}] {p['title'][:52]} -> "
              f"{md.get('country','?')} n={md.get('n_cases',0)}/{md.get('n_controls',0)} "
              f"{md.get('sequencing','?')} ({row['seconds']}s)", flush=True)
    out.close()
    print("DONE_METADATA", flush=True)


if __name__ == "__main__":
    main()
