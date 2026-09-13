#!/usr/bin/env python3
"""Pre-flight for the new-gold GPU run. Everything here runs on the laptop, so the
only untested step left is the GPU itself.

    python dryrun_gold_run.py            # exits non-zero if any check fails

Checks, in the order they would bite:
  1. GBNF grammar: every rule on ONE line. LlamaGrammar.from_string() does not
     validate -- a multi-line `root` rule parses "successfully" and then segfaults
     during sampling (rc=139, no traceback). This is a text check precisely
     because the library's own check is useless.
  2. extract_input_gold.json: shape, required keys, no empty text, no duplicates.
  3. smart_truncate at the chosen n_ctx: how many papers actually get cut.
  4. The prompt template formats for every paper (a stray brace would KeyError on
     paper 1 of 265, 15 minutes into a paid instance).
  5. parse_output() accepts what the grammar can emit.
  6. Checkpoint resume: write, re-read, confirm the skip set matches.
  7. Watchdog scripts: present, executable, and the two copies still agree.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval_gold as R                                   # noqa: E402

N_CTX = 32768              # what the plan launches with
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


print("1. GBNF grammar")
rules = [ln for ln in R.GRAMMAR_STR.strip().splitlines() if ln.strip()]
names = [ln.split("::=")[0].strip() for ln in rules if "::=" in ln]
check("every non-blank grammar line defines exactly one rule",
      all("::=" in ln for ln in rules), f"{len(rules)} lines, {len(names)} rules: {names}")
check("no rule body is continued onto the next line",
      len(names) == len(rules))
check("root rule is a single line",
      sum(1 for ln in rules if ln.strip().startswith("root ")) == 1)
check("root is defined", "root" in names)
# a rule body split across lines shows up as a line with no '::=' — belt and braces
check("no orphan continuation lines", not [ln for ln in rules if "::=" not in ln],
      "the failure mode is silent: from_string() succeeds, sampling segfaults rc=139")

print("\n2. extract_input_gold.json")
path = R.DATASETS["goldset"]
check("dataset path resolves", os.path.exists(path), path)
papers = json.load(open(path))
check("is a list of records", isinstance(papers, list) and bool(papers), f"{len(papers)} papers")
need = {"title", "text", "disease", "taxa_enriched", "taxa_depleted", "link"}
missing = [p["title"][:40] for p in papers if not need <= set(p)]
check("every record has the keys run_eval reads", not missing, str(missing[:3]))
check("no empty text", not [p for p in papers if not (p.get("text") or "").strip()])
titles = [p["title"] for p in papers]
check("titles unique (resume + merge both key on title)",
      len(set(titles)) == len(titles), f"{len(titles) - len(set(titles))} dupes")
check("every paper has at least one gold taxon",
      not [p for p in papers if not (p["taxa_enriched"] or p["taxa_depleted"]).strip()])

print(f"\n3. smart_truncate at n_ctx={N_CTX}")
budget = (N_CTX - R.MAX_TOKENS - 400) * 4
cut = [(p["title"][:50], len(p["text"]), len(R.smart_truncate(p["text"], n_ctx=N_CTX)))
       for p in papers]
truncated = [c for c in cut if c[2] >= budget]
refs_stripped = sum(1 for c, p in zip(cut, papers) if c[2] < len(p["text"]))
print(f"  char budget {budget:,} | reference list stripped on {refs_stripped}/{len(papers)}")
check("fewer than 5% of papers hit the hard cap",
      len(truncated) / len(papers) < 0.05,
      f"{len(truncated)} truncated" + (f": {truncated[0][0]}" if truncated else ""))

print("\n4. prompt formatting")
bad = []
for p in papers:
    try:
        s = R.PROMPT_TEMPLATE.format(text=R.smart_truncate(p["text"], n_ctx=N_CTX))
        if "{text}" in s:
            bad.append(p["title"][:40])
    except Exception as e:
        bad.append(f"{p['title'][:40]}: {type(e).__name__}")
check("PROMPT_TEMPLATE.format succeeds for all 265", not bad, str(bad[:3]))

print("\n5. parse_output on grammar-shaped output")
ok = True
for raw in ['{"disease": "Parkinson\'s disease", "taxa_enriched": ["Akkermansia"], "taxa_depleted": []}',
            '{"disease":"MS","taxa_enriched":[],"taxa_depleted":["Faecalibacterium prausnitzii"]}',
            '```json\n{"disease":"AD","taxa_enriched":["Bacteroides"],"taxa_depleted":[]}\n```']:
    try:
        d = R.parse_output(raw)
        ok &= set(d) == {"disease", "taxa_enriched", "taxa_depleted"}
    except Exception:
        ok = False
check("parse_output round-trips the grammar's shapes", ok)

print("\n6. checkpoint resume")
with tempfile.TemporaryDirectory() as d:
    ck = os.path.join(d, "t.checkpoint.jsonl")
    with open(ck, "w") as f:
        for p in papers[:3]:
            f.write(json.dumps({"title": p["title"], "predicted_enriched": ""}) + "\n")
    done = {}
    for line in open(ck):
        done[json.loads(line)["title"]] = 1
    todo = [p for p in papers if p["title"] not in done]
check("resume skips exactly the checkpointed papers", len(todo) == len(papers) - 3,
      f"{len(todo)} would run")

print("\n7. watchdog")
w1 = os.path.join(HERE, "gpu_watchdog.sh")
w2 = os.path.expanduser("~/.brev-watchdog/watchdog.sh")
check("kg/gpu_watchdog.sh exists and is executable",
      os.path.exists(w1) and os.access(w1, os.X_OK))
check("~/.brev-watchdog/watchdog.sh exists and is executable",
      os.path.exists(w2) and os.access(w2, os.X_OK))
check("the two copies are identical",
      os.path.exists(w1) and os.path.exists(w2)
      and open(w1).read() == open(w2).read())
body = open(w1).read() if os.path.exists(w1) else ""
for cond, label in [(r"hard deadline", "deadline kill"),
                    (r"DONE\*\)", "DONE marker kill"),
                    (r"FAILED\*\)", "FAILED marker kill"),
                    (r"unreachable for 20", "unreachable kill"),
                    (r"brev delete", "delete (not stop) fallback")]:
    check(f"watchdog implements: {label}", bool(re.search(cond, body)))
check("watchdog pulls results every tick, not only at the end",
      body.count("scp -o ConnectTimeout") >= 2)
dest = re.search(r'DEST="([^"]+)"', body)
check("watchdog DEST directory exists",
      bool(dest) and os.path.isdir(os.path.expandvars(dest.group(1))),
      dest.group(1) if dest else "")

print("\n" + "=" * 64)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("all pre-flight checks passed — the GPU is the only untested step")
