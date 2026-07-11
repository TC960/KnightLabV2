#!/usr/bin/env python3
"""
Multi-model microbiome-disease extraction pipeline via Lava Payments proxy.

Dependencies:
    pip install requests python-dotenv


 # TRIED TO DO FOR FREE WITH EXISTING CREDITS FOR    
"""

import base64
import json
import os
import random
import re
import time

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()
LAVA_API_KEY = os.getenv("LAVA_API_KEY")
if not LAVA_API_KEY:
    raise RuntimeError("LAVA_API_KEY not found. Create a .env file with LAVA_API_KEY=your_key")

EXCLUDE_INDICES = {554, 638, 1025, 1946, 508, 979, 1597, 1776}
SAMPLE_SIZE = 4
SEED = 42
MAX_CHARS = 48_000  # ~12k tokens at ~4 chars/token
TIMEOUT = 120
RETRY_MAX = 2
RETRY_BACKOFF = 10
CALL_DELAY = 3

INPUT_FILE = "MAIN_DATA.json"
OUTPUT_FILE = "extraction_results.json"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "Claude Opus 4.6",
        "provider_url": "https://api.anthropic.com/v1/messages",
        "model_string": "claude-opus-4-6",
        "request_format": "anthropic",
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    {
        "name": "GPT-5.2",
        "provider_url": "https://api.openai.com/v1/chat/completions",
        "model_string": "gpt-5.2",
        "request_format": "openai",
        "extra_headers": None,
    },
    {
        "name": "Gemini 3 Pro",
        "provider_url": "https://generativelanguage.googleapis.com/v1beta/chat/completions",
        "model_string": "gemini-3-pro-preview",
        "request_format": "openai",
        "extra_headers": None,
    },
    {
        "name": "Kimi K2.5",
        "provider_url": "https://api.moonshot.cn/v1/chat/completions",
        "model_string": "kimi-k2.5",
        "request_format": "openai",
        "extra_headers": None,
    },
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert medical data extraction specialist with deep knowledge of "
    "microbiome research methodology and statistics. You must output ONLY valid JSON "
    "— no markdown, no explanation, no preamble."
)

USER_PROMPT_TEMPLATE = """You are an expert medical data extraction specialist with deep knowledge of microbiome research methodology and statistics.

Your task: Extract microbiome-disease relationships from research papers with complete accuracy.

## STEP 1: IDENTIFY STUDY TYPE
First, determine what type of study this is:
- Disease characterization (comparing diseased vs healthy)
- Treatment/intervention study (testing drugs, supplements, transplants)
- Observational/longitudinal study
- Other

## STEP 2: EXTRACT DISEASE INFORMATION
- Primary disease or condition being studied
- Related conditions mentioned
- Control groups or comparison conditions

## STEP 3: EXTRACT ALL BACTERIAL CHANGES
For EACH bacterium mentioned with quantitative or qualitative changes:

Extract at ALL taxonomic levels present:
- Phylum level (if mentioned)
- Family level (if mentioned)
- Genus level (if mentioned)
- Species level (if mentioned)

For each bacterium, record:
- Name (exact as written)
- Taxonomic level (phylum/family/genus/species)
- Direction: "increased", "decreased", "unchanged", or "unclear"
- Quantitative data if available (percentages, fold-changes)
- Statistical significance (p-value, confidence level)
- Context (disease vs control, pre vs post treatment, etc.)

## STEP 4: DISTINGUISH CAUSALITY
CRITICAL: Determine if bacterial changes are:
- Associated with DISEASE state (disease vs healthy)
- Result of TREATMENT/INTERVENTION (pre vs post treatment)
- Correlational only
- Unknown/unclear

## STEP 5: VERIFY COMPLETENESS
- Did you extract EVERY bacterium mentioned with changes?
- Did you check all tables, figures, and text?
- Did you note if the paper says "X bacteria and Y others" (indicating incomplete listing)?
- Did you check for contradictions between sections?

## STEP 6: VALIDATE LOGIC
- Do the directions make biological sense?
- Are there any contradictory statements in the paper?
- Is the statistical significance adequate (adjust for multiple comparisons)?
- Were any bacteria mentioned in discussion but not measured in results?

## OUTPUT FORMAT:
Return a JSON object with this structure:

{{
  "study_type": "disease_characterization | treatment_intervention | observational | other",
  "study_design": "brief description",
  "primary_disease": "disease name or null",
  "related_conditions": ["condition1", "condition2"],
  "sample_size": "number or not specified",
  "statistical_methods": "brief description of analysis methods",

  "bacteria_relationships": [
    {{
      "taxon_name": "exact name from paper",
      "taxonomic_level": "phylum | family | genus | species",
      "direction": "increased | decreased | unchanged | unclear",
      "change_context": "disease_vs_control | treatment_effect | temporal | other",
      "quantitative_data": {{
        "disease_group": "percentage or value",
        "control_group": "percentage or value",
        "fold_change": "X-fold or null",
        "p_value": "value or null",
        "statistical_significance": "significant | not_significant | not_reported"
      }},
      "location_in_paper": "Table X | Figure Y | Results section | Discussion",
      "confidence": "high | medium | low",
      "notes": "any important context or caveats"
    }}
  ],

  "extraction_metadata": {{
    "total_bacteria_found": 0,
    "completeness_assessment": "complete | partial | unclear",
    "potential_missing_data": "description if incomplete",
    "contradictions_found": ["list any contradictions"],
    "limitations": ["any extraction limitations"]
  }}
}}

## IMPORTANT RULES:
1. Extract EVERY bacterium mentioned, even if changes are small or not significant
2. If a paper says "10 species changed" but only lists 7, note the missing 3
3. NEVER fabricate data - if direction is unclear, mark as "unclear"
4. Distinguish between disease effects and treatment effects
5. Note if findings didn't reach statistical significance after multiple comparison adjustment
6. Include bacteria mentioned in discussion even if not in main results (note as "discussion_only")
7. If genus-level and species-level data both exist, include both
8. Check for bacteria that DECREASED in one section but are described differently elsewhere

Now, extract from this paper:

Paper text: {text}

Think step-by-step, then provide the complete JSON output. Output ONLY the JSON object, nothing else."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_lava_headers(extra_headers=None):
    """Build headers for the Lava forward proxy."""
    token_obj = json.dumps({"secret_key": LAVA_API_KEY})
    encoded = base64.b64encode(token_obj.encode()).decode()
    headers = {
        "Authorization": f"Bearer {encoded}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers


def call_model(model_config, system_prompt, user_prompt):
    """Call a model through the Lava proxy. Returns raw text response."""
    provider_url = model_config["provider_url"]
    lava_url = f"https://api.lavapayments.com/v1/forward/{provider_url}"
    headers = build_lava_headers(model_config.get("extra_headers"))

    if model_config["request_format"] == "anthropic":
        body = {
            "model": model_config["model_string"],
            "max_tokens": 8192,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    else:  # openai-compatible
        body = {
            "model": model_config["model_string"],
            "max_tokens": 8192,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    last_err = None
    for attempt in range(1, RETRY_MAX + 2):  # 1 initial + RETRY_MAX retries
        try:
            resp = requests.post(lava_url, headers=headers, json=body, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if model_config["request_format"] == "anthropic":
                return data["content"][0]["text"]
            else:
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            last_err = str(e)
            if attempt <= RETRY_MAX:
                print(f"    Attempt {attempt} failed: {last_err}. Retrying in {RETRY_BACKOFF}s...")
                time.sleep(RETRY_BACKOFF)
            else:
                raise RuntimeError(f"All {RETRY_MAX + 1} attempts failed. Last error: {last_err}")


def parse_extraction(raw_text):
    """Parse extraction JSON from model response. Returns dict or None."""
    text = raw_text.strip()
    # Strip markdown fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Models may output reasoning before the JSON — find the first { ... } block
    brace_start = text.find("{")
    if brace_start == -1:
        return None
    # Find matching closing brace by counting
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def is_junk_chunk(chunk):
    """Return True if a chunk should be skipped."""
    text = chunk.strip()
    # Too short
    if len(text) < 30:
        return True
    # 3+ citation-style references
    citation_hits = len(re.findall(r"\[DOI\]|\[PubMed\]|\[PMC", text))
    if citation_hits >= 3:
        return True
    # Starts with DOI pattern
    if re.match(r"10\.\d{4,}/", text):
        return True
    # Author/affiliation junk: 3+ email patterns or repeated institution names
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", text)
    if len(emails) >= 3:
        return True
    return False


def merge_chunks(chunks):
    """Filter junk chunks, merge with smart joining, truncate."""
    clean = [c for c in chunks if not is_junk_chunk(c)]
    if not clean:
        return ""

    parts = [clean[0]]
    for prev, cur in zip(clean, clean[1:]):
        prev_stripped = prev.rstrip()
        if prev_stripped and prev_stripped[-1] in ".!?":
            parts.append(" " + cur)
        else:
            parts.append(". " + cur)

    merged = "".join(parts)
    if len(merged) > MAX_CHARS:
        merged = merged[:MAX_CHARS]
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    with open(INPUT_FILE) as f:
        data = json.load(f)

    keys = list(data.keys())
    valid_keys = [k for i, k in enumerate(keys) if i not in EXCLUDE_INDICES]
    print(f"Total papers: {len(keys)}, valid: {len(valid_keys)}, excluded: {len(keys) - len(valid_keys)}")

    random.seed(SEED)
    sampled_keys = random.sample(valid_keys, SAMPLE_SIZE)
    print(f"Sampled papers (keys): {sampled_keys}")
    for k in sampled_keys:
        print(f"  {k}: {data[k]['name'][:80]}")

    # Prepare texts
    prepared = {}
    for k in sampled_keys:
        paper = data[k]
        chunks = paper["chunks"]
        merged = merge_chunks(chunks)
        prepared[k] = {
            "title": paper["name"],
            "text": merged,
            "orig_chunks": len(chunks),
        }
        print(f"  Paper {k}: {len(chunks)} chunks -> {len(merged)} chars after merge")

    # Run extraction
    results = {}
    for model in MODELS:
        model_name = model["name"]
        results[model_name] = {}
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        for k in sampled_keys:
            info = prepared[k]
            print(f"  [{model_name}] Processing paper {k}: {info['title'][:50]}...")

            user_prompt = USER_PROMPT_TEMPLATE.format(text=info["text"])

            try:
                raw = call_model(model, SYSTEM_PROMPT, user_prompt)
                parsed = parse_extraction(raw)
                if parsed is None:
                    print(f"    WARNING: Could not parse JSON. Storing raw text.")
                    results[model_name][k] = {"error": "JSON parse failed", "raw": raw[:500]}
                else:
                    n_rels = len(parsed.get("bacteria_relationships", []))
                    print(f"    OK: {n_rels} relationships extracted")
                    results[model_name][k] = parsed
            except Exception as e:
                print(f"    FAILED: {e}")
                results[model_name][k] = {"error": str(e)}

            time.sleep(CALL_DELAY)

    # Save results
    output = {
        "papers_sampled": sampled_keys,
        "results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Paper':>8} | {'Model':<20} | Relationships")
    print(f"{'-'*8}-+-{'-'*20}-+-{'-'*15}")
    for k in sampled_keys:
        for model in MODELS:
            mn = model["name"]
            entry = results[mn].get(k, {})
            if "error" in entry:
                count = "ERROR"
            else:
                count = str(len(entry.get("bacteria_relationships", [])))
            print(f"{k:>8} | {mn:<20} | {count}")


if __name__ == "__main__":
    main()
