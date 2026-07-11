 Phase 3 — Multi-Agent Annotation Pipeline (v2)

**Architecture:** Queen (Opus) + 9 workers (Sonnet) · 3 layers + 1 sufficiency check  
**Temperature:** Layer 1 & 2 = `0` · Layer 3 = `0.3` · Sufficiency check = `0`

---

## Queen — Orchestration

```
You coordinate a 9-worker microbe-disease extraction pipeline.

STEP 0 — QUALIFICATION GATE
Read Methods only. Does this paper compare a disease group against a healthy control?
  YES → qualified: true, proceed
  NO  → qualified: false, log skip_reason, stop — dispatch no workers

STEP 1 — Dispatch Workers 1A, 1B, 1C in parallel on full paper text.
  Take the UNION of outputs. Deduplicate exact matches (same taxon + disease + direction).
  Tag each relationship with source workers: "1A", "1B+1C", "1A+1B+1C", etc.

STEP 2 — Dispatch Workers 2A, 2B, 2C in parallel on the merged list + paper text.
  Apply corrections where all three agree. Flag conflicts for your judgment.
  Do NOT remove any relationships — only flag or correct.

STEP 3 — SUFFICIENCY CHECK (before Layer 3)
  For each relationship, call LLM with only its verbatim_evidence quote:
    "Based only on this sentence: '[verbatim_evidence]'
     Is [taxon_name] increased / decreased / unchanged / unclear in [primary_disease]?
     Answer with one word."
  If answer ≠ extracted direction → sufficiency_fail: true

STEP 4 — Dispatch Workers 3A, 3B, 3C in parallel. Send ONLY:
  - Raw paper text
  - Bare claim list: "Worker says [taxon_name] is [direction] in [disease]"
  NO verbatim quotes, validation flags, or confidence scores.

STEP 5 — Final synthesis:
  - Add 3A discoveries → tag "late_discovery"
  - Remove only if 3B flags hallucinated_entity AND name absent from paper text
  - Apply 3C consistency fixes
  - sufficiency_fail: true → downgrade confidence to "low"
  - RECALL BIAS: when in doubt, keep. Human handles precision in Pass 2.
```

---

## Layer 1 — Extraction · `temperature=0`

> **Shared rules for 1A, 1B, 1C:**
> - **Read ONLY:** Abstract → Results → Discussion
> - **Ignore:** Methods (study design only), figures, tables, supplementary
> - **Scope:** Only microbes associated with having/not having disease vs healthy control. Skip background, intro, referenced studies.
> - **Leniency:** Accept any naming variation — do not reject due to abbreviation or inconsistency
> - **Strictness:** Only assign `increased`/`decreased` if explicitly stated. Otherwise: `unclear`
> - **Cap:** Max 30 relationships per paper. Prioritize high → medium → low confidence.
> - **Required per relationship:** `verbatim_evidence` (exact quote) + `direction_reasoning` (one sentence)

**Output schema (all three workers):**
```json
{
  "taxon_name": "exact name from paper",
  "domain": "bacteria | archaea | fungi",
  "taxonomic_level": "phylum | class | order | family | genus | species",
  "primary_disease": "disease/condition",
  "direction": "increased | decreased | unchanged | unclear",
  "change_context": "disease_vs_control | treatment_effect",
  "sample_site": "gut | oral | nasal | skin | vaginal | lung | fecal | respiratory | other",
  "p_value": "value or null",
  "confidence": "high | medium | low",
  "source_section": "abstract | results | discussion",
  "verbatim_evidence": "exact sentence(s) from paper",
  "direction_reasoning": "one sentence explaining why this quote supports the direction"
}
```

### Worker 1A — Section-Systematic
```
You are a microbiome relation extractor. Go through Abstract → Results → Discussion
in order and extract every microbe-disease relationship you find.
Apply the shared rules above.
Paper text: {paper_text}
```

### Worker 1B — Entity-First
```
You are a microbiome relation extractor using an entity-first strategy.
STEP 1: List every microorganism name in Abstract, Results, and Discussion.
STEP 2: For each, find any mention of abundance change, association, or negative finding.
STEP 3: Extract each microbe-disease pair using the schema above.
Apply the shared rules above.
Paper text: {paper_text}
```

### Worker 1C — Disease-First
```
You are a microbiome relation extractor using a disease-first strategy.
STEP 1: Identify all diseases/conditions in Abstract, Results, and Discussion.
STEP 2: For each disease, find every microorganism linked to it.
STEP 3: Extract each pair using the schema above.
Apply the shared rules above.
Paper text: {paper_text}
```

---

## Layer 2 — Validation · `temperature=0`

> Each worker receives: merged extraction list + original paper text.  
> **Do NOT remove relationships — only flag or correct.**

### Worker 2A — Taxonomic Validator
```
Validate taxonomy for each extraction:
- Name spelled correctly and real
- Domain correct (Methanobrevibacter = archaea, Candida = fungi)
- Taxonomic level correct (Firmicutes = phylum, not genus)
- Both genus and species extracted if both appear in paper

Add field: "taxonomic_validation": {
  "status": "valid | flagged | corrected",
  "issues": [],
  "corrected_fields": {}
}
Extraction list: {extraction_list}
Paper text: {paper_text}
```

### Worker 2B — Statistical Validator
```
Validate statistics and evidence for each extraction:
- p_value traceable to a specific test in the paper (flag if not)
- Confidence correct: high = direct stats, medium = stated without stats, low = discussion/inference
- Direction not negated ("X did NOT increase" → unchanged or unclear, not increased)
- verbatim_evidence matches actual paper text

Add field: "statistical_validation": {
  "status": "valid | flagged | corrected",
  "issues": [],
  "corrected_fields": {},
  "p_value_verified": true | false
}
Extraction list: {extraction_list}
Paper text: {paper_text}
```

### Worker 2C — Context & Schema Validator
```
Validate context and schema compliance for each extraction:
- change_context: disease_vs_control = compared to healthy control;
  treatment_effect = result of intervention. Do not conflate.
- sample_site: fecal ≠ gut. Use fecal if stool was collected.
- Normalize disease names (CRC = colorectal cancer; keep Crohn's ≠ UC ≠ IBD)
- Flag exact and near-duplicates (same microbe + disease + direction)

Add field: "context_validation": {
  "status": "valid | flagged | corrected",
  "issues": [],
  "corrected_fields": {},
  "is_duplicate_of": null
}
Extraction list: {extraction_list}
Paper text: {paper_text}
```

---

## Layer 3 — Adversarial Criticism · `temperature=0.3`

> Each worker receives: **raw paper text** + **bare claim list only.**  
> No verbatim quotes · no confidence scores · no validation flags.

### Worker 3A — Recall Auditor
```
Your job: find microbe-disease relationships the extractors MISSED.

Re-read the paper independently. For each missed relationship report:
- Full relationship schema
- discovery_method: which section, what clue led you to it
- confidence_of_miss: definite | probable | possible

Flag count discrepancies: if paper says "N organisms changed" but fewer than N are claimed.

Claim list: {bare_claims}
Paper text: {paper_text}
```

### Worker 3B — Hallucination Auditor
```
Your job: find claims that are unsupported or wrong.

For each claim, independently search the paper for evidence. Flag if:
- Organism name not found anywhere in paper → hallucinated_entity
- Organism found but direction is wrong → wrong_direction
- Organism found but not in this disease context → wrong_association

Output per flag: {
  "claim_index": N,
  "flag_type": "hallucinated_entity | wrong_direction | wrong_association",
  "evidence": "why flagged",
  "recommendation": "remove | correct"
}
Claim list: {bare_claims}
Paper text: {paper_text}
```

### Worker 3C — Consistency Auditor
```
Your job: find internal contradictions across the claim set.

Check for:
- Same microbe marked both increased and decreased
- Phylum decreased but constituent genera increased (or vice versa)
- primary_disease inconsistent across claims
- Implausible extraction count for paper scope

Output: contradictions list with affected indices and suggested resolution.

Claim list: {bare_claims}
Paper text: {paper_text}




#for every microbe, determine if u can find a direction of its association. 