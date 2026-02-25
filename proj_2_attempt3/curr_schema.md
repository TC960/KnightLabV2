# Current Schema — Microbe-Disease Relationships (Minimal)

Simplified extraction schema for Phase 3 benchmarking. Captures the core relationship: which microbes change in which diseases.

Full expanded schema (metabolites, immune markers, pathways, etc.) saved in `schema_full.md` for future phases.

---

## Entity Types

### Microbe

| Field | Type | Values |
|-------|------|--------|
| `taxon_name` | string | Exact name from paper |
| `domain` | enum | `bacteria`, `archaea`, `fungi` |
| `taxonomic_level` | enum | `phylum`, `class`, `order`, `family`, `genus`, `species` |

### Disease

| Field | Type | Values |
|-------|------|--------|
| `primary_disease` | string | Main disease or condition studied |

---

## Relation: microbe_disease_association

A microbe's abundance is reported as changed (or unchanged) in the context of a disease or treatment.

| Field | Type | Values |
|-------|------|--------|
| `direction` | enum | `increased`, `decreased`, `unchanged`, `unclear` |
| `change_context` | enum | `disease_vs_control`, `treatment_effect` |
| `sample_site` | enum | `gut`, `oral`, `nasal`, `skin`, `vaginal`, `lung`, `fecal`, `respiratory`, `other` |
| `p_value` | string or null | Reported p-value, e.g. `"0.003"` |
| `confidence` | enum | `high` (direct statistical evidence), `medium` (stated but no stats), `low` (inferred or discussion only) |

---

## Extraction Rules

- Only extract relationships explicitly stated or directly supported by data
- Do not infer relationships not present in the text
- If both phylum and genus are reported for the same organism, extract as separate entries
- If a microbe is mentioned without directional change, skip it
- One entry per unique microbe-disease-direction combination

---

## Paper-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | string | Unique paper identifier |
| `paper_title` | string | Full title of the paper |
| `study_type` | string | e.g. `case_control`, `treatment_intervention`, `cohort` |
| `primary_disease` | string | Main disease or condition studied |
| `sample_sites` | array of strings | All sample sites in the study |

---

## Per-Relationship Annotation Metadata (`_annotation_metadata`)

Stored inside each relationship object. Not part of the extracted content — used for tracking annotation provenance.

| Field | Type | Description |
|-------|------|-------------|
| `source_agents` | array of strings | Which agents produced this annotation (e.g. `["combined"]`) |
| `agreement_level` | enum | `verified`, `partial`, `disputed` |
| `validation_flags` | array of strings | e.g. `["complex_taxon_name"]`, empty if none |
| `criticism_flags` | array of strings | Issues flagged during review, empty if none |
| `review_notes` | string | Annotator reasoning and context |
| `queen_decision` | enum | `keep`, `remove`, `needs_review` |
| `verbatim_evidence` | string | Exact quote from paper supporting the relationship |

---

## Example Output

```json
{
  "paper_id": "1",
  "paper_title": "...",
  "study_type": "case_control",
  "primary_disease": "oral squamous cell carcinoma",
  "sample_sites": ["oral"],
  "relationships": [
    {
      "id": "1_rel_001",
      "taxon_name": "Fusobacterium nucleatum",
      "domain": "bacteria",
      "taxonomic_level": "species",
      "primary_disease": "oral squamous cell carcinoma",
      "direction": "increased",
      "change_context": "disease_vs_control",
      "sample_site": "oral",
      "p_value": "<0.05",
      "confidence": "high",
      "_annotation_metadata": {
        "source_agents": ["combined"],
        "agreement_level": "verified",
        "validation_flags": [],
        "criticism_flags": [],
        "review_notes": "Abstract explicitly states significant enrichment",
        "queen_decision": "keep",
        "verbatim_evidence": "..."
      }
    }
  ]
}
```