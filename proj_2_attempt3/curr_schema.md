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

## Example Output

```json
{
  "paper_title": "paper title here",
  "relationships": [
    {
      "taxon_name": "Fusobacterium nucleatum",
      "domain": "bacteria",
      "taxonomic_level": "species",
      "primary_disease": "oral squamous cell carcinoma",
      "direction": "increased",
      "change_context": "disease_vs_control",
      "sample_site": "oral",
      "p_value": "0.003",
      "confidence": "high"
    }
  ]
}
```