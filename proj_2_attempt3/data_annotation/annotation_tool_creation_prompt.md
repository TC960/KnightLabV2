# Build a Paper Annotation Tool

## What it does

A single-page web app (standalone HTML file) that helps me annotate microbiome research papers. I load a paper's JSON, read the text, and fill in structured microbe-disease relationships.

## Layout (3 panels)

### Left Panel — Paper Text
- Displays the `paper_text` field from the loaded JSON (or concatenated `chunks` if that's the structure)
- Scrollable, readable, with decent typography
- Taxon names auto-highlighted in blue (match against common suffixes: -aceae, -ales, -ota, -ella, -ium, -us, -coccus, -bacillus, etc., plus italicized Latin binomials)
- Ctrl+F should work within this panel

### Middle Panel — Annotation Form
- A form to add microbe-disease relationships one at a time
- Fields (all from the schema below):
  - `taxon_name` — text input (free text)
  - `domain` — dropdown: bacteria, archaea, fungi
  - `taxonomic_level` — dropdown: phylum, class, order, family, genus, species
  - `primary_disease` — text input (auto-fills from first entry, editable)
  - `direction` — dropdown: increased, decreased, unchanged, unclear
  - `change_context` — dropdown: disease_vs_control, treatment_effect, subgroup_comparison
  - `sample_site` — dropdown: gut, oral, nasal, skin, vaginal, lung, fecal, respiratory, other
  - `source_section` — dropdown: results, discussion, abstract
  - `p_value` — text input (nullable)
  - `confidence` — dropdown: high, medium, low (with tooltip showing definitions)
  - `notes` — textarea
- "Add Relationship" button appends to the list
- After adding, form clears but retains `primary_disease`, `sample_site`, and `change_context` (since these are usually the same across a paper)

### Right Panel — Added Relationships
- Shows all relationships added so far as compact cards
- Each card shows: taxon_name, direction arrow (↑↓→?), taxonomic_level, confidence badge, p_value
- Edit button on each card (loads back into form)
- Delete button on each card
- Running count: "7 relationships added"

## Top Bar
- "Load Paper JSON" button — file picker, loads the JSON and populates the left panel
- Paper title displayed after loading
- Paper ID displayed
- "Download Annotations JSON" button — exports the completed annotation

## Bottom Bar
- Paper-level notes textarea
- Extraction completeness section:
  - `total_reported` — number input ("Paper reports N significant species")
  - `total_extracted` — auto-calculated from relationships list
  - `gap_reason` — text input

## Export JSON Format

```json
{
  "paper_id": "from loaded json",
  "paper_title": "from loaded json",
  "annotator": "stored in localStorage, prompted on first use",
  "date": "auto-filled",
  "time_spent_minutes": "auto-tracked from load to download",
  "relationships": [
    {
      "taxon_name": "Fusobacterium nucleatum",
      "domain": "bacteria",
      "taxonomic_level": "species",
      "primary_disease": "Crohn's disease",
      "direction": "increased",
      "change_context": "disease_vs_control",
      "sample_site": "oral",
      "source_section": "results",
      "p_value": "0.003",
      "confidence": "high",
      "notes": ""
    }
  ],
  "extraction_completeness": {
    "status": "complete | partial",
    "total_reported": 32,
    "total_extracted": 14,
    "gap_reason": "18 species in Figure 3 only"
  },
  "paper_notes": ""
}
```

## Confidence Tooltip Definitions
- **high**: Per-relationship p-value or FDR explicitly reported (e.g., "p = 0.003", "FDR < 0.05")
- **medium**: Direction stated in text but no per-relationship p-value, OR species listed as significant with global FDR threshold
- **low**: Finding from Discussion only, inferred from context, or blends own findings with literature citations

## Technical Requirements
- Single HTML file (inline CSS + JS, no build step)
- Works by opening the file directly in Chrome (file:// protocol)
- Clean, minimal design — no frameworks needed, just good CSS
- localStorage for auto-save (save state every time a relationship is added, restore on page load)
- The paper JSON files will have varying structures, so handle at minimum:
  - `{ "paper_id": "...", "paper_title": "...", "paper_text": "full text string" }`
  - `{ "paper_id": "...", "paper_title": "...", "chunks": ["chunk1", "chunk2", ...] }`
  - `{ "paper_id": "...", "title": "...", "text": "..." }`

## What I do NOT need
- No server, no database, no deployment
- No pre-annotation or LLM extraction
- No Label Studio integration
- No user auth