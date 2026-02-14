# Microbiome-Disease Relation Extraction Schema

> Revised from `prompt_advanced.txt` based on analysis of 60 papers by 4 independent workers.

---

## Entity Types

### 1. Microbe (renamed from "bacteria" to include all domains)

Captures any microorganism with reported changes or associations.

| Field | Type | Description |
|-------|------|-------------|
| `taxon_name` | string | Exact name from paper |
| `domain` | enum | `bacteria` \| `archaea` \| `fungi` |
| `taxonomic_level` | enum | `phylum` \| `class` \| `order` \| `family` \| `genus` \| `species` |
| `direction` | enum | `increased` \| `decreased` \| `unchanged` \| `unclear` |
| `change_context` | enum | `disease_vs_control` \| `treatment_effect` \| `temporal` \| `dose_response` \| `other` |
| `sample_site` | enum | `gut` \| `oral` \| `oral_subsite_[name]` \| `nasal` \| `skin` \| `vaginal` \| `lung` \| `fecal` \| `uterine` \| `respiratory` \| `other` |
| `temporal_context` | string | `cross_sectional` \| `baseline` \| `followup_[timepoint]` \| `day_[N]` \| `month_[N]` \| `year_[N]` |
| `quantitative_data` | object | `{disease_group, control_group, fold_change, p_value, statistical_significance}` |
| `location_in_paper` | string | Table X \| Figure Y \| Results section \| Discussion |
| `confidence` | enum | `high` \| `medium` \| `low` |
| `notes` | string | Important context or caveats |

**Examples from papers:**
- Bacteria: *Fusobacterium nucleatum* increased in oral squamous cell carcinoma (gut, genus)
- Archaea: *Sulfophobococcus* found as core organism in PLS patient saliva (Paper 1280)
- Fungi: *Ascomycota* (84%) dominant in indoor dust from asthma homes (Paper 387)

### 2. Disease/Condition

| Field | Type | Description |
|-------|------|-------------|
| `primary_disease` | string | Main disease or condition studied |
| `related_conditions` | string[] | Other conditions mentioned |
| `disease_site` | string | Body site affected (if different from sample site) |

### 3. Metabolite

Captures metabolites with reported changes or mechanistic roles.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Specific metabolite or class name |
| `metabolite_class` | enum | `SCFA` \| `bile_acid` \| `phenolic` \| `amino_acid` \| `lipid` \| `organic_acid` \| `other` |
| `direction` | enum | `increased` \| `decreased` \| `unchanged` |
| `produced_by` | string[] | Bacteria/microbes linked to production |
| `associated_outcome` | string | Disease or clinical measure affected |
| `sample_type` | enum | `plasma` \| `urine` \| `fecal` \| `tissue` \| `saliva` \| `other` |
| `quantitative_data` | object | `{value, p_value, significance}` |
| `notes` | string | Context |

**Examples from papers:**
- SCFAs decreased in IBD patients, produced by *Faecalibacterium* (Paper 123)
- 4-ethylphenyl sulfate (4EPS) produced by gut bacteria, crosses blood-brain barrier (Paper 1726)
- Nitrite reduced 80% by chlorhexidine killing oral bacteria (Paper 370)
- Oxalate metabolism altered by gut bacteria in kidney stone disease (Paper 1744)

### 4. Immune Marker

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Marker name (e.g., IL-6, TNF-α) |
| `marker_type` | enum | `cytokine` \| `antimicrobial_peptide` \| `immune_cell` \| `immune_protein` \| `other` |
| `direction` | enum | `increased` \| `decreased` \| `unchanged` |
| `associated_microbes` | string[] | Microbes linked to this marker |
| `sample_site` | string | Where measured |
| `quantitative_data` | object | `{value, p_value}` |

**Examples from papers:**
- IL-6, TNF-α, IL-1β elevated alongside dysbiosis (Papers 1326, 1605)
- Intraepithelial lymphocytes as primary outcome in IBS (Paper 1241)
- Surfactant protein D correlates with lung microbiome diversity (Paper 1061)

### 5. Virus (when applicable)

| Field | Type | Description |
|-------|------|-------------|
| `virus_name` | string | Name (e.g., HPV-16, SARS-CoV-2) |
| `associated_microbes` | string[] | Microbes with reported associations |
| `relationship` | enum | `virus_correlates_with_microbe` \| `virus_modifies_microbiome` \| `microbe_affects_viral_outcome` |
| `quantitative_data` | object | `{p_value, effect_size}` |
| `notes` | string | Context |

**Examples from papers:**
- HPV-16 associated with *Gardnerella*, reduced *Lactobacillus* (Paper 306)
- SARS-CoV-2 viral load correlates with bacterial abundance (Paper 1250)
- HPV persistence linked to *Lactobacillus* deficiency; relationship reverses in HIV+ women (Paper 1625)

---

## Relation Types

### 1. microbe_disease_association (original, expanded)
Microbe abundance changed in disease vs control. Now includes `sample_site` and `temporal_context`.

### 2. intervention_affects_microbe (NEW)
Treatment/intervention causes microbiome changes. Distinct from disease association.
- Includes: FMT, probiotics, antibiotics, diet, drugs, surgery
- Example: Broad-spectrum antibiotics reduce diversity; narrow-spectrum preserves it (Paper 1055)

### 3. microbe_produces_metabolite (NEW)
Microbe linked to metabolite production or metabolism.
- Example: Gut bacteria convert dietary tyrosine → 4-ethylphenol → 4EPS (Paper 1726)

### 4. metabolite_affects_outcome (NEW)
Metabolite mediates clinical/physiological outcome.
- Example: Nitrite reduction → blood pressure increase (Paper 370)

### 5. microbe_microbe_interaction (NEW)
Co-occurrence, antagonism, or synergy between microbes.
- Example: *S. aureus* positively associated with *Acinetobacter*, negatively with *Dolosigranulum pigrum* (Paper 43)
- Example: Bacterial-fungal antagonism via antibiotic resistance genes (Paper 1966)

### 6. host_factor_modifies_relationship (NEW)
Host characteristic modifies a microbe-disease relationship.
- Example: *Lactobacillus*-HPV relationship reverses in HIV+ vs HIV- women (Paper 1625)
- Example: FMT effective in female but not male Huntington's mice (Paper 565)

### 7. diversity_outcome_association (NEW)
Community-level diversity metric predicts or associates with clinical outcome.
- Example: Decreased lung microbiome diversity predicts low FVC and early mortality (Paper 1061)

---

## Directionality Conventions

- **microbe → disease**: Microbe abundance is the observation, disease is the context
- **intervention → microbe**: Intervention is cause, microbe change is effect
- **microbe → metabolite → outcome**: Multi-step chain, direction follows causal flow
- **microbe ↔ microbe**: Bidirectional co-occurrence; use `interaction_type` to specify

---

## JSON Output Format

```json
{
  "study_type": "disease_characterization | treatment_intervention | observational | longitudinal | other",
  "study_design": "brief description",
  "primary_disease": "disease name or null",
  "related_conditions": ["condition1", "condition2"],
  "sample_size": "number or not_specified",
  "statistical_methods": "brief description",

  "sample_sites_analyzed": ["gut", "oral"],
  "multi_site_study": true,

  "host_factors": {
    "age_group": "infant | child | adult | elderly | specific_range",
    "sex": "male | female | mixed",
    "diet": "Western | Mediterranean | specific | not_reported",
    "bmi": "value or category or not_reported",
    "comorbidities": ["HIV", "obesity"],
    "behavioral": ["smoking", "alcohol"],
    "geographic": "urban | rural | specific | not_reported"
  },

  "intervention": {
    "type": "FMT | probiotic | antibiotic | diet | drug | surgery | none | other",
    "agent_name": "specific agent or null",
    "dosing": "dose info or null",
    "duration": "duration or null"
  },

  "microbe_relationships": [
    {
      "taxon_name": "exact name from paper",
      "domain": "bacteria | archaea | fungi",
      "taxonomic_level": "phylum | class | order | family | genus | species",
      "direction": "increased | decreased | unchanged | unclear",
      "change_context": "disease_vs_control | treatment_effect | temporal | dose_response | other",
      "sample_site": "gut | oral | nasal | skin | vaginal | lung | fecal | other",
      "temporal_context": "cross_sectional | baseline | followup_[timepoint]",
      "quantitative_data": {
        "disease_group": "percentage or value",
        "control_group": "percentage or value",
        "fold_change": "X-fold or null",
        "p_value": "value or null",
        "statistical_significance": "significant | not_significant | not_reported"
      },
      "location_in_paper": "Table X | Figure Y | Results section | Discussion",
      "confidence": "high | medium | low",
      "notes": "any important context"
    }
  ],

  "diversity_metrics": [
    {
      "metric_type": "alpha_diversity | beta_diversity | shannon | chao1 | observed_species | FB_ratio | other",
      "value": "numeric or null",
      "direction": "increased | decreased | no_change",
      "comparison": "disease_vs_control | pre_vs_post | site_A_vs_site_B",
      "sample_site": "where measured",
      "p_value": "value or null",
      "associated_outcome": "clinical measure or disease state if applicable",
      "notes": ""
    }
  ],

  "metabolites": [
    {
      "name": "metabolite name",
      "metabolite_class": "SCFA | bile_acid | phenolic | amino_acid | lipid | organic_acid | other",
      "direction": "increased | decreased | unchanged",
      "produced_by": ["bacteria names if linked"],
      "associated_outcome": "disease or clinical measure",
      "sample_type": "plasma | urine | fecal | tissue | saliva | other",
      "quantitative_data": {
        "value": "",
        "p_value": "",
        "significance": "significant | not_significant | not_reported"
      },
      "notes": ""
    }
  ],

  "immune_markers": [
    {
      "name": "marker name",
      "marker_type": "cytokine | antimicrobial_peptide | immune_cell | immune_protein | other",
      "direction": "increased | decreased | unchanged",
      "associated_microbes": ["microbe names if linked"],
      "sample_site": "",
      "quantitative_data": {
        "value": "",
        "p_value": ""
      }
    }
  ],

  "virus_associations": [
    {
      "virus_name": "HPV-16",
      "associated_microbes": ["Gardnerella_vaginalis"],
      "relationship": "virus_correlates_with_microbe | virus_modifies_microbiome | microbe_affects_viral_outcome",
      "quantitative_data": {
        "p_value": ""
      },
      "notes": ""
    }
  ],

  "microbe_interactions": [
    {
      "microbe_1": "name",
      "microbe_2": "name",
      "interaction_type": "co_occurrence | antagonism | competition | synergy",
      "strength": "strong | moderate | weak",
      "evidence": "correlation | experimental | network_analysis",
      "quantitative_data": {
        "correlation": "",
        "p_value": ""
      }
    }
  ],

  "mechanistic_pathways": [
    {
      "description": "brief pathway summary",
      "steps": [
        {"entity_type": "microbe | metabolite | immune_marker | outcome", "name": "X", "action": "produces | activates | inhibits | causes | converts_to"}
      ],
      "confidence": "high | medium | low",
      "evidence_location": "Results | Figure X"
    }
  ],

  "predictive_model": {
    "present": false,
    "model_type": "random_forest | logistic_regression | neural_network | other | null",
    "auc": "value or null",
    "prediction_target": "disease | treatment_response | prognosis | null",
    "key_features": ["top predictive bacteria/metabolites"]
  },

  "extraction_metadata": {
    "total_microbes_found": "number",
    "completeness_assessment": "complete | partial | unclear",
    "potential_missing_data": "description if incomplete",
    "contradictions_found": ["list any contradictions"],
    "limitations": ["any extraction limitations"]
  }
}
```

---

## Changes vs Original `prompt_advanced.txt`

| Change | Rationale | Worker Evidence |
|--------|-----------|-----------------|
| `bacteria_relationships` → `microbe_relationships` with `domain` field | Archaea (Paper 1280) and fungi (Paper 387, 1966) are distinct from bacteria but need same extraction structure | W1, W2, W3 |
| Added `sample_site` to microbe relationships | 40-73% of papers across all workers used non-gut/oral sites; multi-site studies lose comparative structure without it | W1, W2, W3, W4 |
| Added `temporal_context` field | Longitudinal studies (Papers 599, 1119, 1888, 356) need temporal dimension | W1, W2, W3, W4 |
| Added `metabolites` array | Found in 19+ papers across all workers; SCFAs, bile acids, phenolics are central findings, not side notes | W1, W2, W3, W4 |
| Added `diversity_metrics` as top-level (promoted from metadata) | Diversity is a PRIMARY outcome in many papers (1061, 1163, 1041, 1412), not just procedural metadata | W1, W2, W3 |
| Added `immune_markers` array | Cytokines (IL-6, TNF-α) and immune cells are key mechanistic intermediaries in 2-4 papers per worker | W1, W2, W3, W4 |
| Added `virus_associations` array | HPV-bacteria and SARS-CoV-2-bacteria interactions found in multiple papers (306, 608, 1250, 1625) | W2, W4 |
| Added `microbe_interactions` array | Bacteria-bacteria co-occurrence/antagonism (Papers 43, 298, 1655, 1966) lost with individual-microbe-only schema | W2, W3, W4 |
| Added `mechanistic_pathways` array | Multi-step causal chains (bacteria→metabolite→disease) fragmented by current schema; 3+ papers per worker | W1, W2, W3, W4 |
| Added `host_factors` section | Sex, age, diet, BMI, comorbidities modify microbe-disease relationships; found across all workers | W1, W2, W3, W4 |
| Added `intervention` section | Current `change_context` too vague; need to distinguish FMT/probiotic/antibiotic/diet effects from disease associations | W1, W3, W4 |
| Added `predictive_model` section | ML-based prognostic studies (AUC 0.72-0.918) are distinct from associative findings | W1, W2, W4 (3/4 vote) |
