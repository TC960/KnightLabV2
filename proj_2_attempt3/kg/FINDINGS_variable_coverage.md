# FINDINGS: Variable Coverage Survey

Survey of **303 papers** (250 from `all_usable_papers.json` + 53 from `new_papers.json`) for extractable study variables that might explain disagreement on microbiome signatures.

## Coverage Table: Which Papers Mention Each Variable?

| Rank | Variable | % Papers | Already Covered? | Example Sentence |
|------|----------|----------|------------------|------------------|
| 1 | recruitment setting | 96.4% | ❌ NEW | 1515/tnsci-2020-0117 Search in PMC Search in PubMed View in NLM Catalog Add to search Dysbiosis characteristics of gut microbiota in cerebral infarction patients Hao Li Hao Li Department of General Surgery, Shanghai Tenth People's Hospital, Tongji... |
| 2 | 16s region | 80.2% | metadata.jsonl: region_16S | Methods Stool samples were collected from 79 CI patients and 98 healthy controls and subjected to 16S rRNA sequencing to identify stool microbes To do this, we analyzed stool samples from 79 CI patients and 98 HCs by 16S rRNA sequencing |
| 3 | diet | 77.2% | metadata.jsonl: diet_controlled | This discrepancy might be due to sample size difference, different regional dietary habits of included patients, or the differences of BPB at the genus level This discrepancy may be due to their classification of different disease statuses or diff... |
| 4 | sample storage | 76.6% | ❌ NEW | On arrival to the laboratory, all samples were immediately aliquoted and stored in a −80℃ freezer DNA extraction was performed within 15 days of sample collection, and the extracted DNA was stored at −20℃ |
| 5 | differential abundance method | 71.6% | ❌ NEW | The data obtained by sequencing were optimized for operational taxonomic units (OTUs) clustering and used for different analysis, including the alpha and beta diversity analyses, the principal coordinates analysis (PCoA), and the linear discrimina... |
| 6 | sequencing platform | 69.0% | metadata.jsonl + extractions_corrected.json: sequencing | Pyrosequencing and bioinformatics analysis Sequencing and bioinformatics analysis were carried out as previously described [ 15 ] The purified products were pyrosequenced using the Miseq system (Illumina, San Diego, California) |
| 7 | antibiotic use | 65.7% | ❌ NEW | Alternatively, researchers demonstrated that gut microbiota plays a role in the outcome of ischemic stroke in an antibiotic-induced mouse model of intestinal dysbiosis [ 5 ] Initial exclusion criteria included cancer, infection, history of intesti... |
| 8 | bmi | 63.0% | ❌ NEW | Common risk factors of CI include hypertension, diabetes, smoking, and obesity [ 3 ] There was no significant difference observed between these two groups regarding proportions of gender, smoking, comorbidities, average age, or body mass index (BMI) |
| 9 | probiotic use | 63.0% | ❌ NEW | Initial exclusion criteria included cancer, infection, history of intestinal disease, or exposure to antibiotics or probiotics within 1 month before sample collection Isolation of cholesterol-lowering lactic acid bacteria from human intestine for ... |
| 10 | bioinformatics pipeline | 62.7% | ❌ NEW | The data obtained by sequencing were optimized for operational taxonomic units (OTUs) clustering and used for different analysis, including the alpha and beta diversity analyses, the principal coordinates analysis (PCoA), and the linear discrimina... |
| 11 | disease severity | 51.5% | ❌ NEW | Conversely, the abundances of LAB in CI patients were increased, and LAB were positively correlated with the disease severity Probiotic VSL#3 reduces liver disease severity and hospitalization in patients with cirrhosis: A randomized, controlled t... |
| 12 | sex gender balance | 49.2% | metadata.jsonl: pct_female | To exclude confounding bias, the authors matched the two groups for well-known confounders, including age, gender, education status, CD4 count, and sexual preference, that underwent further analysis Results PHES reference ranges A total of 200 hea... |
| 13 | multiple testing correction | 41.3% | ❌ NEW | The Bonferroni method was used to correct the false discovery rate (FDR) of the microbiota correlation analysis The P -values were adjusted to control the false discovery rate (FDR) |
| 14 | dna extraction kit | 40.3% | ❌ NEW | DNA extraction DNA extractions were performed using QIAamp DNA Stool Mini Kit (Qiagen, USA) according to the manufacturer’s instructions Total genomic DNA was isolated using DNA Extraction Kit (Qiagen, Düsseldorf, Germany) as per the instructions ... |
| 15 | patient age | 39.3% | metadata.jsonl: mean_age | The study included hospitalized patients with depression in the IS group who met the following criteria: (i) aged 18 years or older; (ii) previously diagnosed with depression according to International Classification of Diseases, 10th version (ICD... |
| 16 | medication | 39.3% | metadata.jsonl: medication_controlled | Dysbiosis Microbial Communities Microbiome Microbiota Microbial Genetics Parkinson's disease Gut Microbiota Influence on Parkinson's Disease Mechanisms Background Parkinson’s disease (PD) is a multisystem and progressive neurodegenerative disease ... |
| 17 | disease duration | 22.1% | ❌ NEW | There was a significant negative correlation between the Bifidobacterium with the duration of illness ( P = 0 Demographic and clinical data were collected including age, gender, residency (rural/urban), occupation, age of onset, duration of illnes... |

## Coverage Ranking

Highest to lowest % of papers mentioning the variable:

1. **recruitment setting**: 96.4% (292/303)
2. **16s region**: 80.2% (243/303)
3. **diet**: 77.2% (234/303)
4. **sample storage**: 76.6% (232/303)
5. **differential abundance method**: 71.6% (217/303)
6. **sequencing platform**: 69.0% (209/303)
7. **antibiotic use**: 65.7% (199/303)
8. **bmi**: 63.0% (191/303)
9. **probiotic use**: 63.0% (191/303)
10. **bioinformatics pipeline**: 62.7% (190/303)
11. **disease severity**: 51.5% (156/303)
12. **sex gender balance**: 49.2% (149/303)
13. **multiple testing correction**: 41.3% (125/303)
14. **dna extraction kit**: 40.3% (122/303)
15. **patient age**: 39.3% (119/303)
16. **medication**: 39.3% (119/303)
17. **disease duration**: 22.1% (67/303)

## Top 6 NEW Variables to Extract

Ranked by coverage × estimated relevance to microbiome disagreement:

**1. recruitment setting** (96.4%, 292 papers)
Recruitment source (hospital vs community) affects microbiota composition; hospital patients often have comorbidities and prior treatments that alter baseline dysbiosis.

**2. sample storage** (76.6%, 232 papers)
Storage conditions (temperature, time-to-freezing, -80C vs -20C) dramatically affect bacterial viability; a known major confounder in metagenomic studies.

**3. differential abundance method** (71.6%, 217 papers)
Statistical test choice (LEfSe vs DESeq2 vs Wilcoxon) changes which taxa are called significant; method harmonization affects reproducibility.

**4. antibiotic use** (65.7%, 199 papers)
Antibiotics directly reshape the microbiome; critical inclusion/exclusion criterion or confounding variable that explains much disagreement.

**5. bmi** (63.0%, 191 papers)
Obesity/BMI is an independent microbiome determinant; uncontrolled BMI differences between case/control groups bias taxa comparisons.

**6. probiotic use** (63.0%, 191 papers)
Probiotic history during the study or washout period affects baseline microbiota; critical to track for case/control matching.

## Interpretation

**Already well-covered in metadata.jsonl:**
- patient_age (mean_age)
- sex/gender balance (pct_female)
- diet control (diet_controlled)
- medication control (medication_controlled)
- 16S hypervariable region (region_16S)
- sequencing platform (sequencing)

**High-coverage NEW variables worth extracting:**
The top 6 new variables are mentioned in >60% of papers and have known mechanistic effects on microbiome composition. Extracting these would improve case–control matching and explain some of the observed disagreement in the literature.

**Medium-coverage candidates:**
- disease_severity (51.5%), sex/gender balance details (49.2%), multiple_testing_correction (41.3%)
- These are in 40–50% of papers; worth extracting if simpler patterns succeed first.

**Low-coverage (skip for now):**
- patient_age raw values (39.3%) — metadata.jsonl already has mean_age
- disease_duration (22.1%) — too sparse to be reliable

---

# AUDIT (2026-08-31) — the percentages above are keyword-hit rates, not coverage

Everything above this line is the original survey, kept as written. It does **not**
support its own conclusions, and the ranking it produces is wrong. Reproduce with
`audit_variable_coverage.py`.

## The tell is in the survey's own examples

| survey claim | the example it cites | what that actually is |
|---|---|---|
| recruitment setting, **96.4%** | *"Department of General Surgery, Shanghai Tenth People's Hospital, Tongji…"* | an **author affiliation** |
| disease severity, **51.5%** | *"Probiotic VSL#3 reduces liver disease severity and hospitalization…"* | a **reference title** |
| antibiotic use, **65.7%** | *"…an antibiotic-induced **mouse** model of intestinal dysbiosis"* | an animal study, cited as background |
| medication, **39.3%** | *"Dysbiosis Microbial Communities Microbiome Microbiota Microbial Genetics Parkinson's disease…"* | keyword soup; **no medication mentioned at all** |

The affiliation case is the exact error already documented for `country` in
`FINDINGS_task0_rescore.md`: recording an author's institution instead of the
cohort's site. Matching "Hospital" anywhere in a paper finds the authors, not the
recruitment setting.

The survey also shipped **no script**, so none of its 17 numbers were reproducible.

## Re-measured at four tiers

- **NAIVE** — keyword anywhere (what the survey reported)
- **BODY** — outside front-matter and the reference list
- **ATTRIBUTED** — in a sentence carrying an own-study cue (*we / this study /
  were recruited / inclusion criteria*) and **not** an animal-model cue
- **VALUED** — a concrete value adjacent (a number, a named kit/test/scale).
  This is the only tier an extractor can work from: a mention is not a datum.

303 papers, same corpus as the survey.

| variable | NAIVE | BODY | ATTRIB | **VALUED** | drop |
|---|---:|---:|---:|---:|---:|
| recruitment setting | 296 (97.7%) | 269 | 202 | **105 (34.7%)** | −63.0 pt |
| differential abundance method | 219 (72.3%) | 217 | 116 | **98 (32.3%)** | −40.0 pt |
| disease severity | 229 (75.6%) | 228 | 144 | **78 (25.7%)** | −49.9 pt |
| sample storage | 209 (69.0%) | 208 | 82 | **53 (17.5%)** | −51.5 pt |
| antibiotic use | 205 (67.7%) | 205 | 84 | **47 (15.5%)** | −52.2 pt |
| bmi | 146 (48.2%) | 146 | 72 | **22 (7.3%)** | −40.9 pt |
| dna extraction kit | 184 (60.7%) | 184 | 66 | **13 (4.3%)** | −56.4 pt |
| probiotic use | 122 (40.3%) | 122 | 34 | **10 (3.3%)** | −37.0 pt |

**Bounds, stated honestly.** The VALUED patterns are strict and will miss real
statements (phrasings the regex does not anticipate), so VALUED is a **lower
bound** and NAIVE an **upper bound**; true coverage sits between. The *ordering*
and the *size of the gap* are the findings, not the exact percentages.

## What changes

1. **The recommended top-6 is wrong.** `bmi` (7.3%) and `probiotic use` (3.3%)
   were ranked #5 and #6 on 63% each. They are the two **worst** variables in the
   set. `dna extraction kit` was told to skip at 40.3%; it is 4.3% — the advice was
   right by accident.
2. **#3 is not new.** The correct datasheet already carries
   `Differential Abundance Test` hand-curated at **306/337 (91%)** — LEfSe (LDA) 120,
   Other 117, Linear/MaAsLin 23, DESeq 13, ANCOM 7. Better than any extractor would
   do, and free. Do not extract it; **join to it**.
3. **Only two candidates survive**, and both need a caveat:
   - **recruitment setting (34.7%)** — genuinely new and the best of the set, but
     one third of papers is thin for a variable meant to explain 226 contested
     edges. Must be extracted as *where subjects came from*, never by matching an
     institution name.
   - **disease severity (25.7%)** — real, and the values are named scales with
     scores (UPDRS, H&Y, EDSS, MMSE, MoCA, NIHSS, ALSFRS). But the scales are
     disease-specific, so it is **not one variable** — it cannot be pooled across
     43 diseases, only used within a disease.

## Power, which is the actual blocker

`analyze_contested.py` already tested 26 study-design categories at n=250 against
contested-edge direction: nothing survived BH correction, best FDR 0.243. Adding a
variable present in a third of papers gives a **smaller** effective n than the
variables that already failed. Extracting `recruitment setting` at 34.7% coverage
buys ~105 papers of signal against 226 contested edges averaging ~4 papers per side.

That does not mean do not extract it — it means **do not expect it to explain the
contested edges**, and pre-register that expectation before spending a GPU pass.

