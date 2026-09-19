# Can effect sizes be extracted from microbiome papers, and can incommensurable statistics be harmonised?

Research note for the KG project. Narrow question: is direction-only edge weighting (paper-count)
a choice we're stuck with, or could a magnitude/effect-size layer be added on top of the existing
~300-paper corpus. Short answer up front: **no, not at the fidelity this graph needs — see Verdict.**
Everything below is the evidence for that.

## 1. What statistics do microbiome case-control papers actually report?

Microbiome differential-abundance analysis is dominated by a handful of methods with very different
outputs, and none of them is a classical standardized effect size:

- **LEfSe (LDA Effect Size)** — by far the most common tool cited in the microbiome case-control
  literature. It runs a Kruskal-Wallis test across classes, a pairwise Wilcoxon test to check
  consistency across subclasses, then builds a **Linear Discriminant Analysis** model and reports,
  per taxon, the **log10-transformed LDA score**: the score is obtained by averaging the difference
  between class means (raw feature values) with the difference between class means projected onto
  the first discriminant axis ([Segata et al. 2011, the original LEfSe paper](https://pubmed.ncbi.nlm.nih.gov/21702898/);
  method described in [the bioBakery LEfSe docs](https://forum.biobakery.org/t/how-to-interpret-lda-effect-size/1794)
  and [Biostars](https://www.biostars.org/p/474121/)). **This is not a standardized effect size in
  the Cohen's-d sense.** It has no known sampling distribution, no closed-form variance, no
  confidence interval, and its scale depends on the raw abundance units and the specific dataset's
  between-class variance — it is explicitly a *ranking* statistic for biomarker discovery within one
  study, not a portable cross-study quantity. The commonly-used significance threshold (LDA score >
  2.0 or > 4.0) is a rule of thumb, not a calibrated cutoff, and [very small per-class sample sizes are
  documented to make both the LEfSe p-value and the LDA score unstable](https://academic.oup.com/bioinformatics/article/40/12/btae707/7908399).
  **Practical consequence for this project: even if we could extract every reported LDA score
  perfectly, we could not compare or pool them across the ~300 papers in our corpus** — a score of
  3.8 in one paper and 3.8 in another do not represent the same magnitude of enrichment unless the
  two studies share abundance normalization, taxonomic depth, and comparable class sizes, none of
  which is guaranteed or even usually stated.

- **DESeq2 / edgeR (log2 fold-change + Wald/LRT p-value)** — the second most common family, borrowed
  from RNA-seq. These *are* on a defined, comparable scale (log2FC is a genuine ratio-of-means
  statistic), but [comparative benchmarking work shows DESeq2, edgeR, LEfSe and metagenomeSeq all
  achieve high sensitivity at the cost of poorly controlled FDR on compositional microbiome
  data](https://pmc.ncbi.nlm.nih.gov/articles/PMC11978113/), and effect estimates are sensitive to
  the normalization/library-size method chosen — which is inconsistently reported.

- **Relative-abundance percentages with a rank-sum or t-test p-value** — very common, especially in
  smaller / less bioinformatics-heavy papers: "genus X was 4.2% ± 1.1% in patients vs 1.8% ± 0.9% in
  controls, p = 0.02." This is the one format that is genuinely convertible to Cohen's d (see §2),
  but only when means and SDs (not just significance) are stated for both groups.

- **Bare significance statements with no magnitude** — a substantial fraction of papers, especially
  when reporting a large taxon table, state only direction and p/q-value ("significantly enriched,
  p < 0.05") with the actual abundance numbers relegated to a supplementary table or a figure axis.
  This is also what our own extraction schema targets, and it is one reason we chose direction-only
  in the first place.

**Independent power evidence.** A 2025 methods paper explicitly modeling microbiome differential-abundance
power ([Investigating statistical power of differential abundance studies, PMC11978113 /
PLOS One 2025](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318820)) using
seven real case-control datasets found that **typical published microbiome studies are underpowered
to detect realistic per-taxon fold changes** — reinforcing that even the "hard" numbers papers do
report (fold-change, LDA) are estimated with wide, usually unreported, uncertainty, which is a second,
independent reason vote-counting-on-direction is more defensible here than it looks at first glance:
the alternative (trusting a single underpowered point estimate per paper) is not obviously better.

## 2. Converting between effect-size metrics — formulas and their assumptions

Standard conversions exist and are implemented in tools like the [Campbell Collaboration effect-size
calculator](https://www.campbellcollaboration.org/calculator/equations), the
[`compute.es` / `escalc` (metafor) R functions](https://search.r-project.org/CRAN/refmans/compute.es/html/p_to_es.html),
and the newer [metaConvert package (127 formulas across 74 input combinations, Gosling et al.,
2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12527507/):

| From → To | Formula (equal-n case) | Key assumption |
|---|---|---|
| Two-sample t-stat → Cohen's d | d = t·√(1/n₁ + 1/n₂) | independent groups, roughly normal, homogeneous variance |
| Cohen's d → Hedges' g | g = d·(1 − 3/(4N−9)) | small-sample bias correction only; same normality assumption as d |
| Cohen's d ↔ correlation r | r = d/√(d²+a), a=4 (equal n) or (n₁+n₂)²/(n₁n₂) (unequal n) | dichotomous grouping variable, continuous roughly-normal outcome |
| Cohen's d ↔ log odds ratio | ln(OR) ≈ d·π/√3 (probit/logit link approximation) | outcome is continuous underlying a normal latent variable; breaks down for skewed/count data (i.e. most abundance data) |
| Cohen's d ↔ AUC / probability of superiority | AUC = Φ(d/√2) | bivariate normality, equal variances — [conversion tables in Ruscio & Mullen 2012](https://scielo.isciii.es/scielo.php?pid=S1889-18612018000100035&script=sci_abstract&tlng=en) |
| p-value + n → Cohen's d | via t: recover t from p and df, then d = t·√(1/n₁+1/n₂) | **requires the exact p-value and the exact test/df used** |

**The p-value route is the one that matters most for a literature-mining project, and it is the
weakest link.** It only works if the paper reports an *exact* p-value from a *known* test
(t-test/Wilcoxon/Kruskal-Wallis all need different back-transformations). In practice biomedical
papers overwhelmingly report **inequality-thresholded p-values** ("p < 0.05", "p < 0.001") rather
than exact values — and an inequality cannot be inverted to a point estimate, only to a (usually
very wide) bound. Converting p < 0.05 with n=20 per arm gives a *minimum* d, not the true d, and the
gap can be enormous for small effect sizes reported as "n.s." or as a bare threshold. This is a
well-known failure mode in meta-analysis methodology (see the Cochrane Handbook's treatment of
[why vote-counting and threshold-based synthesis are discouraged](https://training.cochrane.org/handbook/current/chapter-12)),
and it is compounded here because most of our reported statistics are Wilcoxon/Kruskal-Wallis
(LEfSe's engine), not t-tests — the standard conversion formulas assume parametric tests and are
only approximate for rank-based ones. **Net: p-to-d conversion is usable only for the minority of
observations that (a) report an exact p-value and (b) come from a parametric test — probably under
20% of the "bare significance" observations in this literature, based on the mix in §1.**

## 3. Automated extraction of statistics from biomedical text — state of the art

- **Statcheck** (psychology) — the closest thing to a proven success story, but a poor comparator:
  it works because APA style is extremely rigid (`t(df) = x, p = y`). It achieves
  [96.2–99.9% classification accuracy (consistent/inconsistent), sensitivity 85.3–100%, specificity
  96–100%](https://www.researchgate.net/publication/326119300_The_Validity_of_the_Tool_statcheck_in_Discovering_Statistical_Reporting_Inconsistencies),
  but even in that friendly, single-format environment it only **finds about 60% of the
  null-hypothesis tests in a typical paper** ([LSE Impact
  blog](https://blogs.lse.ac.uk/impactofsocialsciences/2018/02/28/statcheck-a-spellchecker-for-statistics/)).
  Biomedical/microbiome reporting has no equivalent house style — statistics appear in prose, tables,
  and figure legends in a dozen formats per journal.

- **RobotReviewer / Trialstreamer** — the RCT-evidence-synthesis analogue. Trialstreamer
  [continuously extracts PICO elements, sample sizes, and free-text "key findings" from RCT
  abstracts using a mix of ML and rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC7727361/), and
  RobotReviewer automates *risk-of-bias* judgments, not effect-size extraction — telling: even in
  RCTs, the best-established tools stop short of extracting numeric effect sizes from full text and
  instead extract structured metadata around the trial.

- **LLM-based extraction, most directly relevant**: [Automatically Extracting Numerical Results
  from RCTs with LLMs (2024/2025, PMC12448672)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12448672/)
  is the best available accuracy benchmark, because RCT reports are *far more standardized* than
  microbiome papers (structured Results sections, CONSORT-guided reporting) and it still found:
  **GPT-4 exact-match accuracy of 65.5% for binary outcomes and only 48.7% for continuous
  outcomes**, with open-source and biomedical-tuned models (PMC-LLaMA, BioMistral) performing far
  worse or producing zero complete extractions. Failure modes were hallucinated values, wrong
  group/timepoint attribution, and confusing baseline vs. post-intervention numbers — all of which
  would recur, likely worse, in microbiome papers where the "outcome" (a taxon's abundance) is one
  of dozens listed per paper rather than one or two pre-registered primary outcomes. Other 2025 work
  on LLM data extraction for systematic reviews reports similarly mixed numbers: [GPT-4 ~87%
  accuracy / 72% recall in one benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC12892481/), but
  [only 91–94% for simple group-size extraction and 57–71% for event counts](https://pubmed.ncbi.nlm.nih.gov/40107689/)
  in another — accuracy is highly field- and task-specific, and always lower for anything requiring
  arithmetic or cross-referencing multiple numbers.
  Multimodal (image-based) extraction of numbers directly from figures/charts has improved sharply
  with recent frontier models — reported mean absolute errors on chart-digitization benchmarks have
  dropped into low single digits — but published accuracy figures are on clean, single-panel charts,
  not multi-panel LEfSe cladograms or 96-taxon heatmaps, and no biomedical-specific validation of this
  exists yet.

## 4. Do the numbers even live in the text?

No systematic count exists for microbiome papers specifically, but the adjacent RCT literature gives
a proxy, and it is unfavorable: in one annotation study of RCT numerical extraction, **the actual
numbers needed to compute an effect size were found in *tables* far more often than in body text**
(471 of the annotated instances were table-sourced) even though RCTs are the *most* standardized,
text-heavy reporting genre in biomedicine. Microbiome differential-abundance results are structurally
worse for text-mining than RCT primary outcomes, because the standard reporting artifact is a
**LEfSe cladogram or LDA bar chart** (a figure, often with the actual score only readable from an
axis or a caption threshold statement like "LDA > 2.0"), backed by a **supplementary table** listing
every taxon's abundance and q-value — with the body text typically naming only the taxa the authors
consider notable, in prose, without the numbers ("Prevotella was significantly enriched in patients").
This matches the pattern already established for this project's own corpus: only a minority of
sentences carry a taxon + a direction cue, and of those, most do not also carry a magnitude — the
gate our own extraction pipeline uses already reflects this (see root `CLAUDE.md`, recall findings).
**Realistic estimate: magnitude/effect-size numbers are recoverable from body text alone for at most
20–30% of taxon-disease observations; the remainder requires table or figure parsing to even attempt
extraction**, and that is before accounting for extraction accuracy on top.

## 5. Table and figure extraction from scientific PDFs/XML — solved or not?

**Not solved, and PMC XML only rescues part of the problem.**

- **PMC/JATS XML** — where available, this is a genuine win: [Europe PMC has ~3 million full-text
  articles in NISO-JATS XML](https://academic.oup.com/bioinformatics/article/35/18/3533/5305021),
  a format where tables are marked up as structured `<table-wrap>` elements, not images, so a
  correctly-tagged JATS table is close to trivially parseable (tools like
  [`pmcgrab`](https://github.com/rajdeepmondaldotcom/pmcgrab) or `JATSdecoder` do this). **The
  catch is coverage**: this only helps for the subset of our corpus that is (a) open-access PMC and
  (b) has full tables actually tagged in the XML rather than embedded as scanned images or
  supplementary PDFs/Excel files (very common for LEfSe output tables, which routinely ship as
  supplementary spreadsheets *outside* the XML entirely).
- **PDF table extraction (non-PMC papers, or papers where the table is supplementary)** is
  meaningfully worse. A 2025 heterogeneous-document benchmark found accuracy varies enormously by
  tool and document type: rule-based tools like **GROBID lag well behind learning-based approaches**,
  and even the best commercial tool tested (Adobe Extract) only reached **F1 ≈ 0.47** for table
  extraction across a realistic mixed corpus ([Benchmarking Table Extraction from Heterogeneous
  Scientific PDF Documents, 2025](https://arxiv.org/html/2511.16134); consistent with the earlier
  [multi-tool academic-document benchmark](https://link.springer.com/chapter/10.1007/978-3-031-28032-0_31)
  showing GROBID/CERMINE lead on references and metadata but all tools struggle on tables, lists,
  and footers). General multimodal LLMs (Gemini, GPT-4V-class) are starting to close this gap but
  have not been benchmarked on biomedical supplementary tables specifically.
- **Figures (LEfSe cladograms, bar plots)** are strictly harder than tables — extracting a numeric
  LDA score from a bar-chart axis is a chart-digitization problem, not a table-parsing one, and no
  validated tool or benchmark exists for this on microbiome figures specifically (§3).

## 6. Microbiome meta-analyses that did harmonise effect sizes — how, and at what scale?

Two real patterns exist in the literature, and neither is "text-mine effect sizes out of a large
heterogeneous corpus of published papers" — which is the thing this project would need to do:

1. **Reprocess raw data instead of extracting published statistics.** The most influential example,
   [MicrobiomeHD (Duvallet, Gibbons, Gurry, Irizarry & Alm, *Nature Communications* 2017)](https://www.nature.com/articles/s41467-017-01973-8),
   built a cross-disease meta-analysis of **28 case-control 16S studies across 10 diseases** — but
   it did this by **downloading the raw sequencing data for each study and reprocessing all 28
   through one uniform pipeline**, not by extracting or harmonizing the effect sizes each paper
   published. This sidesteps the entire incommensurability problem by never touching the papers'
   own statistics at all. It is the gold-standard approach, and it is exactly the thing we cannot do
   — we have text/PDFs of ~300 papers, not their raw sequencing data, most of which was never
   deposited or is not readily linkable back to the paper.
2. **Standardized-mean-difference meta-analysis, but restricted to a handful of pre-agreed,
   directly-comparable outcomes** — almost always **alpha-diversity indices (Shannon, Chao1)** and
   occasionally **phylum-level relative abundance for the 2–3 phyla every paper reports (Firmicutes,
   Bacteroidetes)**, because those are the only quantities reported with enough consistency (a mean
   and an SD, in the same units, for a comparable grouping) across most papers in a disease area.
   E.g. the [depression gut-microbiota meta-analysis (Nikolova et al., *Translational Psychiatry*
   2023)](https://www.nature.com/articles/s41398-023-02670-5) pooled **44 studies / 2,091 patients
   vs 2,792 controls** but its genus-level findings ("depleted Butyricicoccus... enriched
   Eggerthella...") are explicitly **vote-counted directionally**, not effect-size-pooled — SMD
   pooling was reserved for alpha-diversity and the 2 dominant phyla only. The ankylosing spondylitis
   meta-analysis follows the identical pattern (SMD for Shannon index only). **This is the closest
   published precedent to what we do, and it independently validates direction-only vote-counting
   as the field's actual practice for taxon-level findings, precisely because taxon-level effect
   sizes are not comparable across studies** — the same conclusion this project reached
   independently.

No published microbiome meta-analysis was found that harmonises **taxon-level** effect sizes (LDA
scores, fold-changes) across more than a handful of studies without reprocessing raw data first.

## Verdict

**Effect-size extraction is not worth attempting for this ~300-paper corpus, at least not as a
replacement for direction + paper-count.** The chain of evidence:

- The dominant reported statistic (LEfSe's LDA score) is **not a standardized effect size** and has
  no principled cross-study scale — extracting it perfectly would still not make it poolable (§1).
- p-value-to-effect-size back-conversion, the fallback for papers that report only significance, is
  only valid for exact p-values from known parametric tests, which is a minority case here, and the
  standard conversions materially assume the wrong test family (LEfSe uses Wilcoxon/Kruskal-Wallis,
  not t-tests) (§2).
- The best directly-relevant accuracy benchmark — LLM extraction of RCT numerical results, a far
  more standardized genre than microbiome papers — tops out at **~65% (binary outcomes) and ~49%
  (continuous outcomes)** exact-match with GPT-4-class models (§3), and biomedical-tuned smaller
  models did worse.
- A realistic **20–30% of taxon-level observations** even have a usable magnitude number reachable
  from body text; the rest sit in tables, supplementary spreadsheets, or figures (§4), where
  extraction tooling is markedly worse — even the best benchmarked PDF table tool reaches only
  **F1 ≈ 0.47** on a realistic mixed corpus (§5).
- Compounding these stages (≈30% text-located × ≈50–65% extraction accuracy × further loss from
  format/test-family mismatches in conversion) puts a realistic **usable, correctly-harmonised
  effect size around 10–15% of the ~3,000 observation-level edges** in the current graph — with no
  way to know *which* 10–15% without the same manual verification effort the direction-only
  extraction already required, plus the correctness of a downstream statistical conversion on top.
- The one rigorous precedent for cross-study microbiome effect-size harmonisation
  (MicrobiomeHD) achieved it by **reprocessing raw sequencing data**, not text-mining statistics —
  an option this project doesn't have. The precedent that *does* work from published papers
  (depression/AS-style SMD meta-analyses) **restricts pooling to alpha-diversity and 1-2 phyla and
  vote-counts everything else**, which is functionally the same design choice already made here.

**What it would unlock, if it worked**: Cohen's-d-style pooling would enable real heterogeneity
statistics (I², Q-test), proper random-effects meta-regression against the study-design covariates
this project already collects (country, cohort size, platform, etc. — see the open questions in root
`CLAUDE.md`), and forest plots with confidence intervals instead of vote counts. That is a genuine
capability gap. But given the numbers above, it is not a gap this corpus's raw material (text) can
close — the field's own practice confirms it, by restricting SMD synthesis to the handful of
quantities (diversity indices, dominant phyla) that are actually reported comparably, and
vote-counting everything else. **Recommendation: if magnitude ever becomes a priority, the
cost-effective move is a narrow extension — extract and pool alpha-diversity SMDs for the subset of
papers that report them (a much smaller, higher-fidelity task) — not full taxon-level effect-size
harmonisation across the corpus.**
