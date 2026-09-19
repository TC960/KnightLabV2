# Prior work on detecting, representing, and explaining contradictions in scientific literature

**Scope note.** This is a literature survey commissioned to inform how we represent the 220
CONTESTED edges in the microbe–disease KG (edges where different papers assert opposite
directions for the same taxon–disease pair). It does not re-derive or re-check our own numbers
(84% of papers reporting both directions somewhere; 15.8% contested rate in the human gold vs.
11.7% in our extraction) — those are given facts from the calling task. The question here is
purely: what has the field already built for detecting, representing, and explaining
contradictions in biomedical/scientific text, and which of those frameworks fits our case.

---

## 1. Contradiction detection in biomedical text

**Alamri & Stevenson, "A corpus of potentially contradictory research claims from
cardiovascular research abstracts,"** *Journal of Biomedical Semantics* 7:36 (2016).
[Springer](https://link.springer.com/article/10.1186/s13326-016-0083-z) /
[PDF](https://eprints.whiterose.ac.uk/id/eprint/100099/8/WRRO_100099.pdf). They built a corpus
of 259 MEDLINE abstracts drawn from 24 systematic reviews spanning four cardiovascular topics,
annotated for whether *pairs* of abstracts contain claims that contradict one another (vs. agree,
vs. are simply about different things). Inter-annotator agreement was high. This is the direct
precedent for treating "does paper A's claim conflict with paper B's claim" as an annotatable,
reliable NLP task rather than noise to be averaged away — the corpus (**BioContradiction**, hosted
by Stevenson's group,
[staffwww.dcs.shef.ac.uk/.../bio_contradictions](https://staffwww.dcs.shef.ac.uk/people/M.Stevenson/resources/bio_contradictions/))
became the standard benchmark for downstream contradiction-classification work, including
Alamri's PhD thesis, *The Detection of Contradictory Claims in Biomedical Abstracts*
([full text](https://etheses.whiterose.ac.uk/id/eprint/15893/1/FinalThesis-Alamri.pdf)), and later
systems such as the Springer chapter **"Automated Contradiction Detection in Biomedical
Literature"** ([2018](https://link.springer.com/chapter/10.1007/978-3-319-96136-1_12)), which
trains classifiers (SVM / feature-based, later transformer-based follow-ups) directly on this
corpus to flag abstract pairs as contradictory vs. not. A recent 2026 arXiv entry,
**"HealthContradict: Evaluating Biomedical Knowledge Conflicts in Language Models"**
([arxiv.org/pdf/2512.02299](https://arxiv.org/pdf/2512.02299)), extends this line to ask whether
LLMs *themselves* hold consistent beliefs when biomedical claims conflict — worth a follow-up read
if we ever want to characterize whether our extractor's own confidence correlates with which side
of a contested pair it lands on.

Two structural points carry over directly to our problem: (1) contradiction is defined
*claim-pair-wise*, not paper-wise — a paper can contain both a contradicting and a corroborating
claim relative to different other papers, exactly like our per-edge contest structure; (2) the
Alamri/Stevenson annotation scheme distinguishes contradiction from mere *difference in scope*
(different population, different outcome measure) — a distinction we don't yet make explicit in
our contested-edge representation (see §7 proposal).

**SemRep / SemMedDB and conflicting predications.** SemRep is NLM's rule-based system that parses
biomedical sentences into subject–predicate–object triples; **SemMedDB**
([Kilicoglu et al., *Bioinformatics* 28(23):3158, 2012](https://academic.oup.com/bioinformatics/article/28/23/3158/195282))
is the resulting PubMed-scale repository of these predications, now numbering in the tens of
millions. Two features matter here:

- **Factuality tagging.** Each predication in SemMedDB carries one of seven factuality values —
  FACT, PROBABLE, POSSIBLE, DOUBTFUL, COUNTERFACT, UNCOMMITTED, CONDITIONAL — assigned from
  local negation/hedging cues in the source sentence
  ([PMC7122011, "Patterns and Trends in Semantic Predications"](https://pmc.ncbi.nlm.nih.gov/articles/PMC7122011/)).
  This is the closest existing analogue to a per-observation confidence/hedge tag sitting
  *inside* a triple store, rather than being collapsed before storage — directly relevant to
  whether our contested-edge schema should carry a per-paper certainty/hedge field (our extractor
  already gates on "reported statistical significance," so in principle every surviving edge is
  FACT-level by construction, but the *degree* of significance and any hedging language in the
  source sentence is currently discarded).
- **Conflict detection at scale.** Work built on SemMedDB (surveyed in the "Patterns and Trends"
  review above, and in the "Towards medical knowmetrics" paper,
  [*Scientometrics* 2021](https://link.springer.com/article/10.1007/s11192-021-03880-8)) treats a
  pair of predications sharing subject and object but asserting *opposite* predicates
  (`increases` vs. `decreases`, `treats` vs. `causes` in incompatible directions) as a candidate
  contradiction. One cited figure: **462,188 conflicting triple pairs** detected in SemMedDB this
  way. The method is essentially the SemMedDB analogue of our own "same taxon–disease pair,
  opposite direction" contest rule — i.e., the field's default operationalization of "contradiction
  in a structured KG" is exactly the rule we already use. What SemMedDB work adds that we don't yet
  have is *burstness/temporal* analysis — tracking whether a conflict resolves over time as more
  papers accumulate, versus persists — which is a natural extension for us given we have
  per-edge paper-provenance and (in principle) publication years.

**Negation/uncertainty handling upstream of contradiction detection** (UMLS-adjacent, for
completeness): the classic tools are **NegEx** (Chapman et al. 2001) and the **ConText** algorithm,
which tag clinical/biomedical sentence spans as negated, hypothetical, or historical before
relation extraction; SemRep's factuality tagger (above) is the abstract-literature descendant of
this lineage. We don't currently need a negation detector (our prompt already filters at
extraction time), but it's the standard citation if we ever audit whether the extractor is
correctly distinguishing "not associated" from "no data reported."

---

## 2. How large biomedical KGs handle conflicting assertions

**SemMedDB** (above) is itself the largest example, and its answer is: **keep every predication,
tag factuality, do not resolve.** It performs no truth reconciliation — contradictions are a
downstream analysis to be run *on* the database, not a state the database refuses to represent.
This matches our "keep both sides, never average" policy.

**Hetionet** ([Himmelstein et al., *eLife* 2017](https://github.com/hetio/hetionet); ~47K nodes,
2.25M edges across 24 edge types integrated from 29 source databases) does not carry an explicit
contradiction/negative-edge type in its base schema — it integrates *positive* assertions
(compound treats disease, gene associates with disease, etc.) from curated sources and is largely
silent about disagreement between those sources at the individual-edge level, i.e. conflicts get
resolved implicitly by whichever source's curation "wins" a given predicate slot rather than being
preserved as a queryable disagreement. The more direct precedent is a natural-product-drug-interaction
KG case study
([arxiv.org/pdf/2209.11950](https://arxiv.org/pdf/2209.11950)) which explicitly reports finding
*both* `inhibits` and `positively_regulates` edges for the same compound–protein pair pulled from
different literature sources, and on manual review attributes this **directly to genuine
contradictions in the underlying literature** rather than extraction error — i.e. an independent
domain replicate of our own finding that disagreement is a property of the literature, not an
artifact of the pipeline. It's a useful citation for the "this isn't just us" argument but doesn't
offer a representational solution beyond noting the conflict exists.

On the **representation-theoretic** side (rather than case studies), there's an active line on
**uncertain knowledge graph embedding (UKGE)** and its extensions
([survey: "Uncertainty Management in the Construction of Knowledge Graphs,"
arxiv.org/html/2405.16929v2](https://arxiv.org/html/2405.16929v2)): a UKG stores triples
`(s, p, o, confidence)` with confidence ∈ [0,1] rather than boolean truth, and models like **UKGE**
and **BEUrRE** (box embeddings with probabilistic semantics) learn embeddings that respect those
confidences. More relevantly, **"Improving Knowledge Graph Embeddings through Contrastive Learning
with Negative Statements"** ([arxiv.org/pdf/2510.11868](https://arxiv.org/pdf/2510.11868)) argues
that *explicitly declared negative statements* (X is NOT associated with Y) are almost always
discarded by KG embedding pipelines even when a curated source states them, and shows that feeding
them in as an explicit negative signal (rather than via the closed-world assumption) changes
downstream link prediction. This is the most direct technical parallel to our own contest
structure: a contested edge in our graph is functionally a pair of a **positive** statement
(enriched, paper A) and what is, relative to paper A, a **contradicting** statement (depleted,
paper B) — not merely an absent edge — and generic KG tooling defaults to throwing that structure
away unless it's modeled on purpose.

---

## 3. Claim verification / scientific fact-checking datasets

This is the most directly transferable body of work, because the SUPPORTS / REFUTES / NOT-ENOUGH-INFO
(NEI) label set is structurally identical to "this paper's edge agrees with / contradicts /
is irrelevant to a given taxon–disease direction claim."

- **SciFact** ([Wadden et al., EMNLP 2020, "Fact or Fiction: Verifying Scientific Claims,"
  aclanthology.org/2020.emnlp-main.609](https://aclanthology.org/2020.emnlp-main.609/)): 1,409
  expert-written atomic scientific claims derived from real citation sentences, each paired
  against a corpus of 5,183 abstracts, labeled SUPPORTS / REFUTES / NOINFO with gold *rationale
  sentences*. Critically, to guarantee the dataset actually contains contradictory evidence, the
  authors **manually negated a subset of claims** to convert SUPPORTS pairs into REFUTES pairs —
  i.e. even the field's flagship dataset needed a synthetic step to get enough real disagreement,
  which is a data point in favor of treating our organically-occurring 220 contested edges as a
  valuable, hard-to-manufacture resource rather than a nuisance.
- **MultiVerS** (formerly **LongChecker**;
  [Wadden et al., Findings of ACL 2022, arxiv.org/pdf/2112.01640](https://arxiv.org/pdf/2112.01640);
  [code](https://github.com/dwadden/multivers)) is the current strongest published model on
  SciFact: a Longformer-based model that jointly does evidence-sentence selection and label
  classification over full abstracts (not just extracted sentences), trained with weak supervision
  from citation contexts. Reported SciFact test F1 (label + rationale, the standard combined
  metric): **MultiVerS 0.7248** vs. **VeriSci (original SciFact baseline) 0.47** vs.
  **ARSJoint 0.7123** (numbers per the community leaderboard/comparison tables surfaced via search;
  treat as approximate pending a direct leaderboard check). MultiVerS was trained and released with
  checkpoints for **three** verification datasets — SciFact, **CovidFact**, and **HealthVer** — which
  makes it the natural off-the-shelf model to prototype against if we ever wanted to auto-classify
  a new paper's stance on an existing contested edge.
- **HealthVer** ([Sarrouti et al., Findings of EMNLP 2021,
  aclanthology.org/2021.findings-emnlp.297](https://aclanthology.org/2021.findings-emnlp.297/)):
  14,330 (claim, evidence) pairs built from real-world search-engine health claims about COVID-19,
  verified against scientific abstracts — notable for using "in the wild" claims rather than
  expert-authored ones, closer to how a contested edge in our graph would actually be phrased by a
  non-expert consumer of the KG.
- **COVID-Fact** ([Saakyan et al., ACL 2021,
  arxiv.org/abs/2106.03794](https://arxiv.org/abs/2106.03794)): 4,086 FEVER-style claims about
  COVID-19 with evidence and **automatically generated counter-claims** (rather than
  human-written), showing the field also treats *contradiction generation* (not just detection) as
  tractable to automate — relevant if we ever want to auto-surface "the opposite of this edge,
  stated as a claim" for a KG browsing UI.
- **Citation-Integrity** ([Naik et al./ScienceNLP-Lab, *Bioinformatics* 40(7):btae420, 2024,
  academic.oup.com/bioinformatics/article/40/7/btae420](https://academic.oup.com/bioinformatics/article/40/7/btae420);
  [code](https://github.com/ScienceNLP-Lab/Citation-Integrity)): 100 highly-cited biomedical papers
  and 3,063 citing instances, labeled ACCURATE / NOT_ACCURATE / IRRELEVANT for whether a citing
  paper's claim actually matches what the cited paper says — **39.18%** of citation instances
  carried an accuracy error. Best system (BM25+MonoT5 retrieval, fine-tuned MultiVerS classifier):
  **0.59 micro-F1 / 0.52 macro-F1**; GPT-4 in-context learning: **0.65 micro-F1 / 0.45 macro-F1**,
  better on accurate citations but worse on catching erroneous ones. This is the closest existing
  work to "does paper B's stated direction for taxon X actually reflect what paper B's own data
  show," as opposed to "do papers A and B disagree with each other" — i.e. it targets a
  *misreporting* failure mode we have not separately audited (a paper could misstate its own
  finding's direction, independent of whether the underlying biology is genuinely inconsistent
  across cohorts).

**Overall accuracy ceiling to note**: across this whole line, best-in-class F1/micro-F1 on
three-way SUPPORTS/REFUTES/NEI-style labeling tops out around **0.6–0.75** even with
purpose-built long-document models — i.e. the task the calling context might eventually want
("automatically classify whether a new paper's edge is consistent with the existing contested
pair") is not a solved problem, and any such automation on our KG should be validated, not assumed.

---

## 4. Irreproducibility as a signal, not noise

- **Youyou, Yang & Uzzi**, and related work on **text-based ML prediction of psychology
  replication** ([PNAS, cross-validated on 14,000+ articles across six psychology subfields], and
  the follow-up critique **"The limitations of machine learning models for predicting scientific
  replicability,"** *PNAS* 2023
  ([pnas.org/doi/10.1073/pnas.2307596120](https://www.pnas.org/doi/10.1073/pnas.2307596120)):
  the original models predict binary replication success from *narrative text alone* at
  **~70% cross-validated accuracy (AUC 0.77)**, and effect-size direction/magnitude at
  Spearman ρ ≈ 0.38, with no access to the actual statistics — i.e. hedging language, framing, and
  discourse structure in a paper's own text carry real signal about whether its finding will hold
  up. The 2023 critique paper is important to cite alongside it: it argues these models are
  overfit to a very small labeled set (<500 manual replications existed at the time) and don't
  generalize as well as first claimed. Net takeaway for us: **the idea that a paper's own
  language predicts reliability is real but fragile at current data sizes** — encouraging as
  motivation, not as a ready-made tool we could point at our 271 papers (nowhere near enough
  labeled replications to train on).
- **DARPA SCORE** ("Systematizing Confidence in Open Research and Evidence," 2019–2022): generated
  machine-derived and human-forecast **confidence scores** for ~3,000 social/behavioral-science
  claims (2009–2018, 60 journals), validated against a ~5% actually-replicated subsample. Two
  companion papers matter: **"Predicting replicability — analysis of survey and prediction market
  data from large-scale forecasting projects"** (*PLOS ONE* 2021,
  [journals.plos.org/plosone/article?id=10.1371/journal.pone.0248780](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0248780))
  and **"Are replication rates the same across academic fields? Community forecasts from the
  DARPA SCORE programme"** (*Royal Society Open Science* 2020,
  [royalsocietypublishing.org/rsos/article/7/7/200566](https://royalsocietypublishing.org/rsos/article/7/7/200566/95673/Are-replication-rates-the-same-across-academic)).
  Prediction markets and expert surveys, run *before* replication attempts, forecast replication
  outcomes better than chance and reveal that the research community holds latent, extractable
  information about which findings are shaky — the DARPA program treats "will this replicate" as
  itself a predictable quantity worth eliciting rather than something only discoverable
  post-hoc. **Replication Markets** ([arxiv.org/pdf/2005.04543](https://arxiv.org/pdf/2005.04543))
  is the companion technical report on running these markets at scale for the DARPA program and is
  a useful methods reference if we ever wanted to run an internal "which contested edges will
  resolve" forecasting exercise among domain experts.
- The **Reproducibility Project: Psychology** (Open Science Collaboration, *Science* 2015) is the
  foundational empirical result underlying all of the above (100 studies re-run, ~36–39%
  replicated depending on criterion) — it's the reason "disagreement rate" became a first-class
  quantity fields started trying to predict rather than just deplore.

None of this line offers a ready-made *classifier* we could apply to our 220 contested edges (the
labeled replication data that exists is in psychology/social-science, not microbiome 16S/shotgun
studies, and is far too small to transfer). What it does offer is **conceptual license**: treating
disagreement rate as a measurable, structured property of a literature — with its own base rate,
its own predictors, and its own forecasting task — is an established research program, not a
post-hoc rationalization for "we didn't clean up our contested edges."

---

## 5. Argumentation mining and evidence aggregation

**Argumentation mining** treats a body of text as a directed graph of claims connected by
**support** and **attack** edges (survey: **"Large Language Models in Argument Mining: A
Survey,"** [arxiv.org/html/2506.16383v1](https://arxiv.org/html/2506.16383v1)). Recent work
explicitly targets *scientific* literature and *inter-document* structure — e.g. **"Mining
Inter-Document Argument Structures in Scientific [Literature],"**
([TGDK 3.3.4](https://drops.dagstuhl.de/storage/08tgdk/tgdk-vol003/tgdk-vol003-issue003/TGDK.3.3.4/TGDK.3.3.4.pdf)),
which builds graphs where an argument component in one paper explicitly supports or attacks a
claim in *another* paper — structurally exactly what a contested edge is (paper B's "depleted"
finding attacks paper A's "enriched" finding, both anchored to the same taxon–disease claim as the
central node they disagree about).

The formal backbone for this is **Dung's abstract argumentation framework** (1995) extended to
**bipolar argumentation frameworks (BAFs)**, which add an independent *support* relation alongside
*attack* ([overview via ResearchGate summaries above]). The version most relevant to us is the
**Quantitative Bipolar Argumentation Framework (QBAF)**: each argument (here: "taxon X is enriched
in disease Y," "taxon X is depleted in disease Y") gets a **base score**, attack/support edges
from other arguments (here: individual paper-observations) modify that score under a **gradual
semantics**, and the framework produces a final **strength** for each pole plus, critically, an
**explanation** of *why* — via Argument Attribution Explanations / Relation Attribution
Explanations, which assign credit to individual supporting/attacking edges (Shapley-value or
removal-based; see **"Contrastive Explanations in Quantitative Bipolar Argumentation
Frameworks,"** [arxiv.org/html/2609.02399](https://arxiv.org/html/2609.02399), and **"Argument
Attribution Explanations in QBAFs"**). QBAFs have been applied specifically to **truth discovery**
— aggregating conflicting claims from multiple unreliable sources into a strength-weighted verdict
while keeping every source's contribution inspectable (**"An Empirical Study of Quantitative
Bipolar Argumentation Frameworks for Truth Discovery,"**
[orca.cardiff.ac.uk/id/eprint/170179](https://orca.cardiff.ac.uk/id/eprint/170179/1/Potyka%20and%20Booth%202024%20-%20An%20empirical%20study%20of%20quantitative%20bipolar%20argumentation%20frameworks%20for%20truth%20discovery.pdf)).
This is, almost exactly, our problem: N unreliable-in-aggregate sources (papers), each casting a
directional vote, no ground truth, and a desire for both a summary strength *and* a transparent
accounting of who voted which way and why.

---

## 6. Microbiome-literature-specific disagreement

- **Duvallet, Gibbons, Gurry, Irizarry & Alm, "Meta-analysis of gut microbiome studies identifies
  disease-specific and shared responses"** (*Nature Communications* 2017,
  [nature.com/articles/s41467-017-01973-8](https://www.nature.com/articles/s41467-017-01973-8)) —
  the standard prior meta-analysis showing many microbe–disease associations are inconsistent
  across independently-collected case-control cohorts even for the *same* disease, and that a
  meaningful share of the literature's disagreement is a real biological/cohort effect rather than
  pure noise. Directly relevant background for our "container: contested is a finding, not an
  error" stance.
- **Tierney, Kelly, Fung et al. (Duvallet's lab), "Systematically assessing microbiome–disease
  associations identifies drivers of inconsistency in metagenomic research"** (*PLOS Biology*
  2022, [journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3001556](https://journals.plos.org/plosbiology/article?id=10.1371%2Fjournal.pbio.3001556)):
  the most quantitatively precise prior number in this space. Testing 581 previously-reported
  microbe–disease associations across 15 public cohorts (2,343 individuals) and millions of
  modeling-strategy combinations, they find **roughly 1 in 3 taxa show substantial sign
  inconsistency**, that inconsistency is strongly modeling-strategy-dependent (choice of
  confounders adjusted for — sequencing depth, BMI, glucose, cholesterol — flips direction for
  many taxa), and that **>90% of published type-1/type-2-diabetes associations are non-robust** by
  their criterion. This is the single most citable "disagreement rate in this literature is large
  and partly explainable by analysis choices, not just biology" result, and gives us an
  external base rate (~33%) to compare our own ~11–16% contested-edge rate against — ours is lower,
  plausibly because we're comparing *qualitative direction across independent published papers*
  (which already survived each paper's own significance filter) rather than *re-analysis
  robustness across confounder-adjustment choices* on the same raw cohorts, a stricter and
  different kind of "inconsistency."
- Other named drivers repeatedly cited across the general microbiome-heterogeneity literature
  (obesity meta-analysis review, [tandfonline.com/doi/full/10.1080/19490976.2024.2304900](https://www.tandfonline.com/doi/full/10.1080/19490976.2024.2304900);
  T2D review, [mdpi.com/2673-4540/7/8/153](https://www.mdpi.com/2673-4540/7/8/153)): cohort
  composition/geography, diet, DNA-extraction efficiency by taxon, sequencing platform and depth,
  OTU-vs-ASV and LEfSe-vs-DESeq2 pipeline choice, and rarefaction strategy. Notably, **this is
  almost exactly the 24-variable list our own root-CLAUDE.md `FINDINGS_paper_discordance.md`
  already tested against paper-level discordance and found null at MDEs of 16–22%** — so the field's
  standard causal story for microbiome disagreement is one we've already looked for in our own data
  and not found (with the caveat, already on record, that our corpus's paper-level SD of
  discordance, 3.4pp, may simply be too small to resolve most of these effects). Worth stating
  explicitly in any write-up that cites this section: the literature's usual explanations and our
  own null results are in tension, and the honest reading is "our corpus can't adjudicate it,"
  not "these variables don't matter."

---

## 7. Proposal: representing a contested edge so its disagreement is queryable and explainable

The framework that fits best is a **Quantitative Bipolar Argumentation Framework**, with the
factuality-tagging discipline of **SemMedDB** and the SUPPORTS/REFUTES vocabulary of **SciFact**
layered on top for interoperability with existing claim-verification tooling. Concretely, for
each contested (taxon, disease) pair:

1. **Two poles, not one edge.** Model `(taxon, disease, ENRICHED)` and
   `(taxon, disease, DEPLETED)` as two separate argument nodes anchored to the same claim slot,
   rather than one edge with a "contested" flag. This is what makes the disagreement *queryable*
   instead of collapsed into a boolean.
2. **Each supporting paper is an edge into its pole, carrying provenance, not just a vote.**
   Reuse fields we already have (paper id, section, sentence) and add, per the SemMedDB precedent,
   a lightweight **factuality/hedge tag** pulled from the source sentence (plain FACT vs.
   hedged/PROBABLE) — cheap to add since we already extract the sentence, and it lets a future
   query distinguish "12 papers flatly say enriched, 1 hedges" from "13 papers flatly disagree
   50/50."
3. **A base score plus a gradual-semantics strength, not a raw count.** Under QBAF, each pole's
   final strength is a function of its supporting-paper count *and* their individual weights —
   this is the slot where paper-level covariates we already have on hand for a subset of papers
   (cohort size, country, sequencing platform, from the existing `maindata_screen.json` /
   `metadata.jsonl` work) could act as per-edge weights if/when the field's causal story (§6) is
   ever actually confirmed on this corpus; until then, weight = 1 uniformly, which is equivalent
   to today's plain evidence-count scheme and costs nothing to adopt now.
4. **Attribution, not just aggregation, is the explainability payoff.** QBAF's Argument/Relation
   Attribution Explanations give a principled way to answer "why is DEPLETED stronger here" with
   "because papers 14, 88, and 203 supply 70% of the attribution to that pole" — directly usable
   in the KG's existing per-edge provenance UI, and a strictly richer answer than the current flat
   list of contributing papers.
5. **A three-way label at the pair level, in SciFact vocabulary, for interoperability.** Emit
   each contested pair as SUPPORTS-majority / REFUTES-split / NEI (too few papers to call) so the
   representation is directly consumable by, or comparable against, any future SciFact-style
   verifier (e.g. MultiVerS) if we ever want to auto-classify how a *new* incoming paper's stance
   slots into an existing contested pair, rather than hand-curating that decision every time.
6. **Keep it a bipartite structure, never fold to a signed scalar.** All of the above still
   respects the standing decision to keep contested edges rather than average them — the QBAF
   strength score is a *summary annotation on top of* the full bipartite argument graph, not a
   replacement for it; the underlying two-pole-plus-provenance structure remains queryable at full
   resolution regardless of what the summary strength says.

This gives three things the current flat "220 contested edges, evidence count per side" schema
does not: (a) a place to put per-paper hedge/confidence without overloading the direction field,
(b) an attribution-based explanation of *why* one side currently looks stronger, ready-made from
the QBAF explainability literature rather than invented from scratch, and (c) a label vocabulary
(SUPPORTS/REFUTES/NEI) that is directly compatible with the external scientific-claim-verification
tooling (SciFact/MultiVerS) surveyed above, in case we ever want to validate or extend contested
edges semi-automatically against new papers.
