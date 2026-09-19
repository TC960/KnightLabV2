# 209 — Schema Extension: What, If Anything, Can Be Added

**Status:** complete. §§1–4 rest on repo artifacts; §5 is unverified prior knowledge (see its header).
**Date:** 2026-09-18
**Question:** The graph models one relation over one entity pair. What extensions, if any, can be added without repeating the MicrobioRel 6%-precision failure?

---

## 1. Verdict and ranked recommendation

**Verdict: adopt exactly one extension — negative findings — and only as a gated pilot that
must clear a pre-registered precision bar before any corpus-scale run. Reject or defer
everything else. Body site is already closed by measurement and must not be re-opened.**

### The admission test

MicrobioRel did not fail because 22 labels are too many. It failed because most of its labels had
**no sentence-level falsifier**. Nobody can look at one sentence and say whether
`physically_related_to(X, Y)` is wrong, so nothing was wrong until a stratified sample of 50 was
drawn from 6,363 relations and returned 6% precision. The two corollaries recorded in
`../../CLAUDE.md` are the ones that bind here: process–process relations were **0%** precise and
drove 73% of errors, and **raising the confidence threshold lowered precision** — the model's own
confidence was anti-calibrated, so no new field may ever be gated on model confidence.

So the admission test for any extension is not "is it interesting" but:

> **Can you write down, *before* extraction, the adjudication procedure that would mark a single
> sampled assertion of this field WRONG, using only sentences from the source paper?**

If not, reject outright. This is stricter than "is it useful" and it is the test the current schema
passes: direction is falsifiable against the paper's own sentences, which is exactly how reading
fidelity ≥86.6% [81.7, 91.3] was established (`FINDINGS_direction_audit.md`) without depending on
the flawed in-house gold or on Disbiome/Peryton.

### What the cost actually is

Per `research/208_extraction_frontier.md` §1, a full corpus pass is ~1.7 GPU-hours. **Compute is
free; the scarce resource is adjudication-hours.** The real price of an extension is the precision
audit it obligates, forever, on every future corpus revision. And those hours are already spoken
for: 248 of 271 contributing papers have never been screened for study design
(`FINDINGS_corpus_screen.md`), and the edge-recall short-list is 3 papers holding 12 of 15 confirmed
misses (`FINDINGS_edge_recall.md`). Both are known-value work with no schema risk. **Any extension
must beat those for the same hours.** Only one does.

### Ranking

| # | Candidate | Falsifiable per sentence? | Grounds? | New entity type? | Verdict |
|---|---|---|---|---|---|
| 1 | **Negative findings** (`null` direction) | Yes — the significance sentence is the falsifier | Unchanged (NCBI + MONDO) | No | **ADOPT, pilot-gated** |
| 2 | Verbatim statistic transcription (*not* effect size) | Yes — string match against source text | Unchanged | No | Defer; cheap, low value |
| 3 | Body site as first-class dimension | n/a | UBERON | No | **REJECT — measured null** |
| 4 | Direction qualifiers (treatment/longitudinal/dose) | Partly | Unchanged | No, but new *contrast* | Defer; it is a second relation wearing the first one's clothes |
| 5 | Taxon–taxon co-occurrence | No | Taxa ground; the *relation* does not | No | **REJECT — and it needs no extraction at all** |
| 6 | Metabolites / pathways (SCFAs, bile acids) | **No** | ChEBI (necessary, not sufficient) | **Yes** | **REJECT — this is the 0%-precision shape** |

**1. Negative findings — adopt.** Same entity pair, same three gates, one extra value in an existing
enum. It adds no node type, no predicate, no grounding surface, and no new failure mode; the entire
delta is that a taxon the extractor currently drops silently can instead be asserted as tested-and-null.
It is also the only candidate that changes what the project can *claim* rather than what it can
display: the corpus today records 1,720 enriched and 1,372 depleted votes and **zero nulls**, which
means every publication-bias question is currently unaskable. Detailed design, including the recall
asymmetry that makes this harder than it looks, is §2.

*Pilot (falsifiable, pre-registered).* Extend the `samgated-v1` prompt with a third direction value
and an explicit textual trigger. Run on a random 60-paper subsample (~25 GPU-minutes). Adjudicate
every null assertion produced, capped at 100, against the source sentences; double-adjudicate a
30-item overlap, mirroring the direction audit's 25/28 independent agreement.
**Go/no-go: precision on null assertions ≥0.85.** That is the bar the existing schema already
meets (86.6%); an extension that degrades the graph's weakest measured component is not worth having.
Also pre-register the **non-interference check**: enriched/depleted output on those same 60 papers
must be unchanged, edge-for-edge, versus the current prompt. A new value that perturbs the two
existing ones is a regression regardless of its own precision. If either check fails, discard the
extension — do not tune the prompt and re-measure on the same subsample.

**2. Verbatim statistic transcription — defer.** The standing refusal of effect sizes is correct as
stated: LEfSe LDA, fold-change and p-values are incommensurable and pooling them would be invented
precision. But there is a strictly weaker move that is *not* effect-size modelling — record, as an
edge attribute, the statistic **as written** (test name + value + verbatim span), with no
normalisation and no pooling. It is transcription, so precision is measurable by exact string
containment in the source text, i.e. it is the one candidate that can be audited nearly for free.
It is ranked below negative findings only because its value is speculative: it enables
within-test-family comparison (LEfSe-vs-LEfSe) and little else, and it tempts exactly the pooling
the project has already refused. Defer until something concrete needs it.

**3. Body site — reject, on evidence.** See §3; it was measured and returned a clean null.

**4. Direction qualifiers — defer, with a warning.** "Enriched after treatment" and "enriched vs
healthy control" are not the same relation with a tag on it; they are different comparisons, and
collapsing them into one edge type would silently break the disease-vs-healthy-control gate that
makes the existing edges comparable to Disbiome and Peryton. If ever pursued, it must be a
**separate edge type with its own precision audit and its own agreement baseline**, not a qualifier
field — and there is no instrument for that baseline today.

**5. Taxon–taxon co-occurrence — reject, twice over.** First, an *extracted* taxon–taxon relation is
`Interacts_with` / `physically_related_to` under a new name; those are the generic predicates the
standing instruction names. Second, and decisively: the interesting part needs no extraction at all.
Which taxa co-occur in which paper, and whether they agree on direction, is already computable
deterministically from `graph.json` — that is how the 89%-within-paper-vs-54% containment figure in
`../../CLAUDE.md` was obtained. **Spending extraction and adjudication hours to re-derive something
the existing edges already encode is the worst trade on this list.**

**6. Metabolites / pathways — reject.** It grounds (ChEBI), and it still fails. The claim
"*Faecalibacterium* produces butyrate, which is reduced in IBD" is three-place and its middle hop is
a mechanistic assertion usually made in the Discussion, often citing a *different* paper, and not
covered by any of the three gates — the significance gate does not apply to it, the
disease-vs-healthy-control contrast does not apply to it, and main-text-only does not constrain it.
That is the definition of a process entity, and process relations were 0% precise. There is also no
instrument: reading fidelity, edge recall and the curated-database joins all score the *taxon* half
of a two-place edge, and none of them would notice a wrong mechanistic hop — the same blind spot that
let 209 edges carry wrong MONDO ids until 2026-09-14. **Rejected on the admission test, not on
appetite.**

### The null option is the runner-up, and it is close

"Add nothing" loses to negative findings only because nulls are cheap, reversible, and unlock a claim
the project currently cannot make at all. It beats every other candidate on this list. Schema
discipline is this project's principal methodological asset: the closed schema is the reason attempt3
is at ~73% database agreement where the open-schema predecessor was at 6% precision, and that
contrast is more publishable than any additional field would be. **If the §2 pilot misses its bar,
the correct outcome is to add nothing and spend the hours on the 248 unscreened papers.**

## 2. Negative findings, assessed in depth

### 2.1 What is being added

One value. `direction ∈ {enriched, depleted}` becomes `{enriched, depleted, null}`, where `null`
means **this paper tested this taxon in this disease-vs-healthy-control comparison and reported no
significant difference**. Entity pair unchanged, grounding unchanged, contrast gate unchanged,
main-text gate unchanged. The significance gate is not relaxed — it is *inverted* for this value:
today "if significance is unclear or unreported for a taxon, omit it" silently discards two very
different situations, a taxon the paper never tested and a taxon the paper tested and found flat.
Only the second becomes extractable; the first stays out.

The trigger must be **explicit textual assertion of non-significance**, never inference. Admissible:
"no significant difference in *Bacteroides* between groups", "*Prevotella* did not differ (p = 0.41)",
"*Akkermansia* was comparable between patients and controls". Inadmissible, and worth naming in the
prompt as negatives: a taxon simply absent from the results; a taxon present in a figure but not
discussed; a non-significant trend described as "tended to increase" (that is a direction without
significance — the existing gate already drops it, and it must keep dropping it); and any
model-internal judgement that a difference "looks" small. **No confidence threshold anywhere** —
MicrobioRel's confidence was anti-calibrated, and a new field is precisely where that mistake would
recur.

### 2.2 Precision — easy, and the only part that is easy

A `null` assertion carries its own falsifier: it claims a specific sentence exists. Adjudication is
"find the sentence, or mark it wrong", which is the same instrument as `FINDINGS_direction_audit.md`
and needs no new tooling. Three error classes to score separately, because they have different fixes:
(a) **fabricated** — no non-significance sentence for that taxon exists; (b) **misattributed** — such
a sentence exists but concerns a different contrast (treated vs untreated, timepoint 1 vs 2, a
subgroup); (c) **gate leakage** — the sentence is from a table caption or supplementary reference,
violating main-text-only. Class (b) is the one to watch: it is the exact failure the
disease-vs-healthy-control gate exists to prevent, and non-significance statements are more often
made about secondary comparisons than directional claims are.

### 2.3 Recall — the hard half, and the asymmetry bites harder here

The documented asymmetry is that **significance can be confirmed from visible text but never
refuted**, so the existing instrument can confirm a miss but cannot refute one
(`../../CLAUDE.md`, recall section). For nulls this gets worse in two specific ways, and both must be
stated in any figure the project publishes:

1. **The existing sentence filter is blind to nulls by construction.**
   `relation_sentences_clean.json` keeps only sentences containing a taxon *and* a direction cue.
   A sentence saying "*Roseburia* did not differ between groups" has a taxon and **no** direction
   cue, so it is invisible to every candidate generator built on that file — including the one behind
   `edge_recall_packets.json`. A null-recall audit that reuses it would measure approximately zero
   and report approximately 100% recall. **A separate non-significance cue lexicon is mandatory
   before any recall number is computed** ("no significant difference", "did not differ", "not
   significant", "NS", "p > 0.05", "comparable between", "similar in both groups", "unchanged").

2. **The main-text-only gate is systematically harsher on nulls than on positives.** Papers put
   significant findings in the narrative and park non-significant ones in tables and supplementary
   files. So the achievable ceiling on null recall is not 100% — it is the fraction of nulls that are
   stated in main-text prose at all, and that fraction is unknown and probably well under half.
   **Reporting a null-recall figure without that ceiling would be meaningless**, and worse, would
   look like a failure of the extractor when it is a property of the gate.

Given that, the recall plan is three tiers, and only the first produces a number to quote:

- **Tier A — confirmable-miss lower bound.** Scan a random sample of papers with the
  non-significance lexicon; enumerate (paper, taxon) candidates; adjudicate each with the gates
  applied; count *confirmed* misses only. Same asymmetry as the paper-level recall work: this can
  confirm a miss, never refute one, so the output is a lower bound and **must be quoted as a range**.
- **Tier B — paired within-paper check.** Restrict to papers that report both significant and
  non-significant results for taxa that appear elsewhere in the graph. If the extractor catches the
  positives and drops the nulls in the same paper, the loss is attributable to the new field rather
  than to reading failure. This is the diagnostic tier, not a headline number.
- **Tier C — the ceiling.** Hand-read ~15 papers and record, for every non-significance statement,
  whether it lives in main-text prose, a table, a figure, or supplementary. **This number must be
  published alongside Tier A or Tier A is uninterpretable.** It also decides whether the
  main-text-only gate should ever be relaxed for this field — a question that should be answered by
  this measurement and not by intuition.

### 2.4 The missing baseline, and a second go/no-go

The corpus records **zero** nulls, so there is no prior yield to compare against and no way to know
in advance whether the pilot returns 5 nulls or 500. That makes yield itself a decision variable.
Add to the §1 pilot: **if the 60-paper subsample yields fewer than ~20 adjudicable null assertions,
stop even at high precision** — a field that fires rarely cannot support any of the claims in §2.5,
and it would still impose an audit obligation on every future corpus pass. A sparse null field is
worse than no null field, because it invites exactly the inference it cannot support: reading absence
of a recorded null as evidence of absence.

### 2.5 What the project could then claim — and what it still could not

Could:

- **A denominator for contested edges.** 217 pairs are contested, and `206_edge_confidence.md`
  establishes that `contested` is a refusal to assert direction, not a confidence score. Nulls give
  those pairs an abstention count: 3-up/3-down with 8 papers reporting null is a different object
  from 3-up/3-down with none, and today the two are indistinguishable.
- **A denominator for the 77.7% of edges resting on a single paper.** A one-paper edge with several
  independent nulls on the same pair is weak evidence; a one-paper edge nobody else tested is merely
  under-studied. Same graph today.
- **A directly testable selective-reporting signature.** Compute the decisive:null ratio per taxon
  and test whether high-salience genera (*Akkermansia*, *Faecalibacterium*, *Prevotella*) are
  reported decisively more often than obscure ones, conditioning on how often each is measured at
  all. That is a within-literature publication-bias measurement the graph cannot attempt today.
- **A power-based account of discordance, which 24 study-design variables failed to provide.**
  `FINDINGS_paper_discordance.md` reports 24 variables and 24 nulls, with the paper-level SD of
  discordance at only 3.4 points. If nulls concentrate on the same pairs that contest, the story is
  statistical power rather than protocol — a candidate explanation that is currently untestable. Be
  honest about power: that SD and the existing MDEs of 4–7 points mean this corpus may well not
  resolve it either, and the pre-registration should say so before the test is run.

Could **not**:

- **Classical publication bias.** That concerns studies never published, and no corpus of published
  papers can see them. What is measurable here is *within-paper reporting completeness*, which is a
  different and narrower claim, and the write-up must use the narrower words.
- **Funnel plots or bias-corrected pooling.** Those need effect sizes and standard errors, which the
  project refuses as invented precision (§1, candidate 2). Adding nulls does not change that, and the
  temptation to pool once nulls exist is the main way this extension could go wrong downstream.
- **"Taxon X is unrelated to disease Y."** An extracted null is one paper's failure to detect, not
  evidence of no association. The node-level claim requires pooling across papers with power
  weighting — i.e. effect sizes again. **Nulls must be stored and displayed as per-paper
  observations, never aggregated into an edge-level "no association" verdict.**

## 3. The groundability test, and what body site already measured

### 3.1 Groundability is necessary, and it is nearly useless as a filter

The current schema works partly because both endpoints resolve to controlled vocabularies with
stable identifiers — taxa to NCBI Taxonomy via `taxonomy.py` (76% resolved; the residual 212 of 883
are 16S clade labels and taxa absent from the cached taxdump, per `FINDINGS_taxon_spelling.md`),
diseases to MONDO via `mondo.py`. That is what makes the Disbiome and Peryton joins possible at all,
and the standing rule **never join on another database's identifier** — which cost 209 edges a wrong
MONDO id until 2026-09-14 — is a rule about *how* to ground, not a substitute for grounding.

Applying the test to the candidate list:

| Candidate | Endpoints ground? | To what | Relation falsifiable? |
|---|---|---|---|
| Negative findings | unchanged | NCBI + MONDO | **Yes** |
| Verbatim statistic | n/a — a literal, not an entity | none needed | **Yes** (string containment) |
| Body site | yes | UBERON | n/a — it is a paper attribute |
| Direction qualifiers | yes | **no vocabulary for "comparison type"** | Partly |
| Taxon–taxon | yes, both | NCBI on both ends | **No** |
| Metabolites / pathways | yes | ChEBI; GO/KEGG/MetaCyc | **No** |

**The result is that groundability rejects nothing on this list.** Every candidate's endpoints
resolve to something. Taxon–taxon grounds *better* than the current schema — NCBI on both ends —
and is the worst idea here. That is the whole point: MicrobioRel's entities were largely groundable
too, and it still returned 6% precision, because `physically_related_to` has no sentence that can
disprove it. **Falsifiability rejected three candidates; groundability rejected zero.** Record this
explicitly so no future proposal is waved through on "but it maps to ChEBI" — that sentence is
evidence of nothing.

Two specific grounding caveats worth keeping even though they did not drive any decision here. First,
metabolites ground to ChEBI as *compounds*, but papers overwhelmingly write the class ("SCFAs") not
the member ("butyrate"); mapping class language onto ChEBI members is a judgement call made once per
assertion, i.e. a second place for precision to leak silently. Second, direction qualifiers have no
ontology at all for the comparison type, so they would require a bespoke enum — and bespoke enums
growing one value at a time is exactly the mechanism by which a closed schema becomes a 22-label
open one.

### 3.2 Body site was already measured, and the answer is null

`body_site.py` assigned a sampled site to all 281 contributing papers and, unusually for a metadata
field here, **scored the assignment before trusting it**. 211 papers carry the independent
LLM label from `metadata.jsonl`; the keyword scanner covers the remaining 70, and it was validated at
**230/249 = 92.4% agreement** against the LLM labels on the papers holding both. The decision rule is
itself a measured finding: plain argmax over site cues scored 84.3% and its dominant error was
*co-sampling* — stool-microbiome studies that draw serum or plasma for metabolomics talk about blood
constantly and get read as blood studies. Narrowing the context window made it worse (83.5%, 79.9%,
converting errors into `unknown`); giving any stool cue outright priority fixed it at 92.4%.

The distribution it produced settles the question:

| site | papers |
|---|---|
| stool | 274 |
| oral | 4 |
| gut biopsy | 1 |
| nasal | 1 |
| blood | 1 |

**97.5% of the corpus is stool.** And the downstream test in `bodysite_effect.json` is not merely
non-significant, it is empty:

| source | n common pairs | agree, all sites | agree, gut only | gained | lost | McNemar p |
|---|---|---|---|---|---|---|
| Disbiome | 160 | 114 | 114 | 0 | 0 | 1.0 |
| Peryton | 122 | 91 | 91 | 0 | 0 | 1.0 |

Restricting the external validation to gut-only papers changed **zero** pairs in either direction
against either database. The motivating hypothesis — that site collisions depress agreement with the
gut-weighted curated databases, as in the *Rothia*/Parkinson's case where a saliva study lands on the
same node as gut records — is real for that individual adjudication and **has no measurable
corpus-level effect**, because there is essentially no non-stool variance for it to act through.
Honest caveat in the other direction: the scanner's known failure modes (3 dual-site papers collapsing
to stool, 1 genuine blood study outvoted by its stool arm) mean the non-stool count is slightly
understated — but even several times understated, a handful of papers cannot move 160 pairs.

### 3.3 Therefore

**Keep body site exactly where it is: a per-paper metadata attribute, not a schema dimension.** It
earns its place as an adjudication aid — it is what turned one apparent contradiction of the curated
databases into a correctly-explained site mismatch — and promoting it to a first-class edge dimension
would add a UBERON grounding surface, a per-edge audit obligation, and a node-splitting decision, in
exchange for a difference measured at exactly zero.

The reframing that matters: **body site is a corpus-composition problem, not a schema problem.** The
only thing that would make it worth re-opening is a corpus where non-stool papers are a material
fraction — say above 10–15%, which at present would mean deliberately ingesting oral, nasal and skin
microbiome literature. That is a decision about which papers to screen, and it belongs with the
corpus-scaling work in `203_corpus_scaling.md`, not here. Until the corpus composition changes,
`bodysite_effect.json` is the answer and the analysis should not be re-run.

## 4. Candidate extensions vs. the MicrobioRel failure mode

`mega_dump/proj_2` (`PHASE2_FINAL_ANALYSIS_REPORT.md`) is the control experiment this project already
ran, in the same lab, on the same literature. It has three distinct failure signatures, and it is
worth checking each candidate against each rather than against the headline number.

**Signature 1 — process entities (0% precision, 73% of errors).** A process entity is one whose
assertion is not checkable against a measurement reported in the paper. Metabolites and pathways are
this shape: "*Faecalibacterium* → butyrate → gut barrier integrity" is not a measured contrast, it is
a mechanism, and the paper's own statistics do not adjudicate it. Direction qualifiers are a milder
case — a treatment-response contrast *is* measured, but it is a different measurement than the one
the schema's gates describe. Negative findings, verbatim statistics and body site introduce no
process entity at all.

**Signature 2 — generic predicates (another ~50% of errors).** `part_of`,
`physically_related_to`, `Associated_with`, `Marker/Mechanism`. The defining property is that the
predicate does not constrain what evidence would satisfy it. Taxon–taxon co-occurrence is this,
renamed: "co-occurs with" admits correlation, co-culture, phylogenetic proximity, shared habitat and
mere co-mention, and a reader cannot tell which was meant. Metabolite edges would need `produces`
/ `affects`, which are the same shape. No other candidate adds a predicate at all — negative findings
adds a *value* to an existing one, which is categorically different.

**Signature 3 — anti-calibrated confidence.** Raising the threshold *lowered* precision. This is the
most transferable lesson and it constrains the implementation of even the accepted extension: **no
new field may be filtered by a model-reported confidence score, ever.** Gate on explicit text or do
not gate. It also forbids the obvious-looking remedy if the §2 pilot misses its bar — "keep only
high-confidence nulls" is precisely the move that failed before.

**Signature 4 — the meta-failure.** 6% precision was discoverable only by drawing a stratified sample
of 50 from 6,363 relations *after* the extraction existed. The arithmetic is worth stating plainly:
MicrobioRel's 6,363 relations at 6% precision are roughly **380 sound facts**; this graph's 2,008
edges at ≥86.6% reading fidelity are roughly **1,740**. The open schema produced three times the
volume and about a quarter of the truth. **Edge count is not a measure of progress**, and any
extension proposal that leads with how many new edges it would create should be read as arguing
against itself.

Net mapping:

| Candidate | Sig. 1 process | Sig. 2 generic predicate | Sig. 3 confidence risk | Verdict |
|---|---|---|---|---|
| Negative findings | no | no (new *value*, not predicate) | avoidable by construction | adopt, gated |
| Verbatim statistic | no | no | none | defer |
| Body site | no | no | none | reject on measurement (§3) |
| Direction qualifiers | partial | no, but new contrast | moderate | defer |
| Taxon–taxon | no | **yes** | high | reject |
| Metabolites / pathways | **yes** | **yes** | high | reject |

## 5. What the field does

**Provenance warning: this section is written from prior knowledge, not from sources checked during
this session — the session's web-search budget was exhausted before it was drafted.** Nothing above
depends on it; §§1–4 rest on repo artifacts only. Treat the claims below as leads to verify before
any of them is repeated in a paper, and verify them if this section is ever load-bearing.

**The curated microbe–disease databases record only positive findings.** Disbiome and Peryton both
store a direction (elevated / reduced) for associations that were reported as significant; neither,
as far as this project's joins have ever shown, carries a "tested, no difference" record. That is the
main reason §2's publication-bias claims would be novel rather than derivative — and also the reason
they cannot be validated externally the way direction is. **A null field would have no Disbiome or
Peryton counterpart to agree with**, so its only available instrument is the paper's own sentences,
which is exactly the instrument §2 specifies. This is worth stating in any write-up: the extension
would be measurable but not externally corroborable.

**Adding a negation value to an existing relation is the field's standard move, not an exotic one.**
Biomedical event-extraction shared tasks (the BioNLP/GENIA line) have long annotated negation and
speculation as *modifiers* on events rather than as separate relation types — the same architecture
§2 proposes. The relevant transferable result, and the reason for the §2 pilot bar, is that
modifier-level performance in those tasks ran materially below plain event detection: detecting that
something is asserted is easier than detecting that it is denied. A project that expects null
extraction to be as accurate as direction extraction is expecting the wrong thing.

**Explicit-cue detection is the established design.** The NegEx / ConText line of work and the
BioScope corpus established that negation and hedging in biomedical text are handled well by explicit
lexical cues and scope rules, and poorly by inference. That is independent support for §2.1's
insistence that `null` fires only on an explicit non-significance statement.

**Large biomedical KGs that do carry negation are a cautionary precedent, not an encouraging one.**
SemMedDB's predications carry a negation flag derived from SemRep, and the flag's reliability is
generally regarded as weaker than the predication's. The lesson is the one §2.4 already draws from a
different direction: a sparse, noisy negation field invites over-reading and is worse than none.

**The framing to borrow from clinical evidence synthesis is "outcome reporting bias", not
"publication bias".** Systematic-review methodology distinguishes studies never published from
outcomes measured but not reported within published studies; only the second is visible to a
literature-mining project. §2.5 already uses the narrower words, and this is the literature that
justifies them.

**Precedent for the refusal, too.** The GWAS Catalog's practice of recording only associations past a
significance threshold is the standard example of a curated resource whose selection rule is well
understood and openly stated. That is the defensible posture for this project if the pilot fails:
a closed, positively-selected schema with the selection rule documented is a legitimate and
well-precedented artifact — which is the substance of the "add nothing" option in §1.
