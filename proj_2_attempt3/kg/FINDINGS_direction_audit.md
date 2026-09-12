# Reading fidelity, measured without the gold standard and without Disbiome or Peryton

**2026-09-12.** Scripts: `audit_direction_witness.py`, `witness_discordance.py`,
`clean_abbrev.py`. Data: `audit_direction_witness.json`,
`witness_discordance.json`, `clean_abbrev.json`,
`disagree_adjudication_{opus,haiku}.json`.

---

## Why this was worth doing

Every number we have for "is the extractor faithful to the paper" is compromised
or narrow:

| signal | problem |
|---|---|
| in-house gold standard, F1 ~0.59 | the gold is under audit and known unreliable; 162 of 250 papers have blank taxa columns |
| Disbiome 73.0% / Peryton 72.5% | only half independent — 43 and 24 of our papers are cited by them, and those back half the decisive pairs (`FINDINGS_independence.md`) |
| the 33 verified misspellings, 2026-09-11 | genuinely independent, but 33 taxa |

`FINDINGS_independence.md` decomposed the curated-database agreement into ~90%
**reading fidelity** (papers both sides read) and ~55% **cross-literature
reproducibility**. The reading-fidelity half is the cleanest evidence the project
has — but it is measured only on the ~half of decisive pairs the curators happen
to share with us.

This audit measures the same quantity over **the whole graph**, using only the
papers' own sentences. It depends on neither the gold nor either database.

**What it measures:** whether the extractor read the sentence correctly.
**What it does not measure:** whether the paper is right.

---

## Headline

> **86.6% cue agreement (181/209), 95% CI [81.7%, 91.3%]**, paper-cluster
> bootstrap over 122 papers. Residual disagreement is **not** concentrated in
> particular papers (permutation p = 0.28).
>
> All 28 residual disagreements were read individually, twice and independently.
> **Zero are confirmed extraction errors.** So 86.6% is a **lower bound**, not an
> estimate.

That sits on top of the 87.5% (Disbiome) / 96.8% (Peryton) same-paper figures and
is consistent with them, which is the point: an independent instrument lands in
the same place.

---

## Method, and the three ways the first version of it was wrong

For every `(paper, taxon, direction)` observation backing an edge — 3,077 of them
— find the sentences in that paper naming that taxon, and compare the direction
words beside it with what we extracted.

Attribution is the whole difficulty. "Firmicutes decreased while Proteobacteria
increased" carries both cues. So scoring is restricted to **strict witnesses**:
the sentence names exactly one taxon and all its cues share one polarity.

The first version scored **84.0%** and called the other 16% disagreements.
Reading them showed most were *the audit's* fault. Three tiers, all reported:

| tier | what it scores | result |
|---|---|---|
| **T0** every strict witness | the naive number | 375/442 = **0.848** |
| **T1** T0 minus sentences that are not the paper's own result | **headline** | 181/209 = **0.866** |
| **T2** T1 with a comparison-frame correction | built, measured, **rejected** | 158/204 = 0.774 |

**T1 — background is not a result.** "Chang et al. indicated that acute ischemic
stroke patients with good outcomes exhibited an increased abundance of *Blautia*"
and "some studies have shown that Bacteroidetes … are reduced (Zhuang et al.,
2018)" describe *other people's cohorts*. Scoring our extraction against them is
a category error. A sentence counts as the paper's own result only if it carries
a statistical or measurement cue (p/q/FDR, LEfSe, LDA, AUC, "significantly",
"we found", a figure or table reference) **and** carries no citation marker or
third-party reporting verb.

**T2 — a frame corrector that made things worse, and why.** "*Lactobacillaceae*
was most abundant in control group" carries an UP cue but means DEPLETED IN
DISEASE, so flipping control-framed sentences ought to help. It does the
opposite, 0.866 → 0.774. The reason is that the dominant construction in this
literature names controls as the **reference**, not the subject — "lower in PD
patients compared with the healthy controls", "compared to control participants,
AD participants exhibited decreased Actinobacteria" — and those need no flip.
Of the 54 own-result witnesses the detector called control-framed, **41 were
already correct unflipped.** Only the locative minority genuinely inverts, and no
regex separated the two. Recorded so the next session does not rebuild it.

---

## The 28 residual disagreements: adjudicated twice, independently

Categories were fixed in advance; the orchestrator (Opus) wrote its verdicts
**before** reading the independent Haiku pass. **Exact category agreement 25/28**,
and all three differences are between two *artefact* categories — none moves
anything toward "extraction error".

| category | Opus | Haiku | what it means |
|---|---|---|---|
| `LOCATIVE_CONTROL` | 12 | 11 | the abundance is attributed to the control group, so "depleted in disease" is correct and the raw cue is the inverted one |
| `OTHER_CONTRAST` | 9 | 9 | the sentence compares treatment arms, timepoints, symptom or converter subgroups, or a regression on a continuous score — not disease vs healthy control |
| `TAXON_MISMATCH` | 4 | 4 | the direction belongs to a different organism named in the same sentence |
| `BACKGROUND_LEAK` | 2 | 2 | background that slipped past the provenance filter |
| `UNRESOLVED` | 1 | 2 | truncated or undecidable from the sentence |
| **`EXTRACTION_ERROR`** | **0** | **0** | — |

Worked examples of each, verbatim:

- *locative* — "Bacteria from the genus **Faecalibacterium** were significantly
  more abundant in the mucosa of controls than PD." We extracted *depleted*. We
  are right; the cue word is "more".
- *other contrast* — "**Megamonas** was significantly higher in the bloating
  group than in the non-bloating group" — a symptom subgroup inside the injured
  cohort, silent on the disease-vs-control direction.
- *taxon mismatch* — "Concerning **Proteobacteria**, '*Candidatus* Blochmannia'
  was significantly reduced in PD patients." The reduction is Blochmannia's.
- *background leak* — "the findings in the present study were inconsistent with
  another previous study **25**, which reported a reduced relative abundance of
  genus *Bifidobacterium*". The bare numeric citation "study 25" carries no
  brackets, so the citation regex missed it.

**Reading this honestly:** the 13.4% residual is dominated by what the audit
cannot see — chiefly that a paper contains many contrasts and this instrument
cannot tell which one a sentence is about. Two independent reads found no
extraction error in 28 cases. The true reading fidelity is above 86.6%; how far
above, this instrument cannot say.

---

## A defect found on the way, and fixed: one letter was deciding which genus a paper meant

The audit's first disagreement list contained *Phascolarctobacterium* in
Alzheimer's, "supported" by sentences about **P. gingivalis**. That is
*Porphyromonas gingivalis*, the periodontal pathogen — in an oral-vs-gut AD paper
largely about it — filed under a gut genus.

`relation_sentences.filter_paper` expanded abbreviated binomials with
`alias.setdefault(sci[0].upper(), sci)`: a map from a single **initial letter** to
whichever genus was seen first. A paper naming *Bacteroides*, *Bifidobacterium*
and *Blautia* sent every later "B. ⟨sp⟩" to *Blautia*. `TaxonMatcher.find` then
compounded it — when "⟨that genus⟩ ⟨epithet⟩" failed to resolve it fell back to
the bare genus, so a failed expansion produced a confident wrong hit rather than
no hit.

- **774** of 18,436 taxon mentions came from the abbreviation path.
- **362** sat in a paper where two or more genera share the initial, across
  **78 of 348** papers.
- Repaired: **84 reassigned, 29 already correct, 117 dropped** as unattributable.
  `E. rectale` Escherichia→Eubacterium (×11), `B. uniformis` Blautia→Bacteroides,
  `A. muciniphila` Anaerotruncus→Akkermansia, `R. gnavus` Roseburia→Ruminococcus.

Fixed at source (an ambiguous initial now resolves to nothing) and repaired in
place, because rebuilding needs the NCBI taxdump and `ftp.ncbi.nih.gov` is still
`CONNECT → 403` (sixth session probe). **The repair needed no taxdump**: when a
paper writes "B. adolescentis" it also names *Bifidobacterium* in full somewhere,
and that mention already carries its resolved taxid.

**The curated table was itself briefly weaker than what it audited** — the
failure mode this log has recorded twice before. A first draft keyed on the
epithet alone would have rewritten the correctly species-resolved
"L. salivarius" (*Ligilactobacillus*) to *Streptococcus* and "R. hominis"
(*Roseburia*) to *Dialister*, because those epithets are shared across genera.
Two guards now: never override a species-rank resolution, and require the
candidate genus to start with the mention's own initial.

### Two things this does *not* touch, one of them for an interesting reason

**The knowledge graph is unaffected.** `build_kg.py` never reads this file; it
builds from `extractions_screened.json`.

**The `FINDINGS_cooccurrence.md` null is robust to the defect, structurally.**
Re-running `cooccur_direction.py` on the repaired corpus reproduces every
statistic **bit-for-bit** — pooled p = 0.861 / 0.939, per-edge p = 0.243 / 0.105.
That identity is not a failed rebuild; it was checked. **Zero of 348 papers
change their taxid set.** Reassignment *requires* the true genus to be named in
full in the same paper, and the wrongly-assigned genus was too — that is how it
entered the alias map. The bug moved mentions **between taxa the paper names
anyway**, so a binary incidence profile cannot see it. Per-sentence attribution
can, which is why an audit working sentence-by-sentence found what the
co-occurrence analysis could not.

The lesson generalises: a representation that is immune to a class of error is
also blind to it.

---

## Second question, and a null: does textual provenance predict discordance?

If the audit can mistake background for a result, so could the extractor. Each
observation was classed:

| class | n | meaning |
|---|---|---|
| `own` | 2,109 | at least one own-result sentence names this taxon |
| `background` | 585 | sentences name it, but every one is background or citation |
| `silent` | 383 | no kept sentence names it at all |

Hypothesis: `background` observations are weaker evidence and should disagree
with the rest of the literature more often. Discordance is defined exactly as in
`FINDINGS_paper_discordance.md` (leave-one-out majority within the edge).

**Null.** Over 1,390 decisive observations:

| class | discordance |
|---|---|
| `own` | 285/1032 = **27.6%** |
| `background` | 76/281 = **27.1%** |
| `silent` | 22/77 = 28.6% |

- pooled difference **−0.6 points**, minimum detectable difference at 80% power
  **8.4 points**;
- paired within paper (59 papers carrying both kinds, so paper identity cancels
  exactly; null is a sign flip of each paper's difference) **+5.5 points,
  p = 0.28**, MDE **14.1 points**.

So: **no effect at an MDE of 8 points pooled.** This is the **25th** paper- or
observation-level variable tested against discordance and the 25th null, after
the 24 in `FINDINGS_paper_discordance.md`.

Two readings, both worth keeping:

1. **Good news for the graph.** The 585 background-only observations are not
   detectably worse evidence. The provenance screen does not isolate a defective
   subset to review.
2. **The screen is spent as a lever.** Note the caveat that cuts both ways:
   `relation_sentences.json` keeps 7.5k of 106k sentences, and a taxon reported
   only in a table or a figure is `silent` however well supported. `silent` is
   therefore uninformative; the meaningful contrast is `own` vs `background`, and
   it is flat.

---

## What a PI should take from this

1. **Reading fidelity is ≥86.6% [81.7, 91.3], measured without the gold standard
   and without either curated database.** Two independent reads of all 28
   residual disagreements found zero extraction errors, so the true figure is
   higher. This corroborates the ~90% reading-fidelity half of
   `FINDINGS_independence.md` by an independent route, and it is now the
   project's **fourth** gold-free fidelity signal.
2. **It does not move the 73%/72.5% headline and must not be quoted as doing so.**
   Reading fidelity and cross-literature reproducibility are different quantities;
   the 73% is a blend of the two.
3. **A genuine data defect was found and fixed** — genus abbreviations resolved by
   first-letter collision — but it never reached the graph, and the analysis most
   exposed to it is immune to it for a structural reason.
4. **Discordance remains unexplained after 25 variables.** The binding constraint
   is still n. Nothing in this session changes that.

## Power, plainly

- The headline rests on **209 scoreable observations from 122 papers** — a
  strict-witness subset of 3,077. Widening it means solving sentence-level
  attribution, which the rejected T2 tier shows regex cannot do.
- The 28 residual cases are a **census, not a sample**: every T1 disagreement was
  read.
- The provenance null could not have seen a difference smaller than **8.4
  points** pooled. A real effect of 3 points would have been invisible.
