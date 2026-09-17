# Do the graph's claims name taxa that are actually in the papers?

**2026-09-13, revised 2026-09-17.** Cloud session, CPU only, no GPU, no NCBI
taxdump — but, as of the revision, **with `MAIN_DATA.json`**.

> ## UPDATE 2026-09-17 — the blocker was never real, and the answer is now 100%
>
> **`MAIN_DATA.json.zip` is tracked in git** (`proj_2_attempt3/`, 33 MB). Only the
> unzipped 105 MB `MAIN_DATA.json` is gitignored. Eleven sessions recorded this
> audit as "needs the Mac"; a single `unzip` produces the corpus in any checkout.
> `NEXT_SESSION_PROMPT.md` item 0 — *"the cheapest open lever in the project"* —
> was blocked on a premise that was false the whole time.
>
> **Coverage.** Wiring MAIN_DATA in as a fallback (git full text still wins any
> title collision, so no already-scored observation changes verdict) takes
> not-scoreable observations **795 → 581**, and the 580-edge cohort's
> **144 → 93**. That is a narrowing, **not a closure**: 43 of 271 contributing
> papers are in neither source, so ~19% of observations remain unscoreable.
>
> **Result.** The mention rate does not move because it cannot go up: **every
> scoreable observation names its taxon.** `own` 1,679/1,679, `background`
> 494/494, `silent` 321/321. The 12 flagged "absences" are **all twelve** matcher
> false negatives (see the superseded and retracted sections below) — including
> the one that had been recorded as the project's single confirmed extraction
> error, which is **retracted**. The stability is the point: opening a quarter
> more of the corpus turned up **no** new fabrication candidate, so the
> previously-unscoreable population behaves like the scoreable one rather than
> hiding a reservoir of bad claims.
>
> **A guard came out of it.** The two sources share 16 papers and disagree on 12
> of 179 observations — *all* one-directional (git=mentioned, MAIN_DATA=absent),
> *all* on the two papers where MAIN_DATA holds an abstract-only stub (2.4k/1.4k
> chars against 92k/46k in git). Truncation only deletes text, so **a stub can
> prove a mention but never disprove one**; short MAIN_DATA documents now return
> not-scoreable instead of absent. 295 of 2,019 corpus documents are stubs. See
> `validate_maindata_text.py`.

**Headline (as first published, 2026-09-13).** Of the 580 edges
`FINDINGS_direction_audit.md` sized as beyond the reach of any prose instrument,
**429 of the 436 scoreable ones (98.4%) name their taxon in their own source
paper.** Across the whole extraction, **2,291 of 2,301 claims (99.57%)** do,
against a paper-shuffled null of **15.0% ± 0.8** — a gap of **84.6 points**,
p = 0.0099 (the floor at 100 permutations). ~~Exactly one confirmed extraction
error survived scrutiny~~ (**retracted — there are zero**), and seven of the ten
apparent failures were limitations of my own matcher, not the graph's.

**What this is NOT.** It bounds **fabrication**, not correctness. A taxon can be
named in a paper's introduction and still back a wrong edge — that is precisely
what the `background` provenance class already measures. This number
**does not move 73.0% / 72.5%** (cross-literature agreement) and is **not** a
restatement of the **86.6%** reading fidelity in `FINDINGS_direction_audit.md`.
Three different quantities:

| quantity | what it asks | value |
|---|---|---|
| DB agreement | does the literature replicate? | 73.0 / 72.5% |
| reading fidelity | did we read the direction right? | ≥86.6% |
| **mention (this doc)** | **is the taxon even in the paper?** | **100%** (was 99.57%) |

Mention is the weakest of the three and the easiest to pass. It is worth
measuring only because it reaches a population the other two cannot.

---

## Why this was worth doing

`FINDINGS_direction_audit.md` recorded that 580 edges (28.9% of the graph) rest
on a single paper AND carry no own-result prose witness, and concluded that
"prose filtering cannot reach them". That is true of the **relation** filter,
which requires a taxon and a direction cue *in the same sentence*. It is not
true of a weaker question: **is the taxon named anywhere in the document?**

The distinction matters because `silent` conflates two very different things:

- **silent + mentioned** — reported outside prose, in a table or figure. Invisible
  to the instrument, not unsupported. Direction still unverified.
- **silent + absent** — the taxon does not occur in the only paper backing the
  edge. A candidate fabrication, and individually reviewable.

Nobody had separated them. Before this, the honest description of those 580 edges
was "we cannot say anything". Now it is "98.4% are not fabricated; their direction
remains unverified".

## Method

`verify_taxon_mentions.py` replays every taxon string the extractor emitted in
`extractions_screened.json` against that paper's full text from
`all_usable_papers.json`. Match tiers, strictest first: `exact`, `variant`
(separators collapsed, rank prefixes and `sp.`/`unclassified` tails dropped,
whitespace-insensitive), `abbrev` (`Ak. muciniphila`), and the deliberately weak
`head` (genus only — **excluded** from the strict rate). Deterministic
throughout; per the repo rule, no LLM judgement is used where a string comparison
settles it.

`silent_edge_mentions.py` joins that matcher onto the published per-observation
provenance from `witness_discordance.build()`, so the `own`/`background`/`silent`
classes are the ones already reported, not a re-derivation.

### Two controls, because a raw mention rate proves nothing

**Negative control — is the test vacuous?** Microbiome papers share a core
vocabulary; if *Bacteroides* appears in most of them, "the taxon is in the paper"
could be true by accident. `verify_mentions_null.py` re-pairs each paper's taxon
list with a **different** paper's text, shuffling at the **paper** level (the
repo rule: observations are not independent). Over 100 permutations the null is
**14.96% ± 0.81** (range 12.9–16.6) against a true rate of **99.58%**. The test
is not vacuous. Two supporting numbers: the **rarity-weighted** rate, which
weights each claim by how rare its taxon is corpus-wide, is **99.52%** — so
ubiquitous genera are not carrying the result — and only **2.7%** of claims name
a taxon occurring in more than half the corpus.

**Positive control — is the matcher sound?** `own` and `background` observations
name their taxon *in a sentence* by construction, so they **must** score 100%.
This is an answer known a priori, and it caught a real bug (below). Final:
**own 1520/1520, background 474/474** — zero false negatives on 1,994
observations.

## Results

| provenance | mentioned | absent | rate | not scoreable |
|---|---|---|---|---|
| own | 1520 | 0 | **100.0%** (control) | 589 |
| background | 474 | 0 | **100.0%** (control) | 111 |
| silent | 278 | 10 | **96.5%** | 95 |
| **580 single-paper, no own witness** | **429** | **7** | **98.4%** | **144** |

"Not scoreable" means the paper's full text reaches this repo only through the
gitignored `MAIN_DATA.json`; 795 of 3,077 observations (25.8%) are in that
bucket and are reported as unknown, never as absent.

### The ten residuals, read by hand

All ten come from three papers. Seven are demonstrably present in text
conventions the matcher does not model:

- **Genus factored over a list.** `blautia species wexlerae , faecis and
  massiliensis` — one genus distributed across three epithets
  (*Blautia faecis*, *Blautia wexlerae*). Likewise `agathobacter species faecis
  and sp.` for *Roseburia faecis*, whose node already carries the alias
  *Agathobacter faecis*.
- **Parenthetical abbreviation.** `catabacter (ca.) hongkongensis`,
  `gemmiger (ge.) formicilis`, `phascolarctobacterium (pb.) faecium`,
  `vibrio (vi.) phage pyd38 a`.
- **Short epithets.** `cl. sp cag 273`, `su. sp apc924` — skipped by the
  epithet-anchored matcher, which requires ≥5 characters.

The matcher was **not** loosened further to absorb these. It understates by
design, and the residual is documented instead. Chasing the last seven would be
tuning the instrument to flatter the result.

> **SUPERSEDED 2026-09-17 — the residual was absorbed, and "tuning to flatter the
> result" turned out to be the wrong worry.** Each construction above is a
> *deterministic* fact about how the paper writes a name, not a threshold to be
> relaxed, so three tiers now handle them (`genus_factored`, `gloss_gap`,
> `defined_abbrev`) — the last two resolved against the **paper's own inline
> abbreviation definitions**, and applied only to the node's own genus rather
> than as a global rewrite (this paper defines `sl.` = *Salmonella* while also
> using `sl.` for *Slackia*, so rewriting everywhere would invent taxa).
>
> The real hazard was the opposite of flattery, and it bit on the first attempt:
> a ±400-character proximity version of `genus_factored` **cleared two taxa it
> should not have** — `Roseburia faecis` off a sentence attributing "faecis" to
> *Blautia* and *Agathobacter*, and `Vibrio phage` off the bare word "phage".
> Requiring the genus to *bind* the epithet (same sentence, ≤240 chars, no other
> known genus in between, against a closed vocabulary from the graph's own
> labels) rejects both. `Roseburia faecis` is then re-cleared on sound evidence:
> its node carries the NCBI synonym *Agathobacter faecis* and the paper writes
> "agathobacter species faecis".
>
> **All 12 flagged absences are matcher false negatives. Not one is a
> fabrication.** 10 by tier, 2 by hand (`su. sp. apc924`, carrying the unique
> strain code, and the XIII case retracted below) — those 2 get no tier, because
> a one-off does not justify a rule that could fire wrongly elsewhere.

### ~~One confirmed extraction error~~ — RETRACTED 2026-09-17. There is none.

> **The claim below was wrong, and the way it was wrong is the lesson.** The
> "verbatim" quote that proved it was **truncated exactly one clause before the
> disproof.** Whole-word counts over the source paper: `xii` **1**, `xiii` **1**.
> The full sentence is:
>
> > "...showed positive correlations with groups, whereas clostridiales incerte
> > sedis xii ( r = −0.2625, p = 0.0396) **and xiii ( r = −0.2113, p = 0.0495)**
> > were negatively correlated with groups ( figure 3a )."
>
> *Clostridiales incerte sedis XIII* is reported by the paper, **with its own
> correlation coefficient and its own p-value**, in a prefix-factored
> enumeration — the same construction as the *Bacteroides* list below, with a
> roman numeral in place of a species epithet. The extractor read it correctly.
>
> This repo already knew to distrust supplied quotes: *"several 'verbatim'
> quotes supplied by adjudicators were paraphrases — 7 of 36 edge-level miss
> claims failed the verbatim check"* (SESSION_LOG, 2026-09-16). This one passed
> a verbatim check and was still wrong, because **a true quote can mislead by
> where it stops.** Machine-checking that a quote occurs is not enough; the test
> has to cover the span that would refute the claim. A whole-word count of the
> disputed token — two lines of code — settles it and cannot be truncated.

**Zero confirmed extraction errors in 2,301 claims.** The graph's blast radius
from this instrument is nil: the one Parkinson's placeholder edge it was going to
cost is correct as extracted.

## Two instruments were wrong before the graph was

Consistent with this repo's history, both errors found were in the audit, not the
thing audited.

**My own matcher, caught by the positive control.** The first join scored `own`
at 79.7% — impossible, since those observations name the taxon in a sentence. It
fed `audit_direction_witness.taxon_matchers()`'s forms, which are
`norm_surface`'d (**all** non-alphanumerics stripped, `akkermansiamuciniphila`),
into unstripped text. Every multi-word taxon was forced absent. Squashing needle
and haystack symmetrically fixed it. **The fourth time in this repo an instrument
was weaker than what it audited, and the first caught by a built-in control
rather than a spot-check.** A positive control with an a-priori-known answer is
cheap here and should be standard.

**A subagent's adjudication, overturned by one line of code.** A Haiku agent read
the ten original misses and returned a confident verdict that
`Lachnospiraceae_UCG-001` was a genuine fabrication — the paper "mentions UCG-004
extensively but not UCG-001". The paper contains **both**:

> "g_ haemophilus , g_ lachnospiraceae _ucg-001, g_ parasutterella"

It is written with a space before the underscore, which the agent's manual
searches missed. Its other nine verdicts (abbreviation and whitespace artifacts)
were correct and useful for diagnosis. **`Don't trust a subagent's judgement call
where a deterministic test exists` held again** — the agent was right about the
mechanism and wrong about the one case that mattered.

## Limits

- **Coverage.** 25.8% of observations are unscoreable here because their full
  text is gitignored. The 98.4% figure is over the 436 scoreable of 580; the
  other 144 are unknown, and a run on a machine with `MAIN_DATA.json` would close
  that. This is the single cheapest extension of this result.
- **Mention is not support.** The direction of every one of these 429 edges
  remains unverified by any instrument in this repo. They stay `provisional`,
  which is the correct tier.
- **The null's floor.** p = 0.0099 is 1/101, the smallest value 100 permutations
  can produce; the true gap is 84.6 points against a null SD of 0.8, i.e. ~105
  SDs, so the p-value understates a very large separation rather than sitting
  near a threshold.
- **No new agreement claim.** Nothing here moves Disbiome/Peryton, and it must
  not be reported as doing so.

## Artifacts

`verify_taxon_mentions.py`, `verify_mentions_null.py`, `silent_edge_mentions.py`;
outputs `taxon_mentions.json`, `taxon_mentions_null.json`,
`silent_edge_mentions.json`.
