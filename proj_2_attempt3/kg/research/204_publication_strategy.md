# Publication Strategy Review — KnightLabV2 microbe–disease KG

*Skeptical strategy review. Drafted 2026-09-18, completed 2026-09-19. All six sections written.
Venue facts marked ✅ were fetched and confirmed on 2026-09-19; those marked ⚠️ are from prior
knowledge and must be verified against the venue's own page before acting on them.*

## Contents

1. What is the actual contribution? — ranked, honestly
2. Venue survey
3. Gap list: what is missing to be publishable
4. Related-work landscape
5. The reviewer-2 attack
6. Recommendation (ranked)

---

## 1. What is the actual contribution?

### 1.0 The one-sentence version

The defensible contribution is **a measurement result about how microbe–disease
literature-mining resources are validated** — specifically, that the field's standard
validation move (report agreement with Disbiome/HMDAD/Peryton) is confounded by shared source
papers, and that decomposing it splits a single 73% into ~90% reading fidelity and ~55%
cross-literature reproducibility. The KG is the *instrument* that made that measurement
possible. It is not, on its own, a competitive resource.

### 1.1 Ranking, most defensible first

**(1) The measurement methodology — the independence decomposition. STRONGEST.**

This is the only thing here that is (a) novel, (b) fully measured with stated uncertainty, and
(c) *bad news for other people's published numbers*, which is what makes it worth a reviewer's
attention.

What is actually in hand (`FINDINGS_independence.md`, `check_independence.py`):

- 43/272 papers shared with Disbiome, 24/272 with Peryton — a 9–16% *paper* overlap producing a
  **50.6% / 44.9% overlap in decisive pairs**, because the shared papers are the heavily-cited
  ones. That mechanism — small paper overlap, large evidence overlap — is the generalizable
  point and applies to *every* such validation in this subfield.
- Agreement splits 87.5% vs 58.1% (Disbiome) and 96.8% vs 52.6% (Peryton) on that line.
- The inference is guarded properly: taxon-block permutation p=0.0001 (preserves within-taxon
  correlation), taxon cluster bootstrap CIs excluding zero, survives stratification on evidence
  count, and the disease confound is handled by a within-Parkinson's replication where two
  independent curations land on **59.0%** and **59.6%**.
- A counter-example is logged rather than buried: in MS the gap is absent (72.7 vs 70.6, n=39)
  at an MDE (28.7) that could have seen a Parkinson's-sized effect. That single paragraph is
  worth more to a skeptical reviewer than the headline, because it demonstrates the authors
  looked for their own disconfirmation.

Why it is the strongest card: per the project's own prior-art review
(`research/04_existing_microbiome_kgs.md`, summarized in `00_SYNTHESIS.md` §8), **no published
direction-agreement rate between two independent curated microbe–disease databases could be
found**; the field uses HMDAD/Disbiome as *training targets* for link prediction, not as
mutually-checking references. If that holds under a proper literature sweep, this is a genuine
first, and it is cheap to defend because it requires no claim that our graph is good.

Caveat that must survive into the paper: the result is **established in Parkinson's, not
corpus-wide**. Anyone writing this up who lets it become "literature-mined KGs agree with
curated DBs at only ~55%" has overclaimed, and MS is the counter-evidence a reviewer will find.

**(2) The negative/limits result on inter-study disagreement. STRONG, but as a section, not a paper.**

24–25 study-design variables tested as explanations for directional discordance; 24–25 nulls,
each with an explicit MDE, plus the variance decomposition that explains *why*: paper-level SD
of discordance is **3.4pp** on a 27.6% base (cluster-bootstrap CI [0.0, 6.0], including zero)
against MDEs of 4–7pp (`FINDINGS_paper_discordance.md`). The conclusion the project draws —
"this corpus cannot answer whether kit or pipeline drives disagreement" rather than "they don't"
— is the correct one and is rarer in this literature than it should be.

Two things blunt it as a standalone paper:
- `00_SYNTHESIS.md` §6 concedes the null was **guaranteed by sample size before any analysis
  ran** (Cochrane's ~10 studies per moderator; our edges have 2–4 per side). A reviewer who
  reads that will say the experiment was underpowered by construction, and they will be right.
  It is publishable as *"here is the power floor for this design, computed"* — not as
  *"study design does not explain disagreement"*.
- The related finding that ~83–85% of discordance variance is **edge structure, not paper
  identity** is the more interesting half and is a good argument for the keep-contested-edges
  design decision. Lead with that framing, not with the 24 nulls.

**(3) The extraction method. WEAK as a contribution, necessary as a methods section.**

Nothing about closed-schema + GBNF-grammar-constrained decoding + a local quantized 27B is novel
in 2026. The honest framing is *engineering discipline*: a deliberately narrow schema justified
by a documented prior failure in the same lab (open-schema BioBERT RE at **6% precision, 88%
FPs**, with the diagnosis that process–process relations were 0% precise and generic predicates
contributed another 50% of errors). That lineage is a genuinely good motivating story and almost
nobody publishes their own failed baseline. But it motivates a design; it is not itself a result.

The comparator that hurts: **MINERVA** (Briefings in Bioinformatics 2025) fine-tuned
Biomistral-7B over 129,719 papers → 66,400 edges at RE F1 0.884. Against 271 papers and 2,008
edges, "we built an extractor" is not a competitive claim. Do not make it.

**(4) The resource / KG itself. WEAKEST. Be blunt about this.**

- **Scale is the problem.** 883 taxa / 40 diseases / 2,008 edges from 271 papers, against
  Disbiome's ~10.9k experiments and MINERVA's 66,400 edges from ~130k papers. A resource paper
  competes primarily on coverage, and this loses by 1–2 orders of magnitude.
- **79% of edges are `provisional`** (single paper or a `discriminating` taxon), and that tier
  agrees with the curated DBs at only 66.1% / 61.9%. The project is admirably honest about
  this; a resource reviewer will read it as "four fifths of the resource is a coin-flip-plus".
- **Direction only, no effect size.** Defensible on grounds that source statistics are
  incommensurable (LEfSe LDA vs fold-change vs p-value), but it forecloses meta-analysis
  outright — Q/τ²/I² mechanically require per-study effect size and variance
  (`00_SYNTHESIS.md` §6). A resource that cannot be meta-analyzed has a narrower audience.
- **The disease half is thin.** 28/40 labels resolve to MONDO; MCI correctly resolves to
  nothing. That is handled correctly but it is 40 diseases.
- **No sustainability story yet** — no API, no versioned release, no update commitment. See §3.

The one genuinely distinctive resource property is the **calibration layer**: every edge carries
a confidence tier whose agreement rate with two external databases is *measured*, monotone in
both (93.8/93.3 → 77.8/83.3 → 66.1/61.9), with the cut chosen on Disbiome and Peryton held out
as the out-of-sample check. Neither Disbiome nor Peryton describes any disagreement-resolution
or confidence mechanism at all. That is a real, small, defensible resource contribution — but
it is one table, not a paper.

**(5) "A negative-results paper about reproducibility in microbiome literature." DO NOT lead with this.**

Tempting, and the numbers look striking, but the field is already there and got there with
better instruments:

- Gibbons et al. 2018 (PLOS Comput Biol): 681/1,021 OTUs (67%) differed significantly between
  two studies' **healthy-control cohorts alone**.
- Tierney et al. 2022 (PLOS Biology): ~1 in 3 taxa show sign inconsistency across 581
  associations / 15 cohorts; >90% of T1D/T2D associations not robust to confounder choice.
- Duvallet et al. 2017 (Nat Commun 8:1784): vote counting across re-processed raw data,
  explicitly declining to call associations for conditions with <4 datasets.

Those are re-analyses of **raw sequencing data**. Ours is a re-reading of **published text**, so
our discordance number cannot separate "the field disagrees" from "papers report things in ways
a reader cannot reconcile". Claiming a reproducibility finding invites the fatal question in §5.
Use these papers as *base rates that make our numbers unsurprising* — defensive framing, which
is exactly what §7 of `00_SYNTHESIS.md` proposes — not as a claim of our own.

### 1.2 What the paper should therefore claim

> We build a deliberately narrow-schema, literature-mined microbe–disease graph (883 taxa, 40
> diseases, 2,008 directional edges, 271 papers), and use it as an instrument to audit how such
> resources are validated. We show that agreement with hand-curated databases — the field's
> standard validation — is confounded by source-paper overlap: a 9–16% overlap in papers
> produces a ~50% overlap in the evidence being compared, and agreement splits 87.5%/96.8%
> (shared source) vs 58.1%/52.6% (disjoint), replicating within Parkinson's across two
> independent curations at 59.0% and 59.6%. We report reading fidelity measured against the
> papers' own sentences (≥86.6%, CI [81.7, 91.3]) separately from cross-literature
> reproducibility, and show that these are different quantities that the pooled number blends.

That claim is supportable with what is on disk today. Claims about the KG being a better or
larger resource are not.

### 1.3 Which of the strong numbers are load-bearing, and their weak points

| number | what it rests on | the attack |
|---|---|---|
| independence split (87.5/58.1, 96.8/52.6) | PMID∪DOI∪title join; permutation + cluster bootstrap | disease confound — answered within Parkinson's; MS counter-example is public |
| reading fidelity ≥86.6% [81.7, 91.3] | 181/209 observations, 122 papers, double-adjudicated (25/28 exact) | adjudicators are the same pipeline/authors; no external blind annotator |
| paper-level recall ≥96.1% (99.6% confirmable-only) | gate-aware audit of 42 zero-yield papers | instrument "can confirm a miss but cannot refute one" — stated, but it is a one-sided bound |
| edge-level recall ≤98.1% [96.4, 99.5] | 24 papers / 95 candidates sampled from 378 | candidate generator has ~75% paper-level recall; explicitly an upper bound |
| calibration monotonicity | 2 external DBs, cuts fit on one, checked on the other | tier sizes are lopsided (75 / 135 / 1,607) |
| 24–25 nulls with MDEs | `FINDINGS_paper_discordance.md`, `FINDINGS_variable_coverage.md` | underpowered by construction; the project says so itself |
| F1 0.680 vs in-house gold | flawed human gold under audit | **do not report as accuracy** — see §3.1 |

---

## 2. Venue survey

**How to read this section.** The choice of venue follows directly from §1: the contribution is
a *measurement about validation practice*, with a small resource attached. That makes
resource-first venues (NAR DB Issue, Scientific Data) a poor primary fit and
methodology/evaluation venues a good one — the exact opposite of the instinct to "publish the
KG".

**Provenance of the facts below.** Items marked ✅ were fetched and confirmed on 2026-09-19.
Items marked ⚠️ are from prior knowledge and are the kind of fact that drifts (deadlines,
article-type names, word limits, turnaround). **Verify every ⚠️ against the journal's own page
before acting.** Review timelines are median expectations from field experience, not published
guarantees — no journal in this list commits to one. Per this document's convention, APCs are
described as present/absent, not priced.

### 2.0 The short version

| venue | scope fit | verdict |
|---|---|---|
| **Database (Oxford)** | curation standards + resource + "objective reviews of complementary databases" | **best journal fit — primary candidate** |
| **JBI** | evaluation methodology for biomedical IE | **strong second** |
| **BioNLP workshop (ACL)** | biomedical IE evaluation, small-corpus friendly | **best fast/low-risk outlet** |
| **Briefings in Bioinformatics** | comparative/benchmark methodology; MINERVA's home | plausible, high bar, reviewer pool knows the comparator |
| **BMC Bioinformatics** | catch-all methods + resource | safe fallback, low prestige |
| **Scientific Data** | Data Descriptor for the graph | achievable, but banks the resource and **discards the contribution** |
| **ACL/EMNLP Findings** | NLP evaluation confounds | possible; reviewers will want an NLP-general claim |
| **Bioinformatics (OUP)** | Application Note | resource is too small to carry an AN on utility |
| **NAR Database Issue** | major maintained public database | **no — scale and sustainability bar not met** |
| **mSystems / Microbiome / Gut Microbes** | microbiome biology | **no — they will demand the sequencing validation we don't have** |
| **JAMIA** | clinical/health informatics | scope mismatch (no clinical application) |

### 2.1 Database: The Journal of Biological Databases and Curation (OUP) — PRIMARY

- **Scope fit: high, and for a specific reason.** ✅ Confirmed today, its accepted article types
  include not only database descriptions and updates but explicitly *"articles on curation
  standards and annotation best practices"*, *"objective reviews of complementary databases"*,
  and *"perspective manuscripts on novel approaches and state-of-the-art comparisons"*
  (<https://academic.oup.com/database/pages/About>). The independence decomposition **is** a
  curation-standards result: it says the field's standard way of validating a curated resource
  against another curated resource is confounded by shared source papers. There is no other
  journal in this list whose stated remit names that contribution.
- **Requirements.** A described resource must be publicly accessible; the KG is already published
  at <https://www.mohakprakash.com/KnightLabV2/>. ⚠️ Expect the usual resource hygiene demands:
  a stable URL (a personal domain is a weakness — see §3), a versioned archival deposit
  (Zenodo/figshare DOI), a documented schema, and a licence. ⚠️ An API is *not* formally required
  here the way it effectively is at NAR.
- **OA status:** ✅ fully open access, CC-BY, automatic PMC/UKPMC deposit. APC-bearing; waiver and
  read-and-publish routes exist.
- **Timeline:** ⚠️ typically ~1–3 months to first decision, rolling submission (no annual window).
- **The catch.** Its audience is curators, who will be the most alert readers possible for
  §5.1 (no blind human annotator) and §5.3 C1 (does the overlap confound generalize?). Do not
  submit here without Experiment 1 from §5.5.

### 2.2 Journal of Biomedical Informatics (Elsevier) — STRONG SECOND

- **Scope fit: high.** JBI's core is methods and *evaluation* of biomedical information systems;
  papers whose contribution is "this standard evaluation practice is biased, here is the
  decomposition" are squarely in remit. It is also comfortable with modest corpora when the
  contribution is methodological.
- **Requirements.** ⚠️ Software/data availability statement; no maintained-resource obligation.
  A "Methodological Review" or standard research article both fit.
- **OA status:** ⚠️ hybrid (subscription with an optional OA route); green OA permitted after
  embargo.
- **Timeline:** ⚠️ ~2–4 months to first decision; desk-reject risk is real if the paper reads as
  a domain resource paper rather than an informatics-methods paper. Frame accordingly: the title
  should contain "evaluation" or "validation", not "knowledge graph of".
- **Why it ranks below Database:** no explicit curation-standards remit, so the paper competes
  against general biomedical-NLP methods work where n=271 looks small.

### 2.3 BioNLP workshop (ACL) — BEST FAST OUTLET

- ✅ Confirmed today: BioNLP **2026** ran 3–4 July 2026, co-located with **ACL 2026 in San
  Diego** — i.e. the most recent edition has already passed, and it was held in this lab's own
  city. Submission was 20 April 2026, notification 8 May 2026 (<https://aclweb.org/aclwiki/BioNLP_Workshop>).
  ⚠️ The next realistic cycle is BioNLP 2027 at ACL 2027, with a spring-2027 deadline — confirm
  once the workshop site is up.
- **Format:** ✅ 8-page full papers or 4-page short papers, unlimited references, +1 page
  camera-ready.
- **Scope fit: high.** BioNLP's audience routinely publishes "our evaluation set is contaminated
  / our benchmark is confounded" results, and small corpora are normal there. The
  shared-vs-disjoint-source decomposition would be well received and *well understood*.
- **OA status:** ACL Anthology, free to read, **no APC**. Archival.
- **Timeline:** ~3 weeks from deadline to notification — by far the fastest feedback in this list.
- **Strategic use.** This is the right place to *de-risk*: get the independence result reviewed by
  people competent to break it, then write the extended journal version. ⚠️ Check the workshop's
  and the target journal's dual-submission/extension policies — extending an archival workshop
  paper into a journal article is normal practice in this community, but the journal must be told.

### 2.4 Briefings in Bioinformatics (OUP)

- **Scope fit: moderate-to-good.** BiB publishes comparative evaluations and methodological
  critiques as well as reviews, and it is **the venue that published MINERVA (2025)** — the
  closest method sibling (see §4). That cuts both ways: the reviewer pool will know the
  comparator cold and will ask directly why a 271-paper graph is worth their attention. The
  answer has to be the measurement, delivered in the abstract's first two sentences.
- **Requirements.** ⚠️ No maintained-resource obligation; data/code availability expected.
- **OA status:** ⚠️ hybrid.
- **Timeline:** ⚠️ ~1–3 months to first decision; high desk-reject rate.
- **Verdict:** a reasonable stretch target *after* Experiment 2 in §5.5 — a cross-resource
  replication of the overlap confound (including on MINERVA itself) is exactly the kind of paper
  BiB likes, and it would be published in the journal that published the resource it critiques.

### 2.5 BMC Bioinformatics

- **Scope fit: good but undiscriminating.** Accepts methods, software and database papers; a
  combined "resource + validation methodology" paper fits without contortion.
- **Requirements.** ⚠️ Code/data availability with a permanent identifier; software papers expect
  a working tool.
- **OA status:** fully OA, APC-bearing.
- **Timeline:** ⚠️ ~2–4 months, sometimes longer due to reviewer-sourcing.
- **Verdict:** the safe fallback that will almost certainly publish the work as-is. Costs nothing
  strategically except that the measurement result will be read by fewer of the people whose
  practice it is about.

### 2.6 Scientific Data (Nature Portfolio)

- **Scope fit: mismatched to the contribution, well matched to the artifact.** ⚠️ Data Descriptors
  describe a dataset and its technical validation; they explicitly **do not** carry new scientific
  claims or hypothesis tests. The independence decomposition is a scientific claim, so it would
  have to be cut down to a validation subsection — which inverts §1's ranking.
- **Requirements.** ⚠️ Deposit in an approved repository (figshare/Zenodo/Dryad are accepted for
  data without a domain repository), a full technical-validation section, machine-readable
  metadata. The project is unusually well equipped for the validation section — fidelity, both
  recall bounds, the calibration tiers, and the external-agreement decomposition all belong
  there.
- **OA status:** fully OA, APC-bearing. **Timeline:** ⚠️ ~2–5 months.
- **Verdict:** the right *companion* to a methods paper (bank the graph as a citable dataset with
  a DOI), not the right primary.

### 2.7 Bioinformatics (OUP)

- ✅ Confirmed today: article types are standard research, methods, critical reviews/perspectives,
  database and web-server articles; the journal is fully OA with an APC and discount routes
  (<https://academic.oup.com/nar/pages/General_Instructions> for the NAR family; Bioinformatics
  has its own equivalent page — ⚠️ verify separately).
- **Scope fit: weak.** An Application Note must carry a resource other people will use, judged
  on utility and novelty of the tool. At 883 taxa / 2,008 edges against Disbiome and MINERVA,
  utility is the losing axis (§1.1(4)). A full Original Paper needs methodological novelty in
  the extraction, which §1.1(3) says we do not have.
- **Verdict:** skip unless the RAG/QA layer (`build_rag.py`, `201_kg_grounded_qa.md`) matures into
  a tool with a real user story.

### 2.8 NAR Database Issue — NO

- ✅ NAR confirms "Database articles" as a category and is fully OA with an APC
  (<https://academic.oup.com/nar/pages/General_Instructions>). ⚠️ The Database Issue itself runs
  on an annual cycle with a mid-year submission window for the January issue, and the standing
  expectations are: the resource must be **freely available without registration**, must be
  **actively maintained**, and update papers must demonstrate substantial change since the last.
  Verify the current window on the dedicated Database Issue guidelines page — the URLs this
  session tried (`/nar/pages/database_issue`, `/nar/pages/Database_Issue_Information_For_Authors`)
  both 404, so the page has moved.
- **Why not:** this is the most competitive resource venue in the field and it selects for
  coverage and longevity — precisely the two axes where this project is weakest. There is no
  update commitment, no API, no institutional hosting, and the graph is 1–2 orders of magnitude
  smaller than the incumbents it would sit beside. Submitting is a predictable desk reject and
  spends a year of cycle time.
- **When it becomes reachable:** corpus at 10³–10⁴ papers (see `203_corpus_scaling.md`), an API,
  a named maintainer, and institutional hosting. That is a different project phase, not a revision.

### 2.9 Microbiome (BMC), mSystems (ASM), Gut Microbes (T&F) — NO, for one shared reason

All three are microbiology venues whose reviewers are the people who generated the underlying
data. ⚠️ (mSystems' article-type page returned 403 today; ASM journals moved fully OA in recent
years — verify.) Their reviewers will ask the §5.3 C2 question immediately and in its strongest
form: *you claim the literature disagrees with itself at ~45%; show me that in sequencing data.*
The project has **no** non-literature validation (see `202_sequencing_validation.md`), so the
question is unanswerable today.

- **Microbiome** additionally expects a biological advance; a graph with no new biology is a poor
  match regardless of its engineering.
- **mSystems** is the most reachable of the three — it publishes computational resources and
  methods, and a ⚠️ "Resource Report"-style article type exists in the ASM family — but the same
  objection applies.
- **Gut Microbes** is narrower still (gut-focused, and the graph spans body sites).

**These become the right venues if and only if the sequencing validation happens** — at which
point the paper is a different and much stronger one, because "literature-mined direction agrees
with re-analyzed raw data at X%" is a claim no comparator in §4 has made.

### 2.10 ACL / EMNLP Findings

- **Scope fit: conditional.** ⚠️ Submission runs through ACL Rolling Review (ARR): monthly-ish
  cycles, reviews returned in roughly 6–8 weeks, then commitment to a venue. Findings is archival,
  free to publish, no APC, in the ACL Anthology.
- **The condition.** NLP reviewers will ask what the paper says about NLP, not about microbiome
  curation. The generalizable framing exists — *"when an IE system is evaluated against a curated
  KB, the KB's source documents overlap the system's input corpus, and the overlap in evaluable
  facts is several times the overlap in documents"* — and that is a real, transferable evaluation
  confound with obvious analogues in other domains. But making that claim stick needs Experiment 2
  from §5.5 (replication on another resource) or the reviewers will read it as a domain case study.
- **Verdict:** viable, but BioNLP is the better-matched ACL-family outlet and is far cheaper to
  reach.

### 2.11 JAMIA

- **Scope fit: poor.** JAMIA centres on clinical and health informatics with an application or
  implementation angle. Nothing here touches patients, clinical workflow, or EHR data.
  ⚠️ *JAMIA Open* is more permissive, but the mismatch is scope, not selectivity.
- **Verdict:** skip.

### 2.12 Two venues not on the brief's list, worth considering

- **GigaScience / GigaByte (OUP).** ⚠️ Built for data- and resource-heavy work with strong
  reproducibility requirements (executable environments, deposited data). GigaByte in particular
  publishes short resource papers quickly. A good alternative to Scientific Data for banking the
  artifact, with a more computational readership.
- **PLOS Computational Biology.** ⚠️ Publishes methodological critiques of computational practice,
  and it is where Gibbons et al. 2018 — one of this project's own base-rate citations — appeared.
  Higher bar than Database, but the "your validation is confounded" genre fits its remit, and a
  reviewer pool that already accepted the healthy-control discordance result is a receptive
  audience for the disjoint-literature number.

---

## 3. Gap list: what is missing to be publishable

*Ordered by whether the gap blocks submission, blocks acceptance, or merely costs a revision.
Three specific questions were asked and are answered first.*

### 3.1 Is the flawed gold standard fatal? — **No. It is reframable, and the cleanest move is to drop it.**

**Why it is not fatal:** nothing in §1.2's claim depends on it. The independence decomposition
runs on curated-database joins; reading fidelity runs on the papers' own sentences; both recall
bounds run on source text. The in-house gold contributes exactly one number — F1 0.680 — and that
number's only honest use is as an illustration that gold-standard agreement is not accuracy. The
project can delete it from the manuscript and lose nothing load-bearing.

**Three ways to handle it, best first:**

1. **Report it as a methodological finding with external support.** The claim is: *in this task
   niche, extractor evaluation against a human gold is unreliable, and the literature independently
   shows it.* The support is strong and mostly external — the 2023 PLM comparison
   ([PMC10357883](https://pmc.ncbi.nlm.nih.gov/articles/PMC10357883/)) had to re-annotate **178
   mislabeled instances** in its own gold corpus before benchmarking on it, and our own numbers go
   the same direction (162/250 papers with blank taxa columns; a thorough re-annotation surfacing
   72 taxa the humans missed, of which only 1 never appears in the source text at all; a 15-paper
   benchmark moving 0.64 → 0.84 once corrected). **That is the framing to use**, and it motivates
   the witness-based instruments as the response rather than presenting them as arbitrary.
2. **Drop it entirely.** Costs nothing; loses the rhetorical setup above.
3. **Wait for the annotator's error-rate report** (`CLAUDE.md` records the annotator expects to
   report an error rate rather than a corrected set). If that lands, cite it; it converts anecdote
   into a measured figure. ⚠️ **Do not make the submission timeline depend on it** — it is outside
   this project's control.

**What IS fatal, and is a different thing:** the absence of *any* sound human evaluation. The gold
being broken is survivable; having no human reference at all is what §5.1 says a reviewer will
refuse. The fix is the new blind two-annotator study, not a repair of the old gold. **Do not spend
person-time repairing the old gold** — it was built for a different (pre-KG) task, and repairing it
yields a reference for a task the paper no longer claims.

### 3.2 Does a resource paper require an API and a maintenance commitment? — **Depends entirely on which venue, and this decides §6.**

| venue class | API | maintenance commitment | archival DOI | licence |
|---|---|---|---|---|
| **NAR Database Issue** | effectively yes | **yes, explicit** — this is the bar this project fails | yes | yes |
| **Database (Oxford)** | no | ⚠️ not formally required; reviewers will ask about stability | **yes, in practice** | yes |
| **Scientific Data / GigaByte** | no | no | **yes, mandatory** (approved repository) | yes |
| **JBI / BMC Bioinformatics** | no | no | availability statement with a permanent identifier | yes |
| **BioNLP / Findings** | no | no | a public link suffices | yes |

**What actually exists today:** a published static explorer at a **personal domain**
(<https://www.mohakprakash.com/KnightLabV2/>), a `graph.json`, a `rag_corpus.jsonl`, and code in a
repo. **No licence, no Zenodo/figshare DOI, no versioned release, no schema document, no named
maintainer, no institutional hosting.**

**The minimum bar for every venue except NAR and the microbiology journals** is therefore small and
entirely unblocked:

- a **Zenodo (or figshare) deposit** of the frozen `graph.json` + `rag_corpus.jsonl` + the
  provenance tables, with a DOI and a version tag — **~0.5 person-day**;
- an explicit **licence** on data and code (CC-BY for data, a permissive licence for code) —
  hours, but it needs the lab's/PI's sign-off, so start it early;
- a one-page **schema document** for `graph.json` (node/edge fields, tier definitions, what
  `provisional` and `discriminating` mean) — ~0.5 person-day;
- ⚠️ a decision on **hosting**: a personal domain is a legitimate reviewer concern for a resource
  paper. Either move to institutional hosting or state plainly that the archival Zenodo copy is the
  citable artifact and the site is a convenience viewer. The second is honest and cheap.

**An API is not worth building** for any venue this project should realistically target. If the
target ever becomes NAR, the API is the least of the missing pieces (see §2.8).

### 3.3 Is non-literature (raw sequencing) validation required, or nice to have? — **Required for the microbiology journals; not required anywhere this project should submit; and it may be infeasible regardless.**

`202_sequencing_validation.md` is the authority here and it is unusually clear-eyed:

- **A defensible design exists**: pooled *sign* agreement between graph direction and a
  pre-registered DA coefficient (ANCOM-BC2 primary; ALDEx2 and MaAsLin 3 as pre-registered
  sensitivity), over provenance-screened independent public stool cohorts, against a within-cohort
  label-permutation null, one primary endpoint and ≤8 pre-specified secondaries.
- **The naive version is worse than nothing.** Scoring "not significant in public data" as
  disagreement measures DA power, not graph correctness — MaAsLin 3's own benchmark reports recall
  **0.18 at 50 samples** ([Nickols et al., *Nat Methods* 23:554–564, 2026](https://doi.org/10.1038/s41592-025-02923-9)),
  and our source cohorts have a **median of 40 cases**. It would produce a spuriously damning number
  that looks like a finding.
- **The disease mix is the binding constraint and it cuts against feasibility.** 37 of 40 disease
  nodes are neurological; the top five diseases hold 60% of edges. Public microbiome repositories
  are deep in CRC, IBD, T2D, obesity and cirrhosis — the diseases this graph does *not* contain.
- **There is a mandatory independence screen that could end it in a day.** A public cohort
  deposited by one of our 271 papers is the *same study one layer down*; validating against it
  measures reanalysis stability, not replication — the identical error as the Disbiome/Peryton
  confound, rediscovered in a new medium. The kill criterion is already written: fewer than ~8
  independent cohorts or ~600 independent case+control samples across the top five diseases → stop.

**Therefore:** treat it as **nice to have, with a cheap falsifiable first step**. Spend ~1 person-day
on the provenance screen described in `202_sequencing_validation.md` §5.5 *before* the submission
plan hardens, because the answer changes which venues exist. If it survives, a whole class of
higher-impact venues (§2.9) opens up and the §5.3 C2 objection becomes answerable. If it dies,
that is itself worth a paragraph in the limitations and the plan proceeds unchanged.

### 3.4 The gaps that block acceptance (not submission)

1. **No blind human evaluation.** §5.1. ~3–5 person-days. **Highest priority item in the project.**
2. **The overlap confound is measured on one corpus only.** §5.3 C1. ~2–4 person-days if a
   comparator publishes per-edge PMIDs. This is what separates "a case study" from "a result about
   validation practice", and it is what makes §2.4 (Briefings) and §2.10 (Findings) reachable.
3. **Fidelity is measured on ~7% of observations, non-randomly selected.** §5.2. ~3–5 person-days to
   extend to multi-taxon sentences. Will likely lower a headline number, which is a reason to do it
   first rather than last.
4. **No systematic prior-art search outside the microbiome subfield.** §4.3. ~1 person-day. The
   novelty claim is currently supported by two web-search sweeps; a reviewer citing a concordance
   study from variant curation or a contamination paper from NLP would be embarrassing and cheap
   to prevent.

### 3.5 The housekeeping gaps — small, unglamorous, and genuinely blocking

**The project's own numbers disagree across its own documents.** This is the most likely source of
an avoidable reviewer complaint, and it is visible in three files today:

| quantity | `kg/README.md` | root `CLAUDE.md` | `research/202_…` |
|---|---|---|---|
| edges | 2,034 | 2,008 | 2,008 |
| taxa | 925 | 883 | 883 |
| contributing papers | 272 | 271 | 271 |
| containment links | 723 | 727 | — |
| contested | 217 | 217 | 220 |

These are snapshots from different corpus revisions, and internally the project knows it (the
`SESSION_LOG.md` discipline is good). But a manuscript cannot ship them. Required:

- **Freeze one snapshot**, regenerate every figure and number from it, and give it a version tag
  and a DOI (§3.2). Every number in the paper must trace to that one file. ~1–2 person-days
  including a consistency sweep across the FINDINGS docs that will be cited.
- **Reconcile 217 vs 220 contested** specifically — it is the number attached to the design decision
  the paper defends, so it is the worst one to be inconsistent about.
- **A reproducibility path.** ⚠️ Several instruments are blocked in some environments (NCBI taxdump
  unreachable; the join for the independence result runs on a replay taxonomy cache rather than the
  live taxdump — `FINDINGS_independence.md` "Limits"). That caveat is stated honestly in the repo
  and **must** appear in the manuscript, together with either a shipped cache or a documented way
  to rebuild. A reviewer who cannot rerun the headline analysis will say so.

### 3.6 What is NOT a gap (do not spend time here)

- **More GPU extraction / a 10× corpus.** `203_corpus_scaling.md` §5 says 10× papers does not give
  a 10× better test of the discordance question, for arithmetic reasons. It also does not help the
  contribution in §1, which is a measurement, not a coverage claim.
- **Embeddings, GNNs, link prediction.** `00_SYNTHESIS.md` §2 is decisive: do not build a GNN, and
  the published AUCs in that literature do not survive a node-disjoint split. None of it is needed
  for this paper.
- **Effect-size extraction.** Genuinely valuable and genuinely a different paper. It unlocks
  meta-analysis (§5.4) but is a pilot-then-corpus-scale effort; it should not gate this submission.
- **Repairing the old gold standard.** §3.1.

---

## 4. Related-work landscape

*The detailed survey already exists in `research/04_existing_microbiome_kgs.md` (12 resources,
with sizes, direction semantics, rank handling and disagreement policy each). This section does
not repeat it. It answers the only question that matters for publication: **has someone already
published this, and if so which half?***

### 4.1 The blunt answer

**Yes for the resource. No, so far as two independent searches found, for the measurement.**

- **The KG is scooped.** MINERVA (*Briefings in Bioinformatics* 26(5), Sept 2025,
  [PMC12454267](https://pmc.ncbi.nlm.nih.gov/articles/PMC12454267/),
  [PubMed 40984703](https://pubmed.ncbi.nlm.nih.gov/40984703/), live at
  <https://minervabio.org/>) is an LLM-built microbe–disease knowledge graph: 129,719 papers →
  66,400 associations, 2,941 microbes, 3,299 diseases, fine-tuned Biomistral-7B-AUG, RE
  F1 0.884. That is the same artifact class at ~30× the edges and ~480× the papers, published a
  year ago, in one of the venues on our list. **Any framing of this project as "we built a
  literature-mined microbe–disease KG" is dead on arrival and must not be written.**
- **The measurement is not scooped.** `04_existing_microbiome_kgs.md` §13 searched specifically
  for a published *direction-agreement rate between two independently curated microbe–disease
  databases* and found none. The one paper that combines Disbiome and Peryton — Zhu et al.,
  *Front. Microbiol.* 2021,
  [10.3389/fmicb.2021.685549](https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2021.685549/full)
  — takes their **union** (11,037 associations after dedup) to build a bigger network for
  topological ranking, and removes exact duplicates without ever asking whether the two sources
  agree on direction where they overlap. Everyone else uses HMDAD/Disbiome as *training targets*
  for link prediction, not as mutually-checking references.

So the positioning is forced: **MINERVA is the resource, this is the audit.**

### 4.2 The five comparators that will appear in any reviewer's response, and the precise difference

| comparator | what it is | how this project differs — precisely |
|---|---|---|
| **MINERVA** (BiB 2025) | LLM KG, 66,400 edges / 129,719 papers, Biomistral-7B-AUG, UMLS CUIs | (a) direction semantics differ — MINERVA's ±  is *promotes/inhibits* (causal), ours is *enriched/depleted* (abundance), so the graphs are **not** directly comparable edge-for-edge; (b) MINERVA **collapses** conflicts (in-paper majority vote → impact-factor-weighted average), we keep all 217 contested edges un-averaged; (c) MINERVA reports RE F1 against held-out annotation, never a recall-or-fidelity audit against the source sentences; (d) UMLS CUIs vs NCBI taxids — not joinable without remapping |
| **MINERVA critique** (BiB, Nov 2025, [PMC12780759](https://pmc.ncbi.nlm.nih.gov/articles/PMC12780759/), [PubMed 41396813](https://pubmed.ncbi.nlm.nih.gov/41396813/)) | independent commentary attacking MINERVA on 5 statistical grounds | **This is external corroboration, cite it prominently.** Its objections 2 (sentence-independence violation) and 3 (impact-factor-as-quality) are *exactly* the design choices this project made in the other direction, arrived at independently and for different reasons. It is rare to have a published critique of the nearest comparator that validates your own design decision |
| **Disbiome** (BMC Microbiol 2018, [PMC5987391](https://pmc.ncbi.nlm.nih.gov/articles/PMC5987391/)) / **Peryton** (NAR Database issue 2021, [49(D1):D1328](https://academic.oup.com/nar/article/49/D1/D1328/5932864)) | hand-curated, direction-carrying, NCBI-resolved, ~8.7k and ~7.98k associations | Neither describes **any** disagreement-resolution mechanism, and neither publishes a curator-accuracy audit — Disbiome's 16-item questionnaire scores the *source study's* reporting quality, not the curator's reading. These are the references our decomposition is *about*, not competitors |
| **HMDAD** (BiB 2017) + the MDA link-prediction literature (GCATCMDA, HGNNTMDA, MAGMDA, DuGEL, 2023–2025) | 483 edges frozen at 2014 literature, used as a CV benchmark at AUC 0.91–0.97 | Different question entirely (link prediction on a static table vs. extraction fidelity against text). `00_SYNTHESIS.md` §2 documents why those AUCs do not survive scrutiny (KATZHMDA, a zero-learning topology index, already scores 0.84–0.86; no node-disjoint split anywhere). **Do not benchmark against them; explain why the comparison is invalid** |
| **Sentence-level microbe–disease RE** — Wang et al., *Sci Rep* 2021 ([s41598-021-83966-8](https://www.nature.com/articles/s41598-021-83966-8), F1≈0.74); the PLM comparison 2023 ([PMC10357883](https://pmc.ncbi.nlm.nih.gov/articles/PMC10357883/), BioLinkBERT 0.804±0.036, GPT-3 0.810±0.025) | the task-level state of the art | Puts published SOTA at ~0.80–0.81 sentence F1. **And note the detail worth citing:** the 2023 paper had to re-annotate 178 mislabeled instances in its own gold corpus before use. That is independent, published evidence that gold-standard defects are endemic in this exact niche — which is how §3 should frame our own flawed gold |

### 4.3 The prior art the project has NOT yet accounted for, and a reviewer will

`04_existing_microbiome_kgs.md` swept the *microbiome* literature thoroughly. The novelty claim
in §1 is scoped to that subfield, and it should stay scoped there, because the **method** —
measuring concordance between two independently curated databases and decomposing it by shared
provenance — is not new in biocuration generally. ⚠️ The following are from general knowledge and
need a proper check before they appear in a manuscript, but a reviewer will produce something
from each bucket:

1. **Clinical-variant curation concordance.** Inter-laboratory concordance studies on
   ClinVar/ClinGen variant classifications are a mature genre and are the closest methodological
   analogue anywhere in biology: two expert groups classify the same entities, concordance is
   reported, and discordance is attributed. If a version of the shared-evidence confound has been
   described there, the paper must cite it and claim the *microbiome-specific* instance, not the
   method.
2. **Database-overlap / redundancy analyses in biocuration** (e.g. pathway and PPI database
   overlap studies). Same shape: small entity overlap, disproportionate evidence overlap.
3. **NLP evaluation contamination.** The general form of our finding — *the evaluation reference
   was built from documents that overlap the system's input, so agreement is inflated on the
   overlap* — is the biomedical-IE instance of train/test or benchmark contamination, a large and
   active NLP literature. For a BioNLP or Findings submission (§2.3, §2.10) this framing is a
   strength: it gives the result a home in an existing conversation. For a *Database*/JBI
   submission it is a risk: a reviewer who knows that literature will say "contamination,
   already known". **Pre-empt it in the introduction**: the novel part is not that overlap
   inflates agreement, it is the *ratio* — a 9–16% document overlap producing a ~50% overlap in
   evaluable facts, because curation preferentially selects heavily-reported papers.
4. **Systematic-review overlap metrics.** ⚠️ Methods exist for quantifying primary-study overlap
   across overlapping reviews (the "corrected covered area" family). That is literally the same
   statistic applied to a different object. Worth a look — if one of those metrics fits, adopting
   it costs nothing and makes the paper look properly situated.

**Recommendation:** before submitting, run one focused search pass over buckets 1–4 (~1 person-day).
The cost of missing a close analogue is a reject; the cost of finding one is a paragraph and a
slightly narrower claim.

### 4.4 The three differences worth putting in an abstract

Out of everything in `04_existing_microbiome_kgs.md` §"What is genuinely novel", only three
survive contact with §1's honesty test:

1. **The decomposition itself** — first reported direction-agreement rate between two
   independently curated microbe–disease databases *and* its split by shared vs. disjoint source
   papers (87.5/58.1 and 96.8/52.6; within-Parkinson's 59.0 and 59.6 from two curations that do
   not know about each other).
2. **Self-audited fidelity and recall against the source sentences.** None of the ten curated
   databases surveyed publishes a recall or fidelity audit of its own curators; MINERVA reports
   extraction F1 but not recall against source text. Ours are ≥86.6% [81.7, 91.3] fidelity,
   ≥96.1% paper-level recall, ≤98.1% [96.4, 99.5] edge-level recall, each with its bound
   direction stated. The *uncertainty budget* is the contribution here, not the point estimates.
3. **Contested edges kept, with a measured justification.** 217 contested pairs preserved rather
   than averaged, supported by the within-paper concordance measurement (related taxa agree 89%
   within a paper vs 54% for unrelated) and the variance decomposition (~85% of discordance
   variance is edge structure, not paper identity). MINERVA does the opposite and its published
   critique attacks precisely that.

Everything else in that list — NCBI-taxid discipline, the refusal log, the scale/method niche —
is good engineering and belongs in the methods, not the abstract.

### 4.5 One framing trap to avoid

Do **not** write "we sit in a niche between hand-curated high-trust and LLM-scaled low-trust."
It is an accurate description and a terrible argument: a reviewer reads "middling on both axes".
The niche is only interesting if the paper's claim is about *measurement* — that this is the
mid-scale graph small enough to audit exhaustively and large enough for the audit to have
statistical power, which is why the decomposition could be done here and not on MINERVA. That is
the same fact, stated as a reason the measurement exists rather than as a market position.

---

## 5. The reviewer-2 attack

*Written before §§2–4 deliberately: if the paper cannot survive these three, the venue
question is moot. Each objection is stated in the form a hostile reviewer would actually
write it, then the best rebuttal available **from what is on disk today**, then an explicit
`CANNOT REBUT` block. The `CANNOT REBUT` blocks are the useful part of this section.*

### 5.1 Objection A — "The validation loop is closed: an LLM pipeline certified by LLM readers"

> *The extractor is a quantized 27B LLM. The reading-fidelity audit that certifies it at
> ≥86.6% adjudicates its residual disagreements with two more LLMs (Opus as orchestrator,
> Haiku as the independent second pass). The edge-recall and zero-yield audits are
> LLM-adjudicated too. The only human reference in the paper is the authors' own gold
> standard, which they themselves report as unreliable. There is no point in this pipeline
> at which a human expert who is not an author independently read a sample of the graph.
> The headline is therefore a measure of self-consistency among models from one vendor.*

This is the sharpest available attack and it is substantially correct as stated.

**Best rebuttal in hand — three independent instruments that are not the LLM loop:**

1. **The shared-source cell of the independence decomposition is a human check the authors
   did not perform.** For single-paper edges whose one paper *is* the curated source,
   agreement with Disbiome/Peryton is **85.2% / 94.1%** — that is a professional human
   curator reading the same sentence, and it is adversarial in the sense that the curators
   had no knowledge of this project. That figure lands on top of the LLM-adjudicated 86.6%
   from a different instrument with different failure modes. Two instruments, one answer, is
   the strongest form this argument takes.
2. **Near-zero hallucination is demonstrated non-circularly.** Of Opus 4.8's ~72 extra taxa,
   exactly **one** (`[Eubacterium]_ventriosum_group`) never appears in the source text at all;
   55% sit beside a hard statistical cue (`proj_2_attempt3/CLAUDE.md`). Separately, all **33**
   of the taxon "misspellings" were verified to occur *verbatim* in their source paper —
   the extractor was copying the paper's own error, not inventing a name
   (`FINDINGS_taxon_spelling.md`). String-level fabrication is measured and it is ~0.
3. **The adjudication protocol is better than the objection implies.** Categories were fixed
   in advance; the orchestrator wrote verdicts *before* seeing the second pass; exact category
   agreement was **25/28**, and all three disagreements were between two artifact categories —
   none moved toward `EXTRACTION_ERROR`. Verbatim quotes were required for any miss verdict in
   the recall audits.

**`CANNOT REBUT`:**

- **Opus and Haiku are not independent adjudicators in the sense a reviewer means.** Same
  vendor, same prompt family, plausibly correlated failure modes. 25/28 agreement is
  consistent with two correlated readers, and the project has no way to distinguish that from
  two good ones. There is no statistical dodge here; the only answer is a human.
- **No blind human annotator has ever scored a random sample of the shipped graph.** Not one.
  The in-house gold was built on a different (pre-KG) task, is known to have 162/250 papers
  with blank taxa columns, and is under audit.
- **No inter-annotator agreement figure exists** between any two humans on this schema, so the
  paper cannot state the human ceiling. A reviewer is entitled to ask what two microbiologists
  would score against each other on the `samgated-v1` gate, and the honest answer is "unknown".

**The fix, in person-time.** A stratified random sample of ~150–200 (paper, taxon, direction)
observations — stratified by confidence tier and by whether a strict witness exists — read
blind by **two** annotators who are not the pipeline authors, against the written `samgated-v1`
gate, reporting Cohen's κ and per-tier accuracy. This is roughly 2–4 person-days of a
microbiology-literate annotator plus ~1 day to build the blinding harness. It is the single
highest-leverage unspent item in the whole project: it converts the strongest objection from
unanswerable to answered, requires **no GPU, no taxdump, and no new papers**, and every other
number in the paper gets an external anchor as a side effect. Do it before submitting anywhere.

### 5.2 Objection B — "The fidelity number is measured on 7% of the evidence, selected for being easy"

> *The ≥86.6% reading fidelity rests on 181/209 scoreable observations. The graph contains
> **3,077** observations. So the instrument scores about 7% of the evidence, and the selection
> rule is not random: a "strict witness" is a sentence that names exactly one taxon and whose
> direction cues all share one polarity. Sentences naming several taxa with mixed polarity —
> "Firmicutes decreased while Proteobacteria increased" — are precisely where a reader
> misattributes a direction, and they are excluded by construction. The audit measures
> fidelity on the subset where fidelity is easiest.*

This is the most technically damaging objection because it attacks the number that anchors
the paper's whole framing, and unlike Objection A it cannot be answered by hiring an annotator
for a week — it is a property of the instrument's design.

**Best rebuttal in hand:**

1. **The selection is stated, not hidden.** `FINDINGS_direction_audit.md` reports all three
   tiers including the one that was built and rejected (T0 0.848 / T1 0.866 / T2 0.774), and
   says in terms: "the true reading fidelity is above 86.6%; how far above, this instrument
   cannot say." A reviewer who finds the limitation already written down and quantified is in
   a different mood than one who discovers it.
2. **`TAXON_MISMATCH` — the exact failure mode the objection predicts — is bounded, not
   unmeasured.** It appears in the residual as **4 of 28** cases, and in each the audit, not
   the extractor, was wrong about which organism the sentence concerned.
3. **An independent instrument covers the multi-taxon case indirectly.** The abbreviation-
   expansion defect found during this audit was exactly a misattribution bug — `B. gingivalis`
   filed under *Blautia* — and it was found, quantified (774 of 18,436 mentions on the
   abbreviation path; 362 in ambiguous-initial papers), and repaired (84 reassigned, 29 already
   correct, 117 dropped). So the project has demonstrated it can find attribution errors it is
   not looking for.
4. **Agreement with two human curations is computed over all decisive pairs, not over strict
   witnesses**, so the 87.5%/96.8% same-paper figures are not subject to this selection at all.
   The two instruments have disjoint selection biases and agree.

**`CANNOT REBUT`:**

- **The paper cannot state fidelity on multi-taxon sentences.** No instrument in the repo
  measures it. The honest claim is "fidelity ≥86.6% on unambiguous witness sentences,
  unmeasured elsewhere", and that hedge weakens the headline noticeably when written out.
- **The 7% is not a random 7%.** There is no reweighting or coverage argument available — the
  project cannot show that strict-witness observations are representative of the other 93%,
  because the comparison requires reading the other 93%.
- **Compounding:** the same provenance screen that defines strict witnesses is also the
  candidate generator for the edge-recall audit (~75% paper-level recall by the project's own
  statement), so the recall bound inherits a correlated blind spot. The two headline quality
  numbers are less independent of each other than they look.

**The partial fix, in person-time.** Extend the witness scorer to multi-taxon sentences with
per-taxon cue attachment (dependency parse or an LLM attribution pass with required verbatim
spans), then report fidelity on the *union* with the strict-witness figure as a subgroup. This
raises coverage from ~7% toward ~14% (442 T0 witnesses) and further if relaxed. Estimate 3–5
person-days. It will almost certainly *lower* the headline number — T0 already scores 0.848 —
and it should be done anyway, because a reviewer finding it costs more than reporting it.

### 5.3 Objection C — "Your headline finding is a property of your own small corpus, and you cannot separate 'the field disagrees' from 'you read the unchecked papers wrong'"

This is really two blades of one objection; a good reviewer will swing both.

> **C1 (generality).** *You show that a 9–16% paper overlap produces a ~50% overlap in decisive
> pairs, and that agreement splits 87.5/58.1 and 96.8/52.6 on that line. But this is one
> 271-paper corpus. MINERVA mined 129,719 papers; its overlap with Disbiome is presumably a
> rounding error, so the confound you describe may not exist for anyone else. You have
> demonstrated a property of your corpus and are asking the field to change its validation
> practice on that basis.*
>
> **C2 (identification).** *The disjoint-source rate of ~55% is presented as "cross-literature
> reproducibility". It is equally consistent with your extractor being 55% right on the papers
> no human curator ever checked. The shared bucket is the only bucket where the extractor was
> ever verified; the disjoint bucket is unverified by construction, and it is exactly where
> agreement collapses. The simplest explanation of your own table is that the extractor is good
> where it was checked and bad where it wasn't.*

**Best rebuttal in hand, C1:**

- The **mechanism** is what generalizes, not the magnitude: shared papers are the
  heavily-reported ones, so paper overlap under-states evidence overlap by a factor of ~3–5.
  Any curated database is built preferentially from well-known papers; any literature miner
  that weights by report frequency will re-derive them. The claim to make is *"validation
  overlap must be measured and reported"*, not *"everyone's number is 55%"*.
- Within-Parkinson's replication across **two curations that do not know about each other**
  (59.0% and 59.6%, p=0.0001 each) is the strongest generality evidence available: the same
  disjoint rate arrived at twice independently.
- The MS counter-example is logged rather than buried (72.7 vs 70.6, n=39, MDE 28.7). A
  reviewer who sees the authors publishing their own disconfirmation discounts the rest less.

**Best rebuttal in hand, C2 — this is what the fidelity audit exists for:**

- Reading fidelity is measured **corpus-wide against the papers' own sentences**, not on the
  shared bucket: ≥86.6% over 122 papers, with residual disagreement *not* concentrated in
  particular papers (permutation p = 0.28). If the extractor were collapsing on unchecked
  papers, that instrument would have to be blind to it in a paper-specific way, and the
  permutation test says discordance is not paper-clustered.
- The field's own base rates make ~55% unsurprising without any extraction error: Gibbons
  et al. 2018 (PLOS Comput Biol) found 681/1,021 OTUs (67%) differing significantly between two
  studies' **healthy-control cohorts alone**; Tierney et al. 2022 (PLOS Biology) report ~1 in 3
  taxa sign-inconsistent across 581 associations / 15 cohorts. A ~55% cross-literature
  concordance sits inside that envelope.
- The internal discordance measurement is consistent: 377/1,367 decisive observations (27.6%)
  disagree with the leave-one-out majority *within our own corpus*, where no external curation
  is involved at all.

**`CANNOT REBUT`:**

- **C1 has never been tested on anyone else's resource.** The project has not computed the
  paper-overlap-to-pair-overlap ratio for MINERVA, gutMDisorder, GMrepo, MASI or Amadis against
  Disbiome/Peryton. Until that exists, the generality claim rests on one corpus and the reviewer
  is right to press.
  **This is the second-highest-leverage unspent item** and it is cheap: if a comparator
  publishes per-edge PMIDs, the entire experiment is a join — no GPU, no extraction, ~2–4
  person-days including chasing down provenance files. A single replication on an
  independently-built resource converts §1's headline from "a property of our corpus" to "a
  property of this validation practice", which is the difference between a workshop paper and a
  journal one.
- **C2 cannot be fully closed, ever, by this design.** Distinguishing "the literature disagrees"
  from "we misread the unchecked papers" requires ground truth on the disjoint papers, which
  means human reading of exactly those papers — i.e. §5.1's annotator sample, stratified to
  over-sample **disjoint-source** edges. That is the version of the annotation study worth
  running: it answers A and C2 with one instrument. Without it, the paper must present the ~55%
  as *"an upper bound on cross-literature reproducibility, a lower bound on nothing"*, and must
  not call it a finding about the field.
- **No non-literature validation exists.** Nothing in this project has ever been checked against
  raw sequencing data, so the paper cannot exclude "both sides read the papers correctly and the
  papers are both wrong". See §3 for whether that is required or merely desirable.

### 5.4 Two smaller objections that will still cost a revision cycle

- **"Why should I believe the calibration tiers?"** Tier sizes are lopsided (75 / 135 / 1,607),
  cuts were fit on Disbiome with Peryton as the held-out check — which is defensible — but 79%
  of edges land in the tier that agrees at 66.1% / 61.9%. Expect "most of your resource is a
  coin flip plus". The answer is to lead with the calibration as a *reporting* contribution
  (neither Disbiome nor Peryton ships any confidence mechanism) and never as a coverage claim.
- **"Direction-only forecloses meta-analysis."** True and permanent at this design: Q, τ² and I²
  mechanically require per-study effect size and variance (`00_SYNTHESIS.md` §6), and vote
  counting has power that *decreases* as studies accumulate (Hedges & Olkin 1980, Psych Bull
  88(2):359–369). Pre-empt it by citing Duvallet et al. 2017 (Nat Commun 8:1784), who vote-count
  over *re-processed raw data* and decline to call conditions with <4 datasets. Do not let a
  reviewer be the first to raise Hedges & Olkin.

### 5.5 What this section implies for the submission plan

Two experiments dominate everything else, and neither needs a GPU:

| # | experiment | answers | person-time |
|---|---|---|---|
| 1 | Blind two-annotator sample (~150–200 observations), stratified by tier **and** by shared/disjoint source, with κ | **A**, **C2**, and gives the human ceiling | ~3–5 days |
| 2 | Overlap replication on ≥1 externally-built resource (MINERVA/gutMDisorder/GMrepo vs Disbiome/Peryton) | **C1** — converts the headline to a general claim | ~2–4 days |

Experiment 2 is the one that changes which venues are reachable. Experiment 1 is the one that
stops the paper being desk-rejectable on circularity. The multi-taxon fidelity extension (§5.2)
is third, and will lower a headline number — which is a reason to do it before a reviewer does.

---

## 6. Recommendation

### 6.1 Primary target — *Database: The Journal of Biological Databases and Curation* (OUP)

**The paper is not "our microbe–disease knowledge graph". It is "agreement with curated databases
is a confounded way to validate a literature-mined resource, here is the decomposition, here is
the graph that let us measure it."** Working title shape: *"Shared source papers inflate
database-agreement validation of literature-mined microbe–disease resources"* — note that the
resource is not in the title.

**Why this venue:** it is the only one on the list whose stated remit explicitly covers *curation
standards and annotation best practices* and *objective reviews of complementary databases*
(confirmed 2026-09-19, <https://academic.oup.com/database/pages/About>). It publishes the
resource and the critique in one article, is fully OA with PMC deposit, has rolling submission,
and its readership is the population whose practice the finding is about. Every other journal
either wants a bigger resource (NAR, Bioinformatics), wants biology we do not have (Microbiome,
mSystems, Gut Microbes), or takes the paper while burying it (BMC Bioinformatics).

**Shortest path — 4 items, roughly 8–13 person-days of focused work, none of it needing a GPU:**

| # | item | person-time | why it is on the critical path |
|---|---|---|---|
| 1 | **Blind two-annotator study** — ~150–200 observations, stratified by confidence tier **and** by shared/disjoint source, two annotators who are not pipeline authors, κ reported (§5.1) | 3–5 d | Without it the paper is desk-rejectable on circularity at a curation journal, and it simultaneously answers the C2 identification objection |
| 2 | **Freeze one snapshot + Zenodo DOI + licence + schema doc**, and reconcile the 2,034/2,008 and 217/220 discrepancies across README, `CLAUDE.md` and the research docs (§3.2, §3.5) | 2–3 d | Mandatory for a resource-bearing article; also the cheapest way to lose a reviewer's trust if skipped |
| 3 | **Prior-art sweep outside the microbiome subfield** — variant-curation concordance, database-overlap metrics, NLP contamination, systematic-review overlap statistics (§4.3) | 1 d | The novelty claim currently rests on two in-subfield searches |
| 4 | **Multi-taxon fidelity extension** (§5.2) | 3–5 d | Optional for submission, near-certain to be demanded in review; it will likely *lower* 86.6%, so own it first |

Items 1–3 are the submission bar; item 4 is the revision insurance. **Do item 2 first** — it is
the least interesting and it gates every figure in the manuscript.

**One optional day that could change the plan:** the provenance screen from
`202_sequencing_validation.md` §5.5 (resolve public stool cohorts for the top five diseases to
PMIDs, intersect with the 271 contributing papers). If ~8+ independent cohorts survive, a much
stronger and higher-venue paper becomes possible; if not, the limitation paragraph writes itself.
~1 person-day, and the answer is binary. Run it early.

### 6.2 Fallback — *Journal of Biomedical Informatics*

Same manuscript, reframed as evaluation methodology rather than curation practice: retitle around
*evaluation of biomedical information extraction*, move the resource description into a methods
subsection, and lead the abstract with the document-overlap-to-fact-overlap ratio. JBI is
comfortable with modest corpora when the contribution is methodological, and the desk-reject risk
is almost entirely about framing, not substance.

**Floor, if both decline:** BMC Bioinformatics will publish this work substantially as-is. That is
a real option, not a consolation — but spend the §6.1 items first, because they raise the ceiling
and cost under three weeks.

### 6.3 The two conditional upgrades

- **If Experiment 2 lands** (§5.5 — replicate the overlap confound on an independently built
  resource such as MINERVA, gutMDisorder or GMrepo against Disbiome/Peryton, ~2–4 person-days
  *provided* the comparator publishes per-edge PMIDs), the claim stops being a property of this
  corpus and becomes a property of the validation practice. That upgrade makes **Briefings in
  Bioinformatics** a serious target — and it would publish, in the journal that published MINERVA,
  a measurement that bears directly on MINERVA. ⚠️ Check MINERVA's data release for per-edge
  provenance before budgeting the time; if it ships only aggregated scores, the experiment must
  use a different comparator.
- **If the sequencing screen survives** (§3.3), the microbiology venues in §2.9 open and the paper
  becomes a much bigger one. Do not plan around this; just spend the one day to find out.

### 6.4 Parallel, low-cost de-risking

Submitting the independence result to **ACL Rolling Review** (⚠️ ~monthly cycles, reviews in
roughly 6–8 weeks, free, and a commitment to BioNLP 2027 or Findings can follow) buys expert
adversarial review from people who know evaluation contamination, months before a journal
decision. ✅ Note that BioNLP 2026 already ran (3–4 July 2026, ACL 2026, San Diego), so the
workshop's own next deadline is spring 2027. ⚠️ Confirm dual-submission and prior-publication
policies on both sides before doing this — it is normal practice in the ACL community to extend
an archival workshop paper into a journal article, but the journal must be told, and *Database*'s
policy must be checked rather than assumed.

### 6.5 The four things not to do

1. **Do not submit to the NAR Database Issue.** Predictable desk reject on scale and sustainability;
   costs a year of cycle time (§2.8).
2. **Do not submit to Microbiome, mSystems or Gut Microbes** before the sequencing question is
   answered — their reviewers ask it first and it is unanswerable today (§2.9).
3. **Do not title or frame the paper around the knowledge graph.** MINERVA published that artifact
   at 30× the scale in 2025, in a journal on this list (§4.1).
4. **Do not report F1 0.680 as accuracy anywhere**, and do not spend time repairing the old gold
   (§3.1).

### 6.6 The one-paragraph version

Write the measurement paper, not the resource paper. Spend under three weeks on a blind human
annotation study, a frozen DOI'd snapshot with consistent numbers, and a prior-art sweep outside
the subfield. Send it to *Database (Oxford)*, with *JBI* as the reframed fallback and BMC
Bioinformatics as the floor. Spend one extra day each on the sequencing provenance screen and on
checking whether MINERVA publishes per-edge PMIDs — those two days are the only cheap paths to a
materially better venue.

---
