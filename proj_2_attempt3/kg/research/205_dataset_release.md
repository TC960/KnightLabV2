# 205 — Dataset release: what we may publish, where, and how

**Status:** review, 2026-09-19. Not legal advice — see *How to read this* below.
**Scope:** the KG, the eval/benchmark suite, derived corpus artifacts, and the paper
full text already sitting in the public repo.

---

## How to read this

Three kinds of claim appear below and they are deliberately kept apart, because
conflating them is how projects talk themselves into releases they cannot defend.

- **(i) What licence terms plainly say.** Quoted verbatim, with the URL. These are
  facts about documents. If the quote is here, you can rely on the quote.
- **(ii) Established community practice.** What comparable resources actually do.
  Practice is evidence that something is *tolerated*, not that it is permitted.
- **(iii) Genuinely unsettled.** Places where the honest answer is that reasonable
  people disagree and no court or regulator has settled it. Flagged **[UNSETTLED]**.

Anything marked **[UNSETTLED]** or touching the closed-access material in Part 0
should go to **UCSD Library Scholarly Communication** (copyright and author-rights
consults) and, where the exposure is publisher-facing, the **UCSD Office of
Research Compliance / RCR** — before release, not after. That consult is cheap
relative to the alternative and is specifically the thing a university library
copyright officer exists to do. I am not a lawyer and nothing below is a legal
conclusion.

One framing point that should govern every decision in this document:

> **A permissive licence on data you did not have the right to redistribute is
> worse than no release at all.** It converts a passive exposure into an active
> grant of rights you do not hold, to an unbounded set of downstream users, each
> of whom then has their own problem. It also makes the error un-retractable in
> practice.

---

## Part 0 — What is actually public right now (measured, 2026-09-19)

**The brief understates this by roughly two orders of magnitude.** The brief
describes "~7 MB of paper full text" in `extract_input.json` and
`new_papers.json`. That is a small fraction of the exposure. I measured the
repository directly rather than relying on the description.

`https://github.com/TC960/KnightLabV2` — confirmed **public**, `"private": false`,
no `LICENSE` file anywhere in the tree, default branch `main`, last push
2026-09-17.

### 0.1 Full text live on the default branch right now

Every one of these returns HTTP 200 from the GitHub contents API today:

| Path | Papers | Size | Content |
|---|---:|---:|---|
| `proj_2_attempt3/MAIN_DATA.json.zip` | **2,026** (1,724 with >15k chars) | 32 MB zipped / **100 MB raw** | cleaned full text, `chunks[]` per paper |
| `.../EmilySong_GoldStandardPaper/all_usable_papers.json` | 250 | 11.7 MB | full text in a `text` field |
| `.../EmilySong_GoldStandardPaper/holdout_pool.json` | 57 | 3.5 MB | full text |
| `.../EmilySong_GoldStandardPaper/test_set_v2.json` | 15 | 0.8 MB | full text |
| `.../EmilySong_GoldStandardPaper/gold_standard_final_15.json` | 15 | 0.8 MB | full text |
| `proj_2_attempt3/kg/relation_sentences.json` | 348 | 3.1 MB | 7,545 verbatim sentences |
| `proj_2_attempt3/kg/relation_sentences_clean.json` | 348 | 3.1 MB | 7,509 verbatim sentences |

**Deduplicated: the full text of 1,902 distinct papers is publicly downloadable
from this repository.** (1,724 from `MAIN_DATA.json.zip` + 192 from the
gold-standard JSONs, overlap 14, counting only records above 15k characters so
that abstract-only stubs are excluded.)

`MAIN_DATA.json.zip` has been public since **2026-02-13**;
`all_usable_papers.json` since **2026-04-09**; `relation_sentences_clean.json`
since **2026-09-12**.

### 0.2 Full text reachable in history but not in HEAD

These were deleted from the working tree but the commits are pushed, so the blobs
remain publicly fetchable by SHA (GitHub serves any reachable blob; commit
`a5e1cc5` returns 200 from the API today):

| Path | Size |
|---|---:|
| `proj_2_attempt3/kg/extract_input_gold.json` | 12.4 MB |
| `proj_2_attempt3/kg/extract_input.json` | 5.3 MB |
| `proj_2_attempt3/kg/new_papers.json` | 2.0 MB |

That is ~20 MB in history, not 7 MB — and it is the *smaller* half of the problem.

### 0.3 The one piece of good news, and it is a real one

**The repository has exactly one fork — `AlanThisis/KnightLabV2`, created
2026-02-14, last pushed 2026-02-14 00:25 UTC — and it does not contain the
full-text files** (both paths return 404 on the fork). The fork predates or
narrowly precedes the pushes that matter and was never updated.

This materially changes the remediation calculus. The standard warning that "forks
make history rewriting useless" is the default case; it does **not** apply here as
strongly as it usually does. History rewriting is genuinely viable. See Part 2.

### 0.4 Local state

`main` is **7 commits ahead of `origin/main`**. `prompt_exp_subset.json` (80
papers, 3.4 MB of full text) is tracked locally but **not yet pushed** — it is
currently 404 on GitHub. That file is a decision you still get to make cleanly
rather than a mess you have to clean up. Do not push it.

There is also an untracked `kg/fetch_oa_fulltext.py` and `kg/oa_cache/` in the
working tree, which suggests someone is already fetching OA full text through a
different route. Worth coordinating — if that script pulls from the PMC OA
Subset via an approved service, it is the *correct* acquisition path and should
replace the scraping described in §1.8.

---

## Part 1 — What is actually permissible to redistribute

### 1.1 PMC is three collections, not one, and only one of them is redistributable

This is the distinction that does most of the work. From PMC's copyright page
(<https://pmc.ncbi.nlm.nih.gov/about/copyright/>), quoted verbatim:

> "Articles available from PubMed Central (PMC) are provided by the respective
> publishers or authors."

> "Systematic downloading of batches of articles from the main PMC web site, in
> any way, is prohibited because of copyright restrictions."

And the three buckets:

1. **Open Access Subset** — articles under Creative Commons or similar licences
   that allow "more liberal redistribution and reuse than a traditional
   copyrighted work."
2. **Author Manuscript Collection** — accepted manuscripts deposited under the
   NIH Public Access Policy. PMC states these "often do not include copyright
   statements or license terms" because "Authors own the original copyrights,"
   and that PMC permits reading, text mining, and "other uses consistent with the
   principles of Fair Use."
3. **Everything else in PMC** — "assume that standard copyright protection
   applies, unless the article contains an explicit statement to the contrary."

**Free to read is not free to redistribute.** Bucket 3 is the large majority of
PMC by article count and carries no redistribution right at all. Bucket 2 carries
a *use* permission grounded in fair use, which is not a distribution licence and
notably is **not** a grant PMC is in a position to make on the author's behalf —
PMC is describing what it believes lawful, not licensing you. Only bucket 1 gives
you something you can rely on, and only per-article.

### 1.2 Commercial vs non-commercial inside the OA Subset

From PMC's text-mining page (<https://pmc.ncbi.nlm.nih.gov/tools/textmining/>),
the OA Subset is split into three packages by licence:

- **Commercial use permitted (`oa_comm`):** "CC0, CC BY, CC BY-SA, CC BY-ND"
- **Non-commercial use only (`oa_noncomm`):** "CC BY-NC, CC BY-NC-SA, CC BY-NC-ND"
- **Other (`oa_other`):** articles with custom or untagged licences

Same page, two operative restrictions:

> "The PMC Cloud Service, PMC OAI-PMH Service, E-Utilities and BioC API are the
> only services that may be used for automated retrieval of PMC content."

> "Systematic retrieval (or bulk retrieval) of articles through any other
> automated process is prohibited."

> "Users of this dataset are directly and solely responsible for compliance with
> copyright restrictions."

Two traps in that taxonomy:

- **`oa_comm` includes CC BY-ND.** ND permits verbatim redistribution but
  prohibits distributing *adapted* material. Our corpus is not verbatim — it went
  through a Llama-3-8B cleaning pipeline, was re-chunked, and had References
  stripped. Whether that is an "adaptation" is arguable; it is certainly not
  obviously *not* one. **[UNSETTLED]**
- **`oa_other` is not a safe bucket.** "Untagged" means nobody has determined the
  terms, not that the terms are permissive.

### 1.3 What this corpus actually is — measured, not estimated

I ran the licence determination over all 250 papers in
`all_usable_papers.json`: PMID/DOI → PMC ID via the PMC ID Converter API, then
`efetch db=pmc` per article, parsing the `<permissions>` block (`<license>`
`xlink:href`, `ali:license_ref content-type`, `license-type`, and the
`<license-p>` prose). Papers with no PMC record, or in PMC with no retrievable
body, were then resolved against Unpaywall by DOI for `oa_status` and
`best_oa_location.license`.

**Result, N = 250:**

| Classification | n | % |
|---|---:|---:|
| CC BY | 121 | 48.4% |
| **Closed access — no redistribution right** | **37** | **14.8%** |
| CC BY-NC-ND | 31 | 12.4% |
| CC0 | 16 | 6.4% |
| CC BY-NC | 10 | 4.0% |
| CC BY-NC-ND (publisher site) | 8 | 3.2% |
| CC BY (publisher site) | 6 | 2.4% |
| Custom / publisher-specific terms | 6 | 2.4% |
| CC BY-NC (publisher site) | 2 | 0.8% |
| Unclear — `oa_status=green`, licence `other-oa`/none | 3 | 1.2% |
| Unclear — `oa_status=bronze`, no licence | 2 | 0.8% |
| Unclear — no Unpaywall record / query failed | 5 | 2.0% |
| CC BY-SA (publisher site) | 1 | 0.4% |
| Public domain (publisher site) | 1 | 0.4% |
| CC, unparsed variant | 1 | 0.4% |

**Rolled up:**

| Bucket | n | % |
|---|---:|---:|
| Freely redistributable with attribution (CC BY / CC BY-SA / CC0 / PD) | **145** | **58.0%** |
| Redistributable but non-commercial-restricted (NC, NC-ND) | **51** | **20.4%** |
| **No established redistribution right** (closed, custom, unclear) | **53** | **21.2%** |

So, concretely: **roughly one paper in five in the published corpus is one whose
full text we have no demonstrated right to publish, and 37 of them are outright
closed access.** "Bronze" OA (2 papers) is the sharpest sub-case — free to read on
the publisher's site at the publisher's discretion, with no licence at all and no
commitment to keep it free.

**Caveats on this measurement, stated plainly:**

- The `efetch` test for "is this in the OA Subset" has **false negatives**. My
  first pass under concurrency reported 36 articles as outside the subset; on
  serial re-fetch, 33 of the 36 returned full text and a CC licence. The numbers
  above are the corrected pass. The authoritative test is the PMC OA file lists /
  OA Web Service, not `efetch` — see §1.5. Anyone rerunning this should use the
  authoritative source.
- Unpaywall's `oa_status` describes *reader access*, not licence. `closed` is
  reliable as a negative (a closed paper has no redistribution licence); `gold`
  is not by itself reliable as a positive, which is why licence is read
  separately.
- This is the **250-paper annotated corpus only**. The 1,724-paper
  `MAIN_DATA.json.zip` has **not** been audited. Its source distribution is a
  poor omen rather than a reassurance: 1,477 scraped from `pmc.ncbi.nlm.nih.gov`
  (PMC ≠ OA Subset), **202 from `www.nature.com`** (median 51.7k chars — real
  full text, and Nature research content is predominantly subscription), 25 from
  `link.springer.com`, 10 from `academic.oup.com`. **Assume MAIN_DATA is worse
  than 21% until measured.** The same script will measure it.

### 1.4 The Author Manuscript Collection specifically

The brief is right that this bucket has different and more restrictive terms, and
it is worth being precise about *why*.

For OA Subset articles, the author or publisher has executed a licence — CC BY and
friends are irrevocable grants to the world, and PMC is merely a distribution
channel. Your right to redistribute flows from the licence, not from PMC.

For author manuscripts, no such licence generally exists. PMC's own framing is
that "Authors own the original copyrights" and that uses should be "consistent
with the principles of applicable copyright law" and "the principles of Fair
Use." PMC permits *retrieval* for text mining — the manuscripts are in the PMC
Article Datasets and "may be retrieved in XML and plain text formats" — but
retrieval permission and redistribution permission are different things, and PMC
is not the rightsholder who could grant the latter.

**Practical upshot:** an author manuscript may be freely mined and must not be
redistributed as full text absent a licence statement on the specific article.
Some author manuscripts *do* carry CC licences ("Many author manuscripts have
Creative Commons licenses" — PMC) and those are fine; the rest are not.

**In our 250-paper corpus the audit flagged zero articles carrying the PMC
author-manuscript markers** (`<article-id pub-id-type="manuscript">` or
`self-uri content-type="pmc-manuscript"`). That is a genuine negative result and
one fewer thing to worry about for that subset. It has **not** been checked for
`MAIN_DATA.json.zip`.

### 1.5 How to determine an article's licence programmatically

Four routes, in decreasing order of authority:

1. **PMC OA file lists (authoritative).** The OA Subset is published as
   `oa_comm` / `oa_noncomm` / `oa_other` packages with accompanying file lists
   under the PMC FTP/Cloud service. Membership in `oa_comm` is the strongest
   single signal you can get: it means NLM has classified the article as
   CC0/BY/BY-SA/BY-ND. Use this as the gate. Note the FTP index under
   `/pub/pmc/` currently surfaces only `PMC-ids.csv.gz` and `readme.txt` at the
   top level; the bulk packages are reached through the documented Cloud/OAI
   endpoints rather than by browsing.
2. **`license-type` and the `<permissions>` block in PMC XML** (via E-utilities
   `efetch db=pmc`). This is what I used. The parse must handle all of:
   `<license license-type="...">`, `<license xlink:href="...">`,
   `<ali:license_ref content-type="ccbylicense">`, and free prose inside
   `<license-p>` — articles use different combinations and a regex that only
   checks `xlink:href` on the `<license>` element silently mislabels a large
   minority (it mislabelled 134 of 188 in my first attempt). Add retries: this
   endpoint fails transiently under concurrency and a failure looks exactly like
   "not in the OA Subset."
3. **Unpaywall `oa_status` + `best_oa_location.license`** (by DOI). Best coverage
   for anything outside PMC, which is where our 62 non-PMC papers live. Treat
   `closed` as dispositive against release; treat `bronze` as "no licence, do not
   redistribute"; treat `gold`/`hybrid` + an explicit `cc-*` licence as usable.
4. **Crossref licence metadata.** Useful as a cross-check, but publisher-deposited,
   frequently absent, and frequently records the *TDM* licence URL or an
   embargo-dated licence rather than the content licence. Do not rely on it alone.

**Recommendation: require two independent sources to agree before a paper's text
is released**, and default to exclusion on disagreement. The cost of a false
positive here is asymmetric.

### 1.6 Publisher TDM terms, and whether they reach redistribution

They generally do not. The pattern across the three big publishers is that TDM
rights are framed as *access and analysis* rights for subscribers, coupled with
security obligations that are flatly incompatible with a public repository.

**Springer Nature** (<https://www.springernature.com/gp/researchers/text-and-data-mining>)
— TDM for non-commercial research at subscribing institutions, and its conditions
include that the researcher:

> "store content on a secure internal server without access for third parties"

and limit storage "only for the duration of the TDM project." A public GitHub
repository is the precise negation of "without access for third parties." The 202
`nature.com` full texts in `MAIN_DATA.json.zip` sit directly against this.

**Elsevier** (<https://www.elsevier.com/about/policies-and-standards/text-and-data-mining>)
— permits researchers at subscribing institutions to "text mine full-text content
hosted on Science Direct for non-commercial research purposes" via the full-text
API with a key from `dev.elsevier.com`. The policy pages describe access, not
output redistribution. Historically Elsevier's TDM terms have constrained output
to short snippets and required a specified output licence; **I could not retrieve
the current operative snippet limit or output-licence clause from their
documentation, so I am not going to state a number for it.** If Elsevier content
is in scope, that clause must be read directly — it is the single most
consequential publisher term for a sentence-level release.

Worth noting what Elsevier's own developer portal puts in its footer:

> "Copyright © Elsevier B.V. All rights reserved, including those for text and
> data mining, AI training, and similar technologies."

That is an express rights reservation of exactly the kind Article 4(3) of the DSM
Directive contemplates (§1.7). It does not affect Article 3 research-purposes
mining, but it is a deliberate, machine-readable opt-out against everything else.

**Wiley** follows the same shape — TDM by arrangement for subscribers, with
security and non-redistribution conditions.

**The general rule to carry forward:** *a TDM right is a right to read at scale.
It is not a right to republish what you read, and it usually comes with an
explicit obligation not to.* Nothing in any of these policies converts into
permission to post full text publicly. Whether a TDM right extends to publishing
*extracted sentences* is a different and much closer question — §1.7.

### 1.7 Full text vs sentences vs facts — the three-tier distinction

This is the most important conceptual section in the document, because the three
tiers have genuinely different risk profiles and the project holds artifacts in
all three.

**Tier 3 — structured facts derived from text (the KG). Low risk.**
An edge "*Akkermansia* enriched in Parkinson's disease, 8 papers" is a fact
extracted from literature. Facts are not copyrightable in the US (*Feist
Publications v. Rural Telephone Service*, 499 U.S. 340 (1991) — no copyright in
facts, and no "sweat of the brow" protection for the labour of compiling them).
The *selection and arrangement* of a factual compilation can attract a thin
copyright, which protects our graph, not the sources. The EU adds a **sui generis
database right** (Directive 96/9/EC) which protects substantial investment in
obtaining/verifying/presenting database contents — that right attaches to the
source databases we join against (Disbiome, Peryton) more than to the papers, and
is worth a sentence in the consult.

`graph.json`, `rag_corpus.jsonl` (whose `text` field is *our generated prose*, not
paper text — I checked), the confidence tiers, the taxonomy/MONDO resolutions, the
metadata extractions and the contested-edge structure are all Tier 3. **These are
releasable.** This is the bulk of what makes the project valuable.

**Tier 1 — full text. High risk, and not close.**
Reproducing an entire copyrighted article verbatim is the core of the exclusive
right. There is no fair-use argument that survives the fourth factor when the
copy is complete, is a substitute for the original, and is posted publicly with
no access control. For the 53 papers in §1.3 with no established right — and for
however many of MAIN_DATA's 1,724 fall the same way — there is no defence worth
constructing. This is Part 2's problem.

**Tier 2 — extracted sentences. The genuine grey zone. [UNSETTLED]**
This is where `relation_sentences_clean.json` lives, and where the interesting
question is. Arguments in both directions are real:

*For:* the sentences are selected for a functional, non-expressive purpose
(they contain a taxon and a direction cue); they are scattered and
non-contiguous; they cannot substitute for reading the paper; the use is
transformative in the *Google Books* / *HathiTrust* sense (*Authors Guild v.
Google*, 804 F.3d 202 (2d Cir. 2015) upheld full-text indexing with snippet
display as fair use, and *Authors Guild v. HathiTrust*, 755 F.3d 87 (2d Cir.
2014) similarly for search and accessibility); scientific finding-sentences are
close to the factual end of the idea/expression spectrum.

*Against:* *Google Books* rested heavily on snippet display being **deliberately
crippled** — snippets capped, blacklisted pages, a hard ceiling on how much of any
book any user could assemble. A JSON file that hands you every selected sentence
at once, offline, with no cap, is materially different from a search interface
that grudgingly shows you three lines.

**So I measured how crippled ours actually is.** Across the 348 papers in
`relation_sentences_clean.json`:

| | |
|---|---:|
| sentences released | 7,509 |
| total quoted characters | 1,725,497 |
| mean / median sentence length | 230 / 199 chars |
| **median share of each paper's text quoted** | **9.4%** |
| mean share | 10.4% |
| 90th percentile | 18.8% |
| **maximum share for a single paper** | **51.0%** |
| papers with >10% of text quoted | 157 |
| papers with >20% of text quoted | 28 |
| median sentences per paper | 20 (max 75) |
| "sentences" longer than 500 chars | 261 |
| "sentences" longer than 1,000 chars | 29 |
| longest single "sentence" | **6,017 chars** |

The median case — ~9% of a paper, in 20 scattered ~200-character fragments — is
about as defensible as Tier 2 gets, and is comfortably inside the range that
biomedical NLP resources distribute as a matter of routine practice **(ii)**.

**The tail is the problem, and it is fixable.** 28 papers have more than a fifth of
their text quoted and one has half. The 261 fragments over 500 characters and the
6,017-character maximum are sentence-segmentation failures — they are not
sentences, they are block quotes that the splitter failed to break. They do
nothing for the science and they are exactly what a substantiality analysis
fixates on.

**Concrete, cheap remediation: cap fragment length (a 300–400 character cap drops
the 261 outliers), cap the share of any single paper at some fixed fraction, and
drop or re-split the 29 fragments over 1,000 characters.** That converts the
artifact from "mostly fine with an indefensible tail" to "uniformly defensible,"
costs no science, and is a one-afternoon change to `relation_sentences.py`.
Do that regardless of which release path is chosen.

**EU TDM exception — DSM Directive (EU) 2019/790.** Article 3 (verbatim, via
legislation.gov.uk):

> "Member States shall provide for an exception to the rights provided for in
> Article 5(a) and Article 7(1) of Directive 96/9/EC, Article 2 of Directive
> 2001/29/EC, and Article 15(1) of this Directive for reproductions and
> extractions made by research organisations and cultural heritage institutions
> in order to carry out, for the purposes of scientific research, text and data
> mining of works or other subject matter to which they have lawful access."

> "Copies of works or other subject matter made in compliance with paragraph 1
> shall be stored with an appropriate level of security and may be retained for
> the purposes of scientific research, including for the verification of research
> results."

Article 4(3):

> "The exception or limitation provided for in paragraph 1 shall apply on
> condition that the use of works and other subject matter referred to in that
> paragraph has not been expressly reserved by their rightholders in an
> appropriate manner, such as machine-readable means in the case of content made
> publicly available online."

Read those carefully, because they are often cited for more than they say:

- Article 3 covers **research organisations** (UCSD qualifies) doing TDM on
  content they **lawfully access**. It authorises the *mining*. It does not
  authorise *publication of the corpus*.
- Article 3(2) is the closest thing to a release permission and it is narrow:
  copies may be **retained** — "stored with an appropriate level of security" —
  for research purposes "including for the verification of research results."
  That is a strong argument for keeping a full-text corpus **in a controlled,
  access-restricted archive for reproducibility**. It is not an argument for a
  public GitHub repo, because a public repo is not "an appropriate level of
  security" under any reading.
- Article 4 (general TDM) is defeated by an express reservation — and Elsevier,
  among others, has made one.
- And the EU directive governs conduct in the EU. A US-based project relies on US
  fair use, not on Article 3. Article 3 matters here mainly because our *users*
  may be in the EU and because it is the clearest articulation anywhere of the
  "mine freely, store securely, don't republish" model that this project should
  adopt.

**US posture.** Fair use (17 U.S.C. §107) is fact-specific and decided case by
case; it is a defence, not a permission, and it cannot be determined in advance
with certainty. The TDM-favourable line — *HathiTrust*, *Google Books*, and
*A.V. ex rel. Vanderhye v. iParadigms*, 562 F.3d 630 (4th Cir. 2009) — supports
**ingesting and analysing** full text and supports **limited snippet display**. It
does not support publishing complete copies. The recent wave of generative-AI
training cases has unsettled the edges of this further rather than clarifying
them. **[UNSETTLED]** — and specifically unsettled in a direction that counsels
caution, not confidence.

### 1.8 A second problem the brief does not raise: acquisition

Redistribution is not the only exposure. PMC's terms say bulk retrieval outside
the four approved services "is prohibited," and the corpus provenance
(`mega_dump/dump/`) records scraping from DOI links and article pages —
`MAIN_DATA` URLs are `pmc.ncbi.nlm.nih.gov/articles/...` web pages, i.e. the "main
PMC web site." That is a terms-of-service issue with NLM independent of copyright,
and it applies even to the CC BY articles, where the *content* licence is
unimpeachable but the *retrieval method* was not the sanctioned one.

This is not an emergency and it is very unlikely to be pursued, but it should be
disclosed in the consult and it should be fixed going forward: re-fetch anything
you intend to release through E-utilities / the OA Cloud Service / OAI-PMH, which
is both permitted and easier. The untracked `fetch_oa_fulltext.py` in the working
tree may already do this.

A third, smaller point: **PubMed abstracts and MEDLINE metadata are separately and
more generously licensed.** NLM's terms
(<https://www.nlm.nih.gov/databases/download/terms_and_conditions.html>) permit
republication and redistribution, requiring only that users "acknowledge NLM as
the source of the data by including the phrase 'Courtesy of the U.S. National
Library of Medicine'", not imply NLM endorsement, and either keep the data current
or say clearly that it is not. This is why the dominant biomedical RE benchmarks
are built on abstracts (§6) — it is a deliberate legal choice, not an accident of
convenience.

---

## Part 2 — Practical remediation for the already-public full text

Four options. I have tried to be honest about what each does *not* achieve.

### Option A — Leave it

**What it is:** do nothing; the corpus stays up.

**Argument for:** the exposure is ~7 months old for the largest file; no complaint
has been received; 58% of the audited corpus is CC BY or CC0 and entirely fine;
the practical enforcement risk against a small academic repository is low; and
removal is visible in a way that leaving it is not.

**Argument against:** it is 37 confirmed closed-access papers plus ~200 Nature
full texts against an explicit "no third-party access" condition, and the project
is about to *publicise itself* through a dataset release and a paper. Release
attracts exactly the attention that makes low-probability enforcement risk stop
being low-probability. And doing nothing after measuring the problem is a
different posture from doing nothing before — it is now a knowing choice, which
matters for both institutional and ethical purposes.

**Verdict: not defensible for the closed-access subset once a release is planned.**
Defensible for the CC BY/CC0 subset, which is most of it.

### Option B — Remove from HEAD only (`git rm` + commit)

**What it is:** delete the files, commit, push. History untouched.

**What it achieves:** the files stop being browsable and stop being cloned by
casual users. Google and GitHub search stop surfacing them.

**What it does not achieve:** the blobs remain in history and remain fetchable by
SHA, which is already the situation for `extract_input.json` et al. and is
demonstrably not "removed" — I fetched commit `a5e1cc5` from the public API in
the course of writing this.

**Verdict: insufficient alone, but it is the correct *first* action** because it is
instant, reversible, and stops the bleeding while the rest is arranged.

### Option C — Rewrite history (`git filter-repo`) + force-push

**What it is:** excise the blobs from every commit, force-push, then ask GitHub
Support to garbage-collect and drop cached views.

**The standard warning, verbatim from GitHub**
(<https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>):

> "if you only rewrite your history and force push it, the commits with sensitive
> data may still be accessible elsewhere: In any clones or forks of your
> repository; Directly via their SHA-1 hashes in cached views on GitHub; Through
> any pull requests that reference them."

> "You cannot remove sensitive data from other users' clones of your repository."

GitHub Support can "Dereference or delete any affected PRs on GitHub; Run a
garbage collection on the server to expunge the sensitive data from storage;
Remove cached views" — but:

> "GitHub Support won't remove non-sensitive data, and will only assist in the
> removal of sensitive data in cases where we determine that the risk can't be
> mitigated."

**Copyrighted article text is very likely not "sensitive data" in GitHub's sense**
(their bar is credentials and personal data). So expect to do the rewrite and the
force-push yourself and to *not* get the server-side GC. That leaves cached blob
views reachable for some period.

**But here the usual objection is much weaker than normal**, for the reason
established in §0.3: there is one fork, it predates the files, and it does not
contain them. There are no pull requests carrying them. The realistic residue is
GitHub's own cache plus any clone taken by a person in the last seven months — and
this repo has 0 stars and 1 fork.

**Verdict: this is the option that actually works here**, and it works better here
than the generic warning implies. It is also disruptive: everyone with a clone
(including every agent working in this repo) must re-clone, and the
`mega_dump/proj_2` submodule gitlink and the GitHub Pages build both need
re-checking afterward — the repo has already been broken once by a submodule
issue. **Do not do this without the PI's sign-off, coordinated across everyone
holding a clone, and with a full backup taken first.**

### Option D — Reduce to snippets, and move full text to controlled storage

**What it is:** the substantive fix rather than the cleanup. Specifically:

1. Full text leaves the public repo entirely (Option B then C).
2. Full text is retained in **access-controlled institutional storage** for
   reproducibility — which is precisely the use DSM Article 3(2) describes,
   "stored with an appropriate level of security ... including for the
   verification of research results."
3. What ships publicly is: **(a)** a manifest of DOIs/PMIDs/PMCIDs so anyone can
   re-fetch the corpus themselves, **(b)** the retrieval script (through approved
   APIs), **(c)** the length-capped sentence extracts from §1.7, and **(d)** all
   Tier-3 derived data.
4. Optionally, the *full* text of just the CC BY / CC0 / CC BY-SA subset —
   **145 of 250 audited papers, 58%** — as a genuinely redistributable corpus with
   per-paper licence and attribution recorded. This is a real and valuable
   artifact and it is unambiguously yours to publish.

**This is what the field already does.** It is why BioRED ships 600 PubMed
abstracts rather than full texts, and why full-text biomedical corpora are
overwhelmingly built on the OA Subset. A DOI manifest plus a fetch script is the
accepted, boring, entirely defensible answer **(ii)**.

**Verdict: this is the recommendation.** Options B and C are the cleanup; D is the
design.

---

## Part 3 — Release format and venue

### 3.1 Venue comparison

| Venue | DOI | Versioning | Size | Citable | Fit here |
|---|---|---|---|---|---|
| **Zenodo** | Yes, per version + a concept DOI resolving to latest | Yes, first-class | "up to a 100 files and a total volume of 50GB" per record; higher quotas on request | Yes | **Best fit.** CERN-backed, no fee, arbitrary file types, GitHub release integration, supports restricted/embargoed records — which matters if any full text is ever archived |
| **Figshare** | Yes | Yes | Generous | Yes | Fine. Weaker curation, more commercial. No advantage over Zenodo here |
| **Dryad** | Yes | Yes | Large | Yes | Curated, biology-facing, strong reputation — but scoped to data *underlying a specific publication*, charges a publication fee, and curators will (correctly) query the provenance of scraped text. Use only alongside a paper, and only for the clean derived data |
| **Hugging Face Datasets** | No native DOI (can mint via Zenodo) | Yes, git-based | Large | Weakly | **Best fit for the benchmark**, for reach not for archiving. This is where people who would actually *use* an eval suite look. Supports gating (§3.2) |
| **NCBI/EBI-adjacent** | — | — | — | — | **No fit.** BioSample/BioStudies/ENA take experimental data, not literature-derived KGs. There is no NCBI home for this. BioStudies is the nearest and it is a stretch |

**Recommendation: Zenodo for the archival DOI, Hugging Face for the benchmark's
day-to-day life, GitHub for code, all three cross-linked, Zenodo's DOI the
canonical citation.** Use Zenodo's GitHub integration so a tagged release mints a
version DOI automatically, and cite the **concept DOI** in the paper so the
citation keeps resolving as the graph is corrected — this project has revised its
headline numbers repeatedly and will again.

### 3.2 A note on gating

Hugging Face supports gated datasets: access requests, manual or automatic
approval, custom fields, and `extra_gated_eu_disallowed` for jurisdiction
exclusion. It is well-built and it is a reasonable way to distribute an
NC-restricted derived set with a click-through condition.

**It is not a fix for material you lack the right to distribute.** Gating changes
who gets the file, not whether you were entitled to hand it over. Do not let it
become the excuse that lets the closed-access text stay online.

### 3.3 Serialisation of the graph

Ship **JSON as the primary format** — `graph.json` is already well-shaped
(`meta`/`nodes`/`edges`/`hierarchy`/`papers`), it is what the visualisation and
the RAG layer consume, and it round-trips losslessly.

Add **flat CSV/TSV** (`nodes.csv`, `edges.csv`, `containment.csv`, `papers.csv`).
This is the format that gets *used* — it opens in R, pandas, Excel and Neo4j's
bulk importer without a parser. Hetionet ships TSV alongside JSON for this reason.

**Is RDF/Turtle with an ontology commitment worth it here? No — not yet.**

*For:* the graph already commits to NCBI Taxonomy and MONDO, so the hard part
(entity resolution against persistent identifiers) is done; Biolink Model would
fit the taxon–disease association pattern more or less off the shelf; it is what
Monarch/KG-Hub-adjacent consumers expect.

*Against, and it decides it:*
- **2,008 edges is small.** Nobody needs a SPARQL endpoint for a 3 MB file. The
  federation benefit of RDF accrues at a scale this graph is nowhere near.
- **The interesting structure does not survive naive triplification.** The
  distinguishing features here are *contested* edges (220 pairs where papers
  disagree and no direction is asserted), *evidence counts as weight*, per-edge
  *confidence tiers*, and per-edge *paper-level provenance*. Every one of those is
  edge-level metadata requiring reification, RDF-star, or singleton properties.
  Done carelessly, triplification flattens "220 contested pairs" into 440
  contradictory assertions — destroying the single most defensible design decision
  in the project. That is a worse outcome than not doing it.
- **Effort is better spent elsewhere.** See §3.4.

**If you do it later**, use **Biolink Model** (`biolink:association` reification
handles evidence and provenance natively; it is designed for exactly this), not a
hand-rolled ontology, and emit it as a *derived* artifact from `graph.json` rather
than as the source of truth.

**Neo4j dump:** don't ship one. Version-coupled, opaque, and unnecessary when the
CSVs load in one command.

**OBO:** wrong tool. OBO is for ontologies — term hierarchies — not instance data.
The containment layer is an *induced subgraph of NCBI Taxonomy*, not a new
ontology, and should be shipped as parent/child taxid pairs (which `hierarchy`
already is).

### 3.4 The thing to fix before any release — provenance identifiers

`graph.json`'s `papers` table has 271 entries whose keys are
`country, n_cases, n_controls, seq, site, region, med, diet, kit, platform,
pipeline, feature, da, norm, title, has_meta, has_methods`.

**There is no DOI, no PMID, no PMCID.** Edges reference their sources by
**full title string**:

```json
"papers": ["Disruptions of Anaerobic Gut Bacteria Are Associated with Stroke
           and Post-stroke Infection: a Prospective Case-Control Study", ...]
```

This is the single biggest defect in the release as it stands, and it is not
primarily a legal problem — it is a **FAIR** problem and a usability problem:

- It **fails FAIR F1** ("(meta)data are assigned a globally unique and persistent
  identifier") for the provenance layer, and weakens **R1.2** ("associated with
  detailed provenance") — the provenance is detailed but not resolvable.
- Title strings are a **fragile join key**. This project has already been bitten
  twice by exactly this class of error: 12 duplicate papers found by
  deduplication, and the `Prevotella`/Parkinson's miss from joining on another
  database's identifier.
- Without DOIs, **the Option-D manifest cannot be built**. The manifest *is* the
  release. This is on the critical path.
- It prevents users from doing the obvious thing — clicking through to the paper.

**Add `doi`, `pmid`, `pmcid`, and the measured `license` to every row of the
papers table before release.** The identifiers exist: I resolved 246 of 250 DOIs
and 188 of 250 PMC IDs for the annotated corpus in the course of this review, and
the resolution artifacts are in the scratchpad
(`licence_merged.json`, `licence_classes_final.json`). Carrying the licence field
through also means each edge's provenance states its own redistribution status,
which is a genuinely novel and useful thing for a literature-derived KG to do.

---

## Part 4 — Documentation standards

### 4.1 What to actually produce

Five documents. In priority order:

1. **A Datasheet for Datasets** (Gebru et al., "Datasheets for Datasets," CACM
   64(12), 2021; arXiv:1803.09010). The seven sections — Motivation, Composition,
   Collection Process, Preprocessing/Cleaning/Labeling, Uses, Distribution,
   Maintenance — are a good fit and, unusually, this project can fill them in
   properly. Most datasheets are vague in *Collection Process* and *Uses*; this
   one has `FINDINGS_*.md` covering exactly those. Three sections deserve special
   care:
   - *Composition* — state the corpus licence distribution from §1.3.
   - *Uses* — state the extraction gate explicitly (reported statistical
     significance required, main text only, disease-vs-healthy-control only) and
     state the asymmetry recorded in the root `CLAUDE.md`: **this instrument can
     confirm a miss but cannot refute one.** That sentence belongs in the
     datasheet verbatim.
   - *Distribution* — state what was excluded and why. A datasheet that documents
     a deliberate exclusion for licensing reasons reads as competence.

2. **A Croissant `metadata.json`** (MLCommons, v1.0, published 2024-03-01,
   `http://mlcommons.org/croissant/`). JSON-LD over schema.org, with
   `FileObject` / `FileSet` / `RecordSet` / `Field`. Hugging Face consumes it
   natively, so the benchmark becomes machine-discoverable for free. Cheap to
   produce and increasingly expected for anything calling itself a benchmark.
   Set `license` accurately here — it is a structured field and it will be read
   by machines.

3. **Bioschemas markup** on the published site
   (<https://www.mohakprakash.com/KnightLabV2/>) — `Dataset` profile plus
   `DataCatalog`, as JSON-LD in the page head. This is how life-science dataset
   aggregators find things. Low effort, real discoverability return, and the site
   already exists.

4. **A `LICENSES.md`** with the per-component licence grid from Part 5, plus the
   per-paper licence table. There is currently **no `LICENSE` file at all** in the
   repository, which means the code is technically all-rights-reserved and nobody
   may legally reuse it.

5. **A model/extraction card** for the extraction pipeline — which model
   (Qwopus3.5-27B q4_k_m GGUF), which prompt (`samgated-v1` in
   `eval-v2/run_eval.py`), the gate, and the measured fidelity/recall figures
   *with their stated caveats*.

### 4.2 FAIR, as actually assessed

FAIR is routinely claimed and rarely measured. Against the principles as stated
(<https://www.gofair.foundation/fair-principles>), the release would land roughly:

| | Principle | Status |
|---|---|---|
| **F1** | "(meta)data are assigned a globally unique and persistent identifier" | **Fails today** for the provenance layer (§3.4). Zenodo DOI fixes the dataset; paper DOIs fix the edges |
| F2 | rich metadata | Strong — the `FINDINGS_*.md` corpus is unusually good |
| F3 | metadata include the data's identifier | Trivial once F1 is done |
| F4 | registered/indexed in a searchable resource | Zenodo + HF + Bioschemas |
| A1 | retrievable by identifier over a standard protocol | Yes via Zenodo/HF |
| A2 | "metadata are accessible, even when the data are no longer available" | Zenodo guarantees this. **This is the principle that makes Option D work**: if the full text must be withheld, the *metadata and manifest* still persist, and the dataset remains reconstructible |
| I1 | formal shared knowledge representation | Partial — JSON with documented schema, no RDF (§3.3). Acceptable; be honest rather than claiming otherwise |
| I2 | FAIR vocabularies | **Strong.** NCBI Taxonomy + MONDO, resolved with documented refusals — better than most KGs at this scale |
| I3 | qualified references to other (meta)data | Partial — containment and disease is-a links are qualified; paper links are not until F1 is fixed |
| **R1.1** | "(meta)data are released with a clear and accessible data usage license" | **Fails today.** No LICENSE file anywhere. Part 5 |
| R1.2 | detailed provenance | **Exemplary** — per-edge, per-paper, with direction, counts and contest status. This is the project's strongest FAIR dimension |
| R1.3 | domain-relevant community standards | Partial — Datasheet + Croissant + Bioschemas would close it |

Assess with **FAIR-Checker** or the **F-UJI** automated evaluator and publish the
score in the datasheet. Publishing a mediocre measured score with the failing
indicators named is more credible than an unmeasured "FAIR-compliant" claim, and
two of the three failures above are fixed by adding a LICENSE file and DOIs.

### 4.3 Exemplars to imitate

- **Hetionet** (het.io; Himmelstein et al., *eLife* 2017;6:e26726). The closest
  structural analogue: an integrative biomedical heterogeneous network built from
  many upstream resources with *mixed and incompatible licences*, released as
  "open-source and free to use, barring any upstream restrictions." Two things to
  copy: the **explicit per-source licence accounting** (they enumerate what each
  upstream resource permits rather than asserting a blanket licence over the
  whole), and the **multi-format release** (JSON + TSV + Neo4j + a hosted
  browser). The "barring any upstream restrictions" formulation is itself worth
  borrowing — it is honest about the limits of what the compiler can grant.
- **BioRED** (Luo, Lai, Wei, Arighi & Lu, *Briefings in Bioinformatics* 2022,
  bbac282; `https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/`). The benchmark exemplar.
  Its README carries the NCBI **Public Domain Notice** — "a 'United States
  Government Work' ... cannot be copyrighted ... The National Library of Medicine
  and the U.S. Government have not placed any restriction on its use or
  reproduction" — with a request to cite. Copy the *structure*: corpus + explicit
  annotation guideline PDF + baseline model + source code + train/dev/test split,
  all in one archive with one citation. The annotation guideline as a separate
  first-class document is what makes a benchmark reusable.
- **Monarch Initiative / Biolink Model** — the reference for how a biomedical KG
  should model associations with evidence and provenance, if and when RDF is
  revisited.

---

## Part 5 — Licensing the release itself

Licence **each component separately**. A single blanket licence is where mixed
provenance goes wrong.

| Component | Recommended licence | Reasoning |
|---|---|---|
| **The graph** (`graph.json`, node/edge CSVs, containment, RAG corpus) | **CC0 1.0** | It is a factual compilation. CC0 maximises reuse and avoids licence-stacking downstream. Attribution requested via a CITATION.cff, not compelled. This is what comparable resources do and is standard practice for factual biomedical data |
| **The benchmark / eval suite** (adjudicated fidelity set, recall packets, screen, guidelines) | **CC BY 4.0** | The adjudications and guidelines contain genuine expressive authorship. CC BY keeps it usable while preserving credit — and credit is the whole incentive for a benchmark |
| **The code** (`build_kg.py`, `taxonomy.py`, `mondo.py`, harness) | **MIT** or **BSD-3** | Permissive, standard, no friction. Any is fine; pick one and add the file, because there is currently none |
| **Sentence extracts**, if released | **No open licence.** Distribute under a terms-of-use notice | You do not own the underlying text and cannot license it. State: "these fragments are quoted from the cited works under [fair use / DSM Art. 3]; rights remain with the respective rightsholders; redistribution is the recipient's responsibility." Mirrors PMC's own "directly and solely responsible for compliance" framing |
| **Full text of the CC BY / CC0 subset**, if released | **Per-paper, pass through the original licence** | Ship a `license` column per record and an attribution field. Never relicense CC BY content as CC0 — that is not yours to do |
| **Everything else (closed, NC-ND, unclear)** | **Not released.** Manifest of identifiers only | §1.3, Part 2 |

Three specific warnings:

1. **Do not apply CC BY to the graph *because* the sources are CC BY.** That
   confuses two things. Our edges are our facts about their papers, not copies of
   their expression. Applying CC BY implies we are redistributing their
   copyrighted work, which is a *worse* claim than the true one.
2. **Do not apply an NC licence to the graph** to "cover" the NC-restricted
   sources. NC is sticky, poisons downstream reuse, is notoriously ill-defined,
   and would not cure the underlying problem anyway. Exclude the material instead.
3. **A CC licence on data you lack rights to is an affirmative
   misrepresentation** — it tells every downstream user they may redistribute. If
   any full text ships, its record must carry its *actual* upstream licence.

---

## Part 6 — The benchmark angle

### 6.1 Is this a benchmark, honestly?

The eval suite is the more valuable artifact — the brief's instinct is right, and
for a reason worth naming: **it measures a quantity the existing benchmarks do
not.**

The established biomedical RE benchmarks are annotated-corpus benchmarks over
**abstracts**:

- **BC5CDR** (BioCreative V CDR, 2015) — 1,500 PubMed abstracts, chemical-disease
  relations.
- **ChemProt** (BioCreative VI, 2017) and **DrugProt** (BioCreative VII, 2021) —
  chemical–protein relations, abstracts.
- **BioRED** (2022) — **600 PubMed abstracts**, multiple entity types, multiple
  relation types, plus document-level and *novelty* labels. Currently the
  strongest general-purpose biomedical RE benchmark.
- **BioGRID** and similar are curated *resources*, not benchmarks — no held-out
  split, no inter-annotator agreement, not designed for scoring systems.

Note again what they have in common: **abstracts**. That is substantially a
licensing choice (§1.8), and it is exactly the constraint this project's suite
escapes — and the reason it inherits the problem in Part 0.

**What is genuinely distinctive here:**

1. **Full-text, not abstracts.** The relation is frequently only reported in
   Results with its statistic; the abstract states the headline. This is a real
   gap in the field.
2. **A gated task definition.** "Extract only where statistical significance is
   reported, main text only, disease-vs-healthy-control only" is a much more
   realistic specification than "find all relations," and it is documented and
   enforced.
3. **Reading fidelity measured against the papers' own sentences** — independent
   of any gold standard and any external database. This is methodologically
   unusual and is the project's best idea.
4. **A documented failure mode of the evaluation instrument itself.** "This
   instrument can confirm a miss but cannot refute one," and the finding that
   scoring an extractor without applying its own gate *manufactures* misses (4
   reported, 1 real). Very few benchmarks document their own measurement
   asymmetry. This is the most transferable thing the project has.
5. **A negative-result corpus** — 24-25 variables, 24-25 nulls on discordance, with
   MDEs stated. Rare and valuable.
6. **Honest calibration of its own agreement figures** — the finding that 73%
   Disbiome agreement decomposes into ~90% reading fidelity and ~55%
   cross-literature reproducibility, and should never be quoted pooled.

### 6.2 Be honest about adoption

**Most one-off benchmarks are never used by anyone but their authors.** The
biomedical NLP literature is littered with them. The ones that get adopted share a
small number of properties, and the differentiator is almost never quality:

- **They ride a shared task.** BC5CDR, ChemProt and DrugProt were all BioCreative
  tracks. A competition with participants and a deadline produces adoption;
  posting a tarball does not.
- **They have a leaderboard someone maintains.**
- **They are small enough to run cheaply.** 209 observations across 122 papers is
  a good size. Full-text prompting costs are the friction — ship a truncated
  variant.
- **They ship a baseline and an eval script**, so a newcomer gets a number in
  under an hour.
- **They have an annotation guideline** others can extend.
- **They are legally unencumbered.** A benchmark people cannot redistribute does
  not get adopted, and a full-text benchmark with a licensing cloud over it is
  dead on arrival. **This is the strongest argument for fixing Part 0 properly:
  the legal cleanup is not a compliance chore, it is a precondition for the
  artifact's scientific usefulness.**

### 6.3 What would maximise the odds

In order:

1. **Slot in rather than stand alone.** Report the extractor on **BioRED** as well
   as on this suite. A new benchmark that also gives numbers on the accepted one
   is credible; one that only reports its own is not.
2. **Frame it as a full-text complement to BioRED**, not a competitor. "BioRED
   covers abstracts; this covers gated full-text extraction" is a defensible,
   modest, true claim.
3. **Approach BioCreative.** A track proposal is the single highest-leverage move
   available for adoption, and the gated-extraction task is a good fit.
4. **Ship the guideline as a first-class document** (BioRED's model).
5. **Lead with the measurement-asymmetry finding.** The gate/instrument result
   generalises well beyond microbiome KGs, and it is the sort of thing that gets
   cited even by people who never run the benchmark.
6. **Resolve or retire the in-house gold.** The root `CLAUDE.md` already records
   it as unreliable and under audit. Ship the *adjudicated* fidelity set, which is
   sound, and either publish the error rate for the old gold or leave it out. Do
   not ship a benchmark whose reference is known-flawed without saying so
   prominently.

---

## Recommendation (ranked)

**1. Stop the bleeding on the already-public full text — this week, and before any
release, announcement or paper submission.**

The exposure is **1,902 distinct papers' full text**, not the ~7 MB the brief
describes. Of the 250-paper annotated corpus, **21.2% has no established
redistribution right and 37 papers are outright closed access**; the
1,724-paper `MAIN_DATA.json.zip` is unaudited and its composition (202 Nature
full texts against Springer Nature's explicit "without access for third parties"
condition) suggests it is worse.

Sequence:

  a. **Do not push the 7 local commits** until this is decided. `prompt_exp_subset.json`
     (80 more full texts) is tracked locally and not yet public — keep it that way.
  b. **Audit `MAIN_DATA.json.zip`** with the script used here. You cannot decide
     without the number, and it is a few minutes of API calls.
  c. **`git rm` the full-text files from HEAD and push** (Option B). Instant,
     reversible, stops casual access.
  d. **Take the licence question to UCSD Library Scholarly Communication.** Bring
     the §1.3 table and the §0 inventory — a measured problem gets a fast answer.
     Ask specifically about: the 37 closed-access papers, the ~200 Nature texts,
     the CC BY-ND adaptation question, and the sentence-level artifact.
  e. **With the PI's sign-off, rewrite history** (Option C). The single fork
     predates the files and does not contain them, so this genuinely works here.
     Coordinate across everyone holding a clone; back up first; re-check the
     `mega_dump/proj_2` submodule and the Pages build afterwards.
  f. **Cap the sentence extracts** — 300-400 chars per fragment, a ceiling on
     per-paper share. Drops 261 over-long fragments including one of 6,017
     characters, costs nothing scientifically, and moves the artifact from
     "mostly defensible with an indefensible tail" to uniformly defensible.

Do **not** leave it as-is. The CC BY/CC0 58% would be fine to leave; the rest is
not, and a planned release is exactly what turns a dormant exposure into a live one.

**2. Add persistent identifiers to the papers table.** `graph.json` currently
records provenance by **title string** — no DOI, PMID or PMCID. This fails FAIR F1,
is a fragile join key in a project already bitten twice by fragile joins, and
**blocks the manifest that the entire release strategy depends on**. Add `doi`,
`pmid`, `pmcid` and the measured `license` per paper. The resolutions already
exist from this review. This is on the critical path and is not optional.

**3. Add a LICENSE file — there is none.** The repository is currently
all-rights-reserved by default, which fails FAIR R1.1 and means nobody may legally
reuse even the code. Per-component: **CC0** for the graph, **CC BY 4.0** for the
benchmark, **MIT** for the code, **no open licence** for anything quoted from
papers. Never apply an open licence to text you do not own.

**4. Release under the Option-D design.** Public: the KG, the eval suite, the
code, the length-capped sentence extracts, a DOI manifest, and a fetch script
through PMC-approved APIs. Optionally the full text of the **145 CC BY/CC0/CC
BY-SA papers** as a genuinely redistributable full-text corpus — that is a real
contribution and unambiguously yours. Retain the complete corpus in
access-controlled institutional storage for reproducibility, which is exactly the
use DSM Article 3(2) describes.

**5. Zenodo for the DOI, Hugging Face for the benchmark, GitHub for code.** Cite
Zenodo's **concept DOI** so the citation survives the revisions this project
reliably produces. Skip RDF for now — 2,008 edges does not need it, and naive
triplification would destroy the contested-edge semantics that are the graph's
best design decision. Ship JSON + flat CSVs. Revisit with **Biolink Model** if
federation ever becomes a real requirement.

**6. Document it properly.** Datasheet for Datasets, Croissant `metadata.json`,
Bioschemas on the published site, `LICENSES.md`, an extraction card. Run F-UJI or
FAIR-Checker and publish the measured score with its failures named. Imitate
**Hetionet** for per-source licence accounting and multi-format release, and
**BioRED** for benchmark packaging — corpus, annotation guideline, baseline, code,
one citation.

**7. Give the benchmark a chance at adoption, and be realistic.** Report the
extractor on **BioRED** alongside this suite; frame this as the **full-text,
significance-gated complement** to BioRED's abstracts; ship the annotation
guideline as a first-class document with a baseline and a one-command eval script;
approach **BioCreative** about a shared task. Lead with the
instrument-asymmetry finding — it is the most transferable result the project has.
Most one-off benchmarks are never adopted; the ones that are, ride a shared task
and are legally unencumbered. Recommendation 1 is a precondition for
recommendation 7, not a distraction from it.

---

## Sources

- [PMC Copyright Notice](https://pmc.ncbi.nlm.nih.gov/about/copyright/)
- [PMC Text Mining / PMC Article Datasets](https://pmc.ncbi.nlm.nih.gov/tools/textmining/)
- [PMC Open Access Subset](https://pmc.ncbi.nlm.nih.gov/tools/openftlist/)
- [PMC Author Manuscript Collection](https://pmc.ncbi.nlm.nih.gov/about/authorms/)
- [NLM Terms and Conditions for downloadable data](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html)
- [Directive (EU) 2019/790, Article 3](https://www.legislation.gov.uk/eudr/2019/790/article/3)
- [Directive (EU) 2019/790, Article 4](https://www.legislation.gov.uk/eudr/2019/790/article/4)
- [Elsevier text and data mining policy](https://www.elsevier.com/about/policies-and-standards/text-and-data-mining)
- [Elsevier developer portal — text mining](https://dev.elsevier.com/tecdoc_text_mining.html)
- [Springer Nature text and data mining](https://www.springernature.com/gp/researchers/text-and-data-mining)
- [GitHub — Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Zenodo — file size limits and versioning](https://help.zenodo.org/docs/deposit/manage-files/)
- [Croissant specification v1.0 (MLCommons, 2024-03-01)](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)
- [Hugging Face — gated datasets](https://huggingface.co/docs/hub/datasets-gated)
- [FAIR Principles (GO FAIR Foundation)](https://www.gofair.foundation/fair-principles)
- [BioRED distribution and README](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/)
- [Hetionet](https://het.io/about/)
- Gebru et al., "Datasheets for Datasets," CACM 64(12), 2021 — [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)
- Luo, Lai, Wei, Arighi & Lu, "BioRED: a rich biomedical relation extraction dataset," *Briefings in Bioinformatics*, 2022 — [doi:10.1093/bib/bbac282](https://doi.org/10.1093/bib/bbac282)
- Himmelstein et al., "Systematic integration of biomedical knowledge prioritizes drugs for repurposing," *eLife* 2017;6:e26726 — [doi:10.7554/eLife.26726](https://doi.org/10.7554/eLife.26726)

*Case law referenced for context only, not as legal advice: Feist Publications v.
Rural Telephone Service, 499 U.S. 340 (1991); Authors Guild v. Google, 804 F.3d 202
(2d Cir. 2015); Authors Guild v. HathiTrust, 755 F.3d 87 (2d Cir. 2014); A.V. ex
rel. Vanderhye v. iParadigms, 562 F.3d 630 (4th Cir. 2009).*

*Measurement artifacts supporting Part 0 and §1.3 are in the session scratchpad:
`licence_merged.json` (per-paper final classification), `licence_classes_final.json`,
`licences_per_article.json` (raw PMC `<permissions>` parses), `unpaywall.json`.
Copy them into the repo if the numbers are to be cited.*
