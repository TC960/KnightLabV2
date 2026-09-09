# Five contradictions, one paper: a systematic direction inversion in Disbiome's ALS curation

**Session of 2026-09-09. Script: `adjudicate_db_conflicts.py`. Data: `db_conflicts.json`.**

## Why these five and not the other disagreements

Where the graph disagrees with one curated database, it is an open question who is
wrong. Where **the two curations contradict each other**, at least one of them is
wrong by construction, and neither can serve as the other's reference. That is the
strongest error signal available here, stronger than the 11 doubly-*contradicted*
pairs the previous adjudication looked at, which were cases of both databases
agreeing against us.

`calibrate_agreement.py` found 5 such pairs, all in ALS:

| taxon | Disbiome | Peryton | ours |
|---|---|---|---|
| *Anaerostipes* | enriched | depleted | depleted |
| *Dorea* | depleted | enriched | enriched |
| *Eubacteriales* (Clostridiales) | enriched | depleted | depleted |
| *Lachnospiraceae* | enriched | depleted | depleted |
| *Oscillibacter* | enriched | depleted | depleted |

## The structural question first: they are not five contradictions

Before reading anything, ask which publications the records come from — the cheap
structural question that keeps working in this repo. **All ten records, five from
each database, come from a single paper: PMID 27703453**, Fang et al.,
"Evaluation of the Microbial Diversity in Amyotrophic Lateral Sclerosis Using
High-Throughput Sequencing."

So this is **one curation error, not five independent contradictions** — one paper
read in opposite directions by two curators, on every taxon they both extracted
from it. Reporting it as "5 contradicted pairs" would overstate the evidence by
five-fold, the same mistake the 2026-09-06 session corrected in the "229 opposite
pairs" figure.

## Who is right: the paper says so explicitly

We extracted this paper too, so `relation_sentences.json` carries its verbatim
text. The paper labels its groups **group A = ALS patients, group H = healthy**,
and states the directions outright:

> "Furthermore, the decreased Firmicutes/Bacteroidetes ratio at phylum level using
> LEfSE (LDA > 4.0), together with the **significant increased genus *Dorea***
> (harmful microorganisms) and **significant reduced genus *Oscillibacter*,
> *Anaerostipes*, *Lachnospiraceae*** (beneficial microorganisms) **in ALS
> patients**, indicated that the imbalance in intestinal microflora constitution
> had a strong association with the pathogenesis of ALS."

> "In Figure 4, *Lachnospiraceae* (at family level), *Firmicutes* (at phylum
> level), *Clostridia* (at class level), *Oscillibacter* (at genus level), Family
> XIII (at family level), *Anaerostipes* (at genus level), *Lachnospiraceae* (at
> genus level) and ***Clostridiales*** (at order level) **in group H were
> significant higher than that in group A**, while *Bacteroidetes*, *Bacteroidia*,
> *Bacteroidales*, ***Dorea*** were significant higher [in group A]."

Both sentences say the same thing, and the first needs no knowledge of the group
labels at all — it names "ALS patients" directly.

**Verdict: Peryton and this graph are right on all five; Disbiome has the
direction inverted on all five.** The error is systematic rather than five
separate slips — Disbiome appears to have swapped which group is which for this
paper, which flips every taxon at once, *Dorea* included (they record it depleted
where the other four are recorded enriched — exactly the pattern a group swap
produces, since *Dorea* is the one taxon the paper reports in the opposite
direction from the rest).

Our extractor read all five correctly, including *Eubacteriales*, which required
resolving the paper's "Clostridiales" through the taxonomy.

## Size of it, stated carefully

**5 of our 47 disagreements with Disbiome (10.6%) trace to this one mis-curated
paper.** Correcting them would move Disbiome agreement 127/174 (73.0%) →
132/174 (**75.9%**).

That figure is offered as **an error in the reference, not an accuracy gain for
the graph** — nothing about the graph changed, and the standing rule against
citing corrections as accuracy improvements applies with the sign reversed here.
It is also a caution about the 27% disagreement rate generally: some unknown
fraction of it is the reference being wrong, and this is the first case where that
has been demonstrated rather than assumed.

Note the interaction with `FINDINGS_independence.md`: this paper **is** in our
corpus, so these five are *shared-source* pairs — the bucket that otherwise agrees
87.5%. They are the clearest available demonstration that shared-source
disagreement, when it happens, is worth reading the paper over.

## What to do with it

1. **Report it upstream to Disbiome.** One PMID, five records, an explicit
   quotable sentence. This is the kind of thing a curated database wants.
2. **Do not fold the correction into our agreement figure silently.** If the
   75.9% is quoted anywhere it must be labelled as "after correcting a
   demonstrated reference error", with this document cited.
3. **The method generalises and is cheap.** Any cluster of disagreements that
   traces to a single publication is a candidate curation error rather than N
   findings. `adjudicate_db_conflicts.py` does the tracing; it is worth re-running
   whenever the corpus grows.

## Limits

- Five records is what the two databases both extracted from this paper; neither
  curated all of it, so the true number of inverted records in Disbiome for this
  PMID may be larger.
- This says nothing about Disbiome's accuracy in general. One paper, one error,
  found because two curations happened to overlap on it. The 11 pairs where both
  databases agree *against* us remain the stronger signal about our own errors,
  and were adjudicated separately (`FINDINGS_task3_adjudication.md`).
- Adjudication rests on the filtered relation sentences rather than the full text,
  which is sufficient here because the decisive sentences are summary statements
  the filter is designed to keep — but it would not settle a case that turned on
  a table or a figure.
