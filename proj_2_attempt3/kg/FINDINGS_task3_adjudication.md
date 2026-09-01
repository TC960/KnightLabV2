# Adjudicating the pairs contradicted by BOTH Disbiome and Peryton

*Verdicts with quotes: `adjudication_verdicts.json`. Packets: `build_adjudication_packets.py`
→ `_adjudication/`. Pair list: `doubly_contradicted.json`.*

Two independent curations disagreeing with us is the strongest error signal
available, so each pair was read rather than settled by vote. On the screened
326-paper graph there are **14** such pairs (the list of 11 on record predates the
rebuild). Evidence is the paper's own sentences naming the taxon, pulled from the
relation-sentence substrate validated at 93.9% recall.

## Result: we are mostly right, and that is the finding

| verdict | n |
|---|---:|
| **ours correct — genuine literature dispute** | 9 |
| **rank artifact** (direction right, node wrong) | 1 |
| **body-site mismatch** (not a contradiction at all) | 1 |
| **extraction error — ours wrong** | **1** |
| unadjudicable (no retrievable text) | 2 |

**11 of 12 adjudicable pairs were faithfully extracted from the source papers.**
The strongest available error signal turns up **one** extraction error in fourteen.
So the doubly-contradicted set is not, as assumed, a pile of our mistakes — it is
mostly the literature genuinely disagreeing with itself, plus two *structural*
defects in how the graph is keyed.

### Three of the nine disputes are acknowledged by the source papers themselves

This is the strongest evidence that these are real disagreements and not extraction
noise — the authors saw the conflict and said so:

- **Dorea / Alzheimer's** — *"which was contrary to Liu's findings (Liu et al., 2019)"*
- **Dialister / Parkinson's** — *"the genus Dialister has previously been shown to
  have a higher relative abundance in PD patients in a Southern China population
  [50], which may reflect dietary or other geographical differences"*
- **Halomonas / Alzheimer's** — *"Different from Vogt's and Liu's studies, we found
  that such non-abundant genera as … Halomonas were also enriched"*

*Dialister* is the model case: correctly extracted, large effect (>10-fold), and
the paper names both the contradicting study and a candidate mechanism. That is
what a legitimately contested edge looks like, and averaging it away would destroy
information.

## The one real error

**Phascolarctobacterium / Parkinson's — DROP.** The only sentence naming it says:

> "On genus level, in addition to Faecalibacterium, also Bacteroides, Clostridium,
> **Phascolarctobacterium**, Coprococcus, Odoribacter were **correlated with disease
> stage**."

That is a correlation with severity *within patients*. It states no direction and no
case-vs-control contrast. The "depleted" direction was manufactured. Both databases
are more likely right. This is a specific, checkable extractor failure mode —
**reading a severity correlation as a disease-vs-healthy direction** — and it is
worth a targeted audit, because the extraction prompt is explicitly gated on
disease-vs-healthy-control and this slipped through.

## Two structural defects, both bigger than the error

### 1. Rank placeholders are folded into their parent family

**Erysipelotrichaceae / Parkinson's** looked like a 4-paper contradiction. It is not.
Three of the four papers report **"Erysipelotrichaceae UCG-003"** — an uncultured
*genus-level* SILVA placeholder *inside* the family — which `taxonomy.py` resolved
to the family taxid. The fourth names the family but attributes the change to a
member species (*Eubacterium biforme*). **No paper measures the family aggregate.**
Our family-level "depleted" is an aggregation artifact; the databases record the
family enriched; both can hold at once.

This is systematic, not a one-off (`adjudicate_rankconf.py`):

- **74** distinct placeholder strings folded onto **37** taxids
- **21** edges whose taxon is *only ever* named by a placeholder
- **170** edges with mixed placeholder/exact evidence, **52 of them contested**

Worst offenders: `Lachnospiraceae` absorbs *ND3007 group, ND3008, NK4A136 group,
UCG-001, UCG-004, UCG-008*; `Oscillospiraceae` absorbs six *UCG-/NK4A214* labels;
`Clostridia` absorbs *UCG-014*.

This directly contradicts the project's own load-bearing rule — "synonym folding
(same rank, renamed) and containment (different ranks) are different operations".
A UCG placeholder is a *child*, not a synonym, and is being folded as if it were one.

### 2. Body site is not part of the edge key

**Rothia / Parkinson's** is not a contradiction: both our papers are **saliva/oral**
studies in PD-with-dementia, while the curated records are gut. An oral finding and
a gut finding collide on one node. *Gemella / Parkinson's* comes from the same
saliva paper and is almost certainly the same story. The graph stores body site on
the *paper*, not the *edge*, so nothing prevents this.

## What to do, in priority order

1. **Stop folding rank placeholders into parents.** Give `X UCG-003` its own node
   as a child of `X`, linked by containment. This touches 191 edges and would
   resolve the single worst-looking contradiction in the set. Highest value, and it
   is a bug fix rather than a judgement call.
2. **Put body site in the edge key**, or restrict the graph to stool and mark oral
   studies separately. Two of fourteen doubly-contradicted pairs are this.
3. **Drop the Phascolarctobacterium edge** and audit for the same failure mode:
   edges whose only support is a severity/stage correlation.
4. **Re-run the sentence filter over the full 326-paper corpus.** It currently
   covers only the original 250, which is why 2 of 14 pairs could not be
   adjudicated at all.

## Caveats

- 9 of the 14 rest on **one** paper on our side against **one** curated record on
  theirs. "Who is right" is frequently not answerable at that evidence level, and
  a correctly-extracted single paper disagreeing with a single curated record is a
  disagreement, not an error by either party.
- Adjudication used the filtered relation sentences, not full texts. The filter has
  93.9% recall, so a supporting sentence could in principle have been missed —
  though for the one error found, the absence of any case-control sentence is
  itself the evidence.
