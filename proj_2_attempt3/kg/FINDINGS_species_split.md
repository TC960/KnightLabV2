# The species that were folding into their genus

*2026-09-08. Cloud session, CPU only, no `MAIN_DATA.json`, no NCBI taxdump.*

**The project's top open defect is fixed, and three things about it were wrong.**
It was not blocked on the taxdump. It was not 54 species. And its worst instance
was not the one the docs named.

---

## 1. It was never blocked on the taxdump

Three sessions recorded this as "mechanical on a machine with the NCBI taxdump,
impossible here" because the environment's network policy denies
`ftp.ncbi.nih.gov` (CONNECT → 403). That is still true; it was re-probed, along
with `ftp.ncbi.nlm.nih.gov`, `eutils`, `api.ncbi.nlm.nih.gov`, Ensembl and EBI.
All shut.

But the mapping never required `names.dmp`. **An NCBI taxid is stable across a
rename** — NCBI renames the taxon, it does not renumber it. That makes the taxid
a join key that renaming cannot move, and two sources already in the repo or
reachable sit on either side of it:

| side | source | what it gives |
|---|---|---|
| old name → taxid | **Disbiome** (`disbiome_experiments.json`, already committed) | curated before the 2024-25 reclassifications, so it stores `"Prevotella copri" → 165179` |
| taxid → current name | **`ncbi-taxon-db`** (the NCBI taxonomy redistributed on PyPI, behind `taxoniq`; PyPI is reachable) | `165179 → Segatella copri, rank species` |

Neither source knows about the other. The join is arithmetic, not judgement, and
every entry in `species_synonyms.json` carries the route and the evidence string
that produced it.

Worth stating plainly for the next session: **`ncbi-taxon-db` is not a taxdump
substitute in general.** It has current scientific names only — no synonym table,
which is why `taxoniq.Taxon(scientific_name="Prevotella copri")` raises. Every
one of the genuinely-species strings here failed a direct lookup. It is the
*taxid* side of the join that it serves, and only that.

## 2. It was 24 species, not 54 — and the other 91 must NOT be split

`child_folds.json` classified 115 surface strings that extend the scientific name
they resolved to: 32 rank placeholders (already split), 29 `X sp./spp.`
(correctly folded), and **54 `named_child`, read as "54 real species collapsed
into their genus".**

That reading is wrong, and a mechanical split of all 54 would have damaged the
graph. Only **24** name a species. The rest are things that must stay folded:

- **Two taxa in one string** — `Escherichia / Shigella`, `Escherichia – Shigella`,
  `Escherichia_Shigella`, `Streptococcus salivarius/thermophilus`. A 16S assay
  that cannot separate two genera has not measured either one. These cannot
  become a species node without inventing a measurement.
- **Pipeline cluster labels** — `Clostridium_XlVa`, `Eubacterium_g4`,
  `Prevotella VZCB`, `Lachnospiraceae_NC2004`, `Erysipelotrichaceae CCM`,
  `Clostridium g24 FCEY`, `Turicibacter sp001543345`. RDP/SILVA/GTDB identifiers,
  not binomials.
- **Names no organism** — `Neisseria multispecies`, `bacterium NLAE`,
  `Eikenella species NML 130454`.
- **Strain codes under a parent that is already a species** —
  `Ruminococcus sp. 5_1_39BFAA`, `Lachnospiraceae bacterium MC-35`,
  `Sarcina sp. JB2`. Splitting these makes strain nodes.
- **Phages** — `Escherichia phage TL 2011`, `Vibrio phage pYD38 A`. A phage is not
  a member of the genus it infects; the existing `NOT_CONTAINED` guard exists
  because a previous split hung two of them under bacterial genera.

The resolution ladder in `species_synonyms.py` **is** the classifier — these fail
it and stay put. Nothing on that list was sorted by hand, which matters, because
this repo's standing rule is not to trust a model's judgement call where a
deterministic test exists.

## 3. The cause was not a missing taxdump either

The shipped graph resolves renamed binomials perfectly well: `Clostridium
aldenense` is a node labelled *Enterocloster aldenensis*, `Eubacterium eligens`
is *Lachnospira eligens*, and `Roseburia faecis`/`Agathobacter faecis` pool into
one node. What fails is a specific subset of the 2024-25 reclassifications whose
old binomials the lookup does not return — *Prevotella copri*, *Eubacterium
rectale*, *Bacteroides vulgatus/dorei/plebeius/coprophilus*, *Clostridium
clostridioforme*, *Prevotella buccae/timonensis/shahii*. For those, `resolve()`
falls through to its qualifier-tail trim, **throws the species epithet away**, and
lands the mention on the genus.

**The proof that this is a collision and not a naming quibble: three of these
organisms were already nodes in the graph under their new names.** Papers writing
"Phocaeicola dorei" built a species node; papers writing "Bacteroides dorei" were
folded into genus *Bacteroides*. One organism, two nodes, two ranks, no link
between them. Same for *Holdemanella biformis* / "Eubacterium biforme" and
*Enterocloster clostridioformis* / "Clostridium clostridioforme". This was found
by asking a cheap structural question of the graph — *does any folded string name
an organism that is already a node?* — which is the method that keeps working
here.

## 4. What the split changed

| | before | after |
|---|---|---|
| taxa | 918 | **929** |
| edges | 2,011 | **2,043** |
| containment links | 708 | **719** |
| contested edges | 219 | **215** |
| resolved taxa | 660 | **671** |

Twelve species nodes appear (*Segatella copri*, *Agathobacter rectalis*,
*Phocaeicola vulgatus/coprophilus*, *Hoylesella shahii/timonensis*, *Segatella
buccae*, *Prevotella jejuni*, *Hungatella effluvii*, *Oribacterium sinus*, and two
"ex auct." forms). One node disappears: genus *Oribacterium* had no evidence that
was not the string `Oribacterium sinu`.

**The headline case is *Eubacterium*, not *Prevotella copri*.** *Eubacterium* /
Parkinson's went **4 up / 5 dn (contested) → 2 up / 1 dn**. That contest was
largely two species pulling opposite ways inside one genus node: *E. rectale*
(now *Agathobacter rectalis*, Lachnospiraceae) consistently depleted, *E. biforme*
(now *Holdemanella biformis*, Erysipelotrichaceae) mostly enriched. NCBI no longer
places either in *Eubacterium*, or even in the same family. Pooling them
manufactured a disagreement that the literature never had.

*Prevotella* / Parkinson's — the flagship edge — went 3 up / 14 dn → **2 up /
11 dn**. Direction unchanged, still contested. The hypothesis that *P. copri*
might be carrying that edge, or flipping it, is a **null**: the species agrees in
direction with the genus (5 of 6 papers depleted).

## 5. Does external agreement move? Yes, and the count is better evidence than the ratio

The standing rule is *don't run a structural correction expecting agreement to
move* — five have now moved it by less than this corpus can resolve (~0.013). So
the ratio is reported here only alongside the thing that is actually decidable:
**which decisive pairs entered, left, or flipped.**

| | before | after |
|---|---|---|
| Disbiome overlap / recall | 260 / 51.2% | **269 / 53.0%** |
| Disbiome agreement | 71.7% | **73.1%** |
| Peryton overlap / recall | 220 / 72.6% | **224 / 73.9%** |
| Peryton agreement | 72.5% | **72.7%** |

*(the "before" column is the shipped graph re-scored under the new resolver, so
both sides pass through the same taxonomy; the previously published figures were
71.9% / 72.5%.)*

Enumerated rather than ratioed:

- **Disbiome: 11 decisive pairs entered. All 11 agree. 0 disagreements were
  added. 0 verdicts flipped.** Two agreeing pairs left.
- **Peryton: 1 entered (agrees), 0 left, 0 flipped.**

Against a baseline agreement rate of 0.717, eleven-for-eleven is p = 0.026 — but
those 11 pairs come from only **7 distinct taxa** (*Agathobacter rectalis* appears
in three diseases, *Bacteroides* in three), and clustering on taxon the way this
project clusters on paper gives **p = 0.097**. So: **suggestive, not significant.**
The honest claim is the coverage one, which needs no test — the split added 9 net
decisive Disbiome pairs and 4 net overlapping pairs, and did not create a single
new disagreement.

The two pairs that left are worth naming, because one is a real cost:

- *Faecalibacterium prausnitzii* / MS went 0 up / 7 dn → **1 up / 7 dn**, i.e.
  became contested and dropped out of the decisive set. The fuzzy route folded the
  misspelling `Faecalibacterium prauznitzii` into it, and that paper reported
  enrichment. This is correct behaviour — the paper says what it says — and a
  reminder that de-contesting is not the objective.
- *Prevotella* / Autism left the decisive set as the genus node lost its only
  species-string evidence.

## 6. The fear that blocked this was specific, and it was well-founded

`NEXT_SESSION_PROMPT.md` warned that splitting these in the cloud "would have LOST
the Disbiome/Peryton join for exactly the species that matter." That was right, and
the mechanism was ancestry, not naming: `build_kg.py` builds every containment link
by walking `tax.lineage()`, and the replay cache holds only graph-local parent
links, so a freshly split species would have had **no ancestry at all** and shipped
as a detached node. `species_synonyms.json` therefore stores the full NCBI lineage
per entry, and `taxonomy_cache.lineage()` returns it. One candidate,
*Lawsonibacter phoceensis*, is absent from the 2024 snapshot and is **refused
rather than split detached** — it stays folded, and is listed.

Note the correct ancestor is often *not* the genus the string used to fold into:
*Segatella copri* links to **Prevotellaceae**, since NCBI no longer places it in
*Prevotella* at all.

## 7. Ordering, which is the one subtle thing in the diff

The supplement is consulted **after** `names.dmp` in `taxonomy.py` — NCBI must
always win — but **before** the cached lookup in `taxonomy_cache.py`. There the
"authority" is a replay of `graph.json`, and these 24 entries are precisely what
`graph.json` got wrong: `"Prevotella copri"` sits in the cache as an *alias of the
genus node*. Checked cache-first, the table resolves to 838 forever and is dead
code. It was, for one run, until the resolver was executed and printed
`('838', 'Prevotella', 'genus', 'cached')`.

## 8. Verification

Rebuilt twice → **byte-identical fixed point**. `verify_viz.py` drives the page in
Chromium and asserts against the live DOM and canvas pixels: **19/19 pass**.
`docs/index.html` re-synced.

---

# Part 2 — punctuation was fragmenting concepts across nodes

Written after the above, and it corrects a claim in it. The section above ended by
saying "~20 ambiguous two-genus labels deserve a human decision". Measuring it
instead of estimating it produced a smaller number and a **bigger, different
defect underneath**.

## The finding

Asking which concepts land on more than one node — the same cheap structural
question as before — returns **17**. The worst is the one the estimate had already
noticed, and it is far worse than "needs a decision":

*Escherichia-Shigella* is the standard SILVA/QIIME label for a pair of genera that
16S cannot separate. The corpus writes it seven ways, and the graph filed those
**27 mentions under four different nodes**:

| written as | mentions | filed under |
|---|---:|---|
| `Escherichia-Shigella` | 11 | `t:escherichia-shigella` |
| `Escherichia_Shigella` | 6 | **`t:ncbi:561` — *Escherichia*** |
| `Escherichia/Shigella` | 5 | `t:escherichia/shigella` |
| `Escherichia–Shigella` (en dash) | 2 | `t:escherichia–shigella` |
| `Escherichia / Shigella` | 1 | **`t:ncbi:561` — *Escherichia*** |
| `Escherichia-shigella` | 1 | `t:escherichia-shigella` |
| `Escherichia – Shigella` | 1 | **`t:ncbi:561` — *Escherichia*** |

Two bugs in one place. **Fragmentation** — the same concept split by punctuation
alone, so its evidence never pools. **Misattribution** — and the split is not even
consistent: when the separator happens to be a space or an underscore, `resolve()`
turns it into a space, fails to match, then trims the trailing token *as if it were
a qualifier* and lands the mention on *Escherichia*. A signal from an assay that
could not tell two genera apart was being recorded as evidence about one of them,
depending on which punctuation the paper's authors happened to type.

## The three rules, and why each is narrow

1. **Refuse the qualifier trim when the discarded token is itself a taxon name.**
   "Escherichia" + discard "Shigella" is not a genus with a qualifier; it is two
   genera. Measured over all 1,090 distinct surface strings: **2 strings change**
   (`Escherichia_Shigella`, `Lachnospiraceae_Eubacterium`). A rule, not a rewrite.
2. **Collapse every separator style in the unresolved key.** Hyphen, en dash, em
   dash, slash and underscore all encode the same "A and/or B" join here. This
   touches only the *unresolved* path — anything NCBI resolved has already
   returned — so it cannot merge two taxa the taxonomy told us apart.
3. **Strip square brackets as a FALLBACK.** Brackets are NCBI's own convention for
   a genus known to be misplaced, so `[Eubacterium] siraeum` **is** a scientific
   name and must match as written. Pre-stripping them scores 6 gains and 1 loss;
   as a fallback it is purely additive and scores 6 gains and 0 losses, resolving
   `[ Ruminococcus ]`, `[Eubacterium] ventriosum group`, and — the useful one —
   `[Ruminococcus] gnavus group` → *Mediterraneibacter gnavus*.

## Effect

929 → **925** taxa, 2,043 → **2,034** edges, 719 → **723** containment links,
215 → **217** contested. Five concepts lost a duplicate node; *Escherichia* goes
from 10 edges / 21 papers to **7 edges / 13 papers** as eight papers of joint-genus
evidence leave it for the joint node, which goes to 13 edges / 29 papers.

**Agreement is unchanged, as the standing rule predicts**: Disbiome 73.1% → 73.0%,
Peryton 72.7% → 72.5%, disagreements identical at 47 and 38. The honest cost is
**one Disbiome overlap pair**, lost precisely because we stop crediting
*Escherichia* with evidence the assay could not attribute to it. That is the
correction working, not failing.

## A regression this caught, and how

The first version of rule 2 collapsed separators *before* the rank heuristic, which
guesses "two words means a binomial". `Escherichia-Shigella` is one written token
and is not a species; collapsed first it becomes two, and **~50 unresolved nodes
silently reranked from genus to species**. Nothing errored. It was caught by
diffing the rebuild against the previous graph, which is the only reason it is not
in the shipped artifact — reading the patch would not have found it.

## What is now the top open defect

1. **More papers.** The binding constraint on every remaining statistical question
   is n, not method. Needs a GPU.
2. **A modelling decision, not an analysis**: should a joint two-genus 16S signal
   be attributed to one genus, split across both, or held apart on its own node as
   it now is? The graph no longer decides this by accident; a human should decide
   it on purpose. Same class as disease subtypes as containment.
3. **~8 stray mentions still fragmented by run-together spellings**
   (`ChristensenellaceaeR_7group`, `RuminococcaceaeUCG_005`, `CAG56`,
   `Rumino coccus_1` — a space inserted *inside* a genus name). These need a
   fuzzy match against known names rather than a separator rule, and at 8 mentions
   they are not worth the false-merge risk yet.
