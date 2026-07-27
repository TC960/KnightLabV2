# Improvement ideas — scoped, tested, and what to build next

Autonomous follow-up on the three ideas (from your notes + your friend's suggestions). Each was
scoped against the literature and, where cheap, actually tested on our data. Verdicts are concrete.

---

## 0. Done & shipped this round

**Taxonomy-aware scoring is now baked into `run_eval.py`.**
- New module `taxonomy_match.py`: resolves each taxon to its NCBI lineage (gnparser → taxonkit) and
  counts a match when one taxon is an **ancestor of the other** (collapses genus/family/species rank
  variants), falling back to char-ngram cosine ≥0.5 for names NCBI can't resolve.
- `run_eval.py --metric taxonomy` (now the **default**) uses it; `--metric char` reverts to the old
  behaviour. If the tools aren't installed it prints a warning and falls back to char automatically —
  never hard-fails. One-time setup: `python taxonomy_match.py --setup` (downloads taxonkit + gnparser
  + NCBI taxdump into `/tmp`; re-run after a pod reset).
- Validated: reproduces the Qwopus3.5 = ~0.81 result on the Opus 4.8 benchmark.

---

## Idea 1 — "summarize → keyword sentence → embed in a taxonomy space" (semantic taxon normalization)

**What it is (lit-grounded).** This is exactly **dense-retrieval biomedical entity linking** (SapBERT /
BioSyn) with an **LLM query-expansion front-end** — a known, published recipe ("generative relevance
feedback for entity linking", Bioinformatics 2026). So the idea is sound and has precedent.

**Tested it (LLM-normalization proxy).** 18% of taxon names (43 / 236) fail NCBI resolution — SILVA/GTDB
placeholders, misspellings, abbreviations, bracket/underscore junk. Ran an LLM normalization pass over
those 43:
- **30 / 43 recovered** (70% of the tail; 91% of the 33 that were genuine taxa) — e.g. `Oscillobacter`→
  *Oscillibacter*, `c-Actinobacteria`→*Actinomycetia*, `[Ruminococcus]_gnavus_group`→*Mediterraneibacter
  gnavus*, `Bacteroides vulgatus`→*Phocaeicola vulgatus* (reclassified).
- The remaining 13 are **genuine placeholders** (`RB41`, `UBA1819`, `PAC001212_g`, `MND1`…) + "lactic
  acid bacteria" (a functional group, not a taxon) — truly unrecoverable.

**But — the honest catch.** Applying the normalization end-to-end **did not move eval F1 (+0.000)**.
Reason: tail names that appear in both prediction and gold already match each other by string; the
metric doesn't need NCBI to see that `Butyricoccus` == `Butyricoccus`.

**Verdict / where it actually belongs.** Idea 1 is **not an eval-scoring lever — it's a knowledge-graph
node-canonicalization lever, and there it's essential.** When we build the graph, `Bacteroides
vulgatus`, `Phocaeicola vulgatus`, and `[Ruminococcus]_gnavus_group` / `Mediterraneibacter gnavus`
**must collapse to one node** or the KG fragments. So: **adopt it, but at the KG-assembly stage**, as a
normalization/dedup pass, not in the eval metric. Best implementation per the lit: **gnverifier**
(cascading exact→fuzzy→partial match vs 100+ DBs, handles misspellings/placeholders taxonkit misses)
as the lexical layer + a **SapBERT dense pass** for the semantic residue; keep the LLM step only for the
genuinely ambiguous names, and make it preserve the verbatim string (avoid hallucinated names).

## Idea 2 — "break it apart into smaller steps"

**Strongly supported by the literature; recommend as the next extraction experiment.**
- **RELATE** (arXiv 2509.19057): retrieve candidate relations → **LLM re-ranks with an explicit NONE /
  reject option** → negation handling. This is the textbook fix for our *real* over-extraction false
  positives — a reject option lets the model decline borderline taxa instead of over-listing.
- **Two-stage scenario prompting** (arXiv 2505.01077): decompose into (1) entity/scenario setup →
  (2) relation extraction; reported to rival fine-tuning zero-shot.
- **Microbiome-specific finding** (bioRxiv 2025.08.29.671515): for the **normalization sub-task
  specifically, fine-tuned BERT beats LLMs** — so the right division of labor is *LLM for
  relation/direction, BERT/SapBERT for entity linking*.

**Proposed concrete pipeline to test** (on Qwopus, at scale): **(a)** candidate taxa + the exact
supporting sentence (grounding, à la LangExtract) → **(b)** per-candidate: significance check +
direction (↑/↓) + confirm it's a disease-vs-HC contrast, **with a reject option**. Expect this to lift
precision on the weaker local models (Opus 4.8 is already near-ceiling, so the win shows up on Qwopus).

## Idea 3 — "check how others have done adjacent work"

Done — full lit scan. Highlights worth stealing:
- **Evaluation:** hierarchy-aware / semantic F1 (arXiv 2510.11313) gives partial credit for
  right-lineage-wrong-rank — a principled upgrade over our binary nested match. LLM-as-judge
  (arXiv 2506.00777) as a secondary metric to tell true over-extraction from gold-standard gaps.
- **Validation data:** **Peryton** (NAR 2021) and **Disbiome** — manually curated microbe–disease
  associations *with direction + source*, both covering our **neurodegenerative** disease mix. Use as
  external ground truth for extracted edges. Disbiome's record schema (taxon, disease, direction,
  method, source) ≈ our target edge schema.
- **Downstream:** once edges exist, random-walk-with-restart over the merged
  Peryton+Disbiome+gutMDisorder network (PMC8315281) is the standard link-prediction next step.
- **Tooling:** **gnverifier** (fuzzy taxon verification), **LangExtract** (source-grounded structured
  extraction — lighter than our GBNF grammar, gives free provenance).

---

## Recommended next steps (in order)
1. **Start the KG** — run Qwopus over the full corpus; assemble microbe →↑/↓→ disease edges. This is the
   project's actual goal and everything above now feeds it.
2. **Add the normalization/dedup pass at KG-assembly** (gnverifier + SapBERT + LLM for hard cases) —
   idea 1, in its correct place.
3. **Prototype the decomposed extractor with a reject option** (idea 2 / RELATE) and A/B it against the
   current single-shot prompt on Qwopus — targets our real over-extraction FPs.
4. **Validate edges against Peryton / Disbiome** rather than only our 15-paper benchmark.
