# The taxonomy embeds in hyperbolic space — a sanity check that finally passes

*2026-09-19. The first non-null in the embedding line of work, and it is smaller
than it looks.*

---

## Result

A Poincaré embedding (Nickel & Kiela 2017) trained on the repaired containment
tree recovers taxonomic rank from geometry alone:

```
Spearman(radius, rank depth) = +0.648    p = 9.0e-80
  shuffled-rank null: mean -0.003, 95th pct |rho| 0.077
```

and the ordering is monotonic across every rank with enough members to score:

| rank | n | mean radius | hyperbolic distance from origin |
|---|---:|---:|---:|
| phylum | 20 | 0.9419 | 3.51 |
| class | 18 | 0.9498 | 3.66 |
| order | 29 | 0.9766 | 4.44 |
| family | 80 | 0.9904 | 5.33 |
| genus | 242 | 0.9951 | 6.02 |
| species | 263 | 0.9966 | 6.38 |

Stable across dimension: ρ = +0.617 at d=5, +0.648 at d=10.

**Report hyperbolic distance, not Euclidean radius.** Every rank from family down
sits at radius 0.99-plus and *looks* collapsed onto the boundary. It is not:
distance from the origin is `2·artanh(r)`, so 0.9904 and 0.9966 are 5.33 and 6.38
apart — a full rank-step, not a rounding artifact. Quoting the raw radii would
make a working embedding look degenerate.

---

## What this is NOT

**It is not a discovery about taxonomy.** We already knew every node's rank; it
comes from NCBI. A Poincaré embedding is *trained* to put depth on the radius —
that is its objective function. Recovering rank is the check that the thing
trained correctly, not a finding about the data.

**It does not touch the contested-edge question.** The four nulls on explaining
why papers disagree (26 study-design variables, 34 concepts, methods-only
embeddings, dimensionality variants) are all about the *paper* half of the graph.
This is the *taxon* half. Nothing here moves them, and nothing here should be
quoted as if it did.

**Most of the signal is tree depth, which is not news either.** Depth and rank
correlate at ρ = +0.721 by construction, since `repair_taxonomy_tree.py` builds
depth from full NCBI lineages. Controlling for depth, radius retains
ρ = +0.228 of rank — real but modest, and most plausibly the geometry smoothing
over uneven branch lengths (lineages carry different numbers of intermediate
no-rank nodes, so depth is a noisy proxy for rank and the embedding partially
corrects for it).

So the honest one-line summary: **the instrument now works.** Whether it measures
anything we don't already know is the next question, not this one.

---

## Why it took four attempts

Each bug produced a plausible-looking number rather than an error, which is the
reason this is worth writing down.

**1. Trained on a forest, not a tree.** `graph.json`'s hierarchy only links taxa
that *both* appear in our papers, leaving 20 disconnected fragments with holes in
every lineage. Depth-within-fragment is not rank — a genus hanging straight off a
phylum root sits at depth 1. Result: ρ = **+0.0095, p = 0.81**. The geometry
learned fragment depth perfectly; fragment depth means nothing. Fixed by
`repair_taxonomy_tree.py` (916 nodes, 1 root, 245 inserted Steiner ancestors).

**2. Hand-rolled numeric gradient.** Loss *rose* after epoch 50 and every rank
saturated at radius ~0.99 — genuine boundary collapse, as opposed to the healthy
near-boundary spread above. Fixed with autograd plus norm clipping before the
`arccosh`.

**3. Negatives sampled without excluding true ancestors.** A node's own
grandparent could be drawn as a negative and pushed *away*, fighting the positive
term pulling it close. The result was not weak but **inverted: ρ = −0.178**.
Nickel & Kiela define the negative set as `N(u) = {v : (u,v) ∉ D}` — explicitly
the non-ancestors. *A correlation that comes out backwards is a bug hypothesis,
not a finding.*

**4. Undertrained.** At the original 200 epochs, loss was still falling steadily
(2.32 → 1.68) and every radius sat at 0.4–0.5 — the embedding had barely left the
origin, giving ρ = −0.069, indistinguishable from null. At 3000 epochs loss
reaches 0.16 and the structure appears. **200 epochs would have been recorded as a
fifth null.** Runtime is 80 seconds; there was never a reason to stop early.

A fifth error was caught in the *checking* code rather than the model: the
deflationary depth control unpacked `(parent, child)` backwards, measuring depth
from the leaves and reporting `Spearman(depth, rank) = −0.70`. Deeper cannot mean
less specific, so the script now refuses to run if that correlation is negative.

Two smaller reporting traps, both now handled in code:

- `monotonic: False` was driven entirely by `subspecies` (n=1) and `strain`
  (n=5). Ranks below n=10 are printed but excluded from the ordering claim.
- Steiner nodes are trained on but excluded from scoring — grading the geometry
  on ancestors we inserted ourselves would be grading it on our own scaffolding.
  659 of 916 nodes are scored.

---

## Reproduce

```
python3 repair_taxonomy_tree.py        # -> taxonomy_tree.json (once)
python3 hyperbolic_taxonomy.py --dim 10 --epochs 3000
```

~80 s on CPU. Writes `hyperbolic_taxonomy.json` (statistics, node order) and
`hyperbolic_emb.npy` (916 × 10 coordinates, row *i* ↔ `nodes[i]`).

---

## What this unblocks

The embedding is now available for questions whose answers we do *not* already
have. In rough order of value:

1. **Rank-flexible retrieval.** Sam's ask was to query at species, genus or
   family level. The geometry gives a continuous version of that — but it must
   beat the symbolic baseline (walk the tree, expand descendants) to be worth
   anything, and that comparison has not been run. Expect the baseline to be hard
   to beat on a tree this clean.
2. **Where taxonomy and evidence disagree.** Sibling taxa that sit close in the
   hierarchy but carry opposite directions in the graph are exactly the
   containment cases this project refuses to collapse — *Lachnospiraceae* down,
   *Hungatella* inside it up. A geometry that encodes both could rank those
   conflicts by severity instead of listing them.
3. **Hierarchical multiple testing.** treeclimbR and hierarchical FDR need a real
   tree; they now have one. This is independent of the embedding.

None of these is started. The claim in this document is only that the instrument
is calibrated.
