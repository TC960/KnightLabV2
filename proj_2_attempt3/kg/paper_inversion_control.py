#!/usr/bin/env python3
"""Does the paper-level clustering of minority direction survive a containment control?

`paper_inversion.py` found no single inverted paper (0 of 134 testable survive BH,
and 81 of 134 would have been caught if fully inverted) but DID find that
minority-direction status is clustered by paper far more than the within-edge
shuffle allows: dispersion 164.2 vs null mean 132.7, p = 0.0002.

Before that is called a paper-level effect it has to survive the obvious artifact.
This corpus is already known to carry strong within-paper taxonomic correlation --
`CLAUDE.md`: related taxa agree on direction 89% of the time within a single paper
versus 54% for unrelated taxa. A paper that reports *Lachnospiraceae* and a genus
inside it is making close to one measurement, not two. The within-edge shuffle
permutes each edge independently and therefore CANNOT reproduce that correlation,
so it will manufacture exactly this excess dispersion out of nothing.

CONTROL
-------
Thin the observations so that, within any one paper, no two retained taxa are
taxonomically related. Relatedness is ancestor-descendant closure over the 723
containment links (both directions), plus same-taxon. Selection is deterministic:
within a paper, taxa sorted by key, greedily kept if unrelated to everything kept
so far -- no random subsetting, no cherry-picking.

Then re-run the identical statistic and the identical within-edge permutation on
the thinned set. If the dispersion signal is containment correlation it collapses
to the null. If it survives, papers really do carry a direction offset that is not
explained by which edges they sit on.

Reports the thinned-set power alongside, because thinning costs observations and a
null on a smaller set is only meaningful with its MDE stated.

Writes paper_inversion_control.json.
"""

import json
import random
import sys
from collections import defaultdict, deque

from paper_inversion import (MIN_DECISIVE, N_PERM, SEED, build_observations,
                             load_graph, score, score_permuted)


def ancestor_sets(graph):
    """taxon node id -> set of all ancestor AND descendant node ids (transitive)."""
    children = defaultdict(set)
    parents = defaultdict(set)
    for h in graph["hierarchy"]:
        children[h["parent"]].add(h["child"])
        parents[h["child"]].add(h["parent"])

    def closure(start, adj):
        seen, q = set(), deque([start])
        while q:
            n = q.popleft()
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    q.append(m)
        return seen

    nodes = set(children) | set(parents)
    rel = {}
    for n in nodes:
        rel[n] = closure(n, children) | closure(n, parents)
    return rel


def thin(edges, rel):
    """Drop, per paper, every taxon related to one already kept for that paper."""
    # collect each paper's taxa, in a deterministic order
    by_paper = defaultdict(set)
    for ed in edges:
        for pi, _ in ed["obs"]:
            by_paper[pi].add(ed["source"])

    keep = {}
    for pi, taxa in by_paper.items():
        kept = []
        for t in sorted(taxa):
            if any(t == k or k in rel.get(t, ()) for k in kept):
                continue
            kept.append(t)
        keep[pi] = set(kept)

    out = []
    for ed in edges:
        obs = [(pi, d) for pi, d in ed["obs"] if ed["source"] in keep[pi]]
        if len(obs) >= 2:
            out.append({**ed, "obs": obs})
    return out, keep


def dispersion(per_paper_dis, per_paper_dec, eligible, base):
    s = 0.0
    for p in eligible:
        e_p = max(per_paper_dec.get(p, 0) * base, 1e-9)
        s += (per_paper_dis.get(p, 0) - e_p) ** 2 / max(e_p, 1.0)
    return s


def run(edges, label):
    per_paper, _ = score(edges)
    n_dec = sum(v[0] for v in per_paper.values())
    n_dis = sum(v[1] for v in per_paper.values())
    base = n_dis / n_dec if n_dec else 0.0
    eligible = sorted(p for p, v in per_paper.items() if v[0] >= MIN_DECISIVE)
    print(f"\n[{label}] {len(edges)} edges, "
          f"{sum(len(e['obs']) for e in edges)} observations, "
          f"{n_dec} decisive, {n_dis} disagree ({base:.1%}), "
          f"{len(eligible)} testable papers")
    if not eligible:
        return {"label": label, "n_testable": 0}

    obs_disp = dispersion({p: per_paper[p][1] for p in eligible},
                          {p: per_paper[p][0] for p in eligible}, eligible, base)
    rng = random.Random(SEED)
    null = []
    null_counts = {p: [] for p in eligible}
    for _ in range(N_PERM):
        dis, dec = score_permuted(edges, rng)
        null.append(dispersion(dis, dec, eligible, base))
        for p in eligible:
            null_counts[p].append(dis.get(p, 0))
    pv = (sum(1 for s in null if s >= obs_disp) + 1) / (N_PERM + 1)
    nm = sum(null) / len(null)
    print(f"[{label}] dispersion observed {obs_disp:.1f}  null mean {nm:.1f}  "
          f"p = {pv:.4f}")

    # per-paper, BH
    rows = []
    for p in eligible:
        nc = null_counts[p]
        d = per_paper[p][1]
        rows.append({"paper": p, "n_decisive": per_paper[p][0], "n_disagree": d,
                     "null_mean": round(sum(nc) / len(nc), 2),
                     "p": round((sum(1 for c in nc if c >= d) + 1) / (N_PERM + 1), 5)})
    rows.sort(key=lambda r: r["p"])
    m = len(rows)
    for rank, r in enumerate(rows, 1):
        r["q"] = round(min(1.0, r["p"] * m / rank), 4)
    for i in range(m - 2, -1, -1):
        rows[i]["q"] = min(rows[i]["q"], rows[i + 1]["q"])
    nsig = sum(1 for r in rows if r["q"] < 0.05)
    print(f"[{label}] papers surviving BH q<0.05: {nsig} "
          f"(best q = {rows[0]['q']:.3f})")

    # MDE for the dispersion test: how large a shift in one paper's disagreement
    # count is needed to push the statistic past the null's 95th percentile?
    null_sorted = sorted(null)
    crit = null_sorted[int(0.95 * len(null_sorted))]
    print(f"[{label}] dispersion 95th pct of null = {crit:.1f} "
          f"(observed {obs_disp:.1f})")

    return {"label": label, "n_edges": len(edges),
            "n_obs": sum(len(e["obs"]) for e in edges),
            "n_decisive": n_dec, "n_disagree": n_dis,
            "base_rate": round(base, 4), "n_testable": m,
            "dispersion_observed": round(obs_disp, 2),
            "dispersion_null_mean": round(nm, 2),
            "dispersion_null_p95": round(crit, 2),
            "dispersion_p": round(pv, 5),
            "n_significant_bh05": nsig,
            "papers": rows}


def main():
    graph = load_graph()
    edges = build_observations(graph)
    rel = ancestor_sets(graph)
    print(f"containment closure covers {len(rel)} taxon nodes")

    full = run(edges, "full")

    thinned, keep = thin(edges, rel)
    dropped = sum(len(e["obs"]) for e in edges) - sum(len(e["obs"]) for e in thinned)
    print(f"\nthinning removed {dropped} observations "
          f"({dropped / sum(len(e['obs']) for e in edges):.1%}) as within-paper "
          f"taxonomic relatives")
    ctrl = run(thinned, "containment-controlled")

    # Power of the thinned test: simulate injecting one fully inverted paper and
    # ask whether the DISPERSION statistic would have moved past the null p95.
    print("\nPower of the thinned dispersion test (inject one inverted paper):")
    caught = 0
    tested = 0
    per_paper_t, _ = score(thinned)
    elig_t = sorted(p for p, v in per_paper_t.items() if v[0] >= MIN_DECISIVE)
    base_t = ctrl["base_rate"]
    for p in elig_t:
        flipped = [{**ed, "obs": [(pi, ("d" if d == "e" else "e") if pi == p else d)
                                  for pi, d in ed["obs"]]} for ed in thinned]
        fp, _ = score(flipped)
        disp = dispersion({q: fp[q][1] for q in elig_t},
                          {q: fp[q][0] for q in elig_t}, elig_t, base_t)
        tested += 1
        if disp >= ctrl["dispersion_null_p95"]:
            caught += 1
    print(f"  a single fully-inverted paper pushes the thinned dispersion past "
          f"the null 95th percentile for {caught}/{tested} papers")

    out = {"full": full, "controlled": ctrl,
           "observations_dropped_by_thinning": dropped,
           "thinned_dispersion_power_single_inverted_paper":
               {"caught": caught, "tested": tested}}
    with open("paper_inversion_control.json", "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("\nwrote paper_inversion_control.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
