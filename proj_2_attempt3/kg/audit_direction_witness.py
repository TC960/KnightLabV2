#!/usr/bin/env python3
"""Gold-free, corpus-wide audit of extraction DIRECTION against the papers' own
sentences -- and, as it turns out, a measurement of how much of the graph rests
on sentences that are not the paper's own result.

Why this exists
---------------
Every fidelity number this project has is compromised or partial:
  * the in-house gold standard is under audit and known unreliable;
  * the Disbiome/Peryton agreement is only half independent of our corpus
    (FINDINGS_independence.md) -- honestly decomposed, ~90% *reading fidelity*
    on papers both sides read and ~55% cross-literature reproducibility;
  * the 2026-09-11 spelling sweep was a genuine third signal, but covered 33 taxa.

This goes back to each paper's own sentences for every (paper, taxon, direction)
observation backing an edge and asks whether the direction words next to that
taxon agree with what we extracted. It depends on neither the gold nor the
curated databases.

It measures whether the extractor READ the paper correctly. It says nothing about
whether the paper is right. That is deliberately the same quantity as the "same
paper" half of FINDINGS_independence.md, so the two are comparable.

The instrument was wrong before the thing it audited
----------------------------------------------------
A first version scored 84.0% and called the other 16% disagreements. Reading them
showed most were the AUDIT's fault, not the extractor's, in three ways -- so the
scoring is now tiered, and each tier is reported:

  T0  every strict witness (one taxon in the sentence, one polarity class).
      This is the naive number.
  T1  T0 minus sentences that are not the paper's own result: background,
      mechanism, and claims attributed to other studies. "Chang et al. indicated
      that ... an increased abundance of Blautia" and "some studies have shown
      that Bacteroidetes ... are reduced (Zhuang et al., 2018)" are statements
      about OTHER cohorts; scoring our extraction against them is a category
      error.
  T2  T1 with a comparison-frame correction applied -- BUILT, MEASURED, AND
      REJECTED. "Lactobacillaceae was most abundant in control group" carries an
      UP cue but means DEPLETED IN DISEASE, so flipping such sentences ought to
      help. It does the opposite: agreement falls 0.866 -> 0.774. The reason is
      that the dominant construction in this literature names the controls as the
      REFERENCE, not the subject -- "lower in PD patients compared with the
      healthy controls", "compared to control participants, AD participants
      exhibited decreased Actinobacteria" -- and those need no flip. Of the 54
      own-result witnesses the detector called control-framed, **41 were already
      correct unflipped**. Only the locative minority ("healthy individuals were
      dominated by Bacteroides") genuinely inverts, and no regex separated the
      two. T2 is reported so the next session does not rebuild it.

T1 is the headline. The gap T0 -> T1 measures how badly a naive cue-matching
audit misreads this literature, and it is why the T1 filter doubles as a check on
the graph itself (below).

The second question, which is about the graph rather than the audit
-------------------------------------------------------------------
If the AUDIT can mistake a background sentence for a result, so could the
EXTRACTOR. So this also reports, per observation, whether the paper contains any
own-result sentence naming that taxon at all. An edge whose only textual witness
is a sentence citing somebody else's cohort is a candidate extraction artefact.
That is a screen and not a verdict: `relation_sentences.json` keeps 7.5k of 106k
sentences, and numbers reported only in a table or figure never appear in any
sentence, so "no own-result witness" means "not visible to this instrument", not
"unsupported".

Substrate: relation_sentences_clean.json when present (abbreviation-repaired --
see clean_abbrev.py), else relation_sentences.json. Override with RELSENT=.

Writes audit_direction_witness.json.
"""
import json
import os
import random
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH = os.path.join(HERE, "graph.json")
DEFAULT_RS = "relation_sentences_clean.json"
OUT = os.path.join(HERE, "audit_direction_witness.json")

NPERM = 10000
SEED = 20260912

# ---------------------------------------------------------------------------
# Sentence provenance. A sentence is NOT the paper's own result if it carries a
# citation or attributes the claim to other work.
CITATION = re.compile(
    r"\(\s*[A-Z][A-Za-z\-]+(\s+(?:et\s*al|and|&)[^)]*)?,?\s*\d{4}[a-z]?\s*[;)]"   # (Zhuang et al., 2018)
    r"|\[\s*\d+\s*(?:[,;–\-]\s*\d+\s*)*\]"                                        # [ 8 , 9 ]
    r"|\bet\s*al\s*\.",                                                            # Chang et al .
    re.I,
)
THIRD_PARTY = re.compile(
    r"\b(?:"
    r"previous(?:ly)?\s+(?:studies|research|reports?|work)|"
    r"(?:studies|reports?|research|authors?|investigators?)\s+(?:have\s+)?"
    r"(?:shown|showed|reported|demonstrated|found|indicated|suggested)|"
    r"has\s+been\s+(?:shown|reported|found|demonstrated|observed|associated|linked)|"
    r"have\s+been\s+(?:shown|reported|found|demonstrated|observed|associated|linked)|"
    r"it\s+(?:has|is)\s+(?:been\s+)?(?:reported|shown|known|thought|believed)|"
    r"accumulating\s+evidence|emerging\s+evidence|according\s+to"
    r")\b",
    re.I,
)
# Mechanistic / speculative register: describes what an organism CAN do, not what
# was measured in this cohort.
MECHANISM = re.compile(
    r"\b(?:can|could|may|might|would|should)\s+\w+|"
    r"\b(?:is|are)\s+(?:known|thought|believed|able)\s+to\b|"
    r"\bplays?\s+(?:an?\s+)?(?:important\s+)?role\b|"
    r"\b(?:probably|possibly|presumably|hypothes\w+|speculat\w+)\b",
    re.I,
)
# Positive evidence that the sentence reports a measurement in THIS study.
RESULT_CUE = re.compile(
    r"\b(?:p|q|fdr)\s*[<>=]|\bp[-\s]?value|\blda\b|\blefse\b|\bauc\b|"
    r"\bsignificant(?:ly)?\b|\bfold[-\s]?change\b|\brelative\s+abundance\b|"
    r"\bwilcoxon\b|\bkruskal\b|\bmann[-\s]?whitney\b|\badjusted\b|"
    r"\bin\s+(?:our|this)\s+(?:study|cohort|analysis)\b|\bwe\s+(?:found|observed|detected)\b|"
    r"\bfigure\s*\d|\btable\s*\d|\bfig\s*\.?\s*\d",
    re.I,
)

# The comparison is stated relative to the CONTROL group, which inverts the cue.
CONTROL_FRAME = re.compile(
    r"\b(?:in|among|of|for|than|versus|vs\.?|compared\s+(?:with|to))\s+"
    r"(?:the\s+)?(?:healthy|normal|control|controls|hc\b|non-?(?:patient|case|demented)s?)"
    r"|\bcontrol\s+group\b|\bhealthy\s+(?:controls?|subjects?|individuals?|volunteers?)\b",
    re.I,
)
# ...but if the PATIENT group is also named as the reference, the frame is not a
# clean inversion and we refuse to guess.
PATIENT_FRAME = re.compile(
    r"\b(?:in|among|of|for)\s+(?:the\s+)?(?:patients?|cases?|subjects?\s+with)\b",
    re.I,
)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def norm_surface(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def provenance(sent):
    """'own' | 'background' -- is this the paper reporting its own measurement?"""
    if CITATION.search(sent) or THIRD_PARTY.search(sent):
        return "background"
    if MECHANISM.search(sent) and not RESULT_CUE.search(sent):
        return "background"
    if not RESULT_CUE.search(sent):
        return "background"
    return "own"


def frame(sent):
    """'control' if the sentence states the contrast relative to controls."""
    if CONTROL_FRAME.search(sent) and not PATIENT_FRAME.search(sent):
        return "control"
    return "patient"


def taxon_matchers(node):
    taxid = node.get("taxid") or None
    forms = {norm_surface(s) for s in
             [node.get("label")] + list(node.get("aliases") or []) if s}
    forms.discard("")
    return taxid, forms


def witnesses(rec, taxid, forms):
    out = []
    for s in rec["kept"]:
        for t in s.get("taxa", []):
            surface, tid = t[0], (t[1] if len(t) > 1 else None)
            if (taxid and tid and str(tid) == str(taxid)) or norm_surface(surface) in forms:
                out.append(s)
                break
    return out


def strict_cue(sent):
    """'up'/'down' for a single-taxon, single-polarity sentence, else None."""
    if len(sent.get("taxa", [])) != 1:
        return None
    pol = set(sent.get("pol", []))
    pol.discard("neutral")
    if pol == {"up"}:
        return "up"
    if pol == {"down"}:
        return "down"
    return None


def main():
    rs_name = os.environ.get("RELSENT", DEFAULT_RS)
    rs_path = os.path.join(HERE, rs_name)
    if not os.path.exists(rs_path):
        rs_path = os.path.join(HERE, "relation_sentences.json")
    g = json.load(open(GRAPH))
    rs = json.load(open(rs_path))["papers"]
    by_norm = {norm_title(k): v for k, v in rs.items()}
    papers, nodes = g["papers"], {n["id"]: n for n in g["nodes"]}

    rows = []
    for e in g["edges"]:
        node = nodes.get(e["source"])
        if node is None:
            continue
        taxid, forms = taxon_matchers(node)
        for ev in e.get("ev", []):
            rec = by_norm.get(norm_title(papers[ev["i"]]["title"]))
            if rec is None:
                continue
            ws = witnesses(rec, taxid, forms)
            cues = []
            for s in ws:
                c = strict_cue(s)
                if c:
                    cues.append({"cue": c, "s": s["s"],
                                 "prov": provenance(s["s"]), "frame": frame(s["s"])})
            rows.append({
                "taxon": e["taxon"], "disease": e["disease"],
                "paper_idx": ev["i"], "paper": papers[ev["i"]]["title"],
                "extracted": "up" if ev["d"] == "e" else "down",
                "contested": bool(e.get("contested")),
                "n_witness": len(ws),
                "n_own_witness": sum(1 for s in ws if provenance(s["s"]) == "own"),
                "cues": cues,
            })

    tiers = {
        "T0_all_strict": lambda c: True,
        "T1_own_result": lambda c: c["prov"] == "own",
    }
    out = {"substrate": os.path.basename(rs_path),
           "n_observations": len(rows),
           "n_with_any_witness": sum(1 for r in rows if r["n_witness"] > 0),
           "n_with_own_witness": sum(1 for r in rows if r["n_own_witness"] > 0),
           "n_witness_but_none_own": sum(1 for r in rows
                                         if r["n_witness"] > 0 and r["n_own_witness"] == 0),
           "tiers": {}}

    scored_sets = {}
    for name, keep in tiers.items():
        for apply_frame in ([False, True] if name == "T1_own_result" else [False]):
            key = name + ("_framed" if apply_frame else "")
            scored, dis = [], []
            for r in rows:
                cs = [c for c in r["cues"] if keep(c)]
                if not cs:
                    continue
                eff = set()
                for c in cs:
                    cue = c["cue"]
                    if apply_frame and c["frame"] == "control":
                        cue = "up" if cue == "down" else "down"
                    eff.add(cue)
                if len(eff) != 1:
                    continue
                cue = eff.pop()
                ok = (cue == r["extracted"])
                rec2 = dict(r, cue=cue, agree=ok,
                            sents=[c["s"] for c in cs][:4],
                            frames=[c["frame"] for c in cs])
                scored.append(rec2)
                if not ok:
                    dis.append(rec2)
            n, k = len(scored), sum(1 for r in scored if r["agree"])
            out["tiers"][key] = {
                "n_scored": n, "n_agree": k,
                "agreement": round(k / n, 4) if n else None,
                "n_disagree": len(dis),
                "n_papers": len({r["paper_idx"] for r in scored}),
            }
            scored_sets[key] = (scored, dis)

    head = "T1_own_result"
    scored, dis = scored_sets[head]

    # Is residual disagreement concentrated in particular papers, or spread?
    by_paper = defaultdict(list)
    for r in scored:
        by_paper[r["paper_idx"]].append(r["agree"])
    obs = _dispersion({p: (sum(v), len(v)) for p, v in by_paper.items()})
    flat = [r["agree"] for r in scored]
    sizes = [len(v) for v in by_paper.values()]
    rng = random.Random(SEED)
    ge = 0
    for _ in range(NPERM):
        sh = flat[:]
        rng.shuffle(sh)
        i, sim = 0, {}
        for j, sz in enumerate(sizes):
            sim[j] = (sum(sh[i:i + sz]), sz)
            i += sz
        if _dispersion(sim) >= obs:
            ge += 1
    # Cluster bootstrap CI on the headline rate: resample PAPERS, not
    # observations -- one paper contributes several observations.
    groups = list(by_paper.values())
    boot = []
    rb = random.Random(SEED + 7)
    for _ in range(5000):
        pick = [groups[rb.randrange(len(groups))] for _ in range(len(groups))]
        tot = sum(len(x) for x in pick)
        if tot:
            boot.append(sum(sum(x) for x in pick) / tot)
    boot.sort()
    out["headline_tier"] = head
    out["headline_ci95"] = [round(boot[int(.025 * len(boot))], 4),
                            round(boot[int(.975 * len(boot))], 4)]
    out["paper_clustering_p"] = round((ge + 1) / (NPERM + 1), 5)
    out["disagreements"] = [
        {k: r[k] for k in ("taxon", "disease", "paper", "extracted", "cue",
                           "sents", "frames", "contested")} for r in dis
    ]
    out["contested_split"] = {
        "contested": _rate([r for r in scored if r["contested"]]),
        "uncontested": _rate([r for r in scored if not r["contested"]]),
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"substrate: {out['substrate']}")
    print(f"observations backing edges            : {out['n_observations']}")
    print(f"  with any sentence naming the taxon  : {out['n_with_any_witness']}")
    print(f"  with an OWN-RESULT sentence         : {out['n_with_own_witness']}")
    print(f"  witnessed ONLY by background/citation: {out['n_witness_but_none_own']}")
    print()
    for k, v in out["tiers"].items():
        if v["n_scored"]:
            print(f"  {k:26s} {v['n_agree']:4d}/{v['n_scored']:<4d} = "
                  f"{v['agreement']:.3f}   ({v['n_papers']} papers)")
    print()
    print(f"headline = {head}  95% CI (paper-cluster bootstrap) {out['headline_ci95']}")
    print(f"  contested edges   : {out['contested_split']['contested']}")
    print(f"  uncontested edges : {out['contested_split']['uncontested']}")
    print(f"  disagreement clustered by paper? p = {out['paper_clustering_p']:.4f}")
    print(f"wrote {OUT}")


def _rate(rows):
    if not rows:
        return None
    k = sum(1 for r in rows if r["agree"])
    return [k, len(rows), round(k / len(rows), 4)]


def _dispersion(rates):
    tot_n = sum(n for _, n in rates.values())
    tot_k = sum(k for k, _ in rates.values())
    if not tot_n:
        return 0.0
    p = tot_k / tot_n
    chi = 0.0
    for k, nn in rates.values():
        var = nn * p * (1 - p)
        if var > 0:
            chi += (k - p * nn) ** 2 / var
    return chi


if __name__ == "__main__":
    main()
