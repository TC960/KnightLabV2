#!/usr/bin/env python3
"""Assemble a microbe-disease knowledge graph from the extraction output.

Input : eval-v2/results/<model>__...__all250.json  (one row per paper)
Output: kg/graph.json  {nodes, edges, meta}

Design follows what the established resources (Disbiome, Peryton, MicroPhenoDB)
actually do, plus two properties this corpus forces:

1. **One edge per taxon-disease pair, aggregating papers -- never a consensus
   edge that hides disagreement.** Each edge carries n_up / n_down and the list
   of contributing papers, so a contested pair stays visibly contested. In this
   corpus 119 of 1,729 pairs (~7%, but 45% of the *replicated* pairs) have
   papers pointing both ways; the microbiome replication literature reports
   roughly 1 in 3 taxa flipping sign between cohorts, so contradiction is
   signal, not noise, and must survive into the graph.

2. **No effect sizes.** The extractor returns direction only, and the underlying
   papers report incommensurable statistics (LEfSe LDA, fold-change, p-values)
   that cannot be pooled into a single magnitude. So edge weight is *evidence
   count*, and edge confidence is *directional consistency* -- both computed
   from data we actually have, rather than a fabricated magnitude.

Ranks are preserved rather than collapsed: papers report phylum, genus, species
and OTU-level labels as peers, and there is no accepted convention for merging
them. Rank is a node attribute; downstream consumers can roll up if they want.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                          "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")

# --- disease normalization -------------------------------------------------
# The extractor returns free text ("Alzheimer disease" / "Alzheimer's disease" /
# "AD"), so surface strings must be folded before anything can be counted.
# MONDO ids are the target vocabulary; only the diseases actually present in
# this corpus are mapped, and anything unmapped keeps its cleaned label with
# mondo=None rather than being silently dropped.
DISEASE_MAP = [
    (r"\bparkinson", "Parkinson's disease", "MONDO:0005180"),
    (r"\balzheimer", "Alzheimer's disease", "MONDO:0004975"),
    (r"multiple sclerosis|\bms\b", "Multiple sclerosis", "MONDO:0005301"),
    (r"amyotrophic lateral|\bals\b", "Amyotrophic lateral sclerosis", "MONDO:0004976"),
    (r"mild cognitive impairment|\bmci\b", "Mild cognitive impairment", "MONDO:0005453"),
    (r"\bstroke|cerebral infarct", "Stroke", "MONDO:0005098"),
    (r"huntington", "Huntington's disease", "MONDO:0007739"),
    (r"\bdementia", "Dementia", "MONDO:0001627"),
    (r"spinal muscular atrophy|\bsma\b", "Spinal muscular atrophy", "MONDO:0001516"),
    (r"epilep", "Epilepsy", "MONDO:0005027"),
    (r"autism|\basd\b", "Autism spectrum disorder", "MONDO:0005260"),
    (r"depress", "Depressive disorder", "MONDO:0002050"),
    (r"schizophren", "Schizophrenia", "MONDO:0005090"),
    (r"neuromyelitis", "Neuromyelitis optica", "MONDO:0019100"),
    (r"myasthenia", "Myasthenia gravis", "MONDO:0009688"),
    (r"migraine", "Migraine", "MONDO:0005277"),
    # Three spellings of ONE disease were three separate nodes: "Anti-N-methyl-
    # D-aspartate receptor encephalitis" (22 edges), "NMDAR encephalitis" (16)
    # and "Anti-NMDAR encephalitis" (7) -- one paper each. This is the
    # Bacteroidetes/Bacteroidota case in the disease dimension, and folding it
    # is synonym folding, not a rank collapse: 45 edges become 38, and 4 edges
    # that every view showed as single-paper become replicated, 3 of them
    # CONTESTED -- real inter-study disagreement the fragmentation was hiding.
    # MONDO id deliberately None rather than guessed: this environment's network
    # policy denies EBI/OLS, and a wrong ontology id is worse than no id.
    (r"nmdar?\b|n-methyl-d-aspartate", "Anti-NMDAR encephalitis", None),
]

# rank hints from the naming conventions the papers use
RANK_SUFFIX = [
    (r"^[a-z]__|^[pcofgs]-", None),          # greengenes-style prefix, handled below
    (r"aceae$", "family"), (r"ales$", "order"), (r"ia$|ies$", "class"),
    (r"(ota|etes|bacteria|micrObia)$", "phylum"),
]


def parse_taxa(v):
    if not v or str(v).strip().lower() in ("", "nan", "none"):
        return []
    out = []
    for t in re.split(r"[,;]", str(v)):
        t = re.sub(r"\(.*?\)", "", t)
        t = re.sub(r"p\s*[<>=]\s*[\d.]+", "", t, flags=re.I)
        t = t.strip().strip(".) ").strip()
        if t and t.lower() != "nan" and len(t) > 2:
            out.append(t)
    return out


def norm_disease(s):
    s = (s or "").strip()
    low = s.lower()
    for pat, label, mondo in DISEASE_MAP:
        if re.search(pat, low):
            return label, mondo
    return (s[:1].upper() + s[1:]) if s else "Unspecified", None


# Rank placeholders: labels 16S pipelines emit for a clade they could not name to a
# real taxon -- "Erysipelotrichaceae UCG-003", "Lachnospiraceae ND3007 group",
# "Clostridia UCG-014", "Christensenellaceae R-7 group". taxonomy.py resolves these
# by trimming the qualifier tail, so they land on the PARENT taxid and are pooled as
# if they were the parent itself.
#
# That is a rank collapse wearing a synonym's clothes, and adjudication caught it:
# Erysipelotrichaceae/Parkinson's looked like a 4-paper contradiction of both
# curated databases, but 3 of those 4 papers report "Erysipelotrichaceae UCG-003",
# a genus-level placeholder INSIDE the family. No paper measured the family
# aggregate. Corpus-wide this affects 74 strings over 37 taxids, 21 edges named only
# by a placeholder and 170 mixed, 52 of them contested.
#
# It also breaks the project's own rule: synonym folding (same rank, renamed) and
# containment (different ranks) are different operations. A UCG label is a CHILD.
# So with --split-placeholders these get their own node, linked to the parent by a
# containment edge rather than merged into it.
#
# The first version of this pattern caught the UCG / ND / "group" forms and
# missed the commonest one: SILVA's bare numeric and roman-numeral suffixes.
# "Prevotella 9", "Prevotella_6", "Coprococcus_1", "Ruminiclostridium 5",
# "Clostridium IV", "Clostridiaceae 1" are DISTINCT SILVA genera, and every one
# of them was landing on its parent's taxid. Prevotella/Parkinson's -- the
# highest-weight edge in the graph at 17 papers, and the one the README calls
# load-bearing for the external join -- folds 13 surface strings into one node,
# five of them these placeholders.
#
# Detected corpus-wide by asking which surface strings EXTEND the scientific
# name they resolved to (see child_folds.json): 115 such strings over 52 nodes,
# none flagged. 32 are placeholders of this kind; 29 are "X sp./spp./
# unclassified", where folding to the parent is CORRECT and must not change;
# the remaining 54 are real named species (Prevotella copri, Klebsiella
# pneumonia) whose split needs real taxids and is left to an environment with
# the NCBI taxdump -- see FINDINGS_rank_collapse.md.
PLACEHOLDER = re.compile(
    r"(UCG[-_ ]?\d+|_?group$|ND\d{3,}|R-\d+\b|incertae[ _]sedis|"
    r"sensu[ _]stricto|\bAD\d{3,}\b|\b[A-Z]{1,3}\d{2,}\b|"
    r"[ _]\d{1,3}$|[ _][IVXL]+$|\bcluster[ _]|\bFamily[ _][IVXL]+\b)")

# Guard, agreeing with taxonomy_cache.NOT_CONTAINED. A phage is not a member of
# the genus it infects, and "uncultured X sp. 1" names an unidentified member
# rather than a SILVA rank placeholder -- but both end in a bare number and so
# match the pattern above. Splitting them off produced the only 4 nodes that
# failed to survive a rebuild: the cache (correctly) refuses to hang a phage
# under a bacterial genus, so their parent was unrecoverable and the placeholder
# flag silently evaporated on the second build. Caught by re-running the build
# and diffing, not by reading the pattern.
NOT_PLACEHOLDER = re.compile(r"\b(virus|phage|bacteriophage|uncultured)\b", re.I)

SPLIT_PLACEHOLDERS = False          # set by --split-placeholders
BODY_SITE = {}                      # paper title -> sampled body site
PLACEHOLDER_PARENT = {}             # placeholder node key -> parent taxid


def norm_taxon(t, tax=None):
    """Canonical key + display name + rank.

    With the NCBI taxdump available (kg/taxonomy.py), the key is the taxid, so
    synonyms and renames pool into one node: Bacteroidetes/Bacteroidota both
    become 976, Firmicutes/Bacillota both 1239. Unresolvable names keep their
    surface string as the key and are marked unresolved rather than dropped.

    Without the taxdump this degrades to the old string folding (case + rank
    prefix only), so the builder still runs.
    """
    disp = t.strip()
    if tax is not None and tax.ok:
        tid, sci, rank, how = tax.resolve(disp)
        if tid:
            # A placeholder resolves to its PARENT (the qualifier tail is trimmed),
            # which is detectable: the raw string differs from the scientific name
            # it landed on. Keep it as its own node and remember the parent so a
            # containment link can be added.
            if (SPLIT_PLACEHOLDERS and PLACEHOLDER.search(disp)
                    and not NOT_PLACEHOLDER.search(disp)
                    and disp.lower() != (sci or "").lower()):
                key = "ph:" + re.sub(r"\s+", " ", disp.lower().replace("_", " ")).strip()
                PLACEHOLDER_PARENT[key] = tid
                return key, disp, "clade", "placeholder"
            return f"ncbi:{tid}", sci, (rank or "no rank"), how
    # --- fallback: string folding only ---
    key = disp.lower()
    rank = None
    m = re.match(r"^([pcofgs])[-_]{1,2}", key)      # "o-Clostridia", "f__Rikenellaceae"
    if m:
        rank = {"p": "phylum", "c": "class", "o": "order",
                "f": "family", "g": "genus", "s": "species"}[m.group(1)]
        key = key[m.end():]
        disp = disp[m.end():]
    key = re.sub(r"^[a-z]__", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    if rank is None:
        if len(key.split()) >= 2:
            rank = "species"
        else:
            for pat, r in RANK_SUFFIX:
                if r and re.search(pat, key):
                    rank = r
                    break
            rank = rank or "genus"
    return key, disp, rank, "unresolved"


def load_body_sites():
    """title -> sampled body site, for ALL contributing papers (see body_site.py).

    Deliberately an edge ATTRIBUTE, not part of the edge key. Keying on site was
    the top lever out of the adjudication, and it was tested and rejected: the
    corpus is 97.9% gut (6 non-gut papers in 281), so keying would fragment 58
    mixed edges into singletons to separate evidence that, on a metric sensitive
    enough to see it, moves mean concordance with Disbiome by -0.007 (p = 0.131,
    minimum detectable 0.010) and with Peryton by -0.005 (p = 0.283) -- a null,
    and in the OPPOSITE direction to the hypothesis that oral studies were
    dragging agreement down. As an attribute it still lets a consumer filter to
    gut-only evidence, which is the part that was actually worth having.
    """
    path = os.path.join(HERE, "body_site.json")
    if not os.path.exists(path):
        return {}
    return {t: v.get("site", "") for t, v in json.load(open(path))["papers"].items()}


def load_study_metadata():
    """paper title -> study design fields, from extract_metadata.py output."""
    path = os.path.join(HERE, "metadata.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        try:
            r = json.loads(line)
            if r.get("meta") and not r.get("parse_error"):
                out[r["title"]] = r["meta"]
        except Exception:
            continue
    return out


def norm_title(s):
    """Title key robust to the two ways the same paper entered the corpus twice."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def dedup_rows(rows, verbose=True):
    """Collapse rows that are the SAME PAPER fetched twice.

    12 papers were scraped once from a PubMed link and again from a PMC or
    publisher link. The two copies' titles differ only by a trailing period
    and/or a curly-vs-straight apostrophe ("...disease activity" vs
    "...disease activity."; "Parkinson's" vs "Parkinson’s"), and every paper
    key in this pipeline is the raw title string, so nothing ever collapsed
    them. Because edge weight IS paper count, each duplicate cast two votes:
    9 of them contribute extractions, and an edge resting on one paper could
    present itself as replicated.

    Found by the co-occurrence analysis, not by inspection: 8 paper pairs
    inside contested edges had profile cosine of exactly 1.00.

    The kept copy is the one with the most extracted taxa (the two copies are
    reads of the same text, so the richer read is the more complete one; one
    PMC copy is empty where its PubMed twin is not). Ties break on the title
    string so a rebuild is deterministic.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[norm_title(r.get("title"))].append(r)
    out, dropped = [], []
    for k in sorted(groups):
        g = groups[k]
        if len(g) == 1:
            out.append(g[0])
            continue
        g = sorted(g, key=lambda r: (-(len(parse_taxa(r.get("predicted_enriched"))) +
                                       len(parse_taxa(r.get("predicted_depleted")))),
                                     r.get("title", "")))
        out.append(g[0])
        dropped.extend(g[1:])
    if verbose and dropped:
        print(f"deduplicated {len(dropped)} duplicate paper copies "
              f"({len(rows)} -> {len(out)} rows)")
    # preserve the input ordering of the kept rows
    keep = {id(r) for r in out}
    return [r for r in rows if id(r) in keep], dropped


def build(rows, min_papers=1, tax=None):
    global BODY_SITE
    BODY_SITE = load_body_sites()
    ev = defaultdict(list)
    taxon_disp, taxon_rank, taxon_how = {}, {}, {}
    aliases = defaultdict(set)          # node key -> every surface string that folded into it
    for r in rows:
        dis_raw = (r.get("predicted_disease") or r.get("disease") or "")
        disease, mondo = norm_disease(dis_raw)
        for direction, col in (("enriched", "predicted_enriched"),
                               ("depleted", "predicted_depleted")):
            for raw in parse_taxa(r.get(col)):
                key, disp, rank, how = norm_taxon(raw, tax)
                if not key:
                    continue
                taxon_disp.setdefault(key, disp)
                taxon_rank.setdefault(key, rank)
                taxon_how.setdefault(key, how)
                aliases[key].add(raw)
                ev[(key, disease, mondo)].append(
                    {"dir": direction, "paper": r.get("title", ""), "link": r.get("link", ""),
                     "as_written": raw})

    edges = []
    for (taxon, disease, mondo), obs in ev.items():
        # Papers, not observations. A paper that lists the same taxon twice --
        # literally "Coriobacteriaceae, Coriobacteriaceae", or "Ruminococcus" and
        # "Ruminococcus sp" folding to one taxid -- must still cast ONE vote in
        # that direction. This was previously true of `papers` and `evidence` but
        # NOT of n_up/n_down, which counted raw observations and so inflated the
        # evidence count on 44 of 1,985 edges (2.2%) and the consistency ratio
        # with it. It changes no edge's verdict (measured: 0 of 1,985 flip
        # direction or contested status), so nothing downstream moves -- but
        # "edge weight is evidence count" has to mean what it says.
        c = Counter({d: len({o["paper"] for o in obs if o["dir"] == d})
                     for d in ("enriched", "depleted")})
        up, dn = c["enriched"], c["depleted"]
        n = up + dn
        if n < min_papers:
            continue
        papers = {o["paper"] for o in obs}
        # per-paper direction, so the UI can show WHICH studies said what
        evidence = {}
        for o in obs:
            evidence[o["paper"]] = o["dir"]
        consistency = max(up, dn) / n
        # sorted(papers), not papers: `papers` is a set of title strings, so its
        # iteration order is randomised per process by PYTHONHASHSEED, and a
        # Counter keeps insertion order. That made `sites` key order vary between
        # runs on the ~30 multi-site edges -- identical content, different bytes.
        # Harmless in itself, but it meant the project's own verification rule
        # ("rebuild TWICE and diff") reported a difference on every single build,
        # which is how a real self-erasing fix would have been waved through.
        # Everything else emitted here is already sorted; this was the omission.
        sites = Counter(BODY_SITE.get(p, "") for p in sorted(papers))
        sites.pop("", None)
        edges.append({
            "sites": dict(sites),
            "gut_only": bool(sites) and not (set(sites) - {"stool", "gut biopsy"}),
            "taxon": taxon_disp.get(taxon, taxon), "taxon_key": taxon,
            "rank": taxon_rank.get(taxon, ""),
            "resolved": taxon_how.get(taxon) != "unresolved",
            "disease": disease, "mondo": mondo,
            "direction": "enriched" if up > dn else "depleted" if dn > up else "contested",
            "n_up": up, "n_down": dn, "n_obs": n, "n_papers": len(papers),
            "consistency": round(consistency, 3),
            "contested": bool(up and dn),
            "papers": sorted(papers)[:25],
            "evidence": [{"t": t, "d": d} for t, d in sorted(evidence.items())][:25],
        })

    tax_deg = Counter(e["taxon_key"] for e in edges)
    dis_deg = Counter(e["disease"] for e in edges)
    nodes = (
        [{"id": f"t:{k}", "label": taxon_disp[k], "type": "taxon",
          "taxid": k.split(":")[1] if k.startswith("ncbi:") else None,
          # "resolved" means "has an NCBI taxid". A placeholder deliberately has
          # none -- it is positioned by its containment link to the parent, not by
          # an id -- so it must not inflate the resolved count.
          "resolved": taxon_how[k] not in ("unresolved", "placeholder"),
          "placeholder": taxon_how[k] == "placeholder",
          # Record the parent taxid ON the placeholder node. Without it the only
          # record of the parent is the containment link, which exists only when
          # the parent is ITSELF a node -- so "Polaribacter_1" (no other
          # Polaribacter edge in the corpus) lost its parent on rebuild and
          # decayed into a plain unresolved string node. Storing it makes the
          # replay cache's round-trip exact instead of reconstructed.
          **({"parent_taxid": PLACEHOLDER_PARENT[k]}
             if taxon_how[k] == "placeholder" and k in PLACEHOLDER_PARENT else {}),
          "aliases": sorted(aliases[k]),
          "rank": taxon_rank[k], "degree": tax_deg[k]} for k in tax_deg]
        + [{"id": f"d:{d}", "label": d, "type": "disease",
            "mondo": next((e["mondo"] for e in edges if e["disease"] == d), None),
            "degree": dis_deg[d]} for d in dis_deg]
    )
    for e in edges:
        e["source"], e["target"] = f"t:{e['taxon_key']}", f"d:{e['disease']}"

    # ---- taxonomic hierarchy between taxon nodes -------------------------
    # Papers report at whatever rank they resolved to, so the same disease
    # routinely carries a family AND genera inside it (Lachnospiraceae plus
    # Roseburia, Blautia, Hungatella in Parkinson's). Without an explicit link
    # these are unrelated nodes and the graph cannot express that one contains
    # the other -- which matters because containment is NOT redundancy: in this
    # corpus Lachnospiraceae is depleted (15 papers) while Hungatella inside it
    # is enriched (7). A family shrinking while one genus grows is ordinary
    # biology, and only survives if the nesting is represented rather than
    # collapsed. So we add parent_of edges and let consumers roll up or not.
    hierarchy = []
    # Placeholder nodes hang off the parent they were previously merged INTO, so
    # the containment they always had is now explicit instead of implicit.
    node_ids = {f"t:{k}" for k in tax_deg}
    for key, parent_tid in PLACEHOLDER_PARENT.items():
        child, parent = f"t:{key}", f"t:ncbi:{parent_tid}"
        if child in node_ids and parent in node_ids:
            hierarchy.append({"parent": parent, "child": child,
                              "parent_rank": (tax.rank.get(parent_tid, "")
                                              if tax is not None and tax.ok else ""),
                              "child_rank": "clade"})
    if tax is not None and tax.ok:
        tids = [n["taxid"] for n in nodes if n["type"] == "taxon" and n.get("taxid")]
        lineage = {t: tax.lineage(t) for t in tids}
        present = set(tids)
        for t in tids:
            # nearest ancestor that is itself a node in this graph
            for anc in lineage[t][1:]:
                if anc in present:
                    hierarchy.append({"parent": f"t:ncbi:{anc}", "child": f"t:ncbi:{t}",
                                      "parent_rank": tax.rank.get(anc, ""),
                                      "child_rank": tax.rank.get(t, "")})
                    break
    # ---- paper table: referenced by index so cohort data is stored once ----
    md = load_study_metadata()
    titles = sorted({t for e in edges for t in (x["t"] for x in e["evidence"])})
    pidx = {t: i for i, t in enumerate(titles)}
    link_by_title = {}
    for e in edges:
        for o in e.get("papers", []):
            link_by_title.setdefault(o, "")
    papers_tbl = []
    for t in titles:
        m = md.get(t, {})
        papers_tbl.append({
            "title": t,
            "country": m.get("country", ""),
            "n_cases": m.get("n_cases", 0),
            "n_controls": m.get("n_controls", 0),
            "seq": m.get("sequencing", ""),
            "site": BODY_SITE.get(t, m.get("body_site", "")),
            "region": m.get("region_16S", ""),
            "med": m.get("medication_controlled"),
            "diet": m.get("diet_controlled"),
            "has_meta": t in md,
        })
    for e in edges:
        e["ev"] = [{"i": pidx[x["t"]], "d": x["d"][0]} for x in e["evidence"]]
        del e["evidence"]
    annotate_specificity(nodes, edges)
    return nodes, edges, hierarchy, papers_tbl


def annotate_specificity(nodes, edges):
    """Mark how disease-specific each taxon's direction is.

    WHY. The 2026-09-05 analysis (`FINDINGS_disease_specificity.md`) found that
    ~70% of the directional agreement in this graph is a corpus-wide prior rather
    than disease-specific signal: 59 of 187 taxa reported in >=3 diseases never
    flip direction. *Streptococcus* is enriched in all 12 diseases that report
    it. So "Streptococcus enriched in Parkinson's, 5 papers" reads as a
    Parkinson's finding when it is really a statement about Streptococcus, and
    nothing in the graph said so.

    Edge weight already answers "how much evidence"; these fields answer the
    different question "how much of it is about THIS disease". They are derived
    purely from the edges just built, so they cannot drift out of sync with them.

    Per taxon, counting one vote per disease (a disease whose own edge is
    contested casts no vote, since it has no direction to contribute):
      breadth  -- number of diseases casting a vote
      purity   -- max(up_diseases, down_diseases) / breadth
      class    -- generic       : breadth >= 3 and purity == 1.0
                  discriminating: breadth >= 3 and purity <= 0.6
                  mixed         : breadth >= 3, in between
                  narrow        : breadth < 3, too few diseases to say
    The >=3 floor and the 0.6 cut are reporting thresholds, not test results;
    the underlying counts are emitted so any other cut can be applied.

    Note the vote rule differs deliberately from the exploratory version in
    `disease_specificity.py`, which let a contested disease still vote by its
    majority. Here a contested edge casts no vote at all: if a disease's own
    papers disagree, it has no settled direction to contribute. That is the
    stricter reading and it moves a few counts (Streptococcus is generic across
    11 diseases here, 12 there). Every taxon is annotated either way, so a taxon
    whose every edge is contested gets breadth 0 rather than a missing field --
    an absent field is a trap for whatever consumes this.
    """
    votes = defaultdict(lambda: {"e": 0, "d": 0})
    for e in edges:
        votes[e["taxon_key"]]  # ensure every taxon appears, even if all-contested
        if e["contested"]:
            continue
        votes[e["taxon_key"]]["e" if e["direction"] == "enriched" else "d"] += 1

    stats = {}
    for t, v in votes.items():
        breadth = v["e"] + v["d"]
        purity = max(v["e"], v["d"]) / breadth if breadth else 0.0
        if breadth < 3:
            cls = "narrow"
        elif purity == 1.0:
            cls = "generic"
        elif purity <= 0.6:
            cls = "discriminating"
        else:
            cls = "mixed"
        stats[t] = {
            "breadth": breadth,
            "n_diseases_enriched": v["e"],
            "n_diseases_depleted": v["d"],
            "purity": round(purity, 3),
            "consensus": "enriched" if v["e"] > v["d"] else
                         ("depleted" if v["d"] > v["e"] else "split"),
            "class": cls,
        }

    for n in nodes:
        if n["type"] != "taxon":
            continue
        s = stats.get(n["id"].split("t:", 1)[-1])
        if s:
            n["specificity"] = s
    for e in edges:
        s = stats.get(e["taxon_key"])
        if not s:
            continue
        e["taxon_breadth"] = s["breadth"]
        e["taxon_purity"] = s["purity"]
        e["taxon_class"] = s["class"]
        # Does this edge merely restate the taxon's corpus-wide tendency?
        e["restates_prior"] = bool(
            not e["contested"] and s["class"] == "generic"
            and e["direction"] == s["consensus"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--min-papers", type=int, default=1,
                    help="drop edges supported by fewer than N papers")
    ap.add_argument("--out", default=os.path.join(HERE, "graph.json"))
    # Default ON: merging a UCG placeholder into its parent family is a rank
    # collapse, and it manufactured the worst false contradiction in the graph
    # (see FINDINGS_task3_adjudication.md). --merge-placeholders restores the old
    # behaviour for comparison.
    ap.add_argument("--merge-placeholders", dest="split_placeholders",
                    action="store_false", default=True,
                    help="OLD behaviour: fold rank placeholders (UCG-003, ND3007 "
                         "group) into their parent taxon instead of keeping them "
                         "as their own node")
    ap.add_argument("--no-taxonomy", action="store_true",
                    help="skip NCBI resolution, fold on strings only")
    # Default ON: the same paper scraped twice under two links must not cast two
    # votes on an edge whose weight is defined as a paper count.
    ap.add_argument("--keep-duplicate-papers", dest="dedup",
                    action="store_false", default=True,
                    help="OLD behaviour: keep both copies of a paper that was "
                         "scraped twice under different links")
    a = ap.parse_args()
    global SPLIT_PLACEHOLDERS
    SPLIT_PLACEHOLDERS = a.split_placeholders

    rows = json.load(open(a.input))
    n_raw = len(rows)
    dropped = []
    if a.dedup:
        rows, dropped = dedup_rows(rows)
    tax = None
    if not a.no_taxonomy:
        # Prefer the real taxdump; fall back to replaying graph.json's recorded
        # resolution. The old code fell straight through to string folding when the
        # taxdump was missing, which quietly cost 681 taxid resolutions and all 625
        # containment links while still printing a successful build.
        try:
            from taxonomy_cache import load_taxonomy
            tax = load_taxonomy()
        except Exception as e:
            print(f"taxonomy unavailable ({e.__class__.__name__}) -> string folding only")
    nodes, edges, hierarchy, papers_tbl = build(rows, a.min_papers, tax)
    meta = {
        "source": os.path.basename(a.input),
        "papers_in": n_raw,
        "papers_deduped": len(dropped),
        "papers_contributing": len({e for r in rows for e in [r["title"]]
                                    if parse_taxa(r.get("predicted_enriched")) or
                                    parse_taxa(r.get("predicted_depleted"))}),
        "n_taxa": sum(1 for n in nodes if n["type"] == "taxon"),
        "n_diseases": sum(1 for n in nodes if n["type"] == "disease"),
        "n_edges": len(edges),
        "n_replicated": sum(1 for e in edges if e["n_papers"] > 1),
        "n_contested": sum(1 for e in edges if e["contested"]),
        "n_taxa_resolved": sum(1 for n in nodes if n["type"] == "taxon" and n.get("resolved")),
        "n_hierarchy_links": len(hierarchy),
        "n_papers_table": len(papers_tbl),
        "n_papers_with_metadata": sum(1 for p in papers_tbl if p["has_meta"]),
        "min_papers": a.min_papers,
        "split_placeholders": a.split_placeholders,
        "n_placeholder_nodes": sum(1 for n in nodes
                                   if n["type"] == "taxon" and str(n.get("id","")).startswith("t:ph:")),
        "note": ("Edge weight is evidence count, not effect size: the extractor yields "
                 "direction only and the source papers report incommensurable statistics. "
                 "Contested edges are retained, never merged away."),
    }
    json.dump({"meta": meta, "nodes": nodes, "edges": edges,
               "hierarchy": hierarchy, "papers": papers_tbl}, open(a.out, "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {a.out}")
    top = sorted(edges, key=lambda e: -e["n_papers"])[:10]
    print("\nmost-replicated edges:")
    for e in top:
        flag = " CONTESTED" if e["contested"] else ""
        print(f"  {e['n_papers']:3}p  {e['taxon'][:24]:24} {e['direction']:9} in {e['disease'][:28]:28}"
              f" (up={e['n_up']} dn={e['n_down']}){flag}")


if __name__ == "__main__":
    main()
