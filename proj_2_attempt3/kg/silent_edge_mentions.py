#!/usr/bin/env python3
"""Are the `silent` observations unsupported, or merely invisible to the prose filter?

FINDINGS_direction_audit.md sized 580 edges (28.9% of the graph) that rest on a
single paper AND carry no own-result prose witness, and recorded that "prose
filtering cannot reach them". `silent` there means no sentence passed the
relation filter (taxon AND direction cue in the same sentence) -- NOT that the
taxon is absent from the paper. A taxon reported only as a row in a table is
silent to that instrument however well the paper supports it.

This script asks the weaker, answerable question those 580 edges deserve: is the
taxon named ANYWHERE in the paper's full text? That separates two things the
`silent` label conflates:

  silent + mentioned  -> reported outside prose (table/figure). Not a fabrication.
                         Direction still unverified by any instrument here.
  silent + absent     -> the taxon does not occur in the only paper backing it.
                         A candidate fabrication, and individually reviewable.

Reuses the repo's own provenance logic (`witness_discordance.build`) and its own
surface forms (`audit_direction_witness.taxon_matchers`) so the classification
matches the published one exactly rather than a re-derivation of it.

Full text sources, in preference order. Together they cover **271 of 271**
contributing papers, so as of 2026-09-17 there are no unscoreable observations.

  1. all_usable_papers.json  tracked in git; the datasheet scrape. 211 papers.
  2. extract_input.json      the text the extractor was actually GIVEN, so the
     new_papers.json         correct substrate for a fabrication question. 117
                             papers. UNTRACKED -- see the copyright note below.
  3. MAIN_DATA.json          the 2,026-paper canonical corpus. 33 papers. NOT
                             tracked, but MAIN_DATA.json.zip IS, so a plain
                             `unzip` makes it available in ANY checkout,
                             including the cloud. Eleven sessions recorded this
                             audit as Mac-only believing the corpus unreachable;
                             the zip was in the tree the whole time.

Each source only fills titles the ones above it do not hold, so wiring in a new
source cannot change an already-scored verdict. `validate_text_sources.py`
confirms the stronger claim that source choice is immaterial: 5 of the 6 source
pairs agree EXACTLY (extract_in vs main_data n=303, extract_in vs new_papers
n=629, both 1.0000), and the only disagreements anywhere are the truncated
MAIN_DATA stubs that guard (1) below already traps.

To restore everything in a fresh checkout:

    unzip -o proj_2_attempt3/MAIN_DATA.json.zip MAIN_DATA.json -d proj_2_attempt3/
    git show 254b0a8^:proj_2_attempt3/kg/extract_input.json > extract_input.json
    git show 254b0a8^:proj_2_attempt3/kg/new_papers.json    > new_papers.json

COPYRIGHT -- not optional. Sources 2 and 3 are gitignored on purpose: THIS
REPOSITORY IS PUBLIC and the corpus includes non-open-access articles (254b0a8).
Read them locally; never `git add` them, and never quote their text into a
tracked file. This script writes only verdicts, counts and short evidence spans.

Missing files are skipped, not fatal: with none of them present this degrades
exactly to the 2026-09-13 git-only run (795 unscoreable), which is how the
additivity above was verified.
"""
import json
import os
import re
from collections import Counter, defaultdict

import audit_direction_witness as A
import witness_discordance as W
from verify_taxon_mentions import classify, norm_text, squash

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
MAIN_DATA = os.path.join(HERE, "..", "MAIN_DATA.json")
OUT = os.path.join(HERE, "silent_edge_mentions.json")


# --- Two corrections forced by measurement, 2026-09-17. Both are documented in
# --- FINDINGS_mention_audit.md; neither may be reverted without re-measuring.

# (1) STUB GUARD. validate_maindata_text.py compared the two sources on the 16
# papers they share: 164/179 observations agree, and ALL 12 disagreements are
# git=mentioned / MAIN_DATA=absent on the only two papers where MAIN_DATA holds
# an abstract-only stub (2.4k and 1.4k chars against 92k and 46k in git). Zero
# disagreements ran the other way. Truncation can only DELETE text, so a stub
# may still prove a mention but can never disprove one. 295 of 2,019 MAIN_DATA
# documents are under this threshold; all 17 papers it currently contributes to
# the graph are 36k+, so this guard changes nothing today and exists to stop a
# future stub silently manufacturing a fabrication candidate.
MIN_FULLTEXT_CHARS = 15000

# (2) GENUS-FACTORED BINOMIALS. Papers enumerate congeners with the genus
# factored out of the list -- "8 species (ovatus, fragilis, thetaiotaomicron,
# ... and nordii) belonging to the genus Bacteroides". The extractor correctly
# reconstructs `Bacteroides fragilis`; every tier in verify_taxon_mentions
# looks for the genus adjacent to the epithet and so scores it ABSENT. That is
# a matcher false negative being reported as a candidate fabrication.
# A bare epithet is weak on its own, and a mere proximity window is NOT enough:
# a first cut of this tier used +/-400 chars and wrongly credited `Roseburia
# faecis` to a sentence where the paper attributes "faecis" to *Blautia* and
# *Agathobacter*, and credited `Vibrio phage` to the bare word "phage". Both
# would have been reported as fabrication-cleared.
#
# So the tier requires the genus to BIND the epithet:
#   - the label is a true binomial (exactly two alphabetic tokens);
#   - genus and epithet occur in the SAME sentence, within MAX_SPAN chars;
#   - NO OTHER known genus occurs between them -- an intervening congener means
#     the epithet belongs to that one, not to ours.
# The genus vocabulary is closed and comes from the graph's own node labels.
MAX_SPAN = 240
MIN_EPITHET = 5
# Degenerate "epithets" that are English/structural words, not species names.
STOP_EPITHET = {
    "phage", "virus", "bacterium", "species", "unclassified", "uncultured",
    "group", "clade", "complex", "other", "incertae", "incerte", "sedis",
}


def key(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def genus_factored(forms, text, genus_vocab):
    """Is this binomial present as a genus-factored epithet? Returns the evidence.

    Conservative by construction: anything that is not an unambiguous binomial
    bound to its own genus in one sentence is left as ABSENT for a human to read.
    """
    for f in forms:
        toks = [t for t in re.sub(r"[^A-Za-z ]", " ", f).split() if t]
        if len(toks) != 2:
            continue  # not a clean binomial -- refuse rather than guess
        genus, epithet = toks[0].lower(), toks[1].lower()
        if len(epithet) < MIN_EPITHET or len(genus) < MIN_EPITHET:
            continue
        if epithet in STOP_EPITHET or genus in STOP_EPITHET:
            continue
        for sent in SENT_SPLIT.split(text):
            for em in re.finditer(r"\b%s\b" % re.escape(epithet), sent):
                for gm in re.finditer(r"\b%s\b" % re.escape(genus), sent):
                    lo, hi = sorted([(gm.start(), gm.end()), (em.start(), em.end())])
                    if hi[0] - lo[1] > MAX_SPAN:
                        continue
                    between = sent[lo[1]:hi[0]]
                    rival = [w for w in re.findall(r"[a-z]{5,}", between)
                             if w in genus_vocab and w != genus]
                    if rival:
                        continue  # an intervening congener owns this epithet
                    return re.sub(r"\s+", " ", sent[max(0, lo[0] - 60):hi[1] + 60])
    return None


# (3) TWO MORE CONSTRUCTIONS, both deterministic, found by reading the residue.
# A paper may wedge its own abbreviation gloss into the middle of a name --
# "vibrio (vi.) phage pyd38 a" -- or use a genus abbreviation it defined earlier
# -- "cl. sp cag 273" for *Clostridium sp CAG 273*. Neither is a different taxon.
# Both are resolved against the paper's OWN inline definitions, and both are
# applied only to the node's own genus, never as a global rewrite of the text:
# this paper defines `sl.` = Salmonella while also using `sl.` for *Slackia*, so
# expanding abbreviations everywhere would invent taxa.
ABBREV_DEF = re.compile(r"\b([a-z]{4,})\s*\(\s*([a-z]{1,3})\.\s*\)")
GLOSS = re.compile(r"\(\s*[a-z]{1,3}\.\s*\)\s*")


def abbrev_map(text):
    """Genus abbreviations the paper defines inline, as `genus (ab.)`."""
    m = {}
    for genus, ab in ABBREV_DEF.findall(text):
        m.setdefault(ab, genus)
    return m


def gloss_stripped(forms, text_gloss_sq):
    """Name interrupted by a parenthetical abbreviation gloss."""
    for f in forms:
        if squash(f) and squash(f) in text_gloss_sq:
            return f
    return None


def defined_abbrev(forms, text_sq, amap):
    """Genus written as the abbreviation the paper itself defined for it."""
    rev = {g: a for a, g in amap.items()}
    for f in forms:
        toks = [t for t in re.sub(r"[^A-Za-z0-9 ]", " ", f).split() if t]
        if len(toks) < 2:
            continue
        ab = rev.get(toks[0].lower())
        if not ab:
            continue
        needle = squash(ab + "".join(toks[1:]))
        if needle and needle in text_sq:
            return f"{ab}. {' '.join(toks[1:])}"
    return None


def build_genus_vocab(nodes):
    """Closed vocabulary of genus-like tokens, from the graph's own labels."""
    v = set()
    for n in nodes.values():
        for s in [n.get("label")] + list(n.get("aliases") or []):
            if not s:
                continue
            toks = [t for t in re.sub(r"[^A-Za-z ]", " ", s).split() if len(t) >= 5]
            if toks:
                v.add(toks[0].lower())
    return v - STOP_EPITHET


def main():
    g = json.load(open(os.path.join(HERE, "graph.json")))
    papers = g["papers"]
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = g["edges"]
    genus_vocab = build_genus_vocab(nodes)

    fulltext = {}
    src = {}
    for p in json.load(open(PAPERS)):
        k = key(p.get("title"))
        if k and len(p.get("text") or "") > 500:
            t = norm_text(p["text"])
            fulltext[k] = (t, squash(t), len(p["text"]), squash(GLOSS.sub("", t)), abbrev_map(t))
            src[k] = "git"

    # Sources 2 and 3: the text the extractor was actually GIVEN. UNTRACKED --
    # they were removed from git in 254b0a8 because THIS REPO IS PUBLIC and the
    # corpus includes non-open-access articles. They are in .gitignore. Read them
    # locally, never `git add` them, never quote their text into a tracked file.
    # Restore for local work with:
    #   git show 254b0a8^:proj_2_attempt3/kg/extract_input.json > extract_input.json
    #   git show 254b0a8^:proj_2_attempt3/kg/new_papers.json    > new_papers.json
    # Together with source 1 and 4 these cover 271/271 contributing papers.
    # `validate_text_sources.py` shows 5 of 6 source pairs agree EXACTLY (incl.
    # extract_in vs main_data at n=303 and extract_in vs new_papers at n=629), so
    # which source answers a given paper cannot change its verdict; the only
    # disagreements anywhere are the MAIN_DATA stubs that guard (1) already traps.
    n_extra = 0
    for fn in ("extract_input.json", "new_papers.json"):
        fp = os.path.join(HERE, fn)
        if not os.path.exists(fp):
            continue
        for r in json.load(open(fp)):
            k = key(r.get("title"))
            if not k or k in fulltext or len(r.get("text") or "") <= 500:
                continue
            t = norm_text(r["text"])
            fulltext[k] = (t, squash(t), len(r["text"]),
                           squash(GLOSS.sub("", t)), abbrev_map(t))
            src[k] = "extractor_input"
            n_extra += 1

    # Additive fallback. `setdefault` semantics: source 1 keeps any shared title,
    # so wiring this in cannot change a verdict that was already scoreable.
    n_md = 0
    if os.path.exists(MAIN_DATA):
        for rec in json.load(open(MAIN_DATA)).values():
            k = key(rec.get("name"))
            if not k or k in fulltext:
                continue
            body = "\n".join(rec.get("chunks") or [])
            if len(body) > 500:
                t = norm_text(body)
                fulltext[k] = (t, squash(t), len(body), squash(GLOSS.sub("", t)), abbrev_map(t))
                src[k] = "main_data"
                n_md += 1
    print(f"full text: {len(fulltext)} papers ("
          f"{len(fulltext) - n_md - n_extra} all_usable, "
          f"{n_extra} extractor-input, {n_md} MAIN_DATA)")

    obs = W.build()  # the published per-observation provenance classification

    # Cache mention lookups: (paper_key, node_id) ->
    #   "mentioned" | "genus_factored" | "absent" | None (not scoreable)
    cache = {}
    evidence = {}

    def mentioned(paper_key, node):
        ck = (paper_key, node["id"])
        if ck in cache:
            return cache[ck]
        pair = fulltext.get(paper_key)
        if pair is None:
            cache[ck] = None
            return None
        text, text_sq, rawlen, text_gloss_sq, amap = pair
        # RAW label/aliases, not A.taxon_matchers(): that returns norm_surface'd
        # (whitespace-stripped) forms, which cannot match unstripped text.
        forms = [s for s in [node.get("label")] + list(node.get("aliases") or []) if s]
        verdict = "absent"
        for f in forms:
            tier, _ = classify(f, text, text_sq)
            if tier in ("exact", "variant", "abbrev"):
                verdict = "mentioned"
                break
        if verdict == "absent":
            quote = genus_factored(forms, text, genus_vocab)
            if quote:
                verdict = "genus_factored"
                evidence[ck] = quote
        if verdict == "absent":
            q = gloss_stripped(forms, text_gloss_sq)
            if q:
                verdict = "gloss_gap"
                evidence[ck] = f"matches after removing the paper's `(xx.)` gloss: {q}"
        if verdict == "absent":
            q = defined_abbrev(forms, text_sq, amap)
            if q:
                verdict = "defined_abbrev"
                evidence[ck] = f"paper writes it as `{q}` using its own inline definition"
        if verdict == "absent":
            if src.get(paper_key) == "main_data" and rawlen < MIN_FULLTEXT_CHARS:
                # A stub can prove a mention but never disprove one. See (1) above.
                verdict = None
        cache[ck] = verdict
        return verdict

    # cross-tab provenance x mention
    tab = defaultdict(Counter)
    absent_rows = []
    n_unscoreable = 0

    # edge-level: how many papers back the edge, for the "single-paper" cut
    for o in obs:
        e = edges[o["edge"]]
        node = nodes.get(e["source"])
        if node is None:
            continue
        pk = key(papers[o["paper"]]["title"])
        m = mentioned(pk, node)
        if m is None:
            n_unscoreable += 1
            tab[o["prov"]]["not_scoreable"] += 1
            continue
        tab[o["prov"]][m] += 1
        if m in ("absent", "genus_factored", "gloss_gap", "defined_abbrev"):
            absent_rows.append({
                "taxon": e["taxon"], "disease": e["disease"],
                "direction": o["dir"], "prov": o["prov"],
                "verdict": m,
                "evidence": evidence.get((pk, node["id"])),
                "paper": papers[o["paper"]]["title"],
                "text_source": src.get(pk),
                "n_papers_on_edge": e.get("n_papers"),
                "node_label": node.get("label"),
                "aliases": node.get("aliases") or [],
                "resolved": node.get("resolved"),
                "placeholder": node.get("placeholder"),
            })

    # The specific population FINDINGS_direction_audit.md called unreachable:
    # single-paper edges with no own-result witness.
    own_by_edge = defaultdict(set)
    for o in obs:
        own_by_edge[o["edge"]].add(o["prov"])

    unreachable = Counter()
    for ei, e in enumerate(edges):
        if e.get("n_papers") != 1:
            continue
        provs = own_by_edge.get(ei, set())
        if "own" in provs:
            continue
        node = nodes.get(e["source"])
        if node is None:
            continue
        rows = [o for o in obs if o["edge"] == ei]
        if not rows:
            continue
        pk = key(papers[rows[0]["paper"]]["title"])
        m = mentioned(pk, node)
        unreachable["total"] += 1
        unreachable["not_scoreable" if m is None else m] += 1

    def rate(c):
        """Mention rate. `genus_factored` counts as mentioned -- the taxon IS named,
        with its genus factored into the surrounding clause. `strict` is the old
        definition, kept so the pre-2026-09-17 figures stay reproducible."""
        recovered = c["genus_factored"] + c["gloss_gap"] + c["defined_abbrev"]
        d = c["mentioned"] + c["absent"] + recovered
        if not d:
            return None
        return {
            "mentioned": round((c["mentioned"] + recovered) / d, 4),
            "strict": round(c["mentioned"] / d, 4),
        }

    out = {
        "n_observations": len(obs),
        "n_not_scoreable_no_fulltext": n_unscoreable,
        "by_provenance": {k: dict(v) for k, v in tab.items()},
        "mention_rate_by_provenance": {k: rate(v) for k, v in tab.items()},
        "single_paper_no_own_witness_edges": dict(unreachable),
        "single_paper_no_own_witness_mention_rate": rate(unreachable),
        "absent_observations": sorted(
            absent_rows, key=lambda r: (r["prov"], r["taxon"]))[:400],
        "n_absent_total": sum(1 for r in absent_rows if r["verdict"] == "absent"),
        "n_recovered_by_tier": dict(Counter(
            r["verdict"] for r in absent_rows if r["verdict"] != "absent")),
    }
    json.dump(out, open(OUT, "w"), indent=1)

    print(json.dumps({k: v for k, v in out.items()
                      if k != "absent_observations"}, indent=1))
    print(f"\nwrote {OUT}")
    for want in ("absent", "genus_factored", "gloss_gap", "defined_abbrev"):
        rows = [r for r in absent_rows if r["verdict"] == want]
        if not rows:
            continue
        print(f"\n--- {len(rows)} {want} ---")
        for r in rows:
            print(f"  [{r['prov']:10s}] {r['taxon'][:38]:38s} {r['disease'][:22]:22s} "
                  f"npap={r['n_papers_on_edge']} src={r['text_source']}")
            if r.get("evidence"):
                print(f"      > ...{r['evidence'][:150]}...")


if __name__ == "__main__":
    main()
