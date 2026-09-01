#!/usr/bin/env python3
"""Reduce full papers to the sentences that could actually STATE a relation.

A microbe-disease relation ("Bacteroides is depleted in Alzheimer's") can only be
asserted in a sentence that names a taxon AND carries a direction word. Every
other sentence in a 47k-character paper -- methods, ethics statements, reference
lists, funding -- is noise for the question "why do these two papers disagree?".

That question has already been attacked twice and failed:

  1. hand-picked study-design variables (analyze_contested.py): nothing survived
     cluster-robust permutation + BH (best FDR 0.243);
  2. naive full-text word counting: on Bacteroides/Alzheimer's, 198 terms
     separated the up-papers from the down-papers by >45 points -- but a RANDOM
     split of the same 12 papers separated 184 (p=0.41). With ~8,800 distinct
     terms and 12 papers, hundreds of words align with any split by chance.

Approach (2) failed because of the feature space, not the test. This module is
the fix: shrink the text to relation-bearing sentences, and shrink the vocabulary
to NCBI-resolvable taxa. Both cuts are ~40-70x and neither is a bag of words --
"which other taxa does this paper report" is an interpretable feature.

THE FILTER IS ONLY WORTH USING IF IT KEEPS THE RELATIONS. A filter with a great
reduction ratio and poor recall silently deletes the signal it is supposed to
concentrate, so `--validate` is not optional decoration: it replays every
relation the corpus-scale extractor found and asks whether a surviving sentence
still supports it. Run it before trusting any downstream number.

    python relation_sentences.py --validate          # recall + reduction (do this first)
    python relation_sentences.py --build             # write relation_sentences.json
    python relation_sentences.py --mode strict       # directional cues only

Taxon matching notes
--------------------
* Case-insensitive. Papers write "Bacteroides" in the abstract and "bacteroides"
  in a figure caption; an earlier measurement here only matched capitalised
  candidates and undercounted. The false-positive worry that motivates
  capitalisation ("This", "Data", "China", "Fisher") turns out to be handled
  already by taxonomy.resolve(), which rejects any candidate whose lineage is not
  under Bacteria/Archaea/Fungi/Viruses -- all four of those resolve to nothing.
* Genus abbreviations are expanded per paper: if a paper names Bacteroides
  anywhere, "B. fragilis" later in that paper resolves.
* Ranks above phylum are dropped as features ("Bacteria", "bacterium" are real
  taxids but carry no information about which organisms a paper reports).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
EXTRACTIONS = os.path.join(HERE, "..", "dsmlp_model_prompting", "eval-v2", "results",
                           "qwopus3.5-27b-v3__q4km__samgated-v1__all250.json")
OUT = os.path.join(HERE, "relation_sentences.json")

# --------------------------------------------------------------------------
# direction cues
#
# Tiered deliberately. STRICT is the polarised vocabulary a reader would call a
# "direction word". LOOSE adds the words that carry a comparison without stating
# its sign ("differential abundance of X between groups", "altered", "the X/Y
# ratio") -- these DO introduce relations, and dropping them costs recall, which
# is exactly the trade --validate measures instead of assuming.
UP = [r"increas\w*", r"elevat\w*", r"enrich\w*", r"higher", r"greater", r"more abundant",
      r"over-?represent\w*", r"over-?abundan\w*", r"up-?regulat\w*", r"expand\w*",
      r"overgrow\w*", r"bloom\w*", r"predomin\w*", r"dominat\w*", r"accumulat\w*",
      r"maxim\w*", r"highest", r"most abundant", r"most prevalent",
      r"augment\w*", r"gain\w*", r"prolifera\w*", r"outnumber\w*", r"rais\w*", r"rose",
      r"enrichment", r"abundant"]
DOWN = [r"decreas\w*", r"reduc\w*", r"deplet\w*", r"lower", r"lowest", r"fewer", r"less abundant",
        r"diminish\w*", r"under-?represent\w*", r"down-?regulat\w*", r"declin\w*",
        r"depriv\w*", r"scarce\w*", r"deficien\w*", r"impoverish\w*", r"attenuat\w*",
        r"loss of", r"lost", r"absent", r"absence of", r"disappear\w*", r"suppress\w*",
        r"depletion", r"rarer", r"underrepresent\w*", r"minim\w*", r"least abundant"]
# sign-free comparison cues: a relation is being stated, direction lives elsewhere
NEUTRAL = [r"abundanc\w*", r"differential\w*", r"differ\w*", r"alter\w*", r"chang\w*",
           r"shift\w*", r"proportion\w*", r"prevalenc\w*", r"relative abundance",
           r"significant\w*", r"associat\w*", r"correlat\w*", r"compared (?:to|with)",
           r"versus", r"\bvs\.?\b", r"ratio", r"discriminat\w*", r"characteriz\w*",
           r"biomarker\w*", r"lefse", r"\blda\b", r"p\s*[<>=]", r"level\w*", r"\bOTU",
           r"\bAUC\b", r"optimal", r"marker\w*", r"\bFDR\b", r"\bq\s*[<>=]"]

_UP_RE = re.compile(r"\b(?:%s)\b" % "|".join(UP), re.I)
_DOWN_RE = re.compile(r"\b(?:%s)\b" % "|".join(DOWN), re.I)
_NEU_RE = re.compile(r"(?:%s)" % "|".join(NEUTRAL), re.I)

# --------------------------------------------------------------------------
# sentence splitting
#
# Off-the-shelf splitters are not available offline and biomedical text breaks
# the naive rule anyway: "P < 0.05." vs "B. fragilis" vs "et al." vs "Fig. 2".
# Protect the known abbreviation shapes, then split on terminal punctuation
# followed by a capital or a digit.
_ABBREV = re.compile(
    r"\b(?:et al|e\.g|i\.e|vs|cf|Fig|Figs|Tab|approx|ca|no|No|St|Dr|Prof|Inc|Ltd|"
    r"spp|sp|subsp|var|str|min|max|sec|hr|wk|mo|yr|ref|Refs|Suppl|Supp|Eq)\.")
_INITIAL = re.compile(r"\b([A-Z])\.(?=\s*[a-z])")          # "B. fragilis"
_DECIMAL = re.compile(r"(\d)\.(\d)")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

_REFS = re.compile(r"\n\s*(?:References|REFERENCES|Bibliography|Literature cited)\s*\n")


def strip_references(text):
    """Cut the reference list; it is ~30% of a paper and states no relations."""
    m = None
    for m2 in _REFS.finditer(text):
        m = m2
    if m and m.start() > 0.4 * len(text):
        return text[:m.start()]
    return text


def sentences(text):
    """Split into sentences, keeping the protected abbreviations intact."""
    t = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), text)
    t = _INITIAL.sub(lambda m: m.group(1) + "\x00", t)
    t = _DECIMAL.sub(lambda m: m.group(1) + "\x01" + m.group(2), t)
    out = []
    for chunk in t.split("\n"):
        for s in _SPLIT.split(chunk):
            s = s.replace("\x00", ".").replace("\x01", ".").strip()
            if len(s) >= 25:                 # headings, page furniture, author lines
                out.append(s)
    return out


# --------------------------------------------------------------------------
# taxon matching
_TOKEN = re.compile(r"\[?[A-Za-z][A-Za-z0-9_\-\]]*")
_ABBREV_SP = re.compile(r"\b([A-Z])\.\s*([a-z][a-z\-]{2,})\b")

# real taxa, but useless as a feature: every microbiome paper names them
_TOO_GENERIC = {"bacteria", "bacterium", "archaea", "fungi", "virus", "viruses",
                "eukaryota", "prokaryote", "microorganism", "organism"}
_USEFUL_RANKS = {"phylum", "class", "order", "family", "genus", "species",
                 "subspecies", "strain", "subphylum", "subclass", "suborder",
                 "subfamily", "tribe", "species group", "species subgroup",
                 "no rank", "clade"}
_BAD_RANKS = {"superkingdom", "domain", "kingdom", "realm", "acellular root",
              "cellular root"}


class TaxonMatcher:
    """Longest-match taxon spotter over a sentence, backed by the NCBI taxdump."""

    def __init__(self, tax):
        self.tax = tax
        self.cache = {}                       # candidate string -> hit or None

    def _lookup(self, cand, allow_trim):
        """allow_trim: may resolve() fall back to a prefix of the candidate?

        resolve() trims qualifier tails so that "Clostridium sensu stricto 1"
        lands on Clostridium. Harmless for one token, disastrous across several:
        the 3-gram "Catabacter Howardella Marine_Methylotrophic_Group_3" trimmed
        down to Catabacter, consumed all three tokens, and Howardella -- a taxon
        the extractor did report -- silently vanished. Multi-token candidates
        must therefore match a real NCBI name in full, or not at all.
        """
        key = (cand, allow_trim)
        if key in self.cache:
            return self.cache[key]
        hit = None
        low = cand.lower()
        known = low in self.tax.name2ids
        if known or (allow_trim and ("_" in cand or cand.startswith("["))):
            tid, sci, rank, how = self.tax.resolve(cand)
            if tid and rank not in _BAD_RANKS and sci.lower() not in _TOO_GENERIC:
                hit = (tid, sci, rank)
        self.cache[key] = hit
        return hit

    def find(self, sentence, alias=None):
        """-> list of (surface, taxid, scientific_name, rank), longest match wins."""
        spans = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(sentence)]
        toks = [s[2].strip("[]") for s in spans]
        out, i = [], 0
        while i < len(toks):
            got = None
            for n in (3, 2, 1):
                if i + n > len(toks):
                    continue
                # a binomial never spans a comma or bracket: only whitespace and
                # hyphens may sit between the tokens of one name
                if any(spans[j + 1][0] > spans[j][1] and
                       sentence[spans[j][1]:spans[j + 1][0]].strip(" \t\xa0-") != ""
                       for j in range(i, i + n - 1)):
                    continue
                cand = " ".join(toks[i:i + n])
                if n == 1 and len(cand) < 4:
                    continue
                hit = self._lookup(cand, allow_trim=(n == 1))
                if hit:
                    got = (sentence[spans[i][0]:spans[i + n - 1][1]], *hit)
                    i += n
                    break
            if got:
                out.append(got)
            else:
                i += 1
        # "B. fragilis" -> "Bacteroides fragilis" using genera seen in this paper
        if alias:
            for m in _ABBREV_SP.finditer(sentence):
                g = alias.get(m.group(1).upper())
                if not g:
                    continue
                hit = (self._lookup(f"{g} {m.group(2)}", False)
                       or self._lookup(g, False))
                if hit:
                    out.append((m.group(0), *hit))
        return out


def cue_polarity(sentence, mode="loose"):
    """-> (kept?, set of polarities present) for the sentence."""
    up = bool(_UP_RE.search(sentence))
    dn = bool(_DOWN_RE.search(sentence))
    neu = bool(_NEU_RE.search(sentence)) if mode == "loose" else False
    pol = set()
    if up:
        pol.add("up")
    if dn:
        pol.add("down")
    if neu:
        pol.add("neutral")
    return bool(pol), pol


# --------------------------------------------------------------------------
def load_taxonomy():
    sys.path.insert(0, HERE)
    from taxonomy import Taxonomy
    t = Taxonomy()
    if t.ok:
        return t
    # Fall back to the replay cache (taxonomy_cache.py) where the taxdump cannot be
    # fetched. The bias is ASYMMETRIC and matters differently for the two numbers
    # this module reports, so it must not be papered over:
    #
    #   RECALL is close to unaffected. It asks whether a surviving sentence still
    #   supports a relation the extractor already found, and those taxa are by
    #   construction in graph.json — so the cache resolves them.
    #
    #   REDUCTION is OVERSTATED. The cache knows only the ~1k names in the graph,
    #   so taxa the paper mentions but we never extracted go unresolved, their
    #   sentences are discarded, and the filter looks more aggressive than it is.
    #
    # Treat the reduction ratio as an upper bound until this is rerun on a taxdump.
    from taxonomy_cache import CachedTaxonomy
    c = CachedTaxonomy()
    if not c.ok:
        sys.exit("No taxonomy available: no NCBI taxdump and no graph.json to replay.")
    print("WARNING: no NCBI taxdump — using the graph.json replay cache.")
    print("         recall is ~unaffected; REDUCTION RATIO IS AN UPPER BOUND.\n")
    return c


def filter_paper(text, matcher, mode="loose"):
    """-> (kept_sentences, n_sentences_total, n_chars_total, all_taxids).

    kept sentence = {"s": text, "taxa": [(surface, taxid, sci, rank)], "pol": [...]}
    `all_taxids` is every taxon the matcher saw in ANY sentence, kept or not. It
    is the diagnostic that separates "the cue filter dropped this relation" from
    "the matcher never saw the taxon" -- without it a recall miss is unattributable.
    """
    body = strip_references(text)
    sents = sentences(body)
    # pass 1: collect this paper's genera so abbreviations can be expanded in pass 2
    alias = {}
    prelim = []
    for s in sents:
        hits = matcher.find(s)
        prelim.append(hits)
        for _surf, _tid, sci, rank in hits:
            if rank == "genus" and sci:
                alias.setdefault(sci[0].upper(), sci)
    kept, seen = [], set()
    for s, hits in zip(sents, prelim):
        if _ABBREV_SP.search(s):
            hits = hits + [h for h in matcher.find(s, alias) if h not in hits]
        seen |= {h[1] for h in hits}
        ok, pol = cue_polarity(s, mode)
        if not ok or not hits:
            continue
        kept.append({"s": s, "taxa": hits, "pol": sorted(pol)})
    return kept, len(sents), len(body), seen


def build(mode="loose", limit=None, quiet=False):
    papers = json.load(open(PAPERS))
    if limit:
        papers = papers[:limit]
    tax = load_taxonomy()
    m = TaxonMatcher(tax)
    out, tot_s, tot_c, kept_c = {}, 0, 0, 0
    for i, p in enumerate(papers):
        kept, ns, nc, seen = filter_paper(p.get("text") or "", m, mode)
        out[p["title"]] = {"disease": p.get("disease", ""), "link": p.get("link", ""),
                           "n_sentences": ns, "n_chars": nc, "kept": kept,
                           "taxids_any_sentence": sorted(seen)}
        tot_s += ns
        tot_c += nc
        kept_c += sum(len(k["s"]) for k in kept)
        if not quiet and (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(papers)} papers…", file=sys.stderr)
    n_kept = sum(len(v["kept"]) for v in out.values())
    stats = {"mode": mode, "papers": len(papers), "sentences_total": tot_s,
             "sentences_kept": n_kept, "chars_total": tot_c, "chars_kept": kept_c,
             "sentence_reduction": round(tot_s / n_kept, 1) if n_kept else None,
             "char_reduction": round(tot_c / kept_c, 1) if kept_c else None}
    return out, stats, tax


# --------------------------------------------------------------------------
# recall validation
def _extraction_relations(tax):
    """-> {paper_title: {(taxid_or_surface, direction): as_written}} from the extractor."""
    sys.path.insert(0, HERE)
    from build_kg import parse_taxa, norm_taxon
    rows = json.load(open(EXTRACTIONS))
    rel = defaultdict(dict)
    for r in rows:
        if r.get("parse_error"):
            continue
        for direction, col in (("up", "predicted_enriched"), ("down", "predicted_depleted")):
            for raw in parse_taxa(r.get(col)):
                key, disp, _rank, _how = norm_taxon(raw, tax)
                rel[r["title"]][(key, direction)] = raw
    return rel


def validate(mode="loose", limit=None):
    """Replay every extracted relation and ask whether the filter kept a sentence for it.

    Four nested levels, reported separately because they fail for different reasons:
      in_text     - the taxon string occurs somewhere in the raw paper. Anything
                    below this is unreachable by ANY sentence filter (the model
                    read a table image, paraphrased, or hallucinated).
      seen        - the taxon matcher found it in some sentence. in_text minus
                    this is matcher loss.
      exact       - a KEPT sentence names that exact taxid. seen minus this is
                    what the direction-cue filter actually cost. This is the
                    honest headline recall.
      directional - the kept sentence's cue polarity also agrees with the
                    reported direction. Strictest, and partly unfair: in
                    "Blautia increased while Roseburia fell" the cue that belongs
                    to a taxon is not resolved by a sentence-level test.
      nested      - relaxation: an ancestor/descendant taxid is in a kept
                    sentence. GENEROUS -- a kept "Firmicutes" sentence covers
                    every Firmicutes genus -- so it is a bound, not the number.
    """
    kept_by_paper, stats, tax = build(mode=mode, limit=limit)
    rel = _extraction_relations(tax)
    papers = {p["title"]: p for p in json.load(open(PAPERS))}

    n = Counter()
    misses = []
    for title, doc in kept_by_paper.items():
        rels = rel.get(title, {})
        if not rels:
            continue
        kept_ids, kept_dir, kept_strings = set(), defaultdict(set), []
        for k in doc["kept"]:
            ids = {t[1] for t in k["taxa"]}
            kept_ids |= ids
            for tid in ids:
                kept_dir[tid] |= set(k["pol"])
            kept_strings.append(k["s"].lower())
        kept_lin = defaultdict(set)
        for tid in kept_ids:
            for a in tax.lineage(tid):
                kept_lin[a].add(tid)
        seen_ids = set(doc["taxids_any_sentence"])
        full = (papers.get(title, {}).get("text") or "").lower()

        for (key, direction), as_written in rels.items():
            n["total"] += 1
            surface = as_written.lower().strip()
            in_text = surface in full
            if key.startswith("ncbi:"):
                tid = key.split(":", 1)[1]
                seen = tid in seen_ids
                exact = tid in kept_ids
                nested = exact or bool(set(tax.lineage(tid)) & kept_ids) or tid in kept_lin
                pol = set(kept_dir.get(tid, ()))
                if not exact and nested:
                    for anc in tax.lineage(tid):
                        pol |= kept_dir.get(anc, set())
                    for desc in kept_lin.get(tid, ()):
                        pol |= kept_dir.get(desc, set())
            else:
                # unresolvable surface string ("unidentified_Bacteria"): substring test
                seen = exact = any(surface in s for s in kept_strings)
                nested = exact
                pol = {p for k in doc["kept"] if surface in k["s"].lower() for p in k["pol"]}
            n["in_text"] += in_text
            n["seen"] += seen
            n["exact"] += exact
            n["nested"] += nested
            n["directional"] += bool(nested and (direction in pol or "neutral" in pol))
            if not exact:
                why = ("not in raw text" if not in_text else
                       "matcher missed" if not seen else "cue filter dropped")
                misses.append((title[:52], as_written, direction, why))
    return stats, n, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["strict", "loose"], default="loose")
    ap.add_argument("--validate", action="store_true", help="measure recall + reduction")
    ap.add_argument("--build", action="store_true", help="write relation_sentences.json")
    ap.add_argument("--limit", type=int, help="first N papers only (debug)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.validate:
        for mode in (["strict", "loose"] if a.mode == "loose" else [a.mode]):
            print(f"\n=== mode: {mode} ===", file=sys.stderr)
            stats, rec, misses = validate(mode=mode, limit=a.limit)
            print(f"\nmode={mode}")
            print(f"  sentences {stats['sentences_total']:>8,} -> {stats['sentences_kept']:>6,}"
                  f"  ({stats['sentence_reduction']}x)")
            print(f"  chars     {stats['chars_total']:>8,} -> {stats['chars_kept']:>6,}"
                  f"  ({stats['char_reduction']}x)")
            t = rec["total"]
            print(f"  RECALL over {t} extracted relations")
            print(f"    taxon string anywhere in raw paper        : {rec['in_text']:>5} "
                  f"({rec['in_text']/t:.1%})   <- ceiling for any sentence filter")
            print(f"    taxon seen by matcher in some sentence    : {rec['seen']:>5} "
                  f"({rec['seen']/t:.1%})")
            print(f"    taxon in a KEPT sentence, exact taxid     : {rec['exact']:>5} "
                  f"({rec['exact']/t:.1%})   <- HEADLINE RECALL")
            print(f"       ...as a share of what the matcher saw  : "
                  f"{rec['exact']/rec['seen']:.1%}  (= what the cue filter cost)")
            print(f"    kept sentence cue agrees with direction   : {rec['directional']:>5} "
                  f"({rec['directional']/t:.1%})")
            print(f"    relaxed: ancestor/descendant kept (bound) : {rec['nested']:>5} "
                  f"({rec['nested']/t:.1%})")
            c = Counter(m[3] for m in misses)
            print(f"  {len(misses)} misses by cause: " +
                  ", ".join(f"{k}={v}" for k, v in c.most_common()))
            for m in misses[:8]:
                print(f"    - {m[1][:32]:34} {m[2]:5} {m[3]:18} {m[0]}")
        return

    if a.build:
        docs, stats, _ = build(mode=a.mode)
        json.dump({"stats": stats, "papers": docs}, open(a.out, "w"))
        print(json.dumps(stats, indent=2))
        print(f"wrote {a.out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
