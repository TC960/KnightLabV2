#!/usr/bin/env python3
"""Does every taxon the extractor claimed actually occur in the paper it read?

The strongest falsification available for this graph, and the only fidelity
signal that needs neither the (compromised) in-house gold, nor Disbiome/Peryton,
nor a direction cue. It is a pure string question: the extractor emitted the
surface string `Faecalibacterium` while reading paper P -- does `Faecalibacterium`
appear in P's full text?

Motivation. FINDINGS_direction_audit.md sized 580 edges (28.9%) that rest on one
paper AND have no own-result prose witness, and concluded "prose filtering cannot
reach them". That is true of the RELATION filter, which needs a taxon and a
direction cue in the SAME sentence. It is not true of the weaker question asked
here, which needs only the taxon name anywhere in the document. A table-only
taxon is silent to the relation filter but still mentioned in the text.

Scope note: full text is available in git for the datasheet papers only
(all_usable_papers.json). Papers reachable only through MAIN_DATA.json are
gitignored and are reported as not-scoreable rather than as misses.

Match tiers, strictest first:
  exact   -- the surface string occurs verbatim (after unicode/case/space
             normalisation) in the paper text.
  variant -- a normalised form matches: separators collapsed, rank prefixes
             dropped, `sp.`/`spp.`/`unclassified` suffixes dropped.
  abbrev  -- a binomial `Genus species` occurs as `G. species`.
  head    -- only the leading token (genus/family) occurs. WEAK: confirms the
             concept is in the paper, not the exact taxon.
  MISS    -- none of the above. Candidate fabrication; to be read by hand.

Deterministic by construction. Per the repo rule, no LLM judgement is used where
a string comparison settles it.
"""
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTIONS = os.path.join(HERE, "extractions_screened.json")
PAPERS = os.path.join(HERE, "..", "EmilySong_GoldStandardPaper", "all_usable_papers.json")
OUT = os.path.join(HERE, "taxon_mentions.json")

# Rank prefixes SILVA/QIIME emit; stripped before variant matching.
RANK_PREFIX = re.compile(r"^[dkpcofgs]__+")
# Trailing placeholder words that mean "this genus, unspecified species".
TAIL_NOISE = re.compile(
    r"\b(sp|spp|unclassified|uncultured|group|clade|complex|other|incertae|sedis)\b\.?\s*$",
    re.I,
)
PVAL = re.compile(r"\(?\s*[pq]\s*[<>=]\s*0?\.\d+\s*\)?", re.I)


def nfkc(s: str) -> str:
    """Fold unicode dashes/spaces so the paper and the extraction compare equal."""
    s = unicodedata.normalize("NFKC", s)
    for dash in "‐‑‒–—―−":
        s = s.replace(dash, "-")
    s = s.replace(" ", " ")
    return s


def norm_text(s: str) -> str:
    """Paper-side normalisation: case-fold, collapse whitespace."""
    return re.sub(r"\s+", " ", nfkc(s).lower())


def norm_taxon(s: str) -> str:
    """Extraction-side normalisation, conservative: never changes which taxon is meant."""
    s = nfkc(s).lower().strip()
    s = PVAL.sub(" ", s)
    s = s.strip(" \t\n\r.,;:()[]{}\"'")
    s = RANK_PREFIX.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def variants(t: str):
    """Spellings that denote the SAME taxon. Deliberately narrow -- see the
    repo's recorded near-miss where edit distance merged Oscillospirales into
    Oscillospira. Nothing here changes the identifying tokens."""
    out = {t}
    # separator collapse: Escherichia-Shigella / Escherichia_Shigella / Escherichia/Shigella
    out.add(re.sub(r"[-_/]+", " ", t))
    out.add(re.sub(r"[-_/\s]+", "", t))
    out.add(re.sub(r"[\[\]]", "", t))
    stripped = TAIL_NOISE.sub("", t).strip()
    if stripped and stripped != t:
        out.add(stripped)
        out.add(re.sub(r"[-_/]+", " ", stripped))
    return {v for v in out if len(v) >= 4}


def abbrev_forms(t: str):
    """`Genus species` -> `G. species` / `G.species`, the standard paper shorthand."""
    parts = t.split()
    if len(parts) >= 2 and len(parts[0]) > 2 and parts[1].isalpha():
        initial = parts[0][0]
        rest = " ".join(parts[1:])
        return {f"{initial}. {rest}", f"{initial}.{rest}"}
    return set()


def squash(s: str) -> str:
    """Drop every non-alphanumeric. Used ONLY symmetrically -- a squashed needle
    against a squashed haystack -- so that separator and whitespace conventions
    (`Coprococcus_2` vs `Coprococcus 2`, `Escherichia-Shigella` vs
    `Escherichia/Shigella`) stop mattering. Comparing a squashed needle against
    unsquashed text is a bug: it can never match a multi-word taxon."""
    return re.sub(r"[^a-z0-9]", "", nfkc(s).lower())


def classify(taxon: str, text: str, text_squashed: str = None):
    """Return (tier, evidence_string). text is already norm_text'd.
    text_squashed, when given, enables the separator-insensitive tier."""
    t = norm_taxon(taxon)
    if not t or len(t) < 3:
        return "unscoreable", ""
    if t in text:
        return "exact", t
    for v in sorted(variants(t), key=len, reverse=True):
        if v != t and v in text:
            return "variant", v
    for a in abbrev_forms(t):
        if a in text:
            return "abbrev", a
    if text_squashed is not None:
        sq = squash(t)
        # >=8 chars: short squashed needles match across word boundaries by chance.
        if len(sq) >= 8 and sq in text_squashed:
            return "variant", sq
    head = t.split()[0].strip("[]")
    # A bare head match only means something if the head is a real name, not a
    # fragment like "eubacterium" inside "[eubacterium]".
    if len(head) >= 5 and head in text:
        return "head", head
    return "MISS", ""


def main():
    extractions = json.load(open(EXTRACTIONS))
    papers = json.load(open(PAPERS))

    by_title = {}
    for p in papers:
        key = re.sub(r"[^a-z0-9]", "", (p.get("title") or "").lower())
        if key and len(p.get("text") or "") > 500:
            by_title[key] = p["text"]

    tiers = Counter()
    misses = []
    per_paper = defaultdict(Counter)
    scoreable_papers = set()
    unscoreable_papers = set()
    seen_claims = set()

    for rec in extractions:
        title = rec.get("title") or ""
        key = re.sub(r"[^a-z0-9]", "", title.lower())
        text_raw = by_title.get(key)
        if text_raw is None:
            unscoreable_papers.add(title)
            continue
        scoreable_papers.add(title)
        text = norm_text(text_raw)

        for field, direction in (("predicted_enriched", "enriched"),
                                 ("predicted_depleted", "depleted")):
            raw = rec.get(field) or ""
            if not isinstance(raw, str):
                continue
            for chunk in re.split(r"[,;]", raw):
                taxon = chunk.strip()
                if not taxon:
                    continue
                # One claim = one (paper, taxon) pair. A taxon named in both
                # directions by one paper is a self-contradiction, counted once
                # here; the mention question does not depend on direction.
                claim = (key, norm_taxon(taxon))
                if claim in seen_claims:
                    continue
                seen_claims.add(claim)

                tier, ev = classify(taxon, text)
                tiers[tier] += 1
                per_paper[title][tier] += 1
                if tier == "MISS":
                    misses.append({
                        "paper": title,
                        "taxon": taxon,
                        "normalised": norm_taxon(taxon),
                        "direction": direction,
                        "link": rec.get("link", ""),
                        "disease": rec.get("disease", ""),
                    })

    total = sum(tiers.values())
    found = total - tiers["MISS"] - tiers["unscoreable"]
    strict = tiers["exact"] + tiers["variant"] + tiers["abbrev"]
    denom = total - tiers["unscoreable"]

    summary = {
        "papers_in_extractions": len({re.sub(r'[^a-z0-9]', '', (r.get('title') or '').lower())
                                      for r in extractions}),
        "papers_scoreable": len(scoreable_papers),
        "papers_no_fulltext_in_git": len(unscoreable_papers),
        "claims_scored": denom,
        "tiers": dict(tiers),
        "mention_rate_any": round(found / denom, 4) if denom else None,
        "mention_rate_strict_excl_head": round(strict / denom, 4) if denom else None,
        "n_miss": tiers["MISS"],
    }

    json.dump(
        {"summary": summary,
         "misses": sorted(misses, key=lambda m: (m["paper"], m["taxon"])),
         "per_paper": {k: dict(v) for k, v in per_paper.items()}},
        open(OUT, "w"), indent=1,
    )

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}")
    if misses:
        print(f"\n--- {len(misses)} candidate misses (first 40) ---")
        for m in misses[:40]:
            print(f"  {m['taxon']!r:45s} <- {m['paper'][:70]}")


if __name__ == "__main__":
    main()
