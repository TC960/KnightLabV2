#!/usr/bin/env python3
"""MONDO disease ontology: the disease-side analog of `taxonomy.py`.

Why this exists
---------------
The taxon half of every edge is resolved against an external authority (NCBI, via
`taxonomy.py` reading `names.dmp`). The disease half never was. `build_kg.py`
carries a **hand-written** table of 16 regex -> (label, MONDO id) pairs and
deliberately leaves the other 24 disease labels with `mondo=None` rather than
guessing. Consequently the graph has **713 taxon containment links and 0 disease
hierarchy links**, and "should disease subtypes be modelled as containment?" has
been logged for three sessions as a decision needing a PI because there was no
authority to appeal to.

There is one. MONDO is an is-a disease ontology, and
`github.com/monarch-initiative/mondo/releases` is reachable from this environment
even though `ftp.ncbi.nih.gov` is not (probed in eight sessions). So the disease
hierarchy is a *lookup*, not a judgement call, exactly as the taxon one is.

Conventions copied deliberately from `taxonomy.py` and the repo's rules
-----------------------------------------------------------------------
- **Exact matching only.** A label resolves through its MONDO primary name or an
  `EXACT`/`RELATED` synonym, after a punctuation/case normalisation that is
  documented below. Nothing is fuzzy-matched. An unresolved label is REPORTED as
  unresolved, never guessed -- edit distance would happily merge
  `Cognitive impairment` into `Cognitive impairment, mild`.
- **Obsolete terms are skipped**, and `replaced_by` is followed so a retired id
  does not silently become an orphan.
- **A built-in positive control.** `build_kg.py`'s 16 hand-curated ids must come
  back out of the resolver. Four instruments in this repo have turned out weaker
  than the thing they audited; on 2026-09-13 a built-in control caught one for
  the first time. `--validate` is that control here.

Usage
-----
    python3 mondo.py                # self-test + resolve the graph's 40 labels
    python3 mondo.py --validate     # positive control vs build_kg.py's table

The .obo is cached at ~/.mondo/mondo.obo and is NOT committed (53 MB).
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OBO = os.path.expanduser("~/.mondo/mondo.obo")
GRAPH = os.path.join(HERE, "graph.json")
OUT = os.path.join(HERE, "mondo_resolution.json")

DOWNLOAD_HINT = (
    "mondo.obo not found. Fetch it with:\n"
    "  mkdir -p ~/.mondo && curl -sSL -o ~/.mondo/mondo.obo \\\n"
    "    https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo\n"
    "(purl.obolibrary.org and ebi.ac.uk are both blocked here; the GitHub release is not.)"
)

# ---------------------------------------------------------------------------
# Normalisation. Kept deliberately conservative: it folds the differences that
# are certainly not semantic (case, curly apostrophes, possessives, bracketed
# abbreviations, punctuation-as-separator) and nothing else.
#
# It does NOT fold word order, plurals, or stem words. "Cognitive impairment"
# and "Cognitive impairments" are left distinct because in MONDO a one-token
# difference routinely IS a different disease ("Alzheimer disease 2").
# ---------------------------------------------------------------------------
_PARENS = re.compile(r"\([^)]*\)")
_NONWORD = re.compile(r"[^a-z0-9]+")


def _obo_id(rest: str) -> str:
    """First whitespace-delimited token of an OBO reference line.

    Handles all three shapes MONDO actually emits:
        MONDO:0005180
        MONDO:0005180 ! Parkinson disease
        MONDO:0005071 {source="https://orcid.org/0000-0002-..."} ! nervous ...
    """
    return rest.split("{")[0].split("!")[0].strip().split()[0] if rest.strip() else ""


def norm_label(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = _PARENS.sub(" ", s)          # "Alzheimer's disease (AD)" -> "alzheimer's disease"
    s = s.replace("'s ", " ").replace("'s", " ")   # possessive -> attributive
    s = _NONWORD.sub(" ", s)
    return " ".join(s.split())


class Mondo:
    """Parsed MONDO: labels -> ids, and the is-a DAG."""

    def __init__(self, path=OBO):
        if not os.path.exists(path):
            raise SystemExit(DOWNLOAD_HINT)
        self.name = {}                      # id -> primary name
        self.parents = defaultdict(set)     # id -> set(parent id)   (is_a only)
        self.children = defaultdict(set)
        self.index = defaultdict(set)       # normalised string -> set(id)
        self.obsolete = set()
        self.replaced_by = {}
        self.version = None
        self._parse(path)

    def _parse(self, path):
        term = None
        in_term = False
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("data-version:"):
                    self.version = line.split(":", 1)[1].strip()
                if line.startswith("["):
                    # Only [Term] stanzas; [Typedef] etc. carry no disease data.
                    if in_term and term:
                        self._commit(term)
                    in_term = line.strip() == "[Term]"
                    term = {"id": None, "name": None, "syn": [], "is_a": [],
                            "obsolete": False, "replaced_by": None} if in_term else None
                    continue
                if not in_term or not term:
                    continue
                if line.startswith("id: "):
                    term["id"] = _obo_id(line[4:])
                elif line.startswith("name: "):
                    term["name"] = line[6:].strip()
                elif line.startswith("is_obsolete: "):
                    term["obsolete"] = line.split(":", 1)[1].strip() == "true"
                elif line.startswith("replaced_by: "):
                    term["replaced_by"] = _obo_id(line.split(":", 1)[1])
                elif line.startswith("is_a: "):
                    # "is_a: MONDO:0005180 ! Parkinson disease", but also
                    # 'is_a: MONDO:0005071 {source="https://orcid.org/..."} ! ...'
                    # -- the trailing OBO qualifier block must be stripped or the
                    # id carries it and every ancestor lookup silently misses.
                    # (This was caught by the parser self-test, not by inspection.)
                    term["is_a"].append(_obo_id(line[6:]))
                elif line.startswith("synonym: "):
                    # synonym: "PD" EXACT [...]   /  "..." RELATED ABBREVIATION [...]
                    m = re.match(r'synonym:\s+"((?:[^"\\]|\\.)*)"\s+(\w+)', line)
                    if m:
                        term["syn"].append((m.group(1).replace('\\"', '"'), m.group(2)))
        if in_term and term:
            self._commit(term)

    def _commit(self, t):
        tid = t["id"]
        if not tid or not tid.startswith("MONDO:"):
            return
        if t["obsolete"]:
            self.obsolete.add(tid)
            if t["replaced_by"]:
                self.replaced_by[tid] = t["replaced_by"]
            return
        self.name[tid] = t["name"]
        for p in t["is_a"]:
            if p.startswith("MONDO:"):
                self.parents[tid].add(p)
                self.children[p].add(tid)
        keys = {norm_label(t["name"])}
        for s, scope in t["syn"]:
            if scope in ("EXACT", "RELATED"):
                keys.add(norm_label(s))
        for k in keys:
            if k:
                self.index[k].add(tid)

    # -- resolution ---------------------------------------------------------
    def resolve(self, label):
        """-> (mondo_id | None, how). Exact normalised match only."""
        k = norm_label(label)
        hits = self.index.get(k)
        if not hits:
            return None, "unresolved"
        live = {self.replaced_by.get(h, h) for h in hits}
        live = {h for h in live if h in self.name}
        if len(live) == 1:
            return next(iter(live)), "exact"
        if not live:
            return None, "obsolete_no_replacement"
        # Ambiguous: prefer the term whose PRIMARY name matches, which
        # disambiguates "Dementia" (a grouping) from a synonym of a subtype.
        primary = {h for h in live if norm_label(self.name[h]) == k}
        if len(primary) == 1:
            return next(iter(primary)), "exact_primary"
        return None, "ambiguous:" + ",".join(sorted(live))

    # -- hierarchy ----------------------------------------------------------
    def ancestors(self, tid):
        """Transitive is-a closure, excluding tid itself."""
        seen, stack = set(), list(self.parents.get(tid, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self.parents.get(x, ()))
        return seen

    def relation(self, a, b):
        """Ontological relation of a to b: 'same' | 'descendant' | 'ancestor'
        | 'sibling' (shares a parent) | 'cousin:<n>' | 'unrelated'."""
        if a == b:
            return "same"
        if b in self.ancestors(a):
            return "descendant"       # a is under b
        if a in self.ancestors(b):
            return "ancestor"
        pa, pb = self.parents.get(a, set()), self.parents.get(b, set())
        if pa & pb:
            return "sibling"
        aa, ab = self.ancestors(a), self.ancestors(b)
        if aa & ab:
            return "cousin"
        return "unrelated"

    def distance(self, a, b):
        """Undirected hops in the is-a DAG (BFS), or None if disconnected."""
        if a == b:
            return 0
        adj = lambda x: self.parents.get(x, set()) | self.children.get(x, set())
        seen, frontier, d = {a}, {a}, 0
        while frontier and d < 12:
            d += 1
            nxt = set()
            for x in frontier:
                for y in adj(x):
                    if y == b:
                        return d
                    if y not in seen:
                        seen.add(y)
                        nxt.add(y)
            frontier = nxt
        return None


# ---------------------------------------------------------------------------
# The positive control: build_kg.py's hand-curated table must come back out.
# ---------------------------------------------------------------------------
HAND_CURATED = {
    "Parkinson's disease": "MONDO:0005180",
    "Alzheimer's disease": "MONDO:0004975",
    "Multiple sclerosis": "MONDO:0005301",
    "Amyotrophic lateral sclerosis": "MONDO:0004976",
    "Mild cognitive impairment": "MONDO:0005453",
    "Stroke": "MONDO:0005098",
    "Huntington's disease": "MONDO:0007739",
    "Dementia": "MONDO:0001627",
    "Spinal muscular atrophy": "MONDO:0001516",
    "Epilepsy": "MONDO:0005027",
    "Autism spectrum disorder": "MONDO:0005260",
    "Depressive disorder": "MONDO:0002050",
    "Schizophrenia": "MONDO:0005090",
    "Neuromyelitis optica": "MONDO:0019100",
    "Myasthenia gravis": "MONDO:0009688",
    "Migraine": "MONDO:0005277",
}


def validate(m):
    """Resolve the 16 hand-curated labels and diff against the curated ids."""
    ok = bad = 0
    rows = []
    for label, expect in HAND_CURATED.items():
        got, how = m.resolve(label)
        # An id that MONDO has since merged is not a mismatch: follow it.
        expect_live = m.replaced_by.get(expect, expect)
        agree = got == expect_live
        # ...nor is resolving to a term the curated id is an ancestor of, or
        # vice versa -- but that is a DIFFERENT granularity and must be flagged.
        note = ""
        if not agree and got:
            rel = m.relation(got, expect_live)
            note = f"relation(got, curated)={rel}"
        rows.append({"label": label, "curated": expect, "curated_live": expect_live,
                     "resolved": got, "how": how, "agree": agree,
                     "resolved_name": m.name.get(got), "note": note})
        ok += agree
        bad += (not agree)
    return {"n": len(HAND_CURATED), "agree": ok, "disagree": bad, "rows": rows}


def main():
    m = Mondo()
    print(f"MONDO {m.version}: {len(m.name)} live terms, {len(m.obsolete)} obsolete, "
          f"{sum(len(v) for v in m.parents.values())} is_a edges, "
          f"{len(m.index)} normalised label keys")

    # ---- self-test: three facts that must hold, or the parser is wrong ----
    checks = [
        ("Parkinson's disease resolves", m.resolve("Parkinson's disease")[0] == "MONDO:0005180"),
        ("PD is under nervous system disease",
         "MONDO:0005071" in m.ancestors("MONDO:0005180")),
        ("MCI is not under Alzheimer's",
         "MONDO:0004975" not in m.ancestors(m.resolve("Mild cognitive impairment")[0] or "x")),
    ]
    for what, passed in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {what}")
    if not all(p for _, p in checks):
        sys.exit("parser self-test failed -- do not trust downstream results")

    v = validate(m)
    print(f"\npositive control vs build_kg.py: {v['agree']}/{v['n']} agree")
    for r in v["rows"]:
        if not r["agree"]:
            print(f"  MISMATCH {r['label']!r}: curated {r['curated']} "
                  f"-> resolved {r['resolved']} ({r['resolved_name']}) "
                  f"[{r['how']}] {r['note']}")

    if "--validate" in sys.argv:
        return

    # ---- resolve the graph's own disease labels ----
    g = json.load(open(GRAPH))
    labels = [n["label"] for n in g["nodes"] if str(n.get("id", "")).startswith("d:")]
    papers_per = defaultdict(int)
    for e in g["edges"]:
        papers_per[e["disease"]] = max(papers_per[e["disease"]], 0)
    dpapers = defaultdict(set)
    for e in g["edges"]:
        dpapers[e["disease"]].update(e["papers"])

    res = {}
    for lb in sorted(labels):
        mid, how = m.resolve(lb)
        res[lb] = {"mondo": mid, "how": how, "mondo_name": m.name.get(mid),
                   "n_papers": len(dpapers.get(lb, ()))}
    nres = sum(1 for r in res.values() if r["mondo"])
    print(f"\ngraph disease labels: {nres}/{len(res)} resolved to MONDO")
    for lb, r in sorted(res.items(), key=lambda kv: -kv[1]["n_papers"]):
        flag = " " if r["mondo"] else "!"
        print(f" {flag} {lb[:46]:48s} {r['n_papers']:3d}p  "
              f"{r['mondo'] or r['how']:16s} {r['mondo_name'] or ''}")

    json.dump({"mondo_version": m.version, "control": v, "labels": res},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
