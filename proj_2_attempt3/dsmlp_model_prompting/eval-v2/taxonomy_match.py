#!/usr/bin/env python3
"""Taxonomy-aware taxon matching for the eval harness.

A predicted taxon matches an expected taxon when EITHER
  (a) their NCBI lineages are nested — one is an ancestor of the other
      (collapses rank variants: Bacteroides / Bacteroidaceae / B. vulgatus), OR
  (b) char-ngram cosine >= 0.5 (the original fuzzy fallback, for names NCBI can't resolve).

Tools (gnparser + taxonkit + NCBI taxdump) are resolved from $TAX_BIN / $TAX_DATA
(default /tmp/bin, /tmp/taxdump). Run `python taxonomy_match.py --setup` to fetch them.
If the tools are missing, match_taxa_lca() transparently falls back to char-only,
so the harness never hard-fails.
"""
import os, re, subprocess, csv, io

BIN = os.environ.get("TAX_BIN", "/tmp/bin")
DATA = os.environ.get("TAX_DATA", "/tmp/taxdump")
GNPARSER = os.path.join(BIN, "gnparser")
TAXONKIT = os.path.join(BIN, "taxonkit")
NAMES = os.path.join(DATA, "names.dmp")

TAXONKIT_URL = "https://github.com/shenwei356/taxonkit/releases/download/v0.20.0/taxonkit_linux_amd64.tar.gz"
GNPARSER_URL = "https://github.com/gnames/gnparser/releases/download/v1.15.0/gnparser-v1.15.0-linux-x86.tar.gz"
TAXDUMP_URL = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"


def tools_available():
    return os.path.exists(GNPARSER) and os.path.exists(TAXONKIT) and os.path.exists(NAMES)


def setup():
    os.makedirs(BIN, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    def sh(c): subprocess.run(c, shell=True, check=True)
    sh(f"curl -sL {TAXONKIT_URL} -o /tmp/_tk.tgz && tar xzf /tmp/_tk.tgz -C {BIN}")
    sh(f"curl -sL {GNPARSER_URL} -o /tmp/_gp.tgz && tar xzf /tmp/_gp.tgz -C {BIN}")
    sh(f"curl -sL {TAXDUMP_URL} -o /tmp/_td.tgz && tar xzf /tmp/_td.tgz -C {DATA} names.dmp nodes.dmp")
    print("taxonomy tools ready:", tools_available())


def _run(cmd, inp):
    return subprocess.run(cmd, input=inp, capture_output=True, text=True).stdout


class TaxResolver:
    """Batch-resolves taxon strings to (taxid, ancestor-taxid-set). Cache keyed by lowercased name."""
    def __init__(self):
        self.cache = {}
        self.ok = tools_available()

    def warm(self, names):
        if not self.ok:
            return
        names = sorted({n for n in names if n and n.lower() not in self.cache})
        if not names:
            return
        # 1) gnparser -> canonical simple (col 5)
        canon = {}
        for row in csv.reader(io.StringIO(_run([GNPARSER, "-f", "csv"], "\n".join(names)))):
            if not row or row[0] == "Id":
                continue
            canon[row[1]] = (row[4] if len(row) > 4 and row[4] else row[1])
        # 2) canonical -> taxid.  NCBI names.dmp is properly-cased, so capitalize the
        #    first letter (harmless if already capitalized) before lookup — makes
        #    resolution robust to lowercased inputs from parse_taxa().
        def cap(s):
            return (s[:1].upper() + s[1:]) if s else s
        canon = {k: cap(v) for k, v in canon.items()}
        n2t = {}
        for line in _run([TAXONKIT, "name2taxid", "--data-dir", DATA],
                         "\n".join(sorted(set(canon.values())))).splitlines():
            p = line.split("\t")
            if len(p) >= 2 and p[1] and p[0] not in n2t:
                n2t[p[0]] = p[1]
        # 3) taxid -> ancestor taxid set
        anc = {}
        tids = sorted({t for t in n2t.values() if t})
        if tids:
            for line in _run([TAXONKIT, "lineage", "-t", "--data-dir", DATA],
                             "\n".join(tids)).splitlines():
                p = line.split("\t")
                if len(p) >= 3 and p[2]:
                    anc[p[0]] = set(p[2].split(";"))
        for n in names:
            tid = n2t.get(canon.get(n, n), "")
            self.cache[n.lower()] = (tid, anc.get(tid, set()))

    def nested(self, a, b):
        ta, aa = self.cache.get(a.lower(), ("", set()))
        tb, ab = self.cache.get(b.lower(), ("", set()))
        return bool(ta and tb and (ta in ab or tb in aa))


def _char_sim(pred, exp):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(pred + exp)
    return cosine_similarity(tf[:len(pred)], tf[len(pred):])


def match_taxa_lca(predicted, expected, resolver=None):
    """Same greedy convention as match_taxa, plus a taxonomy-nested fallback per prediction.
    Returns (tp, fp, fn). If resolver is None/unavailable, this is exactly char-only matching."""
    if not predicted and not expected:
        return 0, 0, 0
    if not predicted:
        return 0, 0, len(expected)
    if not expected:
        return 0, len(predicted), 0
    sim = _char_sim(predicted, expected)
    matched, tp, fp = set(), 0, 0
    for i in range(len(predicted)):
        j = int(sim[i].argmax())
        hit = j if float(sim[i][j]) >= 0.5 else -1
        if hit < 0 and resolver is not None and resolver.ok:
            for k in range(len(expected)):
                if resolver.nested(predicted[i], expected[k]):
                    hit = k; break
        if hit >= 0:
            tp += 1; matched.add(hit)
        else:
            fp += 1
    return tp, fp, len(expected) - len(matched)


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup()
    else:
        print("tools_available:", tools_available(), "| BIN:", BIN, "| DATA:", DATA)
        print("run with --setup to download taxonkit + gnparser + NCBI taxdump")
