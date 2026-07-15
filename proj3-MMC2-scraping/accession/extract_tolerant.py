"""Superset extractor: production dictionary PLUS the two false-negative classes the audits found.

Used by the browser re-audit run so a read can never repeat the misses the plain production
dictionary made (whitespace-split accessions, and real codes the +/-60 lookalike filter killed).
Does NOT touch dictionary.py (shared with the other pipeline) — it wraps it.
"""
import re
from . import dictionary

# whitespace-split variants (prefix, 1-3 spaces/newlines, digits)
_SPLIT = [
    ("BioProject",   re.compile(r"(PRJ(?:EB|NA|DB|CA))\s{1,3}(\d{4,})", re.I)),
    ("BioSample",    re.compile(r"(SAM(?:EA|N|D))\s{1,3}(\d{6,})", re.I)),
    ("SRA/ENA/DDBJ", re.compile(r"([SED]R[APRSX])\s{1,3}(\d{5,})", re.I)),
    ("GEO",          re.compile(r"(G(?:SE|SM|PL|DS))\s{1,3}(\d{4,})", re.I)),
    ("dbGaP",        re.compile(r"(phs)\s{1,3}(\d{6})", re.I)),
    ("PRIDE",        re.compile(r"(PXD)\s{1,3}(\d{6})", re.I)),
    ("EGA",          re.compile(r"(EGA[SD])\s{1,3}(\d{6,})", re.I)),
    ("MetaboLights", re.compile(r"(MTBLS)\s{1,3}(\d+)", re.I)),
]
_DEPOSIT = re.compile(
    r"deposit|archiv|accession|bioproject|biosample|sequence read archive|\bsra\b|\bena\b|ddbj|"
    r"european nucleotide|genome[- ]?phenome|\bega\b|repositor|available (in|at|under|through)|"
    r"data availability|submitted to|under the (project|accession)", re.I)


def extract_tolerant(text):
    """Return list of {code, repo, prov} — production hits UNION recovered false-negative classes."""
    out = {}
    # 1) production dictionary (guarded, column-filtered)
    for c in dictionary.extract_codes(text, with_provenance=True):
        if dictionary.in_accession_column(c["repo"]):
            out[c["code"]] = c
    # 2) whitespace-split accessions (require deposit language nearby -> high precision)
    for repo, rx in _SPLIT:
        for m in rx.finditer(text):
            norm = (m.group(1) + m.group(2)).upper()
            if norm in out:
                continue
            at, L = m.start(), len(m.group(0))
            if _DEPOSIT.search(text[max(0, at - 120): at + L + 120]):
                out[norm] = {"code": norm, "repo": repo, "prov": "unclear (ws-split)"}
    # 3) lookalike-collision: a guarded match killed ONLY by the +/-60 lookalike filter, but with
    #    deposit language nearby -> the filter was wrong, keep it.
    for repo, rx, guard in dictionary.DICT:
        if repo in ("figshare", "Zenodo", "Dryad", "GCA"):
            continue
        for m in rx.finditer(text):
            code = m.group(0).upper()
            if code in out:
                continue
            at, L = m.start(), len(m.group(0))
            before = text[at - 1] if at > 0 else " "
            after = text[at + L] if at + L < len(text) else " "
            if guard and (re.match(r"[A-Za-z0-9]", before) or re.match(r"[A-Za-z]", after)):
                continue                                   # genuinely glued -> skip
            if not dictionary.LOOKALIKE.search(text[max(0, at - 60): at + L + 60]):
                continue                                   # production already keeps it -> not our case
            if _DEPOSIT.search(text[max(0, at - 120): at + L + 120]):
                out[code] = {"code": code, "repo": repo, "prov": "unclear (lookalike-fix)"}
    # 4) SRA submission IDs (SUB\d{6,}) — a deposit signal the base dictionary never had
    #    (surfaced by the subagent reject-audit). Require deposit language nearby.
    for m in re.finditer(r"\bSUB\d{6,}\b", text):
        code = m.group(0).upper()
        if code in out:
            continue
        at, L = m.start(), len(m.group(0))
        if _DEPOSIT.search(text[max(0, at - 120): at + L + 80]):
            out[code] = {"code": code, "repo": "SRA-submission", "prov": "own (SUB submission id)"}
    return list(out.values())
