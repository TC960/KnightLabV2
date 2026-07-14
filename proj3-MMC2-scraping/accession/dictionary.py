"""THE dictionary. Single source of truth for repo patterns, guards, provenance, floor reasons.

Imported by the pipeline AND by evals. NEVER copied — the retired Apps Script had three
divergent copies of this in its eval scripts (one missing BioStudies), which is the exact
failure mode this module exists to prevent.

Ported verbatim from prod.gs, plus GCA/GCF genome-assembly capture (config-gated out of the
accession column, since those are almost always reused reference genomes).
"""
import re
from . import config

# (repo, compiled_regex, standalone_guard)
DICT = [
    ("BioProject",   re.compile(r"PRJ(?:EB|NA|DB|CA)\d{4,}", re.I), True),
    ("BioSample",    re.compile(r"SAM(?:EA|N|D)\d{6,}", re.I), True),
    # \d{5,} not {6,}: legacy SRA study accessions (SRP52003) are 5 digits and real.
    # Verified against the cached corpus that this does not inflate false positives.
    ("SRA/ENA/DDBJ", re.compile(r"[SED]R[APRSX]\d{5,}", re.I), True),
    ("GEO",          re.compile(r"G(?:SE|SM|PL|DS)\d{4,}", re.I), True),
    ("dbGaP",        re.compile(r"phs\d{6}(?:\.v\d+\.p\d+)?", re.I), True),
    ("GSA",          re.compile(r"CR[ARX]\d{4,}", re.I), True),
    ("GSA-Human",    re.compile(r"HRA\d{4,}", re.I), True),
    ("GSA-KAP",      re.compile(r"KAP\d{4,}", re.I), True),
    ("NODE",         re.compile(r"OE[PXZS]\d{4,}", re.I), True),
    ("CNGB",         re.compile(r"CN[PSXR]\d{5,}", re.I), True),
    ("MG-RAST",      re.compile(r"mg[mp]\d+\.\d+", re.I), True),
    ("MetaboLights", re.compile(r"MTBLS\d+", re.I), True),
    ("PRIDE",        re.compile(r"PXD\d{6}", re.I), True),
    ("EGA",          re.compile(r"EGA[SD]\d{6,}", re.I), True),
    ("ArrayExpress", re.compile(r"E-[A-Z]{4}-\d+", re.I), True),
    ("BioStudies",   re.compile(r"S-[A-Z]{4}\d+", re.I), True),
    # genome assemblies — captured (repo=GCA) but filtered from the column by default
    ("GCA",          re.compile(r"GC[AF]_\d{9}\.\d+", re.I), True),
    ("figshare",     re.compile(r"10\.6084/m9\.figshare\.\d+", re.I), False),
    ("Zenodo",       re.compile(r"10\.5281/zenodo\.\d+", re.I), False),
    ("Dryad",        re.compile(r"10\.506\d/dryad\.\w+", re.I), False),
]

LOOKALIKE = re.compile(
    r"(refseq|primer|probe|taqman|assay|catalog|cat\.?\s*no|gene (id|expression)|"
    r"mrna reference|\bNM_|\bNR_|\bXM_|\bNP_)", re.I)

OWN = re.compile("|".join([
    r"(deposit|submitt|uploaded|archived|lodged|stored)",
    r"data (availability|accessibility)",
    r"(data|dataset|datasets|reads|raw reads|sequences|sequencing data|raw data|sequence data|"
    r"genomes?|metagenomes?|reads and metadata|files)\s*"
    r"(that support|supporting|generated|produced|used)?[^.]{0,60}"
    r"(are|were|is|was|has|have)\s*(been\s*)?"
    r"(deposited|submitted|uploaded|made (publicly )?available|available|accessible|archived|released|stored|shared)",
    r"available (in|under|at|from|through|via)[^.]{0,90}"
    r"(accession|bioproject|biosample|sra|ena|geo|ddbj|gsa|cngb|node|ega|arrayexpress|pride|"
    r"metabolights|zenodo|figshare|dryad|archive|repositor|database)",
    r"(under|with|assigned)\s*(the\s*)?(accession|bioproject|project|study|submission)\s*"
    r"(number|numbers|code|codes|id|ids|no)?",
    r"(accession|bioproject|project|study)\s*(number|numbers|code|codes|id|ids|no)?[^.]{0,25}(is|are|:|=)",
    r"can be (found|accessed|retrieved|downloaded)\s*(in|under|at|from|via)",
    r"(generated|produced|obtained|sequenced)\s*(in|for|during|as part of)\s*"
    r"(this|the (present|current))\s*(study|work|paper|project)",
    r"\bwe (deposited|submitted|uploaded|have deposited|have submitted|made (the )?data available)",
    r"(reported|described|presented)\s*(in|by)\s*this\s*(paper|study|article)[^.]{0,60}(deposit|available|submitt)",
]), re.I)

REUSE = re.compile("|".join([
    r"(obtained|downloaded|retrieved|acquired|sourced|collected|taken|derived|extracted)\s*from[^.]{0,80}"
    r"(sra|ena|geo|ddbj|bioproject|ncbi|ebi|europe pmc|gsa|cngb|node|mg-?rast|qiita|"
    r"curatedmetagenomicdata|repositor|database|archive|public|published|previous|prior|earlier)",
    r"re-?analy|re-?used|re-?use\b|secondary analys|meta-?analys",
    r"publicly available (data|dataset|datasets|sequenc|reads|genomes?)",
    r"previously (published|deposited|described|reported|generated|released|collected)",
    r"from (a |the )?(previous|prior|earlier|published|existing|original)\s*"
    r"(study|studies|work|paper|cohort|dataset)",
    r"(data|datasets|reads|sequences|samples)\s*(were|was|are|is)\s*(originally\s*)?"
    r"(published|generated|produced|described)\s*(by|in)\s*[^.]{0,40}(et al|\([12]\d{3}\))",
    r"et al\.?\s*\([12]\d{3}\)[^.]{0,60}(PRJ|SR[APRSX]|GSE|accession)",
    r"(existing|third-?party|external|open|archived) (data|dataset|datasets)",
    r"accessed (from|via|through|at)\s*(the\s*)?(sra|ena|geo|ncbi|ebi|repositor|database)",
]), re.I)


def provenance(plain, at, L):
    """own / reused / unclear, via widening windows then paper-level fallback (prod.gs logic)."""
    near = plain[max(0, at - 200): at + L + 200]
    own, reuse = bool(OWN.search(near)), bool(REUSE.search(near))
    if own and not reuse:
        return "own"
    if reuse and not own:
        return "reused"
    wide = plain[max(0, at - 600): at + L + 400]
    own, reuse = bool(OWN.search(wide)), bool(REUSE.search(wide))
    if own and not reuse:
        return "own"
    if reuse and not own:
        return "reused"
    if own and reuse:
        return "unclear (mixed)"
    powner, preuse = bool(OWN.search(plain)), bool(REUSE.search(plain))
    if powner and not preuse:
        return "own (paper-level)"
    if preuse and not powner:
        return "reused (paper-level)"
    return "unclear"


def floor_reason(plain):
    if re.search(r"available (on|upon)( the)?( reasonable)? request|from the corresponding author|"
                 r"not publicly available", plain, re.I):
        return "available on request"
    if re.search(r"no datasets? (were|was) (generated|analy[sz]ed)|no new (data|sequenc)", plain, re.I):
        return "no data generated"
    if re.search(r"(included|contained|presented) (with)?in (this )?(published )?(article|paper)|"
                 r"supplementary (information|material|file)|supporting information", plain, re.I):
        return "data in article/supplement"
    return "no accession in text"


def extract_codes(plain, with_provenance=True):
    """Return list of {code, repo, prov}. Includes GCA (capture); column-filtering is separate."""
    seen, codes = set(), []
    for repo, rx, guard in DICT:
        for m in rx.finditer(plain):
            code = m.group(0).upper()
            at, L = m.start(), len(m.group(0))
            if guard:
                before = plain[at - 1] if at > 0 else " "
                after = plain[at + L] if at + L < len(plain) else " "
                if re.match(r"[A-Za-z0-9]", before) or re.match(r"[A-Za-z]", after):
                    continue
            if LOOKALIKE.search(plain[max(0, at - 60): at + L + 60]):
                continue
            if code in seen:
                continue
            seen.add(code)
            codes.append({"code": code, "repo": repo,
                          "prov": provenance(plain, at, L) if with_provenance else None})
    return codes


def extract_from_annotations(exacts):
    """Mined 'exact' strings run through the dictionary — RefSNP/UniProt/etc drop out for free.
    No provenance (annotations carry no surrounding text)."""
    joined = "  ".join(" %s " % e for e in exacts)
    codes = extract_codes(joined, with_provenance=False)
    for c in codes:
        c["prov"] = "unclear (mined, no context)"
    return codes


def in_accession_column(repo):
    """Whether a repo's codes belong in the accession column (vs capture-only like GCA)."""
    if repo in config.CAPTURE_ONLY_REPOS:
        return repo != "GCA" or config.INCLUDE_GCA_IN_ACCESSION_COLUMN
    return True
