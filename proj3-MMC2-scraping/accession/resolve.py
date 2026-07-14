"""Identifier resolution: normalize dirty DOIs, then backfill pmid/pmcid via EPMC.

The probe found the doi column is dirty ('EA MAY 2021', '...externalicon', doi.org URLs).
The probe also found EPMC resolves DOI->PMID for 86% and DOI->PMCID for 68% of doi-only rows,
so this step is the main lever for the doi-only bucket.
"""
import re
from . import sources

# A DOI is 10.<registrant>/<suffix>. Grab the first valid-looking DOI in the string and stop
# at whitespace; then trim trailing publisher cruft that isn't part of the DOI.
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+")
_TRAILING_CRUFT = re.compile(r"(externalicon|\.full|\.pdf|[.,;)\]]+)$", re.I)


def normalize_doi(raw):
    """Return a clean DOI or None. Strips doi.org prefixes and trailing cruft; rejects non-DOIs."""
    if not raw:
        return None
    s = str(raw).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    m = _DOI_RE.search(s)
    if not m:
        return None  # e.g. 'EA MAY 2021' -> not a DOI
    doi = m.group(0)
    prev = None
    while doi != prev:  # peel repeated trailing cruft
        prev = doi
        doi = _TRAILING_CRUFT.sub("", doi)
    return doi or None


def _digits(x):
    return re.sub(r"[^0-9]", "", str(x or ""))


def resolve(doi_raw, pmid_in, pmcid_in):
    """Fill in missing identifiers. Returns dict with normalized ids + EPMC record flags.

    Cheapest path: if we already have a PMCID, no lookup needed. Otherwise use the identifier
    we have (PMID preferred, else DOI) to fetch the EPMC core record.
    """
    doi = normalize_doi(doi_raw)
    pmid = _digits(pmid_in) or None
    pmcid = None
    if pmcid_in and _digits(pmcid_in):
        pmcid = "PMC" + _digits(pmcid_in)

    out = {"doi": doi, "pmid": pmid, "pmcid": pmcid,
           "source": None, "isOpenAccess": None, "inEPMC": None,
           "hasSuppl": None, "title": None, "resolved_via": None}

    if pmcid:
        out["resolved_via"] = "already_had_pmcid"
        return out

    rec = {}
    if pmid:
        rec = sources.epmc_record_by_pmid(pmid); out["resolved_via"] = "pmid"
    elif doi:
        rec = sources.epmc_record_by_doi(doi); out["resolved_via"] = "doi"
    else:
        out["resolved_via"] = "no_identifier"
        return out

    if rec:
        out["pmid"] = (rec.get("pmid") or out["pmid"] or None)
        rp = (rec.get("pmcid") or "").strip()
        out["pmcid"] = ("PMC" + _digits(rp)) if rp else None
        out["source"] = rec.get("source")
        out["isOpenAccess"] = rec.get("isOpenAccess")
        out["inEPMC"] = rec.get("inEPMC")
        out["hasSuppl"] = rec.get("hasSuppl")
        out["title"] = rec.get("title")
    return out
