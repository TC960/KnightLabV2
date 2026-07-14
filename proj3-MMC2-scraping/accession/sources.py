"""Thin API clients. Every call goes through cache.get — no direct network here."""
import json, urllib.parse, re
from . import cache, config

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_ANN = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
CROSSREF = "https://api.crossref.org/works"

_CORE = "&format=json&resultType=core&pageSize=1"


def _epmc_search(query):
    url = "%s/search?query=%s%s" % (EPMC, urllib.parse.quote(query), _CORE)
    s, b = cache.get(url)
    if s != 200 or not b:
        return {}
    try:
        r = json.loads(b)["resultList"]["result"]
    except Exception:
        return {}
    return r[0] if r else {}


def epmc_record_by_doi(doi):
    return _epmc_search('DOI:"%s"' % doi)


def epmc_record_by_pmid(pmid):
    return _epmc_search("EXT_ID:%s AND SRC:MED" % pmid)


def epmc_fulltext(pmcid):
    """Returns (status, xml). 200+body = full text or (for non-OA) front-matter-only."""
    digits = re.sub(r"[^0-9]", "", str(pmcid))
    if not digits:
        return 0, ""
    return cache.get("%s/PMC%s/fullTextXML" % (EPMC, digits))


def epmc_annotations(article_id):
    """article_id like 'PMC:12775462' or 'MED:41371314'. Returns list of mined 'exact' strings."""
    url = "%s?articleIds=%s&type=%s&format=JSON" % (
        EPMC_ANN, urllib.parse.quote(article_id), urllib.parse.quote("Accession Numbers"))
    s, b = cache.get(url)
    if s != 200 or not b:
        return []
    try:
        data = json.loads(b)
    except Exception:
        return []
    out = []
    for art in data:
        for a in art.get("annotations", []):
            if a.get("type") == "Accession Numbers" and a.get("exact"):
                out.append(a["exact"])
    return out


def ncbi_efetch(pmcid, bypass_cache=False):
    """EPMC-404 fallback. Unauthenticated efetch of PMC full text. Returns (status, xml).
    Non-OA papers return HTTP 200 with front-matter only (see is_non_oa)."""
    digits = re.sub(r"[^0-9]", "", str(pmcid))
    if not digits:
        return 0, ""
    url = "%s?db=pmc&id=%s&rettype=full&retmode=xml" % (config.NCBI_EFETCH, digits)
    if config.NCBI_KEY:
        url += "&api_key=" + config.NCBI_KEY
    return cache.get(url, bucket="ncbi", gap=config.NCBI_THROTTLE_SEC)


def is_non_oa(xml):
    """True if this is the publisher-restricted front-matter-only response (still mine it)."""
    if not xml:
        return False
    return config.NCBI_NONOA_MARK in xml or ("<body" not in xml and "<front" in xml)


def ncbi_preflight():
    """Assert NCBI returns 200 UNAUTHENTICATED before the fallback pass. Fail LOUDLY otherwise —
    a 400 (bad key) or 429 (throttle) must not masquerade as 'no codes found'."""
    # PMC7692476 — a known open-access record used as the liveness probe.
    digits = "7692476"
    url = "%s?db=pmc&id=%s&rettype=full&retmode=xml" % (config.NCBI_EFETCH, digits)
    if config.NCBI_KEY:
        url += "&api_key=" + config.NCBI_KEY
    status, body = cache.get(url, bucket="ncbi", gap=config.NCBI_THROTTLE_SEC)
    if status != 200:
        raise RuntimeError(
            "NCBI preflight FAILED: HTTP %s (not 200).\n"
            "  body[:200]=%r\n"
            "  -> 400 = API key rejected (NCBI_KEY must stay BLANK; do not port the prod.gs key).\n"
            "  -> 429 = IP throttled; lower NCBI rate or wait.\n"
            "Refusing to run the fallback so this does not look like an empty result."
            % (status, body[:200]))
    return True


def crossref_work(doi):
    """Fallback abstract/metadata source. relation[] carries linked datasets when present."""
    s, b = cache.get("%s/%s" % (CROSSREF, urllib.parse.quote(doi)))
    if s != 200 or not b:
        return {}
    try:
        return json.loads(b).get("message", {})
    except Exception:
        return {}


UNPAYWALL = "https://api.unpaywall.org/v2/"


def unpaywall(doi):
    """OA-locator: finds legal open-access full text OUTSIDE PMC (publisher gold/hybrid,
    institutional repositories, preprint servers). Returns the parsed JSON (is_oa, oa_locations)."""
    url = "%s%s?email=%s" % (UNPAYWALL, urllib.parse.quote(doi), config.UNPAYWALL_EMAIL)
    s, b = cache.get(url)
    if s != 200 or not b:
        return {}
    try:
        return json.loads(b)
    except Exception:
        return {}
