"""Central config. No behavior lives here — just knobs, so nothing is a silent default."""
import os

# --- paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV_PATH = os.path.join(REPO, "Copy of MMC2 _Rayyan_output - articles.csv.csv")
CACHE_DIR = os.path.join(HERE, "cache")
RECORDS_PATH = os.path.join(HERE, "records.jsonl")

# --- HTTP ---
USER_AGENT = "accession-pipeline/0.1 (mailto:map.960.20@gmail.com)"
THROTTLE_SEC = 0.10          # min gap between request STARTS => ~10 req/s ceiling (EPMC is forgiving)
FETCH_WORKERS = 12           # concurrent in-flight requests; throughput is capped by THROTTLE_SEC
HTTP_TIMEOUT = 40

# --- NCBI E-utilities (EPMC-404 fallback only) ---
# The prod.gs key ea48f4ef...5708 is INVALID (returns HTTP 400 "API key status invalid") and is
# deliberately NOT ported. Run UNAUTHENTICATED at the shared-IP 3/s limit. A preflight asserts a
# 200 before the pass runs, so an auth/throttle failure fails loudly instead of masquerading as
# an empty result (this cost hours in the Apps Script version).
NCBI_KEY = ""                # intentionally blank — do not paste the invalid prod.gs key here
NCBI_THROTTLE_SEC = 0.35     # ~2.8 req/s, safely under the 3/s unauthenticated shared-IP limit
NCBI_WORKERS = 3
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_NONOA_MARK = "does not allow downloading of the full text"

# --- Unpaywall (OA-outside-PMC fallback) ---
UNPAYWALL_EMAIL = "map.960.20@gmail.com"
OA_CACHE_DIR = os.path.join(HERE, "oa_cache")   # binary OA copies (pdf/html), keyed by url
OA_MAX_BYTES = 30 * 1024 * 1024                 # skip absurdly large downloads
OA_WORKERS = 8
CACHEABLE_STATUS = (200, 404)  # 404 is definitive (not in OA set); never cache transient errors

# --- dictionary behavior ---
# GCA/GCF genome-assembly accessions are captured into records.jsonl (repo=GCA) but, because
# they are almost always REUSED reference genomes rather than a study's own deposit, they are
# excluded from the accession column by default. Flip this to include them — no re-fetch needed,
# the codes are already in records.jsonl.
INCLUDE_GCA_IN_ACCESSION_COLUMN = False
CAPTURE_ONLY_REPOS = {"GCA"}  # captured + stored, filtered out of the column unless flagged in
