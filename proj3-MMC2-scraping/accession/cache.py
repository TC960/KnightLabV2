"""Content-addressed disk cache = the fetch/extract firewall.

Every HTTP response is stored raw, keyed by sha1(url). Extraction reads from here, never
the network, so re-running with a tweaked regex costs seconds instead of re-fetching 13k papers.
Only definitive statuses (200, 404) are cached; transient errors are left uncached so a rerun retries.
"""
import os, json, time, hashlib, threading, urllib.request, urllib.error
from . import config

# Per-host thread-safe token buckets: each serializes only the SCHEDULING of request starts
# while many requests stay in flight. Throughput ceiling per bucket = 1/gap regardless of workers.
# EPMC and NCBI get separate buckets because their rate limits differ (10/s vs 3/s shared-IP).
_buckets = {"default": [0.0], "ncbi": [0.0]}
_locks = {"default": threading.Lock(), "ncbi": threading.Lock()}


def _throttle(bucket, gap):
    lk, nx = _locks[bucket], _buckets[bucket]
    with lk:
        now = time.time()
        start = max(now, nx[0])
        nx[0] = start + gap
    wait = start - now
    if wait > 0:
        time.sleep(wait)


def _path(url):
    return os.path.join(config.CACHE_DIR, hashlib.sha1(url.encode()).hexdigest() + ".json")


def cached_status(url):
    """Return cached status without fetching, or None if not cached."""
    fp = _path(url)
    if os.path.exists(fp):
        try:
            return json.load(open(fp))["status"]
        except Exception:
            return None
    return None


def get(url, bucket="default", gap=None):
    """Cached GET. Returns (status:int, body:str). Body is '' on non-200.
    bucket/gap select which rate-limit token bucket to schedule against."""
    fp = _path(url)
    if os.path.exists(fp):
        try:
            d = json.load(open(fp))
            return d["status"], d["body"]
        except Exception:
            os.remove(fp)  # corrupt entry; refetch
    _throttle(bucket, gap if gap is not None else config.THROTTLE_SEC)
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=config.HTTP_TIMEOUT) as r:
            status, body = r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, body = e.code, ""
    except Exception as e:
        return -1, "ERR:" + str(e)  # transient — do NOT cache
    if status in config.CACHEABLE_STATUS:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        json.dump({"url": url, "status": status, "body": body,
                   "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(fp, "w"))
    return status, body


def stats():
    if not os.path.isdir(config.CACHE_DIR):
        return {"entries": 0}
    return {"entries": len([f for f in os.listdir(config.CACHE_DIR) if f.endswith(".json")])}
