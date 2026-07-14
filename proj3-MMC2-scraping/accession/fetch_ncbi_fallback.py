"""NCBI efetch fallback — fills the cache for rows where EPMC full text 404'd.

Preflight-gated: refuses to run unless an unauthenticated NCBI call returns 200, so an auth or
throttle failure fails loudly instead of looking like 'no codes found'. Idempotent (cached) and
resumable. Writes ncbi.jsonl (row_index -> ncbi_status, non_oa) for tracking; extract.py reads the
cached XML directly.

Run:  python3 -m accession.fetch_ncbi_fallback
"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config, cache, sources

NCBI_RECORDS = os.path.join(config.HERE, "ncbi.jsonl")


def candidates():
    recs = [json.loads(l) for l in open(config.RECORDS_PATH)]
    return [r for r in recs if r.get("pmcid") and r.get("ft_status") == 404]


def done_indices():
    if not os.path.exists(NCBI_RECORDS):
        return set()
    idx = set()
    for line in open(NCBI_RECORDS):
        try:
            idx.add(json.loads(line)["row_index"])
        except Exception:
            pass
    return idx


def fetch_one(r):
    status, xml = sources.ncbi_efetch(r["pmcid"])
    return {"row_index": r["row_index"], "pmcid": r["pmcid"],
            "ncbi_status": status,
            "ncbi_non_oa": sources.is_non_oa(xml) if status == 200 else None,
            "ncbi_bytes": len(xml)}


def main():
    cands = candidates()
    done = done_indices()
    todo = [r for r in cands if r["row_index"] not in done]
    print("EPMC-404 rows: %d  already_done: %d  todo: %d" % (len(cands), len(done), len(todo)), flush=True)
    if not todo:
        print("nothing to do"); return

    print("NCBI preflight (unauthenticated must return 200)...", flush=True)
    sources.ncbi_preflight()
    print("  preflight OK (200)\n", flush=True)

    t0 = time.time()
    n200 = nnonoa = 0
    with ThreadPoolExecutor(max_workers=config.NCBI_WORKERS) as ex, open(NCBI_RECORDS, "a") as out:
        futs = {ex.submit(fetch_one, r): r["row_index"] for r in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"row_index": futs[fut], "error": str(e)}
            out.write(json.dumps(rec) + "\n"); out.flush()
            if rec.get("ncbi_status") == 200:
                n200 += 1
            if rec.get("ncbi_non_oa"):
                nnonoa += 1
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                print("[%4d/%d] %.1f/s  ncbi_200=%d  non_oa=%d"
                      % (k, len(todo), k / el if el else 0, n200, nnonoa), flush=True)
    print("DONE in %.1f min  (200=%d, of which non-OA front-matter=%d)"
          % ((time.time() - t0) / 60, n200, nnonoa), flush=True)


if __name__ == "__main__":
    main()
