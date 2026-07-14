"""Phase 1 — FETCH. Slow, network-bound, run once. Fills the cache and writes records.jsonl.

Target rows (blank accession) are fetched FIRST so yield numbers land early; answered rows
follow so the eval diff has full-text coverage. Idempotent + resumable: every response is
cached, and rows already in records.jsonl are skipped. Safe to kill and re-launch.

Run:  python3 -m accession.fetch_corpus
"""
import csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config, cache, sources, resolve


def _bl(x):
    return (x or "").strip() == ""


def _is_target(r):
    v = (r["Accession Code"] or "").strip().upper()
    return v == "" or v == "ACCESSION_NOT_FOUND"


def load_rows():
    with open(config.CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_indices():
    if not os.path.exists(config.RECORDS_PATH):
        return set()
    idx = set()
    for line in open(config.RECORDS_PATH):
        try:
            idx.add(json.loads(line)["row_index"])
        except Exception:
            pass
    return idx


def fetch_one(i, r):
    res = resolve.resolve(r.get("doi"), r.get("pubmed_id"), r.get("pmc_id"))
    target = _is_target(r)

    ft_status = None
    if res["pmcid"]:
        ft_status, _ = sources.epmc_fulltext(res["pmcid"])

    ann_id, ann_raw = None, 0
    if res["pmcid"]:
        ann_id = "PMC:" + res["pmcid"].replace("PMC", "")
    elif res["pmid"]:
        ann_id = "MED:" + str(res["pmid"])
    if ann_id:
        ann_raw = len(sources.epmc_annotations(ann_id))

    crossref = False
    if target and not res["pmcid"] and res["doi"]:
        crossref = bool(sources.crossref_work(res["doi"]))

    return {
        "row_index": i,
        "url": (r.get("url") or "").strip(),
        "doi_raw": (r.get("doi") or "").strip(),
        "doi": res["doi"], "pmid": res["pmid"], "pmcid": res["pmcid"],
        "source": res["source"], "isOpenAccess": res["isOpenAccess"],
        "inEPMC": res["inEPMC"], "hasSuppl": res["hasSuppl"],
        "resolved_via": res["resolved_via"],
        "ft_status": ft_status, "ann_id": ann_id, "ann_raw_count": ann_raw,
        "crossref": crossref, "target": target,
        "flag_existing": (r.get("flag") or "").strip(),
        "acc_existing": (r.get("Accession Code") or "").strip(),
    }


def main():
    rows = load_rows()
    done = done_indices()
    # target-first ordering, original order preserved within each group
    order = [i for i, r in enumerate(rows) if _is_target(r)] + \
            [i for i, r in enumerate(rows) if not _is_target(r)]
    todo = [i for i in order if i not in done]
    print("total=%d  already_done=%d  todo=%d  (targets first)" % (len(rows), len(done), len(todo)), flush=True)

    t0 = time.time()
    n_pmcid = n_target_done = 0
    # Only the main thread writes records.jsonl (consuming as_completed), so no file lock needed.
    with ThreadPoolExecutor(max_workers=config.FETCH_WORKERS) as ex, \
            open(config.RECORDS_PATH, "a") as out:
        futs = {ex.submit(fetch_one, i, rows[i]): i for i in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"row_index": i, "error": str(e), "target": _is_target(rows[i])}
            out.write(json.dumps(rec) + "\n"); out.flush()
            if rec.get("pmcid"):
                n_pmcid += 1
            if rec.get("target"):
                n_target_done += 1
            if k % 100 == 0 or k == len(todo):
                el = time.time() - t0
                rate = k / el if el else 0
                eta = (len(todo) - k) / rate / 60 if rate else 0
                print("[%5d/%d] %.1f rows/s  pmcid=%d  targets_done=%d  cache=%d  eta=%.0fm"
                      % (k, len(todo), rate, n_pmcid, n_target_done,
                         cache.stats()["entries"], eta), flush=True)
    print("DONE in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
