"""403 recovery, Tier 1 — try ALTERNATE OA locations we never attempted.

Each 403 row 403'd on ONE url (usually the publisher landing page). Unpaywall often lists
several oa_locations. This enumerates them, picks every copy on a DIFFERENT host than the one
that blocked us (preferring repository/green copies and direct PDFs), fetches, and extracts.
Free, no browser. Whatever still fails here is the real browser-tier candidate.

Run:  python3 -m accession.pipe403_tier1
"""
import json, os, re, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config, sources, dictionary
from .unpaywall_pass import oa_fetch, text_from_oa

OUT = os.path.join(config.HERE, "pipe403_tier1.jsonl")


def blocked_host(url):
    m = re.search(r"https?://([^/]+)", url or "")
    return m.group(1).replace("www.", "") if m else ""


def alt_urls(data, avoid_host):
    """All OA urls on a host != the blocked one. Repository/green + url_for_pdf first."""
    if not data or not data.get("is_oa"):
        return []
    locs = (data.get("oa_locations") or [])[:]
    if data.get("best_oa_location"):
        locs.insert(0, data["best_oa_location"])
    # repository copies first, then publisher; pdf url before landing url
    locs.sort(key=lambda l: 0 if l.get("host_type") == "repository" else 1)
    out = []
    for l in locs:
        for key in ("url_for_pdf", "url"):
            u = l.get(key)
            if u and blocked_host(u) != avoid_host and u not in out:
                out.append(u)
    return out


def work(item):
    row, doi, blocked_url = item
    avoid = blocked_host(blocked_url)
    data = sources.unpaywall(doi)
    alts = alt_urls(data, avoid)
    if not alts:
        return {"row": row, "doi": doi, "status": "no-alt", "tried": [], "codes": []}
    for u in alts[:3]:                      # cap attempts per row
        st, blob, ctype = oa_fetch(u)
        if st != 200 or not blob:
            continue
        txt = text_from_oa(blob, ctype)
        if not txt:
            continue
        seen, codes = set(), []
        for c in dictionary.extract_codes(txt, with_provenance=True):
            if dictionary.in_accession_column(c["repo"]) and c["code"] not in seen:
                seen.add(c["code"]); codes.append(c)
        if codes:
            return {"row": row, "doi": doi, "status": "coded", "url": u,
                    "host": blocked_host(u), "codes": codes}
        return {"row": row, "doi": doi, "status": "read-no-code", "url": u,
                "host": blocked_host(u), "codes": []}
    return {"row": row, "doi": doi, "status": "alt-failed", "tried": alts[:3], "codes": []}


def main():
    mr = json.load(open(os.path.join(config.HERE, "manual_review.json")))
    b403 = [(x["row"], x["doi"], x.get("url")) for x in mr if x["reason"] == "publisher 403" and x.get("doi")]
    print("403 rows with a DOI: %d" % len(b403), flush=True)

    res = []
    tally = collections.Counter()
    with ThreadPoolExecutor(max_workers=config.OA_WORKERS) as ex:
        futs = [ex.submit(work, it) for it in b403]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            res.append(r)
            tally[r["status"]] += 1
            if k % 200 == 0:
                print("  [%d/%d] %s" % (k, len(b403), dict(tally)), flush=True)

    res.sort(key=lambda r: r["row"])
    json.dump(res, open(OUT, "w"), indent=1)
    coded = [r for r in res if r["status"] == "coded"]
    print("\n==============  403 TIER-1 (alternate OA location)  ==============")
    print("  outcomes:", dict(tally))
    print("  NEW CODES recovered (no browser needed): %d rows" % len(coded))
    print("  wrote %s" % OUT)
    for r in coded[:25]:
        print("   r%-6d %-22s %s" % (r["row"], r["host"], [c["code"] for c in r["codes"]][:6]))


if __name__ == "__main__":
    main()
