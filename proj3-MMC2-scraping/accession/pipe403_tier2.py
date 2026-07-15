"""403 recovery, Tier 2 — headless Chromium (Playwright) to clear publisher bot-walls.

Plain urllib gets 403 from MDPI/Wiley/OUP (Cloudflare + JS challenge). A real browser engine
renders the challenge and returns the article HTML. This drives headless Chromium over the
403 rows, extracts visible text, and runs the dictionary.

Usage:
  python3 -m accession.pipe403_tier2 --host mdpi.com --limit 20     # prototype a sample
  python3 -m accession.pipe403_tier2 --all                          # full remainder
"""
import argparse, json, os, re, sys, time, collections
from . import config, dictionary

OUT_JSONL = os.path.join(config.HERE, "pipe403_tier2.jsonl")   # line-delimited, appended per row (resumable)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Mask Playwright's automation fingerprint (Akamai/Cloudflare check these).
STEALTH = ("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
           "window.chrome={runtime:{}};"
           "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
           "Object.defineProperty(navigator,'languages',{get:()=>['en-US','en']});")

BLOCK_MARKERS = ("access denied", "just a moment", "verify you are human",
                 "cloudflare", "enable javascript", "403 forbidden", "attention required")


def to_readable(url):
    """Rewrite a blocked download URL to the free HTML article page where we know the mapping."""
    if "mdpi.com" in url:                       # /pdf?ver -> /htm  (PDF is Akamai-walled, HTML isn't)
        return re.sub(r"/pdf(\?.*)?$", "/htm", url)
    return url


def _done_rows():
    """Rows already attempted in a prior (possibly interrupted) run — for resume."""
    if not os.path.exists(OUT_JSONL):
        return set()
    done = set()
    for line in open(OUT_JSONL):
        try:
            done.add(json.loads(line)["row"])
        except Exception:
            pass
    return done


def targets(host=None, limit=None, remaining=False, oa=False):
    import csv
    if oa:
        # OA_AVAILABLE rows still needs-review — read their free Unpaywall copy in a real browser
        rows = list(csv.DictReader(open(os.path.join(config.REPO, "articles.out.csv"),
                                        newline="", encoding="utf-8")))
        up = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "unpaywall.jsonl"))}
        done = _done_rows()
        out = []
        for i, r in enumerate(rows):
            if (r.get("flag") or "").strip() == "OA_AVAILABLE" and r.get("certainty") == "needs-review" \
                    and up.get(i, {}).get("oa_url") and i not in done:
                out.append((i, (r.get("doi") or "").strip(), up[i]["oa_url"]))
        if host:
            out = [t for t in out if host in (t[2] or "")]
        return out[:limit] if limit else out
    mr = json.load(open(os.path.join(config.HERE, "manual_review.json")))
    url_by_row = {x["row"]: (x["doi"], x.get("url")) for x in mr if x["reason"] == "publisher 403"}
    if remaining:
        # only the 403 rows tier-1 could NOT resolve (no alternate OA copy / alt fetch failed)
        t1 = json.load(open(os.path.join(config.HERE, "pipe403_tier1.jsonl")))
        rowset = [r["row"] for r in t1 if r["status"] in ("no-alt", "alt-failed")]
    else:
        rowset = list(url_by_row)
    done = _done_rows()
    rows = [(row, url_by_row[row][0], url_by_row[row][1])
            for row in rowset if row in url_by_row and url_by_row[row][1] and row not in done]
    if host:
        rows = [r for r in rows if host in (r[2] or "")]
    return rows[:limit] if limit else rows


def extract(text):
    seen, codes = set(), []
    for c in dictionary.extract_codes(text, with_provenance=True):
        if dictionary.in_accession_column(c["repo"]) and c["code"] not in seen:
            seen.add(c["code"]); codes.append(c)
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--remaining", action="store_true", help="only the 648 tier-1 could not resolve")
    ap.add_argument("--oa", action="store_true", help="OA_AVAILABLE needs-review rows via their free copy")
    ap.add_argument("--headful", action="store_true")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    tg = targets(a.host, a.limit, remaining=a.remaining, oa=a.oa)
    print("tier-2 targets: %d (host=%s, remaining=%s, already-done skipped)"
          % (len(tg), a.host or "ALL", a.remaining), flush=True)

    sink = open(OUT_JSONL, "a")   # append — never clobber prior progress
    res, tally = [], collections.Counter()
    with sync_playwright() as p:
        # Akamai/Cloudflare pass a HEADFUL, fingerprint-masked browser; headless is blocked.
        browser = p.chromium.launch(headless=not a.headful,
                                    args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA, viewport={"width": 1280, "height": 1800},
                                  locale="en-US")
        ctx.add_init_script(STEALTH)
        page = ctx.new_page()
        for k, (row, doi, url) in enumerate(tg, 1):
            status, codes, nchars = "err", [], 0
            nav = to_readable(url)
            try:
                page.goto(nav, wait_until="domcontentloaded", timeout=45000)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                body = page.inner_text("body")
                nchars = len(body)
                low = body[:600].lower()
                if nchars < 1500 and any(m in low for m in BLOCK_MARKERS):
                    status = "blocked"
                else:
                    codes = extract(body)
                    status = "coded" if codes else "read-no-code"
            except Exception as e:
                status = "err:%s" % type(e).__name__
            tally[status] += 1
            rowres = {"row": row, "doi": doi, "url": url, "nav": nav, "status": status,
                      "nchars": nchars, "codes": codes}
            res.append(rowres)
            sink.write(json.dumps(rowres) + "\n"); sink.flush()   # persist immediately
            print("  [%d/%d] r%-6d %-14s chars=%-7d %s"
                  % (k, len(tg), row, status, nchars,
                     [c["code"] for c in codes][:5] if codes else ""), flush=True)
        browser.close()
    sink.close()

    coded = [r for r in res if r["status"] == "coded"]
    print("\n==============  403 TIER-2 (headless Chromium)  ==============")
    print("  outcomes:", dict(tally))
    print("  rows that yielded a code: %d / %d" % (len(coded), len(res)))
    print("  wrote %s" % OUT_JSONL)
    for r in coded:
        print("   r%-6d %s" % (r["row"], [c["code"] for c in r["codes"]][:6]))


if __name__ == "__main__":
    main()
