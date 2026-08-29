"""Human-in-the-loop headful browser recovery over every needs-review row with a URL.

Design for a long unattended-ish run with occasional captcha help:
  * PERSISTENT profile  -> a challenge you solve once for a domain sticks for all its pages.
  * CRASH RECOVERY      -> if the browser dies (TargetClosedError), relaunch and continue.
  * CHALLENGE WAIT      -> on a Cloudflare/Akamai challenge it PAUSES up to 3 min, printing a
                           prompt, and polls the page until you clear it in the window.
  * INCREMENTAL + RESUME-> appends each row to browser_recover.jsonl; re-running skips done rows.
  * DOMAIN-GROUPED      -> same-domain pages run consecutively so one solve covers the batch.

Targets: needs-review rows, url = Unpaywall oa_url if present else the 403 publisher url.

Run (headful, solve captchas as they appear):
  caffeinate -dis python3 -m accession.browser_recover
"""
import csv, json, os, re, time, random, subprocess, tempfile, collections
from urllib.parse import urlparse
from . import config, dictionary

# sharded parallel run: each worker gets SHARD_ID/NSHARDS + its own CDP_ENDPOINT + own output file.
SHARD = os.environ.get("SHARD_ID")            # None => single run
NSHARDS = int(os.environ.get("NSHARDS", "1"))
REAUDIT = os.environ.get("REAUDIT")           # set => re-audit reaudit_rows.json with tolerant extractor
DELAY = float(os.environ.get("DELAY", "8"))   # base seconds to pause between pages (per worker) + jitter
FAST = os.environ.get("FAST")                 # set => DON'T wait on a challenge; mark blocked, move on
LIMIT = int(os.environ.get("LIMIT", "0"))     # >0 => cap targets per shard (for subset test runs)
# output namespace: a distinct RUN_PREFIX gives an independent done-set (so a fresh pass re-tries
# rows a previous pass only read 'thin', instead of skipping them).
_PREFIX = os.environ.get("RUN_PREFIX") or ("browser_reaudit" if REAUDIT else "browser_recover")
OUT_JSONL = os.path.join(config.HERE,
                         "%s.jsonl" % _PREFIX if SHARD is None else "%s.w%s.jsonl" % (_PREFIX, SHARD))
PROFILE = os.path.join(config.HERE, "browser_profile")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
STEALTH = ("Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
           "window.chrome={runtime:{}};"
           "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
           "Object.defineProperty(navigator,'languages',{get:()=>['en-US','en']});")
BLOCK_MARKERS = ("access denied", "just a moment", "verify you are human", "security verification",
                 "cloudflare", "enable javascript", "attention required", "captcha", "are you a robot")
FULL = 40000   # a real paper; below this we do NOT trust a read


def to_readable(url):
    if "mdpi.com" in url:
        return re.sub(r"/pdf(\?.*)?$", "/htm", url)
    return url


def domain(u):
    try:
        return urlparse(u).netloc.replace("www.", "")
    except Exception:
        return "?"


def targets():
    rows = list(csv.DictReader(open(os.path.join(config.REPO, "articles.out.csv"),
                                     newline="", encoding="utf-8")))
    up = {json.loads(l)["row_index"]: json.loads(l) for l in open(os.path.join(config.HERE, "unpaywall.jsonl"))}
    mr = {x["row"]: x for x in json.load(open(os.path.join(config.HERE, "manual_review.json")))}
    # done-set across this run's files only (browser_recover* for a normal run, browser_reaudit*
    # for a re-audit) so no row is done twice WITHIN a run, but a re-audit doesn't skip rows the
    # first pass already read.
    import glob
    done = set()
    for fp in glob.glob(os.path.join(config.HERE, "%s*.jsonl" % _PREFIX)):
        for l in open(fp):
            try:
                done.add(json.loads(l)["row"])
            except Exception:
                pass
    # re-audit mode: only the specific rows we read >=40k via an un-swept channel
    only = None
    if REAUDIT:
        only = set(json.load(open(os.path.join(config.HERE, "reaudit_rows.json"))))
    out = []
    for i, r in enumerate(rows):
        if r.get("certainty") != "needs-review" or i in done:
            continue
        if only is not None and i not in only:
            continue
        # this worker owns rows where row % NSHARDS == SHARD_ID (balanced, stable partition)
        if SHARD is not None and i % NSHARDS != int(SHARD):
            continue
        url = up.get(i, {}).get("oa_url") or (mr.get(i, {}) or {}).get("url")
        if url:
            out.append((i, (r.get("doi") or "").strip(), url))
    out.sort(key=lambda t: domain(t[2]))   # group by domain so captcha-solves amortize
    return out[:LIMIT] if LIMIT else out


def extract(text):
    if REAUDIT or os.environ.get("TOLERANT"):
        from .extract_tolerant import extract_tolerant   # superset: catches ws-split + lookalike misses
        return extract_tolerant(text)
    seen, codes = set(), []
    for c in dictionary.extract_codes(text, with_provenance=True):
        if dictionary.in_accession_column(c["repo"]) and c["code"] not in seen:
            seen.add(c["code"]); codes.append(c)
    return codes


def is_challenge(body):
    low = body[:800].lower()
    return len(body) < 3000 and any(m in low for m in BLOCK_MARKERS)


def text_from_bytes(data, ctype=""):
    """Bytes -> plain text. PDF via pdftotext, HTML via tag-strip. This is how we read the many
    'thin' rows that were actually PDFs Chrome rendered in its viewer (inner_text sees nothing)."""
    if not data:
        return ""
    if data[:4] == b"%PDF":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data); p = f.name
        try:
            out = subprocess.run(["pdftotext", "-q", p, "-"], capture_output=True, timeout=60).stdout
        except Exception:
            out = b""
        finally:
            os.unlink(p)
        return re.sub(r"\s+", " ", out.decode("utf-8", "replace"))
    head = data[:2000].lower()
    if b"<html" in head or b"<article" in head or "html" in ctype.lower() or "xml" in ctype.lower():
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", data.decode("utf-8", "replace")))
    return ""


def main():
    from playwright.sync_api import sync_playwright, Error as PWError

    tg = targets()
    by_dom = collections.Counter(domain(u) for _, _, u in tg)
    print("browser-recover targets: %d rows across %d domains" % (len(tg), len(by_dom)), flush=True)
    print("top domains:", dict(by_dom.most_common(8)), flush=True)

    sink = open(OUT_JSONL, "a")
    tally = collections.Counter()
    os.makedirs(PROFILE, exist_ok=True)

    cdp = os.environ.get("CDP_ENDPOINT")   # e.g. http://localhost:9222 -> connect to a real Chrome

    with sync_playwright() as p:
        def launch():
            if cdp:
                # CONNECT to an already-running real Chrome (launched with --remote-debugging-port).
                # Not automation-launched -> genuine fingerprint, navigator.webdriver=false ->
                # Cloudflare passes and your manual solves stick.
                br = p.chromium.connect_over_cdp(cdp)
                ctx = br.contexts[0] if br.contexts else br.new_context()
                pg = ctx.pages[0] if ctx.pages else ctx.new_page()
                return ctx, pg
            ctx = p.chromium.launch_persistent_context(
                PROFILE, channel="chrome", headless=False, user_agent=UA, locale="en-US",
                viewport={"width": 1300, "height": 1700},
                args=["--disable-blink-features=AutomationControlled"])
            ctx.add_init_script(STEALTH)
            return ctx, (ctx.pages[0] if ctx.pages else ctx.new_page())

        ctx, page = launch()

        def fetch(nav):
            """Load one URL: handle challenge-pause + crash-relaunch. Returns (status, codes, nchars)."""
            nonlocal ctx, page
            for attempt in (1, 2):
                try:
                    page.goto(nav, wait_until="domcontentloaded", timeout=45000)
                    try:
                        page.wait_for_load_state("networkidle", timeout=4000)
                    except PWError:
                        pass
                    body = page.inner_text("body")
                    if is_challenge(body) and not FAST:
                        print("  >>> CHALLENGE on %s [%s]. SOLVE IT IN THAT CHROME WINDOW..."
                              % (domain(nav), cdp or "main"), flush=True)
                        for _ in range(36):           # up to ~3 min
                            time.sleep(5)
                            try:
                                body = page.inner_text("body")
                            except PWError:
                                break
                            if not is_challenge(body):
                                print("  >>> cleared.", flush=True); break
                    nc = len(body)
                    if is_challenge(body):
                        return "blocked", [], nc
                    cds = extract(body)
                    if cds:
                        return "coded", cds, nc
                    return ("read-no-code" if nc >= FULL else "thin"), [], nc
                except PWError as e:
                    if attempt == 1 and any(s in str(e).lower() for s in ("closed", "crash", "target")):
                        print("  ! page lost, re-acquiring...", flush=True)
                        if not cdp:
                            try:
                                ctx.close()
                            except Exception:
                                pass
                        try:
                            ctx, page = launch()
                        except Exception:
                            time.sleep(3); ctx, page = launch()
                        continue
                    return "err:%s" % type(e).__name__, [], 0
            return "err", [], 0

        def session_fetch(u):
            """GET a URL through the browser's OWN session (carries cookies + Cloudflare clearance,
            so no 403) and turn the bytes into text. This is how we read the PDF the render couldn't."""
            try:
                resp = page.context.request.get(u, timeout=45000)
                if not resp.ok:
                    return ""
                return text_from_bytes(resp.body(), resp.headers.get("content-type", ""))
            except Exception:
                return ""

        for k, (row, doi, url) in enumerate(tg, 1):
            # DOI -> publisher HTML article (has the data-availability statement); the raw oa_url is
            # often a direct-PDF link that renders as empty text. Fall back to it only if HTML is thin.
            cands = (["https://doi.org/%s" % doi] if doi else []) + [to_readable(url)]
            status, codes, nchars, best_nav = "err", [], 0, cands[0]
            for nav in cands:
                st, cds, nc = fetch(nav)
                if cds or nc > nchars:
                    status, codes, nchars, best_nav = st, cds, nc, nav
                if cds or nc >= FULL:
                    break
            # STILL thin & uncoded? the OA copy is very likely a PDF the render couldn't read.
            # Pull it through the browser session and pdftotext it.
            if status != "coded" and nchars < FULL and url:
                txt = session_fetch(url)
                if txt:
                    cds = extract(txt)
                    if cds:
                        status, codes, best_nav = "coded", cds, url
                    elif len(txt) >= FULL:
                        status, best_nav = "read-no-code", url
                    nchars = max(nchars, len(txt))
            tally[status] += 1
            rec = {"row": row, "doi": doi, "url": url, "nav": best_nav, "status": status,
                   "nchars": nchars, "codes": codes}
            sink.write(json.dumps(rec) + "\n"); sink.flush()
            print("  [%d/%d] r%-6d %-14s chars=%-7d %s"
                  % (k, len(tg), row, status, nchars,
                     [c["code"] for c in codes][:5] if codes else ""), flush=True)
            # polite throttle: human-like gap between pages so we don't trip rate limits / Cloudflare
            time.sleep(random.uniform(DELAY, DELAY * 1.8))
        if not cdp:
            ctx.close()          # CDP: leave the user's Chrome open
    sink.close()

    print("\n============== BROWSER RECOVER ==============")
    print("  outcomes:", dict(tally))
    coded = tally.get("coded", 0)
    print("  rows that yielded a code: %d" % coded)


if __name__ == "__main__":
    main()
