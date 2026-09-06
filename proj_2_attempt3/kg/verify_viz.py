"""Drive the built kg.html in a real browser and assert the viewer actually works.

    pip install playwright && python3 verify_viz.py kg.html

WHY THIS EXISTS. Two fixes in this repo have silently erased themselves on
rebuild while printing success, and a blank-canvas bug passed every static
check -- `build_viz.py` emitting well-formed HTML proves nothing about whether
the page renders. So every assertion below reads the rendered DOM or the canvas
pixels back out of Chromium AFTER real user input (tab clicks, select changes,
checkbox toggles), never the source string.

Set BROWSER to a chromium binary if the default path is wrong.
"""
import os, sys, json, pathlib
from playwright.sync_api import sync_playwright

PAGE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg.html")).resolve().as_uri()
BROWSER = os.environ.get("BROWSER", "/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
fails, notes = [], []


def check(name, cond, detail=""):
    (notes if cond else fails).append(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


with sync_playwright() as p:
    b = p.chromium.launch(executable_path=BROWSER)
    pg = b.new_page(viewport={"width": 1280, "height": 1000})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.goto(PAGE)
    pg.wait_for_timeout(3500)

    check("no console/page errors", not errs, str(errs[:3]))

    # --- network canvas actually painted (the historical blank-canvas bug) ---
    ink = pg.evaluate("""() => {
      const c = document.getElementById('net');
      const x = c.getContext('2d');
      const d = x.getImageData(0,0,c.width,c.height).data;
      const seen = new Set();
      for (let i=0;i<d.length;i+=4*97) seen.add(d[i]+','+d[i+1]+','+d[i+2]);
      return {w:c.width, h:c.height, distinct: seen.size};
    }""")
    check("network canvas painted", ink["distinct"] > 5, json.dumps(ink))

    # --- ranked view ---
    pg.click("#tab-rank")
    pg.wait_for_timeout(400)

    def rows():
        return pg.evaluate("""() => [...document.querySelectorAll('#chart .row')].map(r => ({
            tax: r.querySelector('.tax').textContent.trim(),
            scope: (r.querySelector('.scope .chip.cls')||{}).textContent || '',
            hollow: !!r.querySelector('.bar.prior'),
            papers: r.querySelector('.cnt').textContent.trim(),
        }))""")

    base = rows()
    check("ranked rows render", len(base) > 10, f"n={len(base)}")
    check("scope chip on every row", all(r["scope"] for r in base),
          f"missing={sum(1 for r in base if not r['scope'])}")

    n_hollow_default = sum(r["hollow"] for r in base)
    check("hollow (restates-prior) bars present by default", n_hollow_default > 0,
          f"n={n_hollow_default}")

    # --- sort by specificity: discriminating must come first, generic last ---
    pg.select_option("#sortby", "spec")
    pg.wait_for_timeout(300)
    sp = rows()
    order = [r["scope"].split()[0] for r in sp]
    rank = {"flips": 0, "mixed": 1, "narrow": 2, "—": 2, "generic": 3}
    seq = [rank.get(o, 9) for o in order]
    check("specificity sort is monotonic (discriminating->generic)",
          seq == sorted(seq), f"first10={order[:10]} last5={order[-5:]}")
    check("specificity sort surfaces 'flips' first", order[0] == "flips", f"got={order[0]}")
    check("evidence sort differs from specificity sort",
          [r["tax"] for r in base] != [r["tax"] for r in sp])

    # --- hide-prior filter ---
    pg.select_option("#sortby", "ev")
    pg.wait_for_timeout(200)
    pg.check("#hideprior")
    pg.wait_for_timeout(300)
    hid = rows()
    check("hide-prior removes every hollow bar", sum(r["hollow"] for r in hid) == 0,
          f"remaining={sum(r['hollow'] for r in hid)}")
    check("hide-prior actually changes the row set",
          [r["tax"] for r in hid] != [r["tax"] for r in base])
    pg.uncheck("#hideprior")
    pg.wait_for_timeout(300)

    # --- detail panel carries the plain-English specificity sentence ---
    pg.click("#chart .row:first-child")
    pg.wait_for_timeout(400)
    spec_txt = pg.evaluate("() => (document.querySelector('#detail .spec')||{}).textContent || ''")
    check("detail panel shows specificity sentence", "Reported in" in spec_txt,
          repr(spec_txt[:110]))
    check("specificity sentence has no unresolved placeholder", "?" not in spec_txt.split("—")[0],
          repr(spec_txt[:110]))

    # --- rank-conflict filter and its detail block ---
    pg.check("#onlyrc")
    pg.wait_for_timeout(400)
    rc = pg.evaluate("""() => [...document.querySelectorAll('#chart .row')].map(r => ({
        chip: !!r.querySelector('.chip.conf'),
        tax: r.querySelector('.tax').textContent.trim()}))""")
    check("rank-conflict filter returns rows", len(rc) > 0, f"n={len(rc)}")
    check("every filtered row carries the rank chip", rc and all(r["chip"] for r in rc),
          f"missing={sum(1 for r in rc if not r['chip'])}")
    pg.click("#chart .row:first-child")
    pg.wait_for_timeout(400)
    conf = pg.evaluate("() => (document.querySelector('#detail .conflict')||{}).textContent || ''")
    check("detail panel explains the rank conflict", "single study reports both" in conf,
          repr(conf[:90]))
    check("rank-conflict block names the counterpart taxon", " vs its " in conf, repr(conf[:160]))
    pg.uncheck("#onlyrc")
    pg.wait_for_timeout(300)
    check("unchecking rank-conflict filter restores rows",
          len(rows()) > len(rc), f"{len(rc)} -> {len(rows())}")

    # --- table view gained its two columns ---
    heads = pg.evaluate("() => [...document.querySelectorAll('table th')].map(t=>t.textContent.trim())")
    check("table has Diseases + Specificity columns",
          "Diseases" in heads and "Specificity" in heads, str(heads))

    # --- tiles ---
    tiles = pg.evaluate("() => [...document.querySelectorAll('.tile')].map(t=>t.textContent.trim())")
    check("discriminating tile present", any("discriminating" in t for t in tiles), str(tiles))

    pg.screenshot(path=os.environ.get("SHOT_RANK", "/tmp/kg_shot_rank.png"), full_page=False)
    pg.click("#tab-net"); pg.wait_for_timeout(2500)
    pg.screenshot(path=os.environ.get("SHOT_NET", "/tmp/kg_shot_net.png"), full_page=False)
    b.close()

print("\n".join(notes))
print("\n".join(fails))
print(f"\n{len(notes)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
