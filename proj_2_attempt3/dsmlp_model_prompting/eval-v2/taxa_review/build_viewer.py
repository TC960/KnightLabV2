#!/usr/bin/env python3
"""
Build a single self-contained HTML viewer to adjudicate the 15-paper gold-standard
test set (test_set_v2) taxa against the Fable-5 re-annotation.

Question it answers: are the models over-extracting taxa (false positives), or is
Emily's human gold standard too conservative / incomplete?

Colour logic (as requested by the PI):
  GREEN  = taxon is in the human Gold Standard (test_set_v2). Includes gold-only
           taxa AND taxa in the gold ∩ Fable intersection.
  YELLOW = taxon is in the Fable re-annotation ONLY (a candidate false-positive OR
           a candidate gap in the gold standard — that's what the viewer helps decide).

For every taxon the viewer shows the sentence(s) in the paper where it appears, so a
human can judge whether a YELLOW taxon sits near real significance language.

Run:  python build_viewer.py
Out:  test_set_v2_taxa_review.html   (open in any browser, no server needed)
"""
import json, re, html, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GOLD = os.path.join(ROOT, "EmilySong_GoldStandardPaper", "test_set_v2.json")
FABLE = os.path.join(ROOT, "dsmlp_model_prompting", "eval-v2", "results", "fable_gold_15.json")
OUT = os.path.join(HERE, "test_set_v2_taxa_review.html")


def split_taxa(s):
    if not s or str(s).strip().lower() == "nan":
        return []
    return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip() and t.strip().lower() != "nan"]


def variants(t):
    """Surface forms of a taxon name to look for in free text."""
    t = t.strip()
    out = {t}
    if "_" in t:
        out.add(t.replace("_", " "))
        out.add(t.split("_")[0])           # genus token for e.g. Clostridium_sensu_stricto_1
    parts = t.split()
    if len(parts) >= 2 and parts[0][:1].isalpha():
        out.add(parts[0][0] + ". " + " ".join(parts[1:]))   # abbreviated: L. salivarius
    return {v for v in out if len(v) >= 3}


def compile_pat(t):
    vs = sorted(variants(t), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(v) for v in vs) + r")\b", re.IGNORECASE)


SENT_SPLIT = re.compile(r"(?<=[.?!])\s+|\n{2,}")


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def bold(sent, pat):
    return pat.sub(lambda m: "<mark>" + html.escape(m.group(0)) + "</mark>", html.escape(sent))


def highlight_full(text, taxa):
    """Return HTML for the full paper with every taxon occurrence wrapped in a
    span coloured by source (green wins over yellow on overlap)."""
    intervals = []  # (start, end, color, name, dir)
    for tx in taxa:
        for m in tx["_pat"].finditer(text):
            intervals.append((m.start(), m.end(), tx["color"], tx["name"], tx["dir"]))
    # resolve overlaps: sort by start, then green-priority, then longest
    prio = {"green": 0, "yellow": 1}
    intervals.sort(key=lambda x: (x[0], prio[x[2]], -(x[1] - x[0])))
    chosen = []
    last_end = -1
    for iv in intervals:
        if iv[0] >= last_end:
            chosen.append(iv)
            last_end = iv[1]
    out = []
    pos = 0
    for s, e, color, name, d in chosen:
        out.append(html.escape(text[pos:s]))
        arrow = "↑" if d == "up" else "↓"
        title = html.escape(f"{name} ({'enriched' if d=='up' else 'depleted'})")
        out.append(f'<span class="hl {color}" title="{title}">{html.escape(text[s:e])}<sub>{arrow}</sub></span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def main():
    gold = json.load(open(GOLD))
    fable = json.load(open(FABLE))
    gby = {g["link"]: g for g in gold}

    papers = []
    for f in fable:
        g = gby.get(f["link"], {})
        g_en = {t.lower(): t for t in split_taxa(g.get("taxa_enriched"))}
        g_de = {t.lower(): t for t in split_taxa(g.get("taxa_depleted"))}
        f_en = {t.lower(): t for t in split_taxa(f.get("taxa_enriched"))}
        f_de = {t.lower(): t for t in split_taxa(f.get("taxa_depleted"))}
        gset = set(g_en) | set(g_de)
        fset = set(f_en) | set(f_de)

        text = f["text"]
        sents = sentences(text)

        taxa = []
        for key in sorted(gset | fset):
            in_g, in_f = key in gset, key in fset
            if in_g and in_f:
                source, color = "both", "green"
            elif in_g:
                source, color = "gold_only", "green"
            else:
                source, color = "fable_only", "yellow"
            # display name + direction (prefer gold's spelling / direction)
            if key in g_en:
                name, d = g_en[key], "up"
            elif key in g_de:
                name, d = g_de[key], "down"
            elif key in f_en:
                name, d = f_en[key], "up"
            else:
                name, d = f_de[key], "down"
            pat = compile_pat(name)
            hits = [bold(s, pat) for s in sents if pat.search(s)]
            taxa.append({
                "name": name, "dir": d, "source": source, "color": color,
                "found": bool(hits), "sents": hits[:12], "n_hits": len(hits),
                "_pat": pat,
            })

        n_yellow = sum(1 for t in taxa if t["color"] == "yellow")
        n_yellow_found = sum(1 for t in taxa if t["color"] == "yellow" and t["found"])
        full_html = highlight_full(text, taxa)
        for t in taxa:
            del t["_pat"]

        papers.append({
            "title": f["title"], "disease": f.get("disease", ""), "link": f["link"],
            "n_gold": len(gset), "n_fable": len(fset), "n_yellow": n_yellow,
            "n_yellow_found": n_yellow_found,
            "taxa": taxa, "full_html": full_html,
        })

    data_json = json.dumps(papers, ensure_ascii=False)
    htmlout = TEMPLATE.replace("__DATA__", data_json)
    with open(OUT, "w") as fh:
        fh.write(htmlout)

    tg = sum(p["n_gold"] for p in papers)
    tf = sum(p["n_fable"] for p in papers)
    ty = sum(p["n_yellow"] for p in papers)
    tyf = sum(p["n_yellow_found"] for p in papers)
    print(f"wrote {OUT}")
    print(f"{len(papers)} papers | gold taxa={tg} fable taxa={tf} | fable-only(yellow)={ty} found-in-text={tyf}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold-standard taxa review — test_set_v2 vs Fable</title>
<style>
  :root { --green:#1a7f37; --greenbg:#d7f5dd; --yellow:#9a6700; --yellowbg:#fff3c9; --line:#e2e2e2; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c1c1c; }
  header { padding:18px 24px; border-bottom:1px solid var(--line); background:#fafafa; }
  h1 { margin:0 0 6px; font-size:20px; }
  .sub { color:#555; font-size:13px; max-width:900px; }
  .legend { margin-top:10px; font-size:13px; }
  .chip { display:inline-block; padding:2px 8px; border-radius:4px; margin-right:6px; font-weight:600; }
  .chip.green { background:var(--greenbg); color:var(--green); }
  .chip.yellow { background:var(--yellowbg); color:var(--yellow); }
  .layout { display:flex; }
  nav { width:280px; min-width:280px; border-right:1px solid var(--line); height:calc(100vh - 118px);
        overflow:auto; padding:8px; }
  nav button { display:block; width:100%; text-align:left; border:none; background:none; padding:8px 10px;
        border-radius:6px; cursor:pointer; font-size:13px; margin-bottom:2px; }
  nav button:hover { background:#f0f0f0; }
  nav button.active { background:#e8f0fe; font-weight:600; }
  nav .np { font-size:11px; color:#777; }
  main { flex:1; height:calc(100vh - 118px); overflow:auto; padding:18px 26px; }
  .ptitle { font-size:18px; margin:0 0 4px; }
  .pmeta { color:#555; font-size:13px; margin-bottom:12px; }
  .pmeta a { color:#0b62d6; }
  .stats { display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 16px; }
  .stat { background:#f6f6f6; border-radius:8px; padding:8px 14px; min-width:90px; }
  .stat b { display:block; font-size:20px; }
  .stat span { font-size:12px; color:#666; }
  .controls { margin-bottom:14px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  .toggle button { border:1px solid var(--line); background:#fff; padding:6px 14px; cursor:pointer; }
  .toggle button:first-child { border-radius:6px 0 0 6px; }
  .toggle button:last-child { border-radius:0 6px 6px 0; border-left:none; }
  .toggle button.active { background:#1c1c1c; color:#fff; }
  label.f { font-size:13px; color:#333; cursor:pointer; }
  .tax { border:1px solid var(--line); border-radius:8px; margin-bottom:10px; overflow:hidden; }
  .tax.green { border-left:5px solid var(--green); }
  .tax.yellow { border-left:5px solid var(--yellow); }
  .taxhead { padding:8px 12px; display:flex; align-items:center; gap:10px; cursor:pointer; }
  .taxhead:hover { background:#fafafa; }
  .tname { font-weight:600; font-size:15px; }
  .tag { font-size:11px; padding:1px 7px; border-radius:10px; background:#eee; color:#444; }
  .tag.up { background:#e7f6ea; color:var(--green); }
  .tag.down { background:#fdecec; color:#b42318; }
  .tag.notfound { background:#fde3e3; color:#b42318; }
  .tcount { margin-left:auto; font-size:12px; color:#888; }
  .sents { padding:0 14px 10px 30px; display:none; }
  .sents.open { display:block; }
  .sents li { margin:6px 0; color:#333; font-size:13.5px; }
  mark { background:#ffe58a; padding:0 1px; }
  .grouphdr { font-size:13px; font-weight:600; color:#555; margin:16px 0 8px; text-transform:uppercase; letter-spacing:.03em; }
  .full { line-height:1.9; font-size:14px; max-width:900px; }
  .full .hl { padding:0 1px; border-radius:2px; }
  .full .hl.green { background:var(--greenbg); box-shadow:inset 0 -2px 0 var(--green); }
  .full .hl.yellow { background:var(--yellowbg); box-shadow:inset 0 -2px 0 var(--yellow); }
  .full .hl sub { font-size:9px; opacity:.7; }
  .hidden { display:none !important; }
</style>
</head>
<body>
<header>
  <h1>Gold-standard taxa review — <code>test_set_v2</code> (15 papers) vs Fable-5 re-annotation</h1>
  <div class="sub">Adjudication aid: is the human gold standard too conservative, or are the models over-extracting?
    Every taxon is shown in its sentence context so you can judge whether it sits near real significance language.</div>
  <div class="legend">
    <span class="chip green">GREEN</span> in Emily's human gold standard (gold-only + gold∩Fable) &nbsp;·&nbsp;
    <span class="chip yellow">YELLOW</span> Fable re-annotation only — candidate gap or false positive &nbsp;·&nbsp;
    &nbsp;↑ enriched &nbsp;↓ depleted
  </div>
</header>
<div class="layout">
  <nav id="nav"></nav>
  <main id="main"></main>
</div>
<script>
const DATA = __DATA__;
let cur = 0, view = "taxon", onlyYellow = false, onlyNotFound = false;

function buildNav() {
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  DATA.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "navbtn" + (i === cur ? " active" : "");
    b.innerHTML = `<div>${i+1}. ${esc(p.disease || p.title)}</div>
      <div class="np">${esc(shorten(p.title,60))} · gold ${p.n_gold} · fable ${p.n_fable} · <b>${p.n_yellow} yellow</b></div>`;
    b.onclick = () => { cur = i; render(); };
    nav.appendChild(b);
  });
}

function render() {
  buildNav();
  const p = DATA[cur];
  const m = document.getElementById("main");
  const groups = [
    ["Fable-only (YELLOW) — adjudicate these", p.taxa.filter(t => t.color==="yellow")],
    ["Gold ∩ Fable (both agree)", p.taxa.filter(t => t.source==="both")],
    ["Gold only (Fable missed)", p.taxa.filter(t => t.source==="gold_only")],
  ];
  let body;
  if (view === "taxon") {
    body = groups.map(([label, list]) => {
      let items = list;
      if (onlyNotFound) items = items.filter(t => !t.found);
      if (!items.length) return "";
      return `<div class="grouphdr">${label} (${items.length})</div>` + items.map(taxCard).join("");
    }).join("");
    if (onlyYellow) {
      body = `<div class="grouphdr">Fable-only (YELLOW) — adjudicate these</div>` +
        p.taxa.filter(t => t.color==="yellow" && (!onlyNotFound || !t.found)).map(taxCard).join("");
    }
  } else {
    body = `<div class="full">${p.full_html}</div>`;
  }
  m.innerHTML = `
    <div class="ptitle">${esc(p.title)}</div>
    <div class="pmeta">${esc(p.disease)} · <a href="${esc(p.link)}" target="_blank">${esc(p.link)}</a></div>
    <div class="stats">
      <div class="stat"><b>${p.n_gold}</b><span>gold taxa</span></div>
      <div class="stat"><b>${p.n_fable}</b><span>fable taxa</span></div>
      <div class="stat"><b>${p.n_yellow}</b><span>fable-only (yellow)</span></div>
      <div class="stat"><b>${p.n_yellow_found}/${p.n_yellow}</b><span>yellow found in text</span></div>
    </div>
    <div class="controls">
      <div class="toggle">
        <button id="vt" class="${view==='taxon'?'active':''}">By taxon</button>
        <button id="vf" class="${view==='full'?'active':''}">Full paper</button>
      </div>
      <label class="f"><input type="checkbox" id="oy" ${onlyYellow?'checked':''}> only yellow</label>
      <label class="f"><input type="checkbox" id="onf" ${onlyNotFound?'checked':''}> only NOT-found-in-text</label>
    </div>
    <div id="body">${body}</div>`;
  document.getElementById("vt").onclick = () => { view="taxon"; render(); };
  document.getElementById("vf").onclick = () => { view="full"; render(); };
  document.getElementById("oy").onchange = e => { onlyYellow = e.target.checked; render(); };
  document.getElementById("onf").onchange = e => { onlyNotFound = e.target.checked; render(); };
  document.querySelectorAll(".taxhead").forEach(h => h.onclick = () => {
    h.nextElementSibling.classList.toggle("open");
  });
  m.scrollTop = 0;
}

function taxCard(t) {
  const arrow = t.dir === "up" ? "↑ enriched" : "↓ depleted";
  const nf = t.found ? "" : `<span class="tag notfound">NOT in text</span>`;
  const sents = t.found
    ? `<ul>${t.sents.map(s => `<li>${s}</li>`).join("")}</ul>` +
      (t.n_hits > t.sents.length ? `<div class="np">…and ${t.n_hits - t.sents.length} more mention(s)</div>` : "")
    : `<div class="np">No verbatim mention found in scraped full text (may live in a figure/table not captured).</div>`;
  return `<div class="tax ${t.color}">
    <div class="taxhead">
      <span class="tname">${esc(t.name)}</span>
      <span class="tag ${t.dir==='up'?'up':'down'}">${arrow}</span>
      ${nf}
      <span class="tcount">${t.n_hits} mention${t.n_hits===1?"":"s"} ▸</span>
    </div>
    <div class="sents">${sents}</div>
  </div>`;
}

function esc(s){ return (s||"").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function shorten(s,n){ return s && s.length>n ? s.slice(0,n)+"…" : (s||""); }
render();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
