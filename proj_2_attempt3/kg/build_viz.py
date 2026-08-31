#!/usr/bin/env python3
"""Render kg/graph.json into a self-contained HTML explorer.

Two views over the same graph:

- **Network** (default): force-directed node-link, canvas, hand-rolled simulation
  because the artifact CSP blocks CDN libraries. Gated on a paper threshold --
  the full 1,398 edges genuinely is an unreadable hairball, but at >=2 papers it
  is 339 edges over 187 nodes and the structure is legible.
- **Ranked**: diverging bar chart per disease, depleted left / enriched right.
  Better than the network for "what is the evidence for THIS disease", because
  ordering and magnitude are readable in a way node position never is.

Shared encoding, in both views:
  edge color   blue enriched / red depleted / grey dashed contested
  edge width   number of papers (evidence count -- NOT effect size; we have none)
  edge opacity directional consistency
Direction is encoded by position as well as color (bar side; dash pattern), so it
survives colorblindness, greyscale and print. The blue/red poles pass CVD
separation at dE 18.5 (protanopia) against both light and dark surfaces.

Usage: python build_viz.py [--out kg.html]
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """
:root{
  --surface:#fcfcfb; --panel:#ffffff; --line:#e1e0d9;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#898781;
  --up:#1c5cab; --down:#d03b3b; --mixed:#898781;
  --up-soft:#cde2fb; --down-soft:#f7dcdc;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{--surface:#1a1a19;--panel:#222221;--line:#38383550;
        --ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#898781;
        --up:#3987e5;--down:#e66767;--mixed:#898781;
        --up-soft:#1c3d63;--down-soft:#5c2f2f;}
}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#222221;--line:#38383550;
  --ink:#ffffff;--ink-2:#c3c2b7;--ink-3:#898781;
  --up:#3987e5;--down:#e66767;--mixed:#898781;
  --up-soft:#1c3d63;--down-soft:#5c2f2f;}
:root[data-theme="light"]{--surface:#fcfcfb;--panel:#ffffff;--line:#e1e0d9;
  --ink:#0b0b0b;--ink-2:#52514e;--ink-3:#898781;
  --up:#1c5cab;--down:#d03b3b;--mixed:#898781;
  --up-soft:#cde2fb;--down-soft:#f7dcdc;}

*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.5;padding:28px 22px 60px}
.wrap{max-width:1120px;margin:0 auto}
h1{font-family:var(--serif);font-weight:600;font-size:29px;letter-spacing:-.01em;
   margin:0 0 6px;text-wrap:balance}
.sub{color:var(--ink-2);max-width:66ch;margin:0 0 22px;font-size:14.5px}
.eyebrow{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
         color:var(--ink-3);margin-bottom:7px;font-weight:600}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:10px;margin-bottom:22px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:11px 13px}
.tile .n{font-family:var(--mono);font-size:23px;font-variant-numeric:tabular-nums;line-height:1.15}
.tile .l{font-size:11.5px;color:var(--ink-2);margin-top:3px}

.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;margin-bottom:8px;
          padding-bottom:14px;border-bottom:1px solid var(--line)}
label{display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
      color:var(--ink-3);margin-bottom:4px;font-weight:600}
select,input[type=range]{font-family:var(--sans);font-size:14px;padding:6px 8px;
  background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:5px}
select:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--up);outline-offset:2px}
.toggle{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--ink-2)}

.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2);margin:14px 0 4px}
.key{display:flex;align-items:center;gap:6px}
.sw{width:19px;height:9px;border-radius:2px;display:inline-block}

.chart{margin-top:10px}
.row{display:grid;grid-template-columns:190px 1fr 62px;align-items:center;gap:10px;
     padding:2px 0;border-radius:4px}
.row:hover{background:color-mix(in oklab,var(--ink) 5%,transparent)}
.tax{font-size:13px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tax .rank{color:var(--ink-3);font-size:10.5px;margin-left:4px}
.track{position:relative;height:19px}
.axis{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--line)}
.bar{position:absolute;top:3px;height:13px;border-radius:3px}
.bar.up{background:var(--up)}
.bar.down{background:var(--down)}
.bar.faded{opacity:.45}
.cnt{font-family:var(--mono);font-size:12px;color:var(--ink-2);
     font-variant-numeric:tabular-nums;text-align:right}
.chip{display:inline-block;font-size:9.5px;letter-spacing:.05em;padding:1px 5px;border-radius:3px;
      border:1px solid var(--mixed);color:var(--ink-2);margin-left:5px;vertical-align:1px}

.tip{position:fixed;pointer-events:none;background:var(--panel);color:var(--ink);
     border:1px solid var(--line);border-radius:6px;padding:9px 11px;font-size:12.5px;
     max-width:330px;box-shadow:0 6px 22px #0003;opacity:0;transition:opacity .1s;z-index:9}
.tip b{font-weight:600}
.tip .p{color:var(--ink-2);font-size:11.5px;margin-top:5px;line-height:1.42}

table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-3)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
.tablewrap{overflow-x:auto;margin-top:6px}
details{margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
summary{cursor:pointer;font-size:13px;color:var(--ink-2)}
.note{font-size:12.5px;color:var(--ink-2);margin-top:26px;border-top:1px solid var(--line);
      padding-top:14px;max-width:70ch}
.empty{color:var(--ink-3);font-size:14px;padding:26px 0}
.tabs{display:flex;gap:2px;margin:18px 0 0;border-bottom:1px solid var(--line)}
.tab{appearance:none;background:none;border:0;border-bottom:2px solid transparent;
     padding:7px 13px;font-family:var(--sans);font-size:13.5px;color:var(--ink-3);cursor:pointer}
.tab[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--up);font-weight:600}
.pane[hidden]{display:none}
#net{width:100%;height:min(70vh,620px);display:block;border:1px solid var(--line);
     border-radius:7px;background:var(--panel);margin-top:12px}
.hint{font-size:12px;color:var(--ink-3);margin-top:7px}
@media (max-width:640px){.row{grid-template-columns:118px 1fr 48px}.tax{font-size:12px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
const G = window.__KG__;
const $ = s => document.querySelector(s);
const byDisease = {};
G.edges.forEach(e => (byDisease[e.disease] = byDisease[e.disease] || []).push(e));
const diseases = Object.keys(byDisease).sort((a,b)=>byDisease[b].length-byDisease[a].length);

const sel = $("#disease"), minp = $("#minp"), minpv = $("#minpv"), onlyC = $("#onlyc");
diseases.forEach(d => {
  const o = document.createElement("option");
  o.value = d; o.textContent = `${d} (${byDisease[d].length})`;
  sel.appendChild(o);
});
const allOpt = document.createElement("option");
allOpt.value = "__all__"; allOpt.textContent = `All diseases (${G.edges.length})`;
sel.insertBefore(allOpt, sel.firstChild);
sel.value = "__all__";

const tip = $("#tip");
function showTip(e, ed){
  tip.innerHTML = `<b>${ed.taxon}</b> — ${ed.disease}<br>`
    + `${ed.n_papers} paper${ed.n_papers>1?"s":""} · `
    + `<span style="color:var(--up)">${ed.n_up} enriched</span> / `
    + `<span style="color:var(--down)">${ed.n_down} depleted</span>`
    + (ed.contested ? `<br><b>Contested</b> — consistency ${(ed.consistency*100).toFixed(0)}%` : "")
    + `<div class="p">${ed.papers.slice(0,3).map(p=>"· "+p.slice(0,74)).join("<br>")}`
    + (ed.papers.length>3 ? `<br>· +${ed.papers.length-3} more` : "") + `</div>`;
  tip.style.opacity = 1;
  const x = Math.min(e.clientX+14, innerWidth-345), y = Math.min(e.clientY+14, innerHeight-140);
  tip.style.left = x+"px"; tip.style.top = y+"px";
}
const hideTip = () => tip.style.opacity = 0;

function render(){
  const d = sel.value, mp = +minp.value;
  minpv.textContent = mp;
  if (window.__netBuild) window.__netBuild(mp, d);
  let rows = (d === "__all__" ? G.edges : (byDisease[d]||[])).filter(e => e.n_papers >= mp);
  if (onlyC.checked) rows = rows.filter(e => e.contested);
  rows.sort((a,b) => b.n_papers - a.n_papers || b.n_up+b.n_down - (a.n_up+a.n_down));
  rows = rows.slice(0, 60);

  const max = Math.max(1, ...rows.map(e => Math.max(e.n_up, e.n_down)));
  const chart = $("#chart");
  if (!rows.length){ chart.innerHTML = `<div class="empty">No taxa match these filters.</div>`;
                     $("#tbody").innerHTML=""; $("#shown").textContent="0"; return; }
  chart.innerHTML = rows.map((e,i) => {
    const up = e.n_up/max*50, dn = e.n_down/max*50;
    return `<div class="row" data-i="${i}">
      <div class="tax">${e.taxon}<span class="rank">${e.rank||""}</span></div>
      <div class="track"><div class="axis"></div>
        ${e.n_down?`<div class="bar down${e.contested?" faded":""}" style="right:50%;width:${dn}%"></div>`:""}
        ${e.n_up?`<div class="bar up${e.contested?" faded":""}" style="left:50%;width:${up}%"></div>`:""}
      </div>
      <div class="cnt">${e.n_papers}${e.contested?'<span class="chip">split</span>':''}</div>
    </div>`;
  }).join("");
  [...chart.querySelectorAll(".row")].forEach(el => {
    const ed = rows[+el.dataset.i];
    el.addEventListener("mousemove", ev => showTip(ev, ed));
    el.addEventListener("mouseleave", hideTip);
  });
  $("#tbody").innerHTML = rows.map(e => `<tr><td>${e.taxon}</td><td>${e.rank||""}</td>
      <td>${e.contested?"contested":e.direction}</td>
      <td class="num">${e.n_up}</td><td class="num">${e.n_down}</td>
      <td class="num">${e.n_papers}</td>
      <td class="num">${(e.consistency*100).toFixed(0)}%</td></tr>`).join("");
  $("#shown").textContent = rows.length;
}
[sel, minp, onlyC].forEach(el => el.addEventListener("input", render));

// tabs
const tabs = [["tab-net","pane-net"],["tab-rank","pane-rank"]];
tabs.forEach(([tid,pid]) => document.getElementById(tid).addEventListener("click", () => {
  tabs.forEach(([t,p]) => {
    const on = t === tid;
    document.getElementById(t).setAttribute("aria-selected", on);
    document.getElementById(p).hidden = !on;
  });
  if (tid === "tab-net" && window.__netResize) window.__netResize();
}));

render();
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=os.path.join(HERE, "graph.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "kg.html"))
    a = ap.parse_args()
    G = json.load(open(a.graph))
    m = G["meta"]
    payload = {"edges": G["edges"], "hierarchy": G.get("hierarchy", [])}

    net_js = open(os.path.join(HERE, "viz_network.js")).read()
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Microbe–disease knowledge graph</title><style>{CSS}</style></head><body>
<div class="wrap">
<div class="eyebrow">Knight Lab · microbiome literature</div>
<h1>Microbe–disease associations</h1>
<p class="sub">Every taxon–disease pair extracted from {m['papers_in']} papers, aggregated across the
papers that report it. Bar length is the <b>number of papers</b>, not effect size — the extractor
returns direction only, and the underlying studies report incommensurable statistics (LEfSe LDA,
fold-change, p-values) that cannot honestly be pooled into one magnitude.</p>

<div class="tiles">
  <div class="tile"><div class="n">{m['n_edges']:,}</div><div class="l">taxon–disease edges</div></div>
  <div class="tile"><div class="n">{m['n_taxa']:,}</div><div class="l">distinct taxa</div></div>
  <div class="tile"><div class="n">{m['n_diseases']}</div><div class="l">diseases</div></div>
  <div class="tile"><div class="n">{m['n_replicated']}</div><div class="l">seen in &gt;1 paper</div></div>
  <div class="tile"><div class="n">{m['n_contested']}</div><div class="l">contested</div></div>
</div>

<div class="controls">
  <div><label for="disease">Disease</label><select id="disease"></select></div>
  <div><label for="minp">Min papers · <span id="minpv">2</span></label>
       <input type="range" id="minp" min="1" max="8" value="2"></div>
  <div class="toggle"><input type="checkbox" id="onlyc"><label for="onlyc"
       style="margin:0;text-transform:none;letter-spacing:0;font-size:13px">Contested only</label></div>
</div>

<div class="legend">
  <span class="key"><span class="sw" style="background:var(--down)"></span>depleted (bar left)</span>
  <span class="key"><span class="sw" style="background:var(--up)"></span>enriched (bar right)</span>
  <span class="key"><span class="chip">split</span>papers disagree — both arms drawn</span>
</div>

<div class="tabs" role="tablist">
  <button class="tab" id="tab-net"  role="tab" aria-selected="true"  aria-controls="pane-net">Network</button>
  <button class="tab" id="tab-rank" role="tab" aria-selected="false" aria-controls="pane-rank">Ranked</button>
</div>

<div class="pane" id="pane-net" role="tabpanel">
  <canvas id="net"></canvas>
  <p class="hint"><b>Scroll</b> to zoom · <b>drag background</b> to pan · <b>drag a node</b> to
  move and fix it · <b>click</b> to pin its neighbourhood · <b>double-click</b> to reset.
  Solid links are associations (blue enriched, red depleted, grey dashed = papers disagree);
  faint dotted links are <b>taxonomic containment</b> — a family and the genera inside it.
  Containment is not redundancy: <i>Lachnospiraceae</i> is depleted in Parkinson's across 15
  papers while <i>Hungatella</i> inside it is enriched across 7.</p>
</div>

<div class="pane" id="pane-rank" role="tabpanel" hidden>
  <div class="chart" id="chart"></div>
</div>
<div class="tip" id="tip"></div>
<div class="tip" id="nettip"></div>

<details><summary>Table view — <span id="shown">0</span> rows shown</summary>
<div class="tablewrap"><table><thead><tr><th>Taxon</th><th>Rank</th><th>Direction</th>
<th class="num">Enriched</th><th class="num">Depleted</th><th class="num">Papers</th>
<th class="num">Consistency</th></tr></thead><tbody id="tbody"></tbody></table></div></details>

<p class="note"><b>How to read a contested edge.</b> {m['n_contested']} pairs have papers pointing
both ways, and they are kept, never averaged into a single direction. That is deliberate: the
microbiome replication literature reports roughly one taxon in three flipping sign between cohorts,
so disagreement is a finding about the evidence, not noise to be smoothed. Direction is encoded by
position as well as color, so the chart survives colorblindness, greyscale and print.<br><br>
Source: <span style="font-family:var(--mono);font-size:11.5px">{m['source']}</span> ·
{m['papers_contributing']} of {m['papers_in']} papers contributed at least one association.
Associations only — no causal claim.</p>
</div>
<script>window.__KG__={json.dumps(payload)};</script>
<script>{JS}</script>
<script>{net_js}</script>
</body></html>"""
    open(a.out, "w").write(html)
    print(f"wrote {a.out}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
