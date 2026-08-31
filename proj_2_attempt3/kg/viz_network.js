// Force-directed node-link view: canvas, hand-rolled simulation (the artifact CSP
// blocks CDN libraries), with zoom / pan / node-drag.
//
// Two link types are drawn:
//   association  taxon -> disease. blue enriched, red depleted, grey dashed contested.
//                width = number of papers, opacity = directional consistency.
//   containment  taxon -> taxon, faint dotted. Papers report at whatever rank they
//                resolved to, so a family and genera inside it both appear. That is
//                NOT redundancy: Lachnospiraceae is depleted in Parkinson's (15
//                papers) while Hungatella inside it is enriched (7). Drawing the
//                nesting is the only way that reads as biology instead of noise.
(function () {
  const C = document.getElementById("net");
  if (!C) return;
  const ctx = C.getContext("2d");
  let nodes = [], links = [], hier = [], hover = null, pinned = null, raf = null;
  let alpha = 1, W = 0, H = 0;
  const DPR = Math.min(devicePixelRatio || 1, 2);
  // view transform (world -> screen): screen = world * k + [tx,ty]
  let k = 1, tx = 0, ty = 0;
  let dragNode = null, panning = false, last = null, moved = false;

  const css = n => getComputedStyle(document.body).getPropertyValue(n).trim();
  const toWorld = (sx, sy) => [(sx - tx) / k, (sy - ty) / k];

  function resize() {
    const r = C.getBoundingClientRect();
    W = r.width; H = r.height;
    C.width = W * DPR; C.height = H * DPR;
  }

  function build(minPapers, disease) {
    let es = window.__KG__.edges.filter(e => e.n_papers >= minPapers);
    if (disease !== "__all__") es = es.filter(e => e.disease === disease);
    const idx = new Map();
    nodes = []; links = []; hier = [];
    const add = (id, type, label, rank) => {
      if (!idx.has(id)) {
        idx.set(id, nodes.length);
        nodes.push({ id, type, label, rank, deg: 0, fixed: false,
          x: W / 2 + (Math.random() - .5) * Math.min(W, H) * .6,
          y: H / 2 + (Math.random() - .5) * Math.min(W, H) * .6, vx: 0, vy: 0 });
      }
      return idx.get(id);
    };
    es.forEach(e => {
      const s = add("t:" + e.taxon_key, "taxon", e.taxon, e.rank);
      const t = add("d:" + e.disease, "disease", e.disease, "");
      nodes[s].deg++; nodes[t].deg++;
      links.push({ s, t, e });
    });
    (window.__KG__.hierarchy || []).forEach(h => {
      if (idx.has(h.parent) && idx.has(h.child))
        hier.push({ s: idx.get(h.parent), t: idx.get(h.child) });
    });
    k = 1; tx = 0; ty = 0; alpha = 1; pinned = null;
    if (raf) cancelAnimationFrame(raf);
    tick();
  }

  function step() {
    const kk = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.72;
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
        if (d2 < 1) { d2 = 1; dx = Math.random() - .5; dy = Math.random() - .5; }
        const f = (kk * kk) / d2 * 0.55, d = Math.sqrt(d2);
        a.vx += dx / d * f; a.vy += dy / d * f;
        b.vx -= dx / d * f; b.vy -= dy / d * f;
      }
    }
    const pull = (s, t, mult) => {
      const a = nodes[s], b = nodes[t];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(Math.hypot(dx, dy), 1);
      const f = (d * d) / kk / 190 * mult;
      a.vx += dx / d * f; a.vy += dy / d * f;
      b.vx -= dx / d * f; b.vy -= dy / d * f;
    };
    links.forEach(l => pull(l.s, l.t, 1 + Math.min(l.e.n_papers, 8) / 9));
    hier.forEach(h => pull(h.s, h.t, 0.45));   // containment pulls kin together, gently
    nodes.forEach(n => {
      if (n.fixed) { n.vx = n.vy = 0; return; }
      n.vx += (W / 2 - n.x) * 0.0016; n.vy += (H / 2 - n.y) * 0.0016;
      n.x += Math.max(-14, Math.min(14, n.vx * alpha));
      n.y += Math.max(-14, Math.min(14, n.vy * alpha));
      n.vx *= 0.82; n.vy *= 0.82;
    });
    alpha *= 0.985;
  }

  const radius = n => n.type === "disease"
    ? Math.min(26, 11 + Math.sqrt(n.deg) * 1.9)
    : Math.min(13, 3.4 + Math.sqrt(n.deg) * 1.7);

  function draw() {
    const up = css("--up"), down = css("--down"), mixed = css("--mixed");
    const ink = css("--ink"), ink2 = css("--ink-2"), panel = css("--panel"), surf = css("--surface");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.setTransform(DPR * k, 0, 0, DPR * k, DPR * tx, DPR * ty);

    const focus = pinned || hover;
    const near = new Set();
    if (focus) {
      near.add(focus.id);
      links.forEach(l => {
        if (nodes[l.s].id === focus.id) near.add(nodes[l.t].id);
        if (nodes[l.t].id === focus.id) near.add(nodes[l.s].id);
      });
      hier.forEach(h => {
        if (nodes[h.s].id === focus.id) near.add(nodes[h.t].id);
        if (nodes[h.t].id === focus.id) near.add(nodes[h.s].id);
      });
    }

    ctx.setLineDash([1.5, 3]);
    ctx.strokeStyle = ink2;
    hier.forEach(h => {
      const a = nodes[h.s], b = nodes[h.t];
      const on = !focus || a.id === focus.id || b.id === focus.id;
      ctx.globalAlpha = on ? 0.35 : 0.04;
      ctx.lineWidth = 1 / k;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    });
    ctx.setLineDash([]);

    links.forEach(l => {
      const a = nodes[l.s], b = nodes[l.t];
      const on = !focus || a.id === focus.id || b.id === focus.id;
      ctx.globalAlpha = on ? (0.30 + 0.55 * l.e.consistency) : 0.05;
      ctx.strokeStyle = l.e.contested ? mixed : (l.e.direction === "enriched" ? up : down);
      ctx.lineWidth = Math.min(5.5, 0.7 + l.e.n_papers * 0.42) / Math.sqrt(k);
      if (l.e.contested) ctx.setLineDash([4 / k, 3 / k]); else ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    });
    ctx.setLineDash([]);

    nodes.forEach(n => {
      const on = !focus || near.has(n.id);
      ctx.globalAlpha = on ? 1 : 0.16;
      const r = radius(n);
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 6.284);
      ctx.fillStyle = n.type === "disease" ? ink : panel;
      ctx.fill();
      ctx.lineWidth = (n.fixed ? 2.6 : 1.6) / k;
      ctx.strokeStyle = n.type === "disease" ? ink : ink2;
      ctx.stroke();
      const show = n.type === "disease" || r > 8 || (focus && near.has(n.id)) || k > 1.6;
      if (show) {
        const fs = (n.type === "disease" ? 12 : 11) / k;
        ctx.font = `${n.type === "disease" ? "600 " : ""}${fs}px ui-sans-serif,system-ui`;
        ctx.textAlign = "center";
        const label = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
        const y = n.y + r + fs + 1;
        ctx.lineWidth = 3 / k; ctx.strokeStyle = surf;
        ctx.strokeText(label, n.x, y);
        ctx.fillStyle = n.type === "disease" ? ink : ink2;
        ctx.fillText(label, n.x, y);
      }
    });
    ctx.globalAlpha = 1;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function tick() { if (alpha > 0.02) step(); draw(); raf = requestAnimationFrame(tick); }

  function at(sx, sy) {
    const [wx, wy] = toWorld(sx, sy);
    let best = null, bd = 1e9;
    nodes.forEach(n => {
      const d = Math.hypot(n.x - wx, n.y - wy), r = radius(n) + 6 / k;
      if (d < r && d < bd) { bd = d; best = n; }
    });
    return best;
  }

  const tip = document.getElementById("nettip");
  function showTip(ev, f) {
    const rel = links.filter(l => nodes[l.s].id === f.id || nodes[l.t].id === f.id);
    const kin = hier.filter(h => nodes[h.s].id === f.id || nodes[h.t].id === f.id);
    const con = rel.filter(l => l.e.contested).length;
    tip.innerHTML = `<b>${f.label}</b>${f.rank ? ` <span style="opacity:.6">${f.rank}</span>` : ""}<br>`
      + `${rel.length} association${rel.length !== 1 ? "s" : ""}`
      + (con ? ` · <b>${con} contested</b>` : "")
      + (kin.length ? ` · ${kin.length} taxonomic link${kin.length !== 1 ? "s" : ""}` : "")
      + `<div class="p">${rel.slice(0, 5).map(l => {
          const o = nodes[l.s].id === f.id ? nodes[l.t] : nodes[l.s];
          return `· ${o.label} — ${l.e.contested ? "contested" : l.e.direction} (${l.e.n_papers}p)`;
        }).join("<br>")}${rel.length > 5 ? `<br>· +${rel.length - 5} more` : ""}</div>`;
    tip.style.opacity = 1;
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - 330) + "px";
    tip.style.top = Math.min(ev.clientY + 14, innerHeight - 180) + "px";
  }

  C.addEventListener("mousedown", ev => {
    const r = C.getBoundingClientRect();
    const n = at(ev.clientX - r.left, ev.clientY - r.top);
    moved = false; last = [ev.clientX, ev.clientY];
    if (n) { dragNode = n; n.fixed = true; } else { panning = true; }
  });
  addEventListener("mousemove", ev => {
    const r = C.getBoundingClientRect();
    if (dragNode) {
      moved = true;
      const [wx, wy] = toWorld(ev.clientX - r.left, ev.clientY - r.top);
      dragNode.x = wx; dragNode.y = wy; dragNode.vx = dragNode.vy = 0;
      alpha = Math.max(alpha, 0.25);
      return;
    }
    if (panning) {
      moved = true;
      tx += ev.clientX - last[0]; ty += ev.clientY - last[1];
      last = [ev.clientX, ev.clientY];
      return;
    }
    if (ev.target !== C) { if (!pinned) tip.style.opacity = 0; hover = null; return; }
    hover = at(ev.clientX - r.left, ev.clientY - r.top);
    C.style.cursor = hover ? "grab" : "default";
    const f = pinned || hover;
    if (f) showTip(ev, f); else tip.style.opacity = 0;
  });
  addEventListener("mouseup", ev => {
    if (dragNode && !moved) {                 // click without drag = pin/unpin
      dragNode.fixed = false;
      pinned = (pinned && pinned.id === dragNode.id) ? null : dragNode;
      // drive the shared detail panel from the node's best-evidenced association
      if (pinned && window.__showDetail) {
        const rel = links.filter(l => nodes[l.s].id === pinned.id || nodes[l.t].id === pinned.id)
                         .map(l => l.e)
                         .sort((a, b) => b.n_papers - a.n_papers);
        if (rel.length) window.__showDetail(rel[0], rel);
      }
    }
    dragNode = null; panning = false;
  });
  C.addEventListener("wheel", ev => {
    ev.preventDefault();
    const r = C.getBoundingClientRect();
    const sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    const [wx, wy] = toWorld(sx, sy);
    const nk = Math.max(0.25, Math.min(6, k * Math.exp(-ev.deltaY * 0.0015)));
    k = nk; tx = sx - wx * k; ty = sy - wy * k;   // zoom about the cursor
  }, { passive: false });
  C.addEventListener("dblclick", () => {
    k = 1; tx = 0; ty = 0; nodes.forEach(n => n.fixed = false); alpha = 0.7;
  });

  window.__netBuild = build;
  window.__netResize = () => { resize(); alpha = Math.max(alpha, 0.5); };
  resize();
  addEventListener("resize", () => { resize(); alpha = Math.max(alpha, 0.4); });
})();
