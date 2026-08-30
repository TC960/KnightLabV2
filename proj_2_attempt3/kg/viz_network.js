// Force-directed node-link view, canvas-rendered, no external libraries
// (the artifact CSP blocks CDNs, so the simulation is hand-rolled).
//
// Only edges with n_papers >= threshold are drawn. At >=2 that is 339 edges over
// 187 nodes, which lays out readably; the full 1,398 is a hairball and is why the
// default threshold is not 1.
(function () {
  const C = document.getElementById("net");
  if (!C) return;
  const ctx = C.getContext("2d");
  let nodes = [], links = [], hover = null, pinned = null, raf = null, alpha = 1;
  let W = 0, H = 0, DPR = Math.min(devicePixelRatio || 1, 2);

  const css = k => getComputedStyle(document.body).getPropertyValue(k).trim();

  function resize() {
    const r = C.getBoundingClientRect();
    W = r.width; H = r.height;
    C.width = W * DPR; C.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function build(minPapers, disease) {
    let es = window.__KG__.edges.filter(e => e.n_papers >= minPapers);
    if (disease !== "__all__") es = es.filter(e => e.disease === disease);
    const idx = new Map();
    nodes = []; links = [];
    const add = (id, type, label, rank) => {
      if (!idx.has(id)) {
        idx.set(id, nodes.length);
        nodes.push({
          id, type, label, rank, deg: 0,
          x: W / 2 + (Math.random() - .5) * Math.min(W, H) * .6,
          y: H / 2 + (Math.random() - .5) * Math.min(W, H) * .6,
          vx: 0, vy: 0
        });
      }
      return idx.get(id);
    };
    es.forEach(e => {
      const s = add("t:" + e.taxon_key, "taxon", e.taxon, e.rank);
      const t = add("d:" + e.disease, "disease", e.disease, "");
      nodes[s].deg++; nodes[t].deg++;
      links.push({ s, t, e });
    });
    alpha = 1;
    if (raf) cancelAnimationFrame(raf);
    tick();
  }

  function step() {
    const k = Math.sqrt((W * H) / Math.max(nodes.length, 1)) * 0.72;
    // repulsion (O(n^2); n<=~200 here so it is fine and avoids a quadtree)
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { d2 = 1; dx = Math.random() - .5; dy = Math.random() - .5; }
        const f = (k * k) / d2 * 0.55;
        const d = Math.sqrt(d2);
        const ux = dx / d * f, uy = dy / d * f;
        a.vx += ux; a.vy += uy; b.vx -= ux; b.vy -= uy;
      }
    }
    // attraction along edges, stronger for well-replicated pairs
    links.forEach(l => {
      const a = nodes[l.s], b = nodes[l.t];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      const f = (d * d) / k / 190 * (1 + Math.min(l.e.n_papers, 8) / 9);
      const ux = dx / d * f, uy = dy / d * f;
      a.vx += ux; a.vy += uy; b.vx -= ux; b.vy -= uy;
    });
    // gravity toward centre + integrate
    nodes.forEach(n => {
      n.vx += (W / 2 - n.x) * 0.0016;
      n.vy += (H / 2 - n.y) * 0.0016;
      n.x += Math.max(-14, Math.min(14, n.vx * alpha));
      n.y += Math.max(-14, Math.min(14, n.vy * alpha));
      n.vx *= 0.82; n.vy *= 0.82;
      const r = n.type === "disease" ? 30 : 12;
      n.x = Math.max(r, Math.min(W - r, n.x));
      n.y = Math.max(r, Math.min(H - r, n.y));
    });
    alpha *= 0.985;
  }

  const radius = n => n.type === "disease"
    ? Math.min(26, 11 + Math.sqrt(n.deg) * 1.9)
    : Math.min(13, 3.4 + Math.sqrt(n.deg) * 1.7);

  function draw() {
    const up = css("--up"), down = css("--down"), mixed = css("--mixed");
    const ink = css("--ink"), ink2 = css("--ink-2"), panel = css("--panel");
    ctx.clearRect(0, 0, W, H);
    const focus = pinned || hover;
    const near = new Set();
    if (focus) {
      near.add(focus.id);
      links.forEach(l => {
        if (nodes[l.s].id === focus.id) near.add(nodes[l.t].id);
        if (nodes[l.t].id === focus.id) near.add(nodes[l.s].id);
      });
    }
    links.forEach(l => {
      const a = nodes[l.s], b = nodes[l.t];
      const on = !focus || a.id === focus.id || b.id === focus.id;
      ctx.globalAlpha = on ? (0.30 + 0.55 * l.e.consistency) : 0.05;
      ctx.strokeStyle = l.e.contested ? mixed : (l.e.direction === "enriched" ? up : down);
      ctx.lineWidth = Math.min(5.5, 0.7 + l.e.n_papers * 0.42);
      if (l.e.contested) ctx.setLineDash([4, 3]); else ctx.setLineDash([]);
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
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = n.type === "disease" ? ink : ink2;
      ctx.stroke();
      const showLabel = n.type === "disease" || r > 8 || (focus && near.has(n.id));
      if (showLabel) {
        ctx.globalAlpha = on ? 1 : 0.16;
        ctx.font = n.type === "disease" ? "600 12px ui-sans-serif,system-ui" : "11px ui-sans-serif,system-ui";
        ctx.textAlign = "center";
        const label = n.label.length > 26 ? n.label.slice(0, 25) + "…" : n.label;
        const y = n.y + r + 12;
        ctx.lineWidth = 3; ctx.strokeStyle = css("--surface");
        ctx.strokeText(label, n.x, y);
        ctx.fillStyle = n.type === "disease" ? ink : ink2;
        ctx.fillText(label, n.x, y);
      }
    });
    ctx.globalAlpha = 1;
  }

  function tick() {
    if (alpha > 0.02) step();
    draw();
    raf = requestAnimationFrame(tick);
  }

  function at(mx, my) {
    let best = null, bd = 1e9;
    nodes.forEach(n => {
      const d = Math.hypot(n.x - mx, n.y - my), r = radius(n) + 6;
      if (d < r && d < bd) { bd = d; best = n; }
    });
    return best;
  }

  const tip = document.getElementById("nettip");
  C.addEventListener("mousemove", ev => {
    const r = C.getBoundingClientRect();
    hover = at(ev.clientX - r.left, ev.clientY - r.top);
    C.style.cursor = hover ? "pointer" : "default";
    const f = pinned || hover;
    if (f) {
      const rel = links.filter(l => nodes[l.s].id === f.id || nodes[l.t].id === f.id);
      const con = rel.filter(l => l.e.contested).length;
      tip.innerHTML = `<b>${f.label}</b>${f.rank ? ` <span style="opacity:.6">${f.rank}</span>` : ""}<br>`
        + `${rel.length} link${rel.length !== 1 ? "s" : ""}`
        + (con ? ` · <b>${con} contested</b>` : "")
        + `<div class="p">${rel.slice(0, 6).map(l => {
            const other = nodes[l.s].id === f.id ? nodes[l.t] : nodes[l.s];
            const d = l.e.contested ? "contested" : l.e.direction;
            return `· ${other.label} — ${d} (${l.e.n_papers}p)`;
          }).join("<br>")}${rel.length > 6 ? `<br>· +${rel.length - 6} more` : ""}</div>`;
      tip.style.opacity = 1;
      tip.style.left = Math.min(ev.clientX + 14, innerWidth - 330) + "px";
      tip.style.top = Math.min(ev.clientY + 14, innerHeight - 170) + "px";
    } else tip.style.opacity = 0;
  });
  C.addEventListener("mouseleave", () => { hover = null; if (!pinned) tip.style.opacity = 0; });
  C.addEventListener("click", ev => {
    const r = C.getBoundingClientRect();
    const n = at(ev.clientX - r.left, ev.clientY - r.top);
    pinned = (pinned && n && pinned.id === n.id) ? null : n;
  });

  window.__netBuild = build;
  window.__netResize = () => { resize(); alpha = Math.max(alpha, 0.5); };
  resize();
  addEventListener("resize", () => { resize(); alpha = Math.max(alpha, 0.4); });
})();
