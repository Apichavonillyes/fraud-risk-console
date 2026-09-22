(function () {
"use strict";
const M = D.meta, TH = D.thresholds;
const $ = (id) => document.getElementById(id);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const DUR = reduce ? 0 : 800;
const C = { fraud: "#ff4757", safe: "#00d9ff", good: "#00e5a0", amber: "#ffb020", violet: "#a58bff",
  ink: "#e6edf3", ink2: "#a9b4c2", muted: "#6f7c8d", line: "#1c2531", grid: "#17202b", panel: "#0f141c", bg: "#0a0e14" };

const fInt = d3.format(","), f$ = (v) => "$" + d3.format(",.0f")(v), f$2 = (v) => "$" + d3.format(",.2f")(v);
const f$s = (v) => d3.format("$.3~s")(v).replace("G", "B");
const pct = (v, dp = 1) => (isFinite(v) ? (v * 100).toFixed(dp) + "%" : "—");
const f3 = (v) => (isFinite(v) ? v.toFixed(3) : "—");
const tFmt = (t) => (t < 0.01 ? t.toFixed(3) : t < 0.1 ? t.toFixed(3) : t.toFixed(2));

const BASE = M.fraud_amount_test;          // loss if no model: every test fraud goes through
const BASE_RATE = M.fraud_raw / M.n_raw;
const S = { model: "xgb", cost: 10, ti: 0, seg: "cluster" };

// ------------------------------------------------------------------ core maths (from precomputed sweeps)
const rowAt = (i, m = S.model, w = "sweep_test") => {
  const r = D.models[m][w][i];
  return { t: r[0], tp: r[1], fp: r[2], fn: r[3], tn: r[4], fnAmt: r[5], tpAmt: r[6] };
};
function met(r, c = S.cost) {
  const al = r.tp + r.fp, P = al ? r.tp / al : NaN, R = r.tp / (r.tp + r.fn);
  const F = isFinite(P) && P + R > 0 ? (2 * P * R) / (P + R) : 0;
  const review = c * al, cost = review + r.fnAmt;
  return Object.assign({}, r, { al, P, R, F, review, cost, saved: BASE - cost, fpr: r.fp / (r.fp + r.tn) });
}
function argminCost(m, c, w) {
  const s = D.models[m][w]; let b = 0, bc = Infinity;
  s.forEach((r, i) => { const v = c * (r[1] + r[2]) + r[5]; if (v < bc) { bc = v; b = i; } });
  return b;
}
const optIdx = (m = S.model, c = S.cost) => argminCost(m, c, "sweep_oof");   // chosen on TRAIN out-of-fold
const testOptIdx = (m = S.model, c = S.cost) => argminCost(m, c, "sweep_test"); // reference only
const series = (m = S.model, c = S.cost) => TH.map((t, i) => met(rowAt(i, m), c));
function idxFromT(t) {
  let b = 0, bd = Infinity; const st = Math.sqrt(t);
  TH.forEach((x, i) => { const d = Math.abs(Math.sqrt(x) - st); if (d < bd) { bd = d; b = i; } });
  return b;
}

// ------------------------------------------------------------------ helpers
function tweenNum(el, to, fmt, dur = DUR) {
  const from = el._v;
  el._v = to;
  cancelAnimationFrame(el._raf);
  if (!dur || from === undefined || !isFinite(from) || !isFinite(to) || from === to) { el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
function countUp(el, to, fmt, dur = reduce ? 0 : 1400) { if (el._v === undefined) el._v = 0; tweenNum(el, to, fmt, dur); }

const tip = $("tip");
function showTip(ev, title, rows) {
  tip.replaceChildren();
  const t = document.createElement("div"); t.className = "tt"; t.textContent = title; tip.append(t);
  rows.forEach(([label, val, color]) => {
    const r = document.createElement("div"); r.className = "tr";
    const lk = document.createElement("span"); lk.className = "lk";
    if (color) { const i = document.createElement("i"); i.style.background = color; lk.append(i); }
    lk.append(document.createTextNode(label));
    const b = document.createElement("b"); b.textContent = val;
    r.append(lk, b); tip.append(r);
  });
  tip.classList.add("on"); placeTip(ev);
}
function placeTip(ev) {
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 16, y = ev.clientY + 16;
  if (x + w > innerWidth - 8) x = ev.clientX - w - 16;
  if (y + h > innerHeight - 8) y = ev.clientY - h - 16;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => tip.classList.remove("on");

function mkSvg(el, h) {
  const w = Math.max(240, Math.floor(el.clientWidth));
  d3.select(el).selectAll("svg").remove();
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("height", h);
  return { svg, w, h, defs: svg.append("defs") };
}
function vgrad(defs, id, color, a0, a1) {
  const g = defs.append("linearGradient").attr("id", id).attr("x1", 0).attr("x2", 0).attr("y1", 0).attr("y2", 1);
  g.append("stop").attr("offset", "0%").attr("stop-color", color).attr("stop-opacity", a0);
  g.append("stop").attr("offset", "100%").attr("stop-color", color).attr("stop-opacity", a1);
}
function glowFilter(defs, id, sd = 3) {
  const f = defs.append("filter").attr("id", id).attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
  f.append("feGaussianBlur").attr("stdDeviation", sd).attr("result", "b");
  const m = f.append("feMerge"); m.append("feMergeNode").attr("in", "b"); m.append("feMergeNode").attr("in", "SourceGraphic");
}
function revealClip(defs, id, x, y, w, h, anim) {
  const r = defs.append("clipPath").attr("id", id).append("rect").attr("x", x).attr("y", y - 6).attr("height", h + 12)
    .attr("width", anim && DUR ? 0 : w);
  if (anim && DUR) r.transition().duration(DUR * 1.4).ease(d3.easeCubicOut).attr("width", w);
  return r;
}
function drawIn(path, anim, delay = 0) {
  if (!anim || !DUR) return;
  const node = path.node(), L = node.getTotalLength();
  path.attr("stroke-dasharray", `${L} ${L}`).attr("stroke-dashoffset", L)
    .transition().delay(delay).duration(DUR * 1.5).ease(d3.easeCubicInOut).attr("stroke-dashoffset", 0)
    .on("end", function () { d3.select(this).attr("stroke-dasharray", null); });
}
const xThr = (w0, w1) => d3.scalePow().exponent(0.5).domain([0, 1]).range([w0, w1]);
function thrAxis(g, x, y, label) {
  const THR_TICKS = x.range()[1] - x.range()[0] < 330 ? [0.01, 0.1, 0.25, 0.5, 1] : [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1];
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${y})`)
    .call(d3.axisBottom(x).tickValues(THR_TICKS).tickFormat((d) => (d < 0.1 ? d : d3.format(".2~f")(d))).tickSize(4));
  g.append("text").attr("class", "alab").attr("x", x.range()[1]).attr("y", y + 28).attr("text-anchor", "end").text(label);
}

// ------------------------------------------------------------------ KPI strip
function kpisStatic() {
  countUp($("k-n"), M.n_raw, fInt);
  $("k-amt").textContent = f$s(M.amount_raw);
  $("k-hrs").textContent = Math.round(M.hours_span);
  countUp($("k-rate"), BASE_RATE * 100, (v) => v.toFixed(3) + "%");
  $("k-fraud").textContent = fInt(M.fraud_raw);
  $("k-acc").textContent = pct(1 - BASE_RATE, 2);
  $("k-tpn").textContent = M.fraud_test;
  $("b-ntest").textContent = fInt(M.n_test);
  $("cm-n").textContent = fInt(M.n_test);
}
function kpisModel(first) {
  const mo = D.models[S.model], oi = optIdx(), om = met(rowAt(oi));
  const f = first ? countUp : tweenNum;
  f($("k-prauc"), mo.pr_auc, (v) => v.toFixed(3));
  $("k-prci").textContent = mo.pr_auc_ci.map((v) => v.toFixed(2)).join("–");
  $("k-roc").textContent = mo.roc_auc.toFixed(3);
  f($("k-saved"), om.saved, f$);
  $("k-base").textContent = f$(BASE);
  $("k-topt").textContent = tFmt(TH[oi]);
}

// ------------------------------------------------------------------ threshold slider
const thr = $("thr");
function sliderSetup() {
  const sc = $("thrscale"); sc.replaceChildren();
  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const s = document.createElement("span");
    s.style.left = `calc(10px + ${Math.sqrt(t)} * (100% - 20px))`;
    s.textContent = t; sc.append(s);
  });
  thr.addEventListener("input", () => setThreshold(idxFromT(Math.pow(+thr.value / 1000, 2)), { fromSlider: true }));
  thr.addEventListener("keydown", (e) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
    if (d) { e.preventDefault(); setThreshold(Math.max(0, Math.min(TH.length - 1, S.ti + d))); }
  });
  $("snap").addEventListener("click", () => setThreshold(optIdx()));
  $("snap50").addEventListener("click", () => setThreshold(idxFromT(0.5)));
}
function placeOptPin() {
  const oi = optIdx();
  $("optpin").style.left = `calc(10px + ${Math.sqrt(TH[oi])} * (100% - 20px))`;
  $("optpin").textContent = `cost-optimal ${tFmt(TH[oi])}`;
}

function setThreshold(i, opts = {}) {
  S.ti = i;
  if (!opts.fromSlider) thr.value = Math.round(Math.sqrt(TH[i]) * 1000);
  thr.setAttribute("aria-valuetext", `threshold ${tFmt(TH[i])}`);
  updateThresholdViews(opts.instant);
}

function updateThresholdViews(instant) {
  const m = met(rowAt(S.ti)), dur = instant ? 0 : 380;
  $("tval").textContent = TH[S.ti].toFixed(3);
  // confusion matrix
  tweenNum($("cm-tp"), m.tp, fInt, dur); tweenNum($("cm-fn"), m.fn, fInt, dur);
  tweenNum($("cm-fp"), m.fp, fInt, dur); tweenNum($("cm-tn"), m.tn, fInt, dur);
  $("cm-tpa").textContent = f$(m.tpAmt) + " stopped";
  $("cm-fna").textContent = f$(m.fnAmt) + " lost";
  $("cm-fpa").textContent = (m.al ? pct(m.fp / m.al, 0) : "0%") + " of alerts";
  $("cm-tna").textContent = pct(m.tn / (m.tn + m.fp), 2) + " of legit";
  const q = (a) => Math.max(0, Math.min(1, a));
  document.querySelector(".cm .tp").style.backgroundColor = `rgba(0,229,160,${0.05 + 0.28 * q(m.R)})`;
  document.querySelector(".cm .fn").style.backgroundColor = `rgba(255,71,87,${0.05 + 0.28 * q(1 - m.R)})`;
  document.querySelector(".cm .fp").style.backgroundColor = `rgba(255,176,32,${0.04 + 0.26 * q(m.al ? m.fp / m.al : 0)})`;
  document.querySelector(".cm .tn").style.backgroundColor = `rgba(0,217,255,.05)`;
  // readouts
  tweenNum($("r-p"), m.P, (v) => pct(v, 1), dur); tweenNum($("r-r"), m.R, (v) => pct(v, 1), dur);
  tweenNum($("r-f1"), m.F, f3, dur); tweenNum($("r-al"), (m.al / M.n_test) * 1e4, (v) => v.toFixed(1), dur);
  tweenNum($("r-cost"), m.cost, f$, dur); tweenNum($("r-saved"), m.saved, (v) => (v < 0 ? "−" : "") + f$(Math.abs(v)), dur);
  $("r-saved").className = "v " + (m.saved >= 0 ? "pos" : "neg");
  // KPI: fraud caught at current threshold
  tweenNum($("k-tp"), m.tp, fInt, dur);
  $("k-rec").textContent = pct(m.R, 1);
  $("k-tpbar").style.width = (m.R * 100).toFixed(1) + "%";
  // charts
  CHARTS.cost.cursor(instant); CHARTS.prt.cursor(instant); CHARTS.gauge.update(instant);
  CHARTS.pr.cursor(instant); CHARTS.roc.cursor(instant);
  const oi = optIdx();
  $("cc-note").textContent = S.ti === oi ? "at cost-optimal threshold" :
    `${m.cost > met(rowAt(oi)).cost ? "+" : "−"}${f$(Math.abs(m.cost - met(rowAt(oi)).cost))} vs. cost-optimal`;
}

// ------------------------------------------------------------------ cost curve (stacked area w/ gradient)
const CHARTS = {};
CHARTS.cost = (function () {
  let ctx = null;
  function build(anim) {
    const el = $("costchart");
    const { svg, w, h, defs } = mkSvg(el, 262);
    const mg = { t: 16, r: 16, b: 34, l: 54 };
    const x = xThr(mg.l, w - mg.r), y = d3.scaleLinear().range([h - mg.b, mg.t]);
    vgrad(defs, "g-loss", C.fraud, 0.8, 0.25); vgrad(defs, "g-rev", C.amber, 0.75, 0.3); glowFilter(defs, "glow-c", 3);
    const clipR = revealClip(defs, "clip-cost", mg.l, mg.t, w - mg.l - mg.r, h - mg.t - mg.b, anim);
    const g = svg.append("g");
    const gy = g.append("g").attr("class", "axis gridl").attr("transform", `translate(${mg.l},0)`);
    thrAxis(g, x, h - mg.b, "threshold (√ scale)");
    const plot = g.append("g").attr("clip-path", "url(#clip-cost)");
    const aLoss = plot.append("path").attr("fill", "url(#g-loss)");
    const aRev = plot.append("path").attr("fill", "url(#g-rev)");
    const lTot = plot.append("path").attr("fill", "none").attr("stroke", C.ink).attr("stroke-width", 2);
    const base = g.append("g");
    base.append("line").attr("stroke", C.muted).attr("stroke-dasharray", "4 4").attr("x1", mg.l).attr("x2", w - mg.r);
    base.append("text").attr("fill", C.ink2).attr("font-size", 10.5).attr("font-family", "var(--mono)").attr("x", w - mg.r - 2).attr("dy", -5).attr("text-anchor", "end");
    const off = g.append("text").attr("fill", C.muted).attr("font-size", 10).attr("font-family", "var(--mono)").attr("x", mg.l + 4).attr("y", mg.t + 10);
    const opt = g.append("g");
    opt.append("line").attr("stroke", C.amber).attr("stroke-dasharray", "2 3").attr("y1", mg.t).attr("y2", h - mg.b);
    opt.append("path").attr("d", d3.symbol(d3.symbolDiamond, 90)()).attr("fill", C.amber).attr("stroke", C.bg).attr("stroke-width", 2).attr("class", "od");
    opt.append("text").attr("fill", C.amber).attr("font-size", 10.5).attr("font-family", "var(--cond)").attr("font-weight", 600).attr("letter-spacing", ".06em").attr("y", mg.t + 2);
    const cur = g.append("g").style("pointer-events", "none");
    cur.append("line").attr("stroke", C.ink).attr("stroke-opacity", 0.35).attr("y1", mg.t).attr("y2", h - mg.b);
    cur.append("circle").attr("r", 5.5).attr("fill", C.bg).attr("stroke", C.ink).attr("stroke-width", 2.5).attr("filter", "url(#glow-c)");
    const curLab = cur.append("g");
    curLab.append("rect").attr("rx", 3).attr("fill", "#0a0e14").attr("stroke", "#2a3747").attr("height", 20);
    curLab.append("text").attr("fill", C.ink).attr("font-size", 11.5).attr("font-family", "var(--mono)").attr("y", 14).attr("x", 7);
    const hover = g.append("line").attr("stroke", C.ink2).attr("stroke-opacity", 0).attr("y1", mg.t).attr("y2", h - mg.b).style("pointer-events", "none");
    svg.append("rect").attr("x", mg.l).attr("y", mg.t).attr("width", w - mg.l - mg.r).attr("height", h - mg.t - mg.b)
      .attr("fill", "transparent").style("cursor", "crosshair")
      .on("pointermove", (ev) => {
        const [px] = d3.pointer(ev), i = idxFromT(Math.max(0, x.invert(px))), d = ctx.data[i];
        hover.attr("x1", x(d.t)).attr("x2", x(d.t)).attr("stroke-opacity", 0.5);
        showTip(ev, `threshold ${tFmt(d.t)}`, [["Fraud lost", f$(d.fnAmt), C.fraud], [`Review (${fInt(d.al)} alerts)`, f$(d.review), C.amber],
          ["Total cost", f$(d.cost), C.ink], ["Saved vs. no model", (d.saved < 0 ? "−" : "") + f$(Math.abs(d.saved)), C.good]]);
      })
      .on("pointerleave", () => { hover.attr("stroke-opacity", 0); hideTip(); })
      .on("click", (ev) => { const [px] = d3.pointer(ev); setThreshold(idxFromT(Math.max(0, x.invert(px)))); });
    ctx = { svg, w, h, mg, x, y, gy, aLoss, aRev, lTot, base, off, opt, cur, curLab, data: null };
    update(false);
  }
  function update(anim) {
    const { x, y, mg, w, h } = ctx;
    const data = (ctx.data = series());
    const ymax = BASE * 1.35; y.domain([0, ymax]);
    ctx.gy.transition().duration(anim ? 500 : 0).call(d3.axisLeft(y).ticks(5).tickFormat((v) => f$s(v)).tickSize(-(w - mg.l - mg.r)));
    const cl = (v) => Math.min(v, ymax * 1.2);
    const areaL = d3.area().x((d) => x(d.t)).y0(y(0)).y1((d) => y(cl(d.fnAmt))).curve(d3.curveMonotoneX);
    const areaR = d3.area().x((d) => x(d.t)).y0((d) => y(cl(d.fnAmt))).y1((d) => y(cl(d.cost))).curve(d3.curveMonotoneX);
    const line = d3.line().x((d) => x(d.t)).y((d) => y(cl(d.cost))).curve(d3.curveMonotoneX);
    const T = (s) => (anim && DUR ? s.transition().duration(600).ease(d3.easeCubicInOut) : s);
    T(ctx.aLoss).attr("d", areaL(data)); T(ctx.aRev).attr("d", areaR(data)); T(ctx.lTot).attr("d", line(data));
    ctx.base.select("line").attr("y1", y(BASE)).attr("y2", y(BASE));
    ctx.base.select("text").attr("y", y(BASE)).text(`no model · ${f$(BASE)} lost`);
    const offIdx = data.findIndex((d) => d.cost <= ymax);
    ctx.off.text(offIdx > 0 ? `↑ off scale below t=${tFmt(data[offIdx].t)} (${f$s(data[0].cost)} at ${data[0].t})` : "");
    const oi = optIdx(), od = data[oi], ox = x(od.t);
    T(ctx.opt.select("line")).attr("x1", ox).attr("x2", ox);
    T(ctx.opt.select(".od")).attr("transform", `translate(${ox},${y(od.cost)})`);
    const lab = ctx.opt.select("text").text(`COST-OPTIMAL · ${f$(od.cost)}`);
    const right = ox > w - 150;
    T(lab).attr("x", right ? ox - 6 : ox + 6).attr("text-anchor", right ? "end" : "start");
    cursor(!anim);
  }
  function cursor(instant) {
    if (!ctx) return;
    const { x, y, w, mg } = ctx, d = ctx.data[S.ti], cx = x(d.t), cy = y(Math.min(d.cost, BASE * 1.35 * 1.2));
    const T = (s) => (instant || !DUR ? s : s.transition().duration(260).ease(d3.easeCubicOut));
    T(ctx.cur.select("line")).attr("x1", cx).attr("x2", cx);
    T(ctx.cur.select("circle")).attr("cx", cx).attr("cy", cy);
    const txt = ctx.curLab.select("text").text(`${f$(d.cost)}`);
    const tw = txt.node().getComputedTextLength() + 14;
    ctx.curLab.select("rect").attr("width", tw);
    const lx = Math.min(Math.max(cx - tw / 2, mg.l), w - mg.r - tw);
    T(ctx.curLab).attr("transform", `translate(${lx},${Math.max(mg.t, cy - 34)})`);
  }
  return { build, update, cursor };
})();

// ------------------------------------------------------------------ precision / recall vs threshold
CHARTS.prt = (function () {
  let ctx = null;
  function build(anim) {
    const { svg, w, h, defs } = mkSvg($("prtchart"), 262);
    const mg = { t: 16, r: 14, b: 34, l: 40 };
    const x = xThr(mg.l, w - mg.r), y = d3.scaleLinear().domain([0, 1]).range([h - mg.b, mg.t]);
    glowFilter(defs, "glow-p", 2.5);
    const g = svg.append("g");
    g.append("g").attr("class", "axis gridl").attr("transform", `translate(${mg.l},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")).tickSize(-(w - mg.l - mg.r)));
    thrAxis(g, x, h - mg.b, "threshold (√ scale)");
    const lf = g.append("path").attr("fill", "none").attr("stroke", C.ink2).attr("stroke-width", 1.25).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);
    const lp = g.append("path").attr("fill", "none").attr("stroke", C.safe).attr("stroke-width", 2);
    const lr = g.append("path").attr("fill", "none").attr("stroke", C.violet).attr("stroke-width", 2);
    const cur = g.append("g").style("pointer-events", "none");
    cur.append("line").attr("stroke", C.ink).attr("stroke-opacity", 0.35).attr("y1", mg.t).attr("y2", h - mg.b);
    const dp = cur.append("circle").attr("r", 5).attr("fill", C.bg).attr("stroke", C.safe).attr("stroke-width", 2.5).attr("filter", "url(#glow-p)");
    const dr = cur.append("circle").attr("r", 5).attr("fill", C.bg).attr("stroke", C.violet).attr("stroke-width", 2.5).attr("filter", "url(#glow-p)");
    const tp = cur.append("text").attr("fill", C.ink).attr("font-size", 11).attr("font-family", "var(--mono)");
    const tr = cur.append("text").attr("fill", C.ink).attr("font-size", 11).attr("font-family", "var(--mono)");
    const hover = g.append("line").attr("stroke", C.ink2).attr("stroke-opacity", 0).attr("y1", mg.t).attr("y2", h - mg.b);
    svg.append("rect").attr("x", mg.l).attr("y", mg.t).attr("width", w - mg.l - mg.r).attr("height", h - mg.t - mg.b)
      .attr("fill", "transparent").style("cursor", "crosshair")
      .on("pointermove", (ev) => {
        const [px] = d3.pointer(ev), i = idxFromT(Math.max(0, x.invert(px))), d = ctx.data[i];
        hover.attr("x1", x(d.t)).attr("x2", x(d.t)).attr("stroke-opacity", 0.5);
        showTip(ev, `threshold ${tFmt(d.t)}`, [["Precision", pct(d.P), C.safe], ["Recall", pct(d.R), C.violet], ["F1", f3(d.F), C.ink2],
          ["Alerts", fInt(d.al), null]]);
      })
      .on("pointerleave", () => { hover.attr("stroke-opacity", 0); hideTip(); })
      .on("click", (ev) => { const [px] = d3.pointer(ev); setThreshold(idxFromT(Math.max(0, x.invert(px)))); });
    ctx = { x, y, mg, w, h, lp, lr, lf, dp, dr, tp, tr, cur, data: null };
    update(false, anim);
  }
  function update(anim, entrance) {
    const { x, y } = ctx, data = (ctx.data = series());
    const mk = (k) => d3.line().defined((d) => isFinite(d[k])).x((d) => x(d.t)).y((d) => y(d[k])).curve(d3.curveMonotoneX)(data);
    const T = (s) => (anim && DUR ? s.transition().duration(600) : s);
    T(ctx.lp).attr("d", mk("P")); T(ctx.lr).attr("d", mk("R")); T(ctx.lf).attr("d", mk("F"));
    if (entrance) { drawIn(ctx.lp, true); drawIn(ctx.lr, true, 150); }
    cursor(!anim);
  }
  function cursor(instant) {
    if (!ctx) return;
    const { x, y, w, mg } = ctx, d = ctx.data[S.ti], cx = x(d.t);
    const T = (s) => (instant || !DUR ? s : s.transition().duration(260).ease(d3.easeCubicOut));
    T(ctx.cur.select("line")).attr("x1", cx).attr("x2", cx);
    const py = isFinite(d.P) ? y(d.P) : y(1), ry = y(d.R);
    T(ctx.dp).attr("cx", cx).attr("cy", py); T(ctx.dr).attr("cx", cx).attr("cy", ry);
    const right = cx > w - 70, ax = right ? cx - 9 : cx + 9, anc = right ? "end" : "start";
    let a = py, b = ry; if (Math.abs(a - b) < 14) { if (a <= b) { a -= 7; b += 7; } else { a += 7; b -= 7; } }
    T(ctx.tp).attr("x", ax).attr("y", Math.max(mg.t + 8, a) - 7).attr("text-anchor", anc);
    T(ctx.tr).attr("x", ax).attr("y", b + 14).attr("text-anchor", anc);
    ctx.tp.text(pct(d.P, 0)); ctx.tr.text(pct(d.R, 0));
  }
  return { build, update, cursor };
})();

// ------------------------------------------------------------------ gauge
CHARTS.gauge = (function () {
  let ctx = null;
  const A0 = -Math.PI / 2, A1 = Math.PI / 2;
  function build(anim) {
    const el = $("gauge"), { svg, w, h, defs } = mkSvg(el, 178);
    const cx = w / 2, cy = 116, R1 = Math.min(98, w / 2 - 16), R2 = R1 - 17;
    glowFilter(defs, "glow-g", 3);
    const g = svg.append("g").attr("transform", `translate(${cx},${cy})`);
    const arc = (r) => d3.arc().innerRadius(r - 6).outerRadius(r + 6).cornerRadius(6);
    [R1, R2].forEach((r) => g.append("path").attr("d", arc(r)({ startAngle: A0, endAngle: A1 })).attr("fill", C.grid));
    // tick marks
    d3.range(0, 1.0001, 0.25).forEach((v) => {
      const a = A0 + v * Math.PI, s = Math.sin(a), c = -Math.cos(a);
      g.append("line").attr("x1", s * (R1 + 10)).attr("y1", c * (R1 + 10)).attr("x2", s * (R1 + 14)).attr("y2", c * (R1 + 14)).attr("stroke", C.muted);
      if (v === 0 || v === 1 || v === 0.5) g.append("text").attr("x", s * (R1 + 24)).attr("y", c * (R1 + 24) + 4).attr("text-anchor", "middle")
        .attr("fill", C.muted).attr("font-size", 9.5).attr("font-family", "var(--mono)").text(v * 100 + "%");
    });
    const pP = g.append("path").attr("fill", C.safe).attr("filter", "url(#glow-g)");
    const pR = g.append("path").attr("fill", C.violet).attr("filter", "url(#glow-g)");
    const f1 = g.append("text").attr("text-anchor", "middle").attr("y", -10).attr("fill", C.ink).attr("font-size", 30).attr("font-family", "var(--mono)").attr("font-weight", 500);
    g.append("text").attr("text-anchor", "middle").attr("y", 8).attr("fill", C.muted).attr("font-size", 10).attr("font-family", "var(--cond)").attr("font-weight", 600).attr("letter-spacing", ".14em").text("F1 SCORE");
    const lg = svg.append("g").attr("transform", `translate(${cx},${cy + 30})`).attr("font-family", "var(--mono)").attr("font-size", 11.5);
    const lp = lg.append("text").attr("x", -8).attr("text-anchor", "end").attr("fill", C.safe);
    const lr = lg.append("text").attr("x", 8).attr("fill", C.violet);
    ctx = { arc, R1, R2, pP, pR, f1, lp, lr, cur: { P: 0, R: 0 } };
    update(!anim);
  }
  function update(instant) {
    if (!ctx) return;
    const m = met(rowAt(S.ti)), P = isFinite(m.P) ? m.P : 0, R = m.R;
    const { arc, R1, R2 } = ctx, from = Object.assign({}, ctx.cur);
    ctx.cur = { P, R };
    const draw = (p, r) => {
      ctx.pP.attr("d", arc(R1)({ startAngle: A0, endAngle: A0 + Math.max(0.001, p) * Math.PI }));
      ctx.pR.attr("d", arc(R2)({ startAngle: A0, endAngle: A0 + Math.max(0.001, r) * Math.PI }));
    };
    const dur = instant || !DUR ? 0 : 420;
    if (!dur) draw(P, R);
    else d3.select(ctx.pP.node()).interrupt().transition().duration(dur).ease(d3.easeCubicOut)
      .tween("g", () => { const ip = d3.interpolate(from.P, P), ir = d3.interpolate(from.R, R); return (t) => draw(ip(t), ir(t)); });
    tweenNum(ctx.f1.node(), m.F, f3, dur);
    ctx.lp.text(`precision ${pct(m.P, 0)}`); ctx.lr.text(`recall ${pct(m.R, 0)}`);
    const d = P - R;
    $("gauge-sub").textContent = Math.abs(d) < 0.06 ? "Balanced: precision and recall within 6 pts." :
      d > 0 ? "Precision-leaning: few false alarms, more fraud slips through." : "Recall-leaning: catches more fraud, more false alarms.";
  }
  return { build, update };
})();

// ------------------------------------------------------------------ PR & ROC curves
function curveChart(elId, key, opts) {
  let ctx = null;
  function build(anim) {
    const { svg, w, h, defs } = mkSvg($(elId), 280);
    const mg = { t: 14, r: 16, b: 38, l: 44 };
    const x = d3.scaleLinear().domain([0, 1]).range([mg.l, w - mg.r]), y = d3.scaleLinear().domain([0, 1]).range([h - mg.b, mg.t]);
    glowFilter(defs, "glow-" + key, 2.5);
    vgrad(defs, "ga-" + key, C.safe, 0.22, 0);
    const g = svg.append("g");
    g.append("g").attr("class", "axis gridl").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")).tickSize(-(w - mg.l - mg.r)));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h - mg.b})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")).tickSize(4));
    g.append("text").attr("class", "alab").attr("x", w - mg.r).attr("y", h - 6).attr("text-anchor", "end").text(opts.xl);
    g.append("text").attr("class", "alab").attr("x", mg.l).attr("y", mg.t - 3).text(opts.yl).attr("dy", -2);
    opts.ref(g, x, y);
    const clipR = revealClip(defs, "clip-" + key, mg.l, mg.t, w - mg.l - mg.r, h - mg.t - mg.b, anim);
    const plot = g.append("g").attr("clip-path", `url(#clip-${key})`);
    const area = plot.append("path").attr("fill", `url(#ga-${key})`);
    const lines = {}, labels = {};
    ["lr", "xgb"].forEach((m) => {
      lines[m] = plot.append("path").attr("fill", "none").attr("stroke-width", 2).attr("stroke-linejoin", "round");
      labels[m] = g.append("text").attr("font-family", "var(--mono)").attr("font-size", 11);
    });
    const cur = g.append("circle").attr("r", 6).attr("fill", C.bg).attr("stroke-width", 2.5).attr("filter", `url(#glow-${key})`).style("pointer-events", "none");
    const hover = g.append("line").attr("stroke", C.ink2).attr("stroke-opacity", 0).attr("y1", mg.t).attr("y2", h - mg.b);
    svg.append("rect").attr("x", mg.l).attr("y", mg.t).attr("width", w - mg.l - mg.r).attr("height", h - mg.t - mg.b).attr("fill", "transparent")
      .on("pointermove", (ev) => {
        const [px] = d3.pointer(ev), xv = Math.max(0, Math.min(1, x.invert(px)));
        hover.attr("x1", x(xv)).attr("x2", x(xv)).attr("stroke-opacity", 0.45);
        const rows = ["xgb", "lr"].map((m) => [D.models[m].label.split(" (")[0], pct(opts.at(D.models[m][key], xv), 1), m === "xgb" ? C.safe : C.violet]);
        showTip(ev, `${opts.xn} ${pct(xv, 1)}`, rows.map((r) => [r[0] + " " + opts.yn, r[1], r[2]]));
      })
      .on("pointerleave", () => { hover.attr("stroke-opacity", 0); hideTip(); });
    ctx = { x, y, w, h, mg, lines, labels, area, cur };
    update(anim);
  }
  function update(anim) {
    const { x, y, w, mg } = ctx;
    const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1])).curve(opts.curve);
    ["lr", "xgb"].forEach((m) => {
      const on = m === S.model, col = m === "xgb" ? C.safe : C.violet, mo = D.models[m];
      ctx.lines[m].attr("d", line(mo[key])).attr("stroke", col).transition().duration(anim === "toggle" ? 400 : 0).attr("stroke-opacity", on ? 1 : 0.4).attr("stroke-width", on ? 2.25 : 1.5);
      if (on) ctx.lines[m].raise();
      const lp = opts.labelPos(m, x, y, w, mg);
      const ci = mo[key + "_auc_ci"] || mo[(key === "pr" ? "pr" : "roc") + "_auc_ci"];
      ctx.labels[m].attr("x", lp[0]).attr("y", lp[1]).attr("text-anchor", lp[2]).attr("fill", col).attr("fill-opacity", on ? 1 : 0.6)
        .text(`${m === "xgb" ? "XGBoost" : "Logistic"} ${opts.aucName} ${mo[key + "_auc"].toFixed(3)}`);
    });
    const act = D.models[S.model][key];
    ctx.area.attr("d", d3.area().x((d) => x(d[0])).y0(y(0)).y1((d) => y(d[1])).curve(opts.curve)(act)).attr("fill", `url(#ga-${key})`);
    d3.select(ctx.area.node().ownerSVGElement).select(`#ga-${key}`).selectAll("stop").attr("stop-color", S.model === "xgb" ? C.safe : C.violet);
    if (anim === true) { drawIn(ctx.lines.xgb, true); drawIn(ctx.lines.lr, true, 200); }
    ctx.cur.attr("stroke", S.model === "xgb" ? C.safe : C.violet);
    cursor(true);
  }
  function cursor(instant) {
    if (!ctx) return;
    const m = met(rowAt(S.ti)), p = opts.point(m);
    (instant || !DUR ? ctx.cur : ctx.cur.transition().duration(260)).attr("cx", ctx.x(p[0])).attr("cy", ctx.y(p[1]));
  }
  return { build, update, cursor };
}
function stepAt(curve, xv) { // precision at recall xv (max precision among points with recall >= xv)
  let best = 0; for (const [r, p] of curve) if (r >= xv - 1e-9 && p > best) best = p; return best;
}
function rocAt(curve, xv) { let v = 0; for (const [f, t] of curve) { if (f <= xv) v = t; else break; } return v; }
CHARTS.pr = curveChart("prchart", "pr", {
  xl: "recall →", yl: "precision", xn: "recall", yn: "precision", aucName: "AP", curve: d3.curveStepAfter,
  at: stepAt, point: (m) => [m.R, isFinite(m.P) ? m.P : 1],
  labelPos: (m, x, y) => (m === "xgb" ? [x(0.03), y(0.2), "start"] : [x(0.03), y(0.1), "start"]),
  ref: (g, x, y) => {
    const prev = M.fraud_test / M.n_test;
    g.append("line").attr("x1", x(0)).attr("x2", x(1)).attr("y1", y(prev)).attr("y2", y(prev)).attr("stroke", C.fraud).attr("stroke-dasharray", "4 3").attr("stroke-opacity", 0.8);
    g.append("text").attr("x", x(1) - 2).attr("y", y(prev) - 5).attr("text-anchor", "end").attr("fill", C.fraud).attr("font-size", 10).attr("font-family", "var(--mono)")
      .text(`no-skill = fraud rate ${(prev * 100).toFixed(2)}%`);
  },
});
CHARTS.roc = curveChart("rocchart", "roc", {
  xl: "false-positive rate →", yl: "true-positive rate", xn: "FPR", yn: "TPR", aucName: "AUC", curve: d3.curveStepAfter,
  at: rocAt, point: (m) => [m.fpr, m.R],
  labelPos: (m, x, y) => (m === "xgb" ? [x(0.97), y(0.2), "end"] : [x(0.97), y(0.1), "end"]),
  ref: (g, x, y) => {
    g.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(1)).attr("y2", y(1)).attr("stroke", C.muted).attr("stroke-dasharray", "4 3");
    g.append("text").attr("x", x(0.55)).attr("y", y(0.55) + 16).attr("fill", C.muted).attr("font-size", 10).attr("font-family", "var(--mono)").attr("transform", `rotate(${-Math.atan2(y(0) - y(1), x(1) - x(0)) * 180 / Math.PI},${x(0.55)},${y(0.55) + 16})`).text("chance (AUC 0.5)");
  },
});

// ------------------------------------------------------------------ feature importance
CHARTS.imp = (function () {
  function build(anim) {
    const mo = D.models[S.model], isX = S.model === "xgb";
    const rows = mo.importance.slice(0, 12);
    const { svg, w, h } = mkSvg($("impchart"), 12 * 22 + 30);
    const mg = { t: 6, r: 58, b: 22, l: 58 };
    const vmax = d3.max(rows, (d) => Math.abs(d.v));
    const x = d3.scaleLinear().domain([0, vmax]).range([mg.l, w - mg.r]).nice();
    const y = d3.scaleBand().domain(rows.map((d) => d.f)).range([mg.t, h - mg.b]).padding(0.28);
    svg.append("g").attr("class", "axis gridl").attr("transform", `translate(0,${h - mg.b})`)
      .call(d3.axisBottom(x).ticks(4).tickFormat(isX ? d3.format(".0%") : d3.format(".2~f")).tickSize(-(h - mg.t - mg.b)));
    const g = svg.append("g");
    const bars = g.selectAll("g.b").data(rows).join("g").attr("class", "b").attr("transform", (d) => `translate(0,${y(d.f)})`);
    bars.append("text").attr("x", mg.l - 8).attr("y", y.bandwidth() / 2 + 4).attr("text-anchor", "end").attr("fill", C.ink2).attr("font-size", 11.5).attr("font-family", "var(--mono)")
      .text((d) => (d.f === "logamt" ? "log(amt)" : d.f));
    const col = (d) => (isX ? C.safe : d.v > 0 ? C.fraud : C.safe);
    const r = bars.append("rect").attr("x", mg.l).attr("y", 0).attr("height", y.bandwidth()).attr("rx", 3).attr("fill", col)
      .attr("fill-opacity", (d, i) => (isX ? 1 - i * 0.05 : 0.9)).attr("width", anim && DUR ? 0 : (d) => x(Math.abs(d.v)) - mg.l);
    if (anim && DUR) r.transition().delay((d, i) => i * 45).duration(DUR).ease(d3.easeCubicOut).attr("width", (d) => x(Math.abs(d.v)) - mg.l);
    bars.append("text").attr("x", (d) => x(Math.abs(d.v)) + 6).attr("y", y.bandwidth() / 2 + 4).attr("fill", C.ink).attr("font-size", 11).attr("font-family", "var(--mono)")
      .text((d) => (isX ? pct(d.v, 1) : (d.v > 0 ? "+" : "−") + Math.abs(d.v).toFixed(2)));
    bars.append("rect").attr("x", 0).attr("width", w).attr("height", y.bandwidth()).attr("y", 0).attr("fill", "transparent")
      .on("pointermove", (ev, d) => showTip(ev, d.f === "logamt" ? "log(1 + amount)" : d.f, isX ? [["Share of total gain", pct(d.v, 1), C.safe]] :
        [["Standardized coefficient", (d.v > 0 ? "+" : "−") + Math.abs(d.v).toFixed(3), col(d)], [d.v > 0 ? "Higher value → more fraud-like" : "Lower value → more fraud-like", "", null]]))
      .on("pointerleave", hideTip);
    $("imp-note").textContent = isX ? "XGBoost · share of total split gain" : "logistic · standardized coefficients";
    const top3 = rows.slice(0, 3);
    $("imp-callout").textContent = isX
      ? `${top3.map((d) => d.f).join(", ")} carry ${pct(d3.sum(top3, (d) => d.v), 0)} of all split gain. Because V1–V28 are anonymized PCA components, this shows which directions separate fraud — not what they mean in business terms.`
      : `Red bars push the score toward fraud as the feature rises; teal bars push it toward fraud as the feature falls. V14 and V10 rank in the top 3 for both models, which is a useful consistency check.`;
  }
  return { build };
})();

// ------------------------------------------------------------------ segments: treemap + rate list
const RATE_MAX = d3.max(Object.values(D.segments).flat(), (d) => d.rate);
const rateColor = d3.scaleSequentialSqrt(d3.interpolateRgbBasis(["#15212e", "#40243a", "#9a2c40", "#ff4757", "#ffb4ac"])).domain([0, RATE_MAX]);
const segLabel = { cluster: "behavioral cluster", tod: "time of day", amount: "amount tier" };
const todHours = { Overnight: "00–06h", Morning: "06–12h", Afternoon: "12–18h", Evening: "18–24h" };
CHARTS.tree = (function () {
  function build(anim) {
    const rows = D.segments[S.seg];
    const el = $("treemap"), { svg, w, h } = mkSvg(el, 214);
    const root = d3.hierarchy({ children: rows }).sum((d) => d.n).sort((a, b) => b.value - a.value);
    d3.treemap().size([w, h]).paddingInner(2).round(true)(root);
    const nodes = svg.selectAll("g").data(root.leaves()).join("g").attr("transform", (d) => `translate(${d.x0},${d.y0})`);
    const rect = nodes.append("rect").attr("width", (d) => d.x1 - d.x0).attr("height", (d) => d.y1 - d.y0).attr("rx", 3)
      .attr("fill", (d) => rateColor(d.data.rate)).attr("stroke", C.bg).attr("stroke-width", 0);
    if (anim && DUR) {
      rect.attr("opacity", 0).attr("transform", (d) => `translate(${(d.x1 - d.x0) / 2},${(d.y1 - d.y0) / 2}) scale(0.2)`)
        .transition().delay((d, i) => i * 70).duration(DUR).ease(d3.easeBackOut.overshoot(0.9)).attr("opacity", 1).attr("transform", "translate(0,0) scale(1)");
    }
    nodes.each(function (d) {
      const bw = d.x1 - d.x0, bh = d.y1 - d.y0, g = d3.select(this);
      const light = d.data.rate / RATE_MAX > 0.45;
      const ink = light ? "#1a0b0e" : C.ink, ink2 = light ? "rgba(26,11,14,.75)" : C.ink2;
      if (bw > 70 && bh > 40) {
        g.append("text").attr("x", 8).attr("y", 17).attr("fill", ink).attr("font-size", 12).attr("font-family", "var(--cond)").attr("font-weight", 600).text(d.data.key);
        g.append("text").attr("x", 8).attr("y", 36).attr("fill", ink).attr("font-size", bw > 120 && bh > 60 ? 18 : 13).attr("font-family", "var(--mono)").text(pct(d.data.rate, 3));
        if (bh > 64 && bw > 110) g.append("text").attr("x", 8).attr("y", 54).attr("fill", ink2).attr("font-size", 10.5).attr("font-family", "var(--mono)").text(`${fInt(d.data.fraud)} of ${fInt(d.data.n)}`);
      } else if (bw > 34 && bh > 18) {
        g.append("text").attr("x", 5).attr("y", 13).attr("fill", ink).attr("font-size", 10).attr("font-family", "var(--mono)").text(pct(d.data.rate, 2));
      }
      g.append("rect").attr("width", bw).attr("height", bh).attr("fill", "transparent")
        .on("pointerenter", function () { rect.filter((n) => n === d).attr("stroke", C.ink).attr("stroke-width", 1.5); })
        .on("pointermove", (ev) => showTip(ev, d.data.key + (d.data.desc ? " · " + d.data.desc : S.seg === "tod" ? " · " + todHours[d.data.key] : ""), [
          ["Fraud rate", pct(d.data.rate, 3), rateColor(d.data.rate)], ["Lift vs. base", (d.data.rate / BASE_RATE).toFixed(1) + "×", null],
          ["Transactions", fInt(d.data.n), null], ["Fraud cases", fInt(d.data.fraud), null], ["Fraud $", f$(d.data.famt), null]]))
        .on("pointerleave", function () { rect.attr("stroke-width", 0); hideTip(); });
    });
    // ranked list
    const list = $("ratelist"); list.replaceChildren();
    const sorted = rows.slice().sort((a, b) => b.rate - a.rate);
    sorted.forEach((d, i) => {
      const r = document.createElement("div"); r.className = "rl";
      const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = d.key;
      const sm = document.createElement("small"); sm.textContent = d.desc || (S.seg === "tod" ? todHours[d.key] : ""); nm.append(sm);
      const bar = document.createElement("span"); bar.className = "bar"; const bi = document.createElement("span");
      bi.style.background = rateColor(d.rate); bi.style.width = "0%"; bar.append(bi);
      const rv = document.createElement("span"); rv.className = "r"; rv.textContent = pct(d.rate, 3);
      const lx = document.createElement("span"); lx.className = "x"; lx.textContent = (d.rate / BASE_RATE).toFixed(1) + "×";
      r.append(nm, bar, rv, lx); list.append(r);
      requestAnimationFrame(() => requestAnimationFrame(() => { bi.style.width = ((d.rate / RATE_MAX) * 100).toFixed(1) + "%"; }));
      r.addEventListener("pointermove", (ev) => showTip(ev, d.key, [["Fraud rate", pct(d.rate, 3), rateColor(d.rate)], ["Share of volume", pct(d.n / M.n_raw, 1), null], ["Share of fraud", pct(d.fraud / M.fraud_raw, 1), null]]));
      r.addEventListener("pointerleave", hideTip);
    });
    const top = sorted[0];
    let txt = `${top.key}${S.seg === "tod" ? " (" + todHours[top.key] + ")" : ""} holds ${pct(top.n / M.n_raw, 1)} of transactions but ${pct(top.fraud / M.fraud_raw, 1)} of fraud: a ${(top.rate / BASE_RATE).toFixed(1)}× lift over the ${pct(BASE_RATE, 3)} base rate.`;
    if (S.seg === "cluster") txt += ` Caveat: k-means silhouette is only ${M.kmeans.silhouette[M.kmeans.k]}, so these are loose pattern groupings, not natural customer types. Fraud labels were not used to build them.`;
    if (S.seg === "amount") txt += ` The “U” shape (micro and large tiers both elevated) is why a single amount cut-off makes a poor rule.`;
    $("seg-callout").textContent = txt;
    $("seg-note").textContent = `all ${fInt(M.n_raw)} transactions · base rate ${pct(BASE_RATE, 3)}`;
  }
  return { build };
})();

// ------------------------------------------------------------------ heatmap (hour × amount tier)
CHARTS.heat = (function () {
  const MIN_N = 300;
  function build(anim) {
    const el = $("heatmap"), { svg, w, defs } = mkSvg(el, 250), h = 250;
    const mg = { t: 8, r: 6, b: 24, l: 62 }, topH = 58, gap = 8;
    const tiers = ["<$10", "$10–50", "$50–200", "$200–1K", "$1K+"];
    const x = d3.scaleBand().domain(d3.range(24)).range([mg.l, w - mg.r]).paddingInner(0.08);
    const y = d3.scaleBand().domain(d3.range(5)).range([mg.t + topH + gap, h - mg.b]).paddingInner(0.08);
    const cellRates = D.heatmap.filter((d) => d.n >= MIN_N).map((d) => d.fraud / d.n);
    const cmax = d3.quantile(cellRates.sort(d3.ascending), 0.97);
    const col = d3.scaleSequentialSqrt(d3.interpolateRgbBasis(["#131c27", "#40243a", "#9a2c40", "#ff4757", "#ffb4ac"])).domain([0, cmax]).clamp(true);
    // marginal: hourly fraud rate bars
    const hr = D.hourly.map((d) => ({ h: d.h, n: d.n, fraud: d.fraud, rate: d.fraud / d.n }));
    const ym = d3.scaleLinear().domain([0, d3.max(hr, (d) => d.rate)]).range([mg.t + topH, mg.t + 10]);
    svg.append("text").attr("class", "alab").attr("x", mg.l - 8).attr("y", mg.t + 22).attr("text-anchor", "end").text("rate");
    svg.append("text").attr("class", "alab").attr("x", mg.l - 8).attr("y", mg.t + 34).attr("text-anchor", "end").text("by hour");
    const mb = svg.append("g").selectAll("rect").data(hr).join("rect").attr("x", (d) => x(d.h)).attr("width", x.bandwidth()).attr("rx", 2)
      .attr("fill", (d) => col(d.rate)).attr("y", anim && DUR ? mg.t + topH : (d) => ym(d.rate)).attr("height", anim && DUR ? 0 : (d) => mg.t + topH - ym(d.rate));
    if (anim && DUR) mb.transition().delay((d) => d.h * 25).duration(DUR).ease(d3.easeCubicOut).attr("y", (d) => ym(d.rate)).attr("height", (d) => mg.t + topH - ym(d.rate));
    mb.on("pointermove", (ev, d) => showTip(ev, `hour ${String(d.h).padStart(2, "0")}:00–${String(d.h).padStart(2, "0")}:59`, [
      ["Fraud rate", pct(d.rate, 3), col(d.rate)], ["Lift vs. base", (d.rate / BASE_RATE).toFixed(1) + "×", null], ["Transactions", fInt(d.n), null], ["Fraud cases", fInt(d.fraud), null]]))
      .on("pointerleave", hideTip);
    // grid
    svg.append("g").selectAll("text").data(tiers).join("text").attr("x", mg.l - 8).attr("y", (d, i) => y(i) + y.bandwidth() / 2 + 4).attr("text-anchor", "end")
      .attr("fill", C.ink2).attr("font-size", 10.5).attr("font-family", "var(--mono)").text((d) => d);
    const cells = svg.append("g").selectAll("rect").data(D.heatmap).join("rect")
      .attr("x", (d) => x(d.h)).attr("y", (d) => y(d.t)).attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 2)
      .attr("fill", (d) => col(d.fraud / d.n)).attr("fill-opacity", (d) => (d.n < MIN_N ? 0.35 : 1))
      .attr("stroke", (d) => (d.n < MIN_N ? C.line : "none")).attr("stroke-dasharray", "2 2");
    if (anim && DUR) cells.attr("opacity", 0).transition().delay((d) => d.h * 28 + d.t * 40).duration(450).attr("opacity", 1);
    cells.on("pointermove", function (ev, d) {
      d3.select(this).attr("stroke", C.ink).attr("stroke-dasharray", null);
      showTip(ev, `${String(d.h).padStart(2, "0")}h · ${D.amt_tiers[d.t]}`, [["Fraud rate", pct(d.fraud / d.n, 3), col(d.fraud / d.n)], ["Fraud cases", fInt(d.fraud), null],
        ["Transactions", fInt(d.n), null], d.n < MIN_N ? ["Small sample: rate unstable", "", null] : ["Lift vs. base", (d.fraud / d.n / BASE_RATE).toFixed(1) + "×", null]]);
    }).on("pointerleave", function (ev, d) { d3.select(this).attr("stroke", d.n < MIN_N ? C.line : "none").attr("stroke-dasharray", "2 2"); hideTip(); });
    svg.append("g").selectAll("text").data([0, 3, 6, 9, 12, 15, 18, 21]).join("text").attr("x", (d) => x(d) + x.bandwidth() / 2).attr("y", h - 8)
      .attr("text-anchor", "middle").attr("fill", C.muted).attr("font-size", 10).attr("font-family", "var(--mono)").text((d) => String(d).padStart(2, "0") + "h");
    // legend
    const lg = $("heatlegend"); lg.replaceChildren();
    const lw = Math.min(260, w - 20), ls = d3.select(lg).append("svg").attr("viewBox", `0 0 ${w} 30`).attr("height", 30);
    const gid = "hl-grad", gr = ls.append("defs").append("linearGradient").attr("id", gid);
    d3.range(0, 1.01, 0.1).forEach((t) => gr.append("stop").attr("offset", t * 100 + "%").attr("stop-color", col(t * cmax)));
    ls.append("rect").attr("x", mg.l).attr("y", 2).attr("width", lw).attr("height", 8).attr("rx", 2).attr("fill", `url(#${gid})`);
    const lx = d3.scaleLinear().domain([0, cmax]).range([mg.l, mg.l + lw]);
    ls.append("g").attr("class", "axis").attr("transform", "translate(0,11)").call(d3.axisBottom(lx).ticks(4).tickFormat((v) => (v * 100).toFixed(1) + "%").tickSize(3)).select(".domain").remove();
    ls.append("text").attr("x", mg.l + lw + 10).attr("y", 10).attr("fill", C.muted).attr("font-size", 10).attr("font-family", "var(--sans)").text(`faded = under ${MIN_N} tx`);
    const peak = hr.slice().sort((a, b) => b.rate - a.rate)[0];
    const night = hr.filter((d) => d.h < 6), nightRate = d3.sum(night, (d) => d.fraud) / d3.sum(night, (d) => d.n);
    const byCount = hr.filter((d) => d.h !== peak.h).sort((a, b) => b.fraud - a.fraud)[0];
    const hh = (h) => String(h).padStart(2, "0") + "h";
    $("heat-callout").textContent = `Fraud rate peaks at ${hh(peak.h)} (${pct(peak.rate, 2)}, ${(peak.rate / BASE_RATE).toFixed(0)}× base): ${peak.fraud} frauds landing when legitimate volume is near its low. ${hh(byCount.h)} has a similar count (${byCount.fraud}) but only ${pct(byCount.rate, 2)}, because daytime volume is ${(byCount.n / peak.n).toFixed(0)}× higher. Overnight as a block runs at ${pct(nightRate, 2)}.`;
  }
  return { build };
})();

// ------------------------------------------------------------------ amount distribution
CHARTS.amt = (function () {
  function build(anim) {
    const el = $("amtchart"), { svg, w, h, defs } = mkSvg(el, 262);
    const mg = { t: 22, r: 14, b: 36, l: 42 };
    const E = D.amount_hist.edges.map((e, i) => (i === 0 ? 0.3 : e));
    const nl = d3.sum(D.amount_hist.legit), nf = d3.sum(D.amount_hist.fraud);
    const bins = D.amount_hist.legit.map((c, i) => ({ i, a: E[i], b: E[i + 1], ra: D.amount_hist.edges[i], rb: D.amount_hist.edges[i + 1], l: c / nl, f: D.amount_hist.fraud[i] / nf, lc: c, fc: D.amount_hist.fraud[i] }));
    const x = d3.scaleLog().domain([0.3, 26000]).range([mg.l, w - mg.r]);
    const y = d3.scaleLinear().domain([0, d3.max(bins, (d) => Math.max(d.l, d.f)) * 1.1]).range([h - mg.b, mg.t]);
    vgrad(defs, "g-al", C.safe, 0.45, 0.04); vgrad(defs, "g-af", C.fraud, 0.6, 0.06);
    revealClip(defs, "clip-amt", mg.l, mg.t, w - mg.l - mg.r, h - mg.t - mg.b, anim);
    svg.append("g").attr("class", "axis gridl").attr("transform", `translate(${mg.l},0)`).call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(".0%")).tickSize(-(w - mg.l - mg.r)));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - mg.b})`)
      .call(d3.axisBottom(x).tickValues([0.3, 1, 10, 100, 1000, 10000]).tickFormat((v) => (v < 0.5 ? "$0" : "$" + d3.format("~s")(v))).tickSize(4));
    svg.append("text").attr("class", "alab").attr("x", w - mg.r).attr("y", h - 4).attr("text-anchor", "end").text("transaction amount (log)");
    svg.append("text").attr("class", "alab").attr("x", mg.l).attr("y", mg.t - 8).text("share of class");
    const pts = (k) => bins.flatMap((d) => [[d.a, d[k]], [d.b, d[k]]]);
    const area = d3.area().x((d) => x(d[0])).y0(y(0)).y1((d) => y(d[1]));
    const line = d3.line().x((d) => x(d[0])).y((d) => y(d[1]));
    const plot = svg.append("g").attr("clip-path", "url(#clip-amt)");
    plot.append("path").attr("d", area(pts("l"))).attr("fill", "url(#g-al)");
    plot.append("path").attr("d", line(pts("l"))).attr("fill", "none").attr("stroke", C.safe).attr("stroke-width", 1.75);
    plot.append("path").attr("d", area(pts("f"))).attr("fill", "url(#g-af)");
    plot.append("path").attr("d", line(pts("f"))).attr("fill", "none").attr("stroke", C.fraud).attr("stroke-width", 2);
    const A = D.amount_stats;
    [[A.fraud_median, C.fraud, "fraud median", 0], [A.legit_median, C.safe, "legit median", 1]].forEach(([v, c, lab, k]) => {
      svg.append("line").attr("x1", x(v)).attr("x2", x(v)).attr("y1", mg.t).attr("y2", h - mg.b).attr("stroke", c).attr("stroke-dasharray", "3 3").attr("stroke-opacity", 0.8);
      svg.append("text").attr("x", x(v) + (k ? 5 : -5)).attr("y", mg.t + 8).attr("text-anchor", k ? "start" : "end").attr("fill", c).attr("font-size", 10.5).attr("font-family", "var(--mono)").text(`${lab} ${f$2(v)}`);
    });
    const hover = svg.append("rect").attr("fill", C.ink).attr("fill-opacity", 0).attr("y", mg.t).attr("height", h - mg.t - mg.b).style("pointer-events", "none");
    svg.append("rect").attr("x", mg.l).attr("y", mg.t).attr("width", w - mg.l - mg.r).attr("height", h - mg.t - mg.b).attr("fill", "transparent")
      .on("pointermove", (ev) => {
        const [px] = d3.pointer(ev), v = x.invert(px), d = bins.find((b) => v >= b.a && v < b.b) || bins[bins.length - 1];
        hover.attr("x", x(d.a)).attr("width", Math.max(1, x(d.b) - x(d.a))).attr("fill-opacity", 0.06);
        const rng = d.i === 0 ? "$0.00–0.49" : `${f$2(d.ra)}–${f$2(d.rb)}`;
        showTip(ev, rng, [["Legit share", `${pct(d.l, 1)} (${fInt(d.lc)})`, C.safe], ["Fraud share", `${pct(d.f, 1)} (${fInt(d.fc)})`, C.fraud],
          ["Fraud rate in bin", d.lc + d.fc ? pct(d.fc / (d.lc + d.fc), 2) : "—", null]]);
      })
      .on("pointerleave", () => { hover.attr("fill-opacity", 0); hideTip(); });
    $("amt-callout").textContent = `Median fraud is ${f$2(A.fraud_median)} vs. ${f$2(A.legit_median)} for legitimate spend, and ${fInt(A.fraud_under_1)} of ${M.fraud_raw} frauds (${pct(A.fraud_under_1 / M.fraud_raw, 0)}) are $1 or less, a pattern often linked to card testing. Fraud also has a heavier tail (mean ${f$(A.fraud_mean)} vs. ${f$(A.legit_mean)}).`;
  }
  return { build };
})();

// ------------------------------------------------------------------ live replay feed
const Feed = (function () {
  const list = $("feedlist"), rows = D.feed; let k = 0, timer = null, paused = false;
  const st = { n: 0, tp: 0, fp: 0, fn: 0 };
  const hhmm = (T) => { const d = Math.floor(T / 86400) + 1, s = T % 86400; return `D${d} ${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`; };
  function push(animate = true) {
    const [id, T, amt, hour, px, pl, cls, clu] = rows[k % rows.length]; k++;
    const p = S.model === "xgb" ? px : pl, flag = p >= TH[S.ti];
    st.n++; if (flag && cls) st.tp++; else if (flag) st.fp++; else if (cls) st.fn++;
    const el = document.createElement("div");
    el.className = "tx" + (flag ? " hot" : "") + (!flag && cls ? " missed" : "");
    if (!animate) el.style.animation = "none";
    const t = document.createElement("span"); t.className = "t"; t.textContent = hhmm(T);
    const m = document.createElement("span"); m.className = "m";
    const a = document.createElement("div"); a.className = "a"; a.textContent = f$2(amt);
    const c = document.createElement("div"); c.className = "c"; c.textContent = `#${id} · ${clu}`;
    m.append(a, c);
    const sc = document.createElement("span"); sc.className = "sc";
    const bar = document.createElement("span"); bar.className = "bar"; const bi = document.createElement("span"); bi.style.width = (Math.sqrt(p) * 100).toFixed(1) + "%"; bar.append(bi);
    const pv = document.createElement("span"); pv.className = "p"; pv.textContent = p < 0.001 ? "<0.001" : p.toFixed(3);
    sc.append(pv, bar);
    const ch = document.createElement("span");
    if (flag && cls) { ch.className = "chip flag"; ch.textContent = "Caught"; }
    else if (flag) { ch.className = "chip fp"; ch.textContent = "False alarm"; }
    else if (cls) { ch.className = "chip miss"; ch.textContent = "Missed"; }
    else { ch.className = "chip pass"; ch.textContent = "Pass"; }
    el.append(t, m, sc, ch);
    list.prepend(el);
    while (list.children.length > 28) list.lastChild.remove();
    $("f-n").textContent = fInt(st.n); $("f-tp").textContent = st.tp; $("f-fp").textContent = st.fp; $("f-fn").textContent = st.fn;
  }
  function loop() { timer = setTimeout(() => { if (!paused && !document.hidden) push(); loop(); }, 400 + Math.random() * 400); }
  function start() {
    for (let i = 0; i < 12; i++) push(false);
    $("feedfoot").textContent = `Replays ${fInt(rows.length)} held-out test transactions: all ${M.fraud_test} test frauds plus ${fInt(rows.length - M.fraud_test)} random legitimate ones. Fraud is oversampled to ${pct(M.feed_fraud_share, 1)} of the stream (vs. ${pct(BASE_RATE, 2)} in reality) so alerts are visible. Scores are real out-of-sample model outputs; decisions use the current threshold.`;
    if (!reduce) loop();
    else { paused = true; setPausedUI(); }
    $("pause").addEventListener("click", () => {
      paused = !paused; setPausedUI();
      if (!paused && !timer) loop();
    });
  }
  function setPausedUI() { $("pause").textContent = paused ? "Resume" : "Pause"; $("live").classList.toggle("paused", paused); $("live").lastChild.textContent = paused ? "Replay paused" : "Live replay"; }
  return { start };
})();

// ------------------------------------------------------------------ method notes
function perfCallout() {
  const x = D.models.xgb, l = D.models.lr, ci = (m) => m.pr_auc_ci.map((v) => v.toFixed(3)).join("–");
  $("pr-callout").textContent = `Average precision (PR-AUC): XGBoost ${x.pr_auc.toFixed(3)} [${ci(x)}] vs. logistic ${l.pr_auc.toFixed(3)} [${ci(l)}]. A no-skill model scores ${(M.fraud_test / M.n_test).toFixed(4)}, the fraud rate. The 95% intervals overlap a little, so with ${M.fraud_test} test frauds the gap is likely but not certain. The ring marks the current threshold.`;
}
function methodNotes() {
  const x = D.models.xgb, l = D.models.lr, oi = optIdx("xgb", 10), ti = testOptIdx("xgb", 10);
  const om = met(rowAt(oi, "xgb"), 10), tm = met(rowAt(ti, "xgb"), 10);
  const el = $("method");
  el.innerHTML = `
  <div><h3>Data</h3>
    <p><b>${fInt(M.n_raw)}</b> card transactions over ${Math.round(M.hours_span)} hours (Sep 2013), <b>${M.fraud_raw}</b> labeled fraud (${pct(BASE_RATE, 3)}). Features V1–V28 are PCA components released without meaning; only <code>Time</code> and <code>Amount</code> are raw.</p>
    <p>${fInt(M.dups_removed)} exact duplicate rows were dropped <b>before</b> splitting so no row sits in both train and test, leaving ${fInt(M.n_model)} rows (${M.fraud_model} fraud) for modeling. Segment charts use all ${fInt(M.n_raw)} rows.</p>
    <p>The dataset does not document its currency (likely EUR); amounts are shown with “$” for readability.</p></div>
  <div><h3>Models &amp; validation</h3>
    <p>70/30 stratified split, seed ${M.seed}: train ${fInt(M.n_train)} (${M.fraud_train} fraud), test ${fInt(M.n_test)} (${M.fraud_test} fraud). Inputs: V1–V28, log(1+amount), hour.</p>
    <p><b>XGBoost</b>: depth 4, 300 trees, learning rate 0.05. <b>Logistic regression</b>: standardized inputs, L2, C = 0.1. No resampling or class weights, so scores stay on a natural probability scale and the threshold does the balancing.</p>
    <p>Test-set PR-AUC: XGBoost ${x.pr_auc.toFixed(3)} [${x.pr_auc_ci.map((v) => v.toFixed(3)).join("–")}], logistic ${l.pr_auc.toFixed(3)} [${l.pr_auc_ci.map((v) => v.toFixed(3)).join("–")}] (95% bootstrap, 300 resamples). With only ${M.fraud_test} test frauds, intervals are wide. Treat the third decimal as noise.</p></div>
  <div class="assume"><h3>Cost-model assumptions</h3>
    <p><b>Missed fraud (FN)</b> costs its full transaction amount: no chargeback recovery.</p>
    <p><b>Every alert (TP or FP)</b> costs a fixed review / customer-friction fee, default <b>$10</b> (adjust it in the top bar, $1–25). Caught fraud loses $0.</p>
    <p><b>Threshold selection</b>: the cost-optimal threshold is picked on 5-fold out-of-fold predictions from the <b>training</b> set, then applied unchanged to test. At $10 that is t = ${tFmt(om.t)} → ${f$(om.cost)} test cost. The test set’s own minimum (t = ${tFmt(tm.t)}, ${f$(tm.cost)}) is shown for reference only; choosing it would be tuning on test data.</p>
    <p>Not modeled: attrition from declined good customers, label delay, fraud-ring effects, operations capacity.</p></div>
  <div><h3>Engineered segments</h3>
    <p><b>Time of day</b>: hour = (Time mod 86,400) / 3,600, bucketed 00–06 / 06–12 / 12–18 / 18–24. Clock alignment is inferred from the volume trough, not documented.</p>
    <p><b>Amount tiers</b>: fixed cut-points set near the quantiles (25th pct ${f$2(M.amount_quantiles["0.25"])}, median ${f$2(M.amount_quantiles["0.5"])}, 90th ${f$2(M.amount_quantiles["0.9"])}, 99th ${f$2(M.amount_quantiles["0.99"])}).</p>
    <p><b>Pattern clusters</b>: k-means on standardized V1–V28 (labels unused), k chosen from 4–6 by silhouette (k = ${M.kmeans.k}, silhouette ${M.kmeans.silhouette[M.kmeans.k]}). Each is named by its two most extreme centroid components. None of these are merchant categories or customer types.</p></div>
  <div><h3>Why PR-AUC, not accuracy</h3>
    <p>A model that flags nothing is ${pct(1 - BASE_RATE, 2)} accurate and catches zero fraud. ROC-AUC is also flattering here: its false-positive rate divides by ~85K legitimate transactions, so a few hundred false alarms barely move it.</p>
    <p>Precision–recall asks the operational question, “of the alerts we raise, how many are real?”, and a no-skill model scores only the base rate (${(M.fraud_test / M.n_test).toFixed(4)}).</p></div>
  <div><h3>Limitations</h3>
    <p>Two days of data from one issuer, 2013. A random split is used rather than an out-of-time split, so performance on future weeks would likely be lower. Hyperparameters were set once and not tuned. Pipeline: Python 3 · pandas · scikit-learn · XGBoost, exported to static JSON. Nothing is retrained in the browser.</p></div>`;
}

// ------------------------------------------------------------------ wiring
function buildAll(anim) {
  CHARTS.cost.build(anim); CHARTS.prt.build(anim); CHARTS.gauge.build(anim);
  CHARTS.pr.build(anim); CHARTS.roc.build(anim); CHARTS.imp.build(anim);
  CHARTS.tree.build(anim); CHARTS.heat.build(anim); CHARTS.amt.build(anim);
}
document.querySelectorAll("[data-model]").forEach((b) => b.addEventListener("click", () => {
  if (S.model === b.dataset.model) return;
  const wasOpt = S.ti === optIdx();
  S.model = b.dataset.model;
  document.querySelectorAll("[data-model]").forEach((x) => x.setAttribute("aria-pressed", x === b));
  if (wasOpt) S.ti = optIdx();
  thr.value = Math.round(Math.sqrt(TH[S.ti]) * 1000);
  kpisModel(false); placeOptPin();
  CHARTS.cost.update(true); CHARTS.prt.update(true); CHARTS.pr.update("toggle"); CHARTS.roc.update("toggle"); CHARTS.imp.build(true);
  $("perf-note").textContent = `active: ${D.models[S.model].label}`;
  updateThresholdViews(false);
}));
document.querySelectorAll("[data-seg]").forEach((b) => b.addEventListener("click", () => {
  S.seg = b.dataset.seg;
  document.querySelectorAll("[data-seg]").forEach((x) => x.setAttribute("aria-pressed", x === b));
  CHARTS.tree.build(true);
}));
$("cost").addEventListener("input", (e) => {
  const wasOpt = S.ti === optIdx();
  S.cost = +e.target.value;
  $("costout").textContent = "$" + S.cost; $("cc-c").textContent = "$" + S.cost;
  if (wasOpt) { S.ti = optIdx(); thr.value = Math.round(Math.sqrt(TH[S.ti]) * 1000); }
  kpisModel(false); placeOptPin(); CHARTS.cost.update(true); updateThresholdViews(false);
});

S.ti = optIdx();
thr.value = Math.round(Math.sqrt(TH[S.ti]) * 1000);
kpisStatic(); kpisModel(true); sliderSetup(); placeOptPin(); methodNotes(); perfCallout();
$("perf-note").textContent = `active: ${D.models[S.model].label}`;
buildAll(true);
updateThresholdViews(true);
Feed.start();
let lastW = document.body.clientWidth, rt;
new ResizeObserver(() => {
  clearTimeout(rt);
  rt = setTimeout(() => { const nw = document.body.clientWidth; if (Math.abs(nw - lastW) > 4) { lastW = nw; buildAll(false); updateThresholdViews(true); } }, 150);
}).observe(document.body);
})();
