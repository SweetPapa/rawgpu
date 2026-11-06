// rawgpu Studio: the recipe gallery, the run form, and a live view of each run.
"use strict";

// ---------------------------------------------------------------------------------------------- helpers
const $ = (s, el = document) => el.querySelector(s);
const NS = "http://www.w3.org/2000/svg";

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "value") e.value = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : String(c));
  return e;
}

function s(tag, attrs, ...kids) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) e.setAttribute(k, v);
  for (const c of kids.flat(Infinity)) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

function icon(name, cls) {
  const svgEl = s("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", class: cls });
  svgEl.innerHTML = ICONS[name] || "";
  return svgEl;
}

const ICONS = {
  gpt: '<path d="M2.5 4h11M2.5 7h11M2.5 10h7M2.5 13h4"/>',
  jepa: '<circle cx="4" cy="11" r="2"/><path d="M6 10c2-1 3-4 5.5-5.5"/><circle cx="12" cy="4.5" r="1.6" stroke-dasharray="1.4 1.2"/><path d="M8.5 13.5h5"/>',
  probe: '<path d="M2 3h12l-4.5 5.5V13L6.5 14V8.5z"/>',
  critic: '<path d="M2.5 11a5.5 5.5 0 0 1 11 0"/><path d="M8 11l2.6-3.4"/><path d="M2.5 13.5h11"/>',
  eg2_contrastive: '<circle cx="5" cy="8" r="2.5"/><circle cx="11" cy="8" r="2.5"/><path d="M7.5 8h1"/>',
  check: '<path d="M3 8.5l3 3 7-7"/>',
  x: '<path d="M4 4l8 8M12 4l-8 8"/>',
  pause: '<path d="M5.5 3.5v9M10.5 3.5v9"/>',
  clock: '<circle cx="8" cy="8" r="5.5"/><path d="M8 5v3.2l2 1.3"/>',
  spin: '<path d="M8 2.5a5.5 5.5 0 1 1-5.5 5.5"/>',
  alert: '<path d="M8 2 14.5 13.5h-13z"/><path d="M8 6.5v3.2m0 2v.1"/>',
  info: '<circle cx="8" cy="8" r="5.5"/><path d="M8 7.5v3.5M8 5.3v.1"/>',
  stop: '<rect x="4" y="4" width="8" height="8" rx="1.5"/>',
  play: '<path d="M5 3.5v9l7-4.5z"/>',
  copy: '<rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 10.5V4a1 1 0 0 1 1-1h6.5"/>',
  trash: '<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8.5h5.6l.7-8.5"/>',
  back: '<path d="M10 3 5 8l5 5"/>',
  therm: '<path d="M8 2.5a1.5 1.5 0 0 1 1.5 1.5v5.3a3 3 0 1 1-3 0V4A1.5 1.5 0 0 1 8 2.5z"/>',
};

const STATES = {
  running: ["spin", "Training"], queued: ["clock", "Queued"], done: ["check", "Done"], failed: ["x", "Failed"],
  stopped: ["pause", "Stopped"], cancelled: ["x", "Cancelled"],
};

function stateEl(st, pill) {
  const [ic, label] = STATES[st] || ["info", st];
  return h("span", { class: (pill ? "pill " : "state ") + st }, icon(ic, st === "running" ? "spin" : ""), label);
}

function glyph(recipe, cls) {
  return h("span", { class: "glyph " + (cls || "") }, icon(recipe));
}

function fmt(v, spec) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof v !== "number") return String(v);
  if (spec) {
    const m = /^(,)?\.(\d+)([f%])$/.exec(spec);
    if (m) {
      const d = +m[2];
      if (m[3] === "%") return (v * 100).toFixed(d) + "%";
      return m[1] ? v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : v.toFixed(d);
    }
  }
  const a = Math.abs(v);
  if (a >= 10000) return Math.round(v).toLocaleString();
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(a >= 10 ? 1 : 2);
  if (a === 0) return "0";
  if (a >= 0.001) return v.toPrecision(3);
  return v.toExponential(2);
}

function dur(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return sec + " s";
  if (sec < 3600) return Math.floor(sec / 60) + " min " + (sec % 60 ? (sec % 60) + " s" : "");
  return Math.floor(sec / 3600) + " h " + Math.round((sec % 3600) / 60) + " min";
}

function ago(t) {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return Math.round(d / 60) + " min ago";
  if (d < 86400) return Math.round(d / 3600) + " h ago";
  return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

async function api(path, body, method) {
  const opt = body !== undefined || method ? { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) } : {};
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.hidden = true), 3500);
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = ["--s1", "--s2", "--s3", "--s4"];

// ------------------------------------------------------------------------------------------------ charts

function tickFmt(ticks, spec) {
  if (spec && /%$/.test(spec)) return (v) => fmt(v, ".0%");
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9) + (Math.abs(step / Math.pow(10, Math.floor(Math.log10(step)))) % 1 ? 1 : 0)));
  return (v) => (Math.abs(v) >= 10000 ? Math.round(v).toLocaleString() : v.toFixed(d));
}

function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

// A line chart: one y axis, thin lines, end labels, a crosshair with a tooltip of every series at that x.
class LineChart {
  constructor(el, opt) {
    this.el = el;
    this.opt = Object.assign({ height: 220, xLabel: "step" }, opt);
    this.data = {};
    this.hoverX = null;
    this.box = h("div", { class: "cbox" });
    this.tip = h("div", { class: "tip", hidden: true });
    const legend = this.opt.series.length > 1 ? h("div", { class: "legend" }, this.opt.series.map((sr, i) =>
      h("span", {}, h("i", { class: sr.dash ? "dash" : "", style: `background:var(${sr.color || SERIES[i % 4]});color:var(${sr.color || SERIES[i % 4]})` }), sr.label))) : null;
    el.append(legend || "", this.box);
    this.box.append(this.tip);
    new ResizeObserver(() => this.draw()).observe(this.box);
  }

  update(data) {
    this.data = data;
    this.draw();
  }

  draw() {
    const o = this.opt, W = this.box.clientWidth || 400, H = o.height;
    const series = o.series.map((sr, i) => ({ ...sr, color: css(sr.color || SERIES[i % 4]), pts: (this.data[sr.key] || []).filter((p) => p[1] !== null && isFinite(p[1])) }));
    const all = series.flatMap((sr) => sr.pts);
    [...this.box.querySelectorAll("svg, .empty")].forEach((e) => e.remove());
    if (!all.length) {
      this.box.style.height = H + "px";
      this.box.append(h("div", { class: "empty" }, o.empty || "Waiting for the first numbers…"));
      return;
    }
    this.box.style.height = "";
    const logY = o.y === "log" && all.every((p) => p[1] > 0);
    const ty = (v) => (logY ? Math.log10(v) : v);
    let [x0, x1] = o.xDomain || [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
    if (x1 === x0) { x0 -= 1; x1 += 1; }
    let [y0, y1] = o.yDomain || [Math.min(...all.map((p) => ty(p[1]))), Math.max(...all.map((p) => ty(p[1])))];
    if (!o.yDomain) {
      const pad = (y1 - y0) * 0.08 || Math.abs(y1) * 0.1 || 1;
      y0 -= pad; y1 += pad;
      if (!logY && y0 < 0 && Math.min(...all.map((p) => p[1])) >= 0) y0 = 0;
    }
    const multi = series.length > 1 && !o.noEndLabels;
    const ml = 46, mr = multi ? 92 : 14, mt = 8, mb = 24;
    const X = (v) => ml + ((v - x0) / (x1 - x0)) * (W - ml - mr);
    const Y = (v) => mt + (1 - (ty(v) - y0) / (y1 - y0)) * (H - mt - mb);
    const Yt = (t) => mt + (1 - (t - y0) / (y1 - y0)) * (H - mt - mb);
    const g = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": o.title || "chart" });
    const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
    let yt = niceTicks(y0, y1, 4);
    if (logY) {  // powers of ten, or 1-2-5 within them when the range is narrow
      yt = [];
      for (let e = Math.floor(y0); e <= Math.ceil(y1); e++) for (const m of y1 - y0 < 2.2 ? [1, 2, 5] : [1]) yt.push(e + Math.log10(m));
    }
    const yf = tickFmt(yt, o.fmt);
    for (const t of yt) {
      if (t < y0 || t > y1) continue;
      grid.append(s("line", { x1: ml, x2: W - mr, y1: Yt(t), y2: Yt(t) }));
      axis.append(s("text", { x: ml - 7, y: Yt(t) + 3.5, "text-anchor": "end" }, logY ? fmt(Math.pow(10, t)) : yf(t)));
    }
    const xt = niceTicks(x0, x1, Math.max(2, Math.floor((W - ml - mr) / 90))), xf = tickFmt(xt);
    for (const t of xt) {
      if (t < x0 || t > x1) continue;
      axis.append(s("text", { x: X(t), y: H - 6, "text-anchor": "middle" }, o.xfmt ? o.xfmt(t) : xf(t)));
    }
    g.append(grid, axis);
    const ends = [];
    series.forEach((sr) => {
      if (!sr.pts.length) return;
      const d = sr.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
      g.append(s("path", { d, class: "line", stroke: sr.color, "stroke-dasharray": sr.dash ? "5 4" : null }));
      if (sr.pts.length <= 40 && !sr.dash) for (const p of sr.pts) g.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: sr.color, stroke: "none" }));
      const last = sr.pts[sr.pts.length - 1];
      ends.push({ y: Y(last[1]), x: X(last[0]), label: sr.label, v: last[1], color: sr.color });
    });
    if (multi) {  // direct labels at the ends, pushed apart
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
      for (const e of ends) g.append(s("text", { x: W - mr + 8, y: e.y + 4, class: "endlabel" }, e.label));
    }
    // hover
    const xs = [...new Set(all.map((p) => p[0]))].sort((a, b) => a - b);
    const cross = s("line", { class: "cross", y1: mt, y2: H - mb, visibility: "hidden" });
    const dots = s("g");
    g.append(cross, dots);
    const hit = s("rect", { class: "hit", x: ml, y: mt, width: Math.max(1, W - ml - mr), height: H - mt - mb });
    g.append(hit);
    const show = (xv) => {
      this.hoverX = xv;
      cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("visibility", "visible");
      dots.replaceChildren();
      this.tip.replaceChildren(h("div", { class: "h" }, (o.xName || o.xLabel) + " " + (o.xfmt ? o.xfmt(xv) : fmt(xv))));
      for (const sr of series) {
        if (!sr.pts.length) continue;
        let best = sr.pts[0];
        for (const p of sr.pts) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        if (Math.abs(best[0] - xv) > (x1 - x0) * 0.04 && sr.pts.length > 1) continue;
        dots.append(s("circle", { cx: X(best[0]), cy: Y(best[1]), r: 4.2, fill: sr.color, stroke: css("--surface"), "stroke-width": 2 }));
        this.tip.append(h("div", { class: "r" }, h("span", {}, h("i", { style: `background:${sr.color}` }), sr.label), h("b", {}, fmt(best[1], o.fmt))));
      }
      this.tip.hidden = false;
      const tx = X(xv), tw = this.tip.offsetWidth;
      this.tip.style.left = (tx + 12 + tw > W ? tx - tw - 12 : tx + 12) + "px";
      this.tip.style.top = mt + "px";
    };
    hit.addEventListener("mousemove", (ev) => {
      const r = g.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W;
      const xv = x0 + ((px - ml) / (W - ml - mr)) * (x1 - x0);
      let best = xs[0];
      for (const v of xs) if (Math.abs(v - xv) < Math.abs(best - xv)) best = v;
      show(best);
    });
    hit.addEventListener("mouseleave", () => { this.hoverX = null; cross.setAttribute("visibility", "hidden"); dots.replaceChildren(); this.tip.hidden = true; });
    this.box.insertBefore(g, this.tip);
    if (this.hoverX !== null && xs.includes(this.hoverX)) show(this.hoverX);
  }
}

// Vertical bars with a tooltip per bar
function barChart(el, bars, opt = {}) {
  const box = h("div", { class: "cbox" }), tip = h("div", { class: "tip", hidden: true });
  el.append(box);
  box.append(tip);
  const draw = () => {
    [...box.querySelectorAll("svg")].forEach((e) => e.remove());
    const W = box.clientWidth || 400, H = opt.height || 160, ml = 38, mr = 8, mt = 8, mb = 22;
    const max = Math.max(1, ...bars.map((b) => b.v));
    const bw = (W - ml - mr) / bars.length;
    const g = s("svg", { viewBox: `0 0 ${W} ${H}`, height: H });
    const grid = s("g", { class: "grid" }), axis = s("g", { class: "axis" });
    for (const t of niceTicks(0, max, 3)) {
      const y = mt + (1 - t / max) * (H - mt - mb);
      grid.append(s("line", { x1: ml, x2: W - mr, y1: y, y2: y }));
      axis.append(s("text", { x: ml - 6, y: y + 3.5, "text-anchor": "end" }, fmt(t)));
    }
    g.append(grid, axis);
    bars.forEach((b, i) => {
      const bh = (b.v / max) * (H - mt - mb), x = ml + i * bw + 1, y = H - mb - bh;
      const r = s("rect", { x, y, width: Math.max(1, bw - 2), height: Math.max(0, bh), rx: Math.min(4, bw / 4), fill: css(b.color || "--s1") });
      g.append(r);
      if (i % Math.ceil(bars.length / 10) === 0 || i === bars.length - 1) axis.append(s("text", { x: x + bw / 2 - 1, y: H - 6, "text-anchor": "middle" }, b.label));
      const hitR = s("rect", { x: ml + i * bw, y: mt, width: bw, height: H - mt - mb, class: "hit" });
      hitR.addEventListener("mousemove", () => {
        tip.replaceChildren(h("div", { class: "h" }, b.title || b.label), h("div", { class: "r" }, h("span", {}, opt.valueLabel || "count"), h("b", {}, fmt(b.v))));
        tip.hidden = false;
        tip.style.left = Math.min(W - tip.offsetWidth, x + bw) + "px";
        tip.style.top = mt + "px";
      });
      hitR.addEventListener("mouseleave", () => (tip.hidden = true));
      g.append(hitR);
    });
    box.insertBefore(g, tip);
  };
  new ResizeObserver(draw).observe(box);
}

// A top-down drawing of the JEPA arena: paths as polylines in arena pixels
function arena(size, layers) {
  const pad = 1.5;
  const g = s("svg", { viewBox: `${-pad} ${-pad} ${size + 2 * pad} ${size + 2 * pad}`, role: "img", "aria-label": "arena" });
  g.append(s("rect", { x: 0, y: 0, width: size, height: size, rx: 1.2, class: "box", "stroke-width": 0.25 }));
  for (const L of layers) {
    const col = css(L.color);
    if (L.path && L.path.length) {
      g.append(s("polyline", { points: L.path.map((p) => p[0] + "," + p[1]).join(" "), stroke: col, "stroke-width": 0.32, fill: "none", "stroke-dasharray": L.dash ? "0.9 0.7" : null }));
      const a = L.path[0];
      g.append(s("circle", { cx: a[0], cy: a[1], r: 0.7, fill: col, stroke: "none" }));
      const z = L.path[L.path.length - 1];
      g.append(s("circle", { cx: z[0], cy: z[1], r: 0.7, fill: css("--surface"), stroke: col, "stroke-width": 0.3 }));
    }
    if (L.ring) g.append(s("circle", { cx: L.ring[0], cy: L.ring[1], r: 3, stroke: col, "stroke-width": 0.45, fill: "none", "stroke-dasharray": "0.8 0.6" }));
  }
  return h("div", { class: "arena" }, g);
}

// ------------------------------------------------------------------------------------------------ state

const S = { recipes: [], byId: {}, runs: [], current: null, card: null, view: null, run: null };

async function boot() {
  S.recipes = await api("/api/recipes");
  for (const r of S.recipes) S.byId[r.id] = r;
  $("#new-run").onclick = () => (location.hash = "#/");
  window.addEventListener("hashchange", route);
  await refreshRuns();
  route();
  refreshCard();
  setInterval(tick, 1500);
  setInterval(refreshCard, 5000);
}

async function tick() {
  await refreshRuns().catch(() => {});
  if (S.view && S.view.kind === "run") await updateRun().catch(() => {});
}

async function refreshRuns() {
  const r = await api("/api/runs");
  S.runs = r.runs;
  S.current = r.current;
  renderRunList();
}

async function refreshCard() {
  try { S.card = await api("/api/card"); } catch { S.card = { state: "unknown" }; }
  const c = S.card, el = $("#card-chip");
  el.replaceChildren();
  const dot = h("span", { class: "dot " + (c.state === "training" ? "busy" : c.state === "idle" ? "on" : "") });
  const label = { training: "Training", idle: "Ready", off: "Not connected", busy: `In use${c.holder ? " by " + c.holder : ""}`,
                  "no-cli": "rawgpu CLI not built", unknown: "Unknown" }[c.state] || c.state;
  el.append(dot, h("span", {}, "RX 590 · ", h("b", {}, label)));
  if (c.temperature_c !== undefined && c.temperature_c !== null)
    el.append(h("span", { class: c.temperature_c >= 80 ? "hot" : "" }, `${c.temperature_c} °C`));
  if (c.engine_mhz > 0) el.append(h("span", {}, `${c.engine_mhz} MHz`));
}

function renderRunList() {
  const el = $("#runs");
  el.replaceChildren();
  if (!S.runs.length) {
    el.append(h("div", { class: "empty-runs" }, "No runs yet. Pick a recipe to start one."));
    return;
  }
  const active = S.runs.filter((r) => r.state === "running" || r.state === "queued");
  const rest = S.runs.filter((r) => !(r.state === "running" || r.state === "queued"));
  const item = (r) => {
    const meta = S.byId[r.recipe] || {};
    const sel = S.view && S.view.kind === "run" && S.view.id === r.id;
    const pct = r.total ? Math.min(100, ((r.step || 0) / r.total) * 100) : 0;
    const key = (meta.tiles || [])[0];
    const val = key && r.summary ? r.summary[key.key] : undefined;
    return h("div", { class: "run-item" + (sel ? " on" : ""), onclick: () => (location.hash = "#/run/" + r.id) },
      glyph(r.recipe),
      h("div", { style: "min-width:0" },
        h("div", { class: "t" }, r.name),
        h("div", { class: "s" }, stateEl(r.state), h("span", {}, "·"),
          h("span", {}, r.state === "running" && r.total ? `${Math.round(pct)}%` : val !== undefined && key ? `${key.label} ${fmt(val, key.fmt)}` : ago(r.created))),
        r.state === "running" ? h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })) : null));
  };
  if (active.length) el.append(h("h4", {}, "Active"), ...active.map(item));
  if (rest.length) el.append(h("h4", {}, "Recent"), ...rest.map(item));
}

function route() {
  const hsh = location.hash.replace(/^#\/?/, "");
  const [kind, id] = hsh.split("/");
  const q = new URLSearchParams(location.hash.split("?")[1] || "");
  if (kind === "run" && id) openRun(id);
  else if (kind === "new" && id) newRun(id.split("?")[0], q.get("from"));
  else home();
  renderRunList();
}

// ------------------------------------------------------------------------------------------------ views

function home() {
  S.view = { kind: "home" };
  const main = $("#main");
  main.replaceChildren();
  const groups = {};
  for (const r of S.recipes) (groups[r.group] = groups[r.group] || []).push(r);
  main.append(h("div", { class: "gallery" },
    h("div", { class: "page-head" }, h("div", { class: "title" }, h("h1", {}, "Train a model on the RX 590"),
      h("div", { class: "intro" }, "Every recipe runs on ", h("b", {}, "rawgpu.nn"),
        ", autograd on the card with each kernel checked against PyTorch. Runs queue, since one program at a time can have the GPU, and they keep checkpoints, so a stopped run resumes where it left off."))),
    Object.entries(groups).map(([g, list]) => [h("h2", {}, g), h("div", { class: "recipes" }, list.map((r) =>
      h("button", { class: "recipe", type: "button", onclick: () => (location.hash = "#/new/" + r.id) },
        glyph(r.id), h("div", {}, h("h3", {}, r.name), h("div", { class: "tag" }, r.tagline), h("p", {}, r.description)))))])));
}

function fieldFor(p, value) {
  const id = "f-" + p.key;
  let input;
  if (p.type === "choice") {
    input = h("select", { id }, p.choices.map((c) => h("option", { value: c, selected: String(c) === String(value) }, c)));
  } else if (p.type === "bool") {
    input = h("input", { id, type: "checkbox", checked: !!value });
  } else {
    input = h("input", { id, type: p.type === "int" || p.type === "float" ? "text" : "text", value: value ?? "", spellcheck: "false",
                         inputmode: p.type === "int" ? "numeric" : p.type === "float" ? "decimal" : null,
                         placeholder: p.type === "path" ? "~/path/to/data" : null });
  }
  input.dataset.key = p.key;
  input.dataset.type = p.type;
  return h("div", { class: "field" + (p.type === "path" || p.type === "str" ? " wide" : "") },
    h("label", { for: id }, p.label, p.unit ? h("span", { class: "unit" }, " (" + p.unit + ")") : null),
    input, p.help ? h("div", { class: "help" }, p.help) : null);
}

async function newRun(recipe, from) {
  const meta = S.byId[recipe];
  if (!meta) return home();
  S.view = { kind: "new", recipe };
  let values = {}, name = "";
  if (from) {
    try {
      const r = await api("/api/runs/" + from);
      values = r.config.params;
      name = r.config.name + " (copy)";
    } catch { /* a missing run: start from the defaults */ }
  }
  const main = $("#main");
  main.replaceChildren();
  const basic = meta.params.filter((p) => !p.advanced), adv = meta.params.filter((p) => p.advanced);
  const err = h("span", { class: "form-err" });
  const nameIn = h("input", { type: "text", id: "f-name", value: name, placeholder: meta.name });
  const start = h("button", { class: "btn primary", type: "button" }, icon("play"), S.current ? "Queue run" : "Start training");
  start.onclick = async () => {
    const params = {};
    for (const el of main.querySelectorAll("[data-key]")) params[el.dataset.key] = el.dataset.type === "bool" ? el.checked : el.value;
    start.disabled = true;
    err.textContent = "";
    try {
      const r = await api("/api/runs", { recipe, params, name: nameIn.value });
      await refreshRuns();
      location.hash = "#/run/" + r.id;
    } catch (e) {
      err.textContent = e.message;
      start.disabled = false;
    }
  };
  main.append(
    h("div", { class: "page-head" }, glyph(recipe),
      h("div", { class: "title" }, h("h1", {}, meta.name), h("div", { class: "sub" }, meta.description)),
      h("div", { class: "acts" }, h("button", { class: "btn ghost", type: "button", onclick: () => (location.hash = "#/") }, icon("back"), "All recipes"))),
    h("div", { class: "panel" },
      h("div", { class: "form" }, h("div", { class: "field wide" }, h("label", { for: "f-name" }, "Name"), nameIn),
        basic.map((p) => fieldFor(p, values[p.key] ?? p.default))),
      adv.length ? h("details", { class: "adv" }, h("summary", {}, "Advanced"), h("div", { class: "form" }, adv.map((p) => fieldFor(p, values[p.key] ?? p.default)))) : null,
      h("div", { class: "form-foot" }, start, err,
        S.current ? h("span", { class: "faint" }, "Another run has the card; this one starts when it finishes.") : null)));
}

// ------------------------------------------------------------------------------------------------ a run

async function openRun(id) {
  S.view = { kind: "run", id };
  S.run = { id, metrics: [], events: [], mnext: 0, enext: 0, charts: [], panelsKey: "" };
  const main = $("#main");
  main.replaceChildren(h("div", { class: "faint" }, "Loading…"));
  try {
    const r = await api("/api/runs/" + id);
    Object.assign(S.run, { config: r.config, status: r.status || {} });
  } catch (e) {
    main.replaceChildren(h("div", { class: "banner critical" }, icon("alert"), h("div", { class: "txt" }, e.message)));
    return;
  }
  buildRun();
  await updateRun(true);
}

function buildRun() {
  const R = S.run, meta = S.byId[R.config.recipe] || { params: [], tiles: [], charts: [], panels: [] };
  R.meta = meta;
  const main = $("#main");
  main.replaceChildren();
  R.el = {
    pill: h("span"), acts: h("div", { class: "acts" }), sub: h("div", { class: "sub" }), banner: h("div"),
    progress: h("div", { class: "panel progress" }), tiles: h("div", { class: "tiles" }), card: h("div", { class: "tiles" }),
    charts: h("div", { class: "charts" }), panels: h("div"), info: h("div", { class: "info-lines" }), play: h("div"),
  };
  main.append(
    h("div", { class: "page-head" }, glyph(R.config.recipe),
      h("div", { class: "title" }, h("h1", {}, R.config.name, R.el.pill), R.el.sub), R.el.acts),
    R.el.banner, R.el.progress,
    h("div", { class: "section" }, R.el.tiles),
    h("div", { class: "section", id: "card-sec" }, h("h2", {}, "The card"), R.el.card));
  // charts
  const big = meta.charts.filter((c) => !c.small), small = meta.charts.filter((c) => c.small);
  R.charts = [];
  const mk = (c, height) => {
    const p = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, c.title), c.unit ? h("span", { class: "unit" }, c.unit) : null));
    const ch = new LineChart(p, { series: c.series, height, fmt: c.fmt, y: c.y, title: c.title });
    R.charts.push({ spec: c, chart: ch });
    return p;
  };
  R.el.charts.append(...big.map((c) => mk(c, 230)));
  main.append(h("div", { class: "section" }, h("h2", {}, "Training"), R.el.charts,
    small.length ? h("div", { class: "small-row" }, small.map((c) => mk(c, 130))) : null), R.el.panels);
  if (meta.playground) main.append(h("div", { class: "section" }, h("h2", {}, "Try it", h("small", {}, "runs on the card when no run is training")), R.el.play));
  // details
  const kv = h("dl", { class: "kv" });
  for (const p of meta.params) {
    const v = R.config.params[p.key];
    kv.append(h("dt", {}, p.label), h("dd", {}, v === "" || v === undefined || v === null ? "–" : String(v) + (p.unit ? " " + p.unit : "")));
  }
  const logBox = h("div", { class: "log" }, "…");
  const logDet = h("details", {}, h("summary", { style: "cursor:pointer;color:var(--text-2);font-weight:550" }, "Log"), logBox);
  logDet.addEventListener("toggle", () => logDet.open && loadLog());
  R.el.log = logBox;
  R.el.logDet = logDet;
  main.append(h("div", { class: "section two" },
    h("div", { class: "panel" }, h("h3", { style: "font-size:13px;margin-bottom:8px" }, "Notes"), R.el.info),
    h("div", { class: "panel" }, h("h3", { style: "font-size:13px;margin-bottom:8px" }, "Parameters"), kv)),
    h("div", { class: "section panel" }, logDet));
}

async function loadLog() {
  const R = S.run;
  try {
    const r = await api(`/api/runs/${R.id}/log`);
    const atEnd = R.el.log.scrollTop + R.el.log.clientHeight >= R.el.log.scrollHeight - 10;
    R.el.log.textContent = r.text || "(empty)";
    if (atEnd) R.el.log.scrollTop = R.el.log.scrollHeight;
  } catch { /* ignore */ }
}

async function updateRun(first) {
  const R = S.run;
  if (!R || !S.view || S.view.id !== R.id) return;
  const [st, m, ev] = await Promise.all([api("/api/runs/" + R.id), api(`/api/runs/${R.id}/metrics?from=${R.mnext}`),
                                         api(`/api/runs/${R.id}/events?from=${R.enext}`)]);
  if (S.view.id !== R.id) return;
  R.status = st.status || {};
  R.config = st.config;
  const newM = m.items.length, newE = ev.items.length;
  R.metrics.push(...m.items);
  R.mnext = m.next;
  R.events.push(...ev.items);
  R.enext = ev.next;
  renderHead();
  if (first || newM) renderMetrics();
  if (first || newE) renderEvents();
  if (first && R.meta.playground) buildPlayground();
  if (R.el.logDet.open && (R.status.state === "running" || first)) loadLog();
}

function series(key) {
  return S.run.metrics.filter((r) => r[key] !== undefined && r[key] !== null).map((r) => [r.step, r[key]]);
}

function latest(key) {
  for (let i = S.run.metrics.length - 1; i >= 0; i--) if (S.run.metrics[i][key] !== undefined) return S.run.metrics[i][key];
  return S.run.status.summary ? S.run.status.summary[key] : undefined;
}

function renderHead() {
  const R = S.run, st = R.status, state = st.state || "queued";
  R.el.pill.replaceChildren(stateEl(state, true));
  const started = st.started, end = st.finished || (state === "running" ? Date.now() / 1000 : null);
  R.el.sub.replaceChildren(`${R.meta.name} · created ${ago(R.config.created || 0)}` + (started ? ` · ran ${dur(end - started)}` : ""));
  // actions
  const A = R.el.acts;
  A.replaceChildren();
  if (state === "running" || state === "queued")
    A.append(h("button", { class: "btn", type: "button", onclick: async () => { await api(`/api/runs/${R.id}/stop`, {}); toast(state === "running" ? "Stopping: saving a checkpoint first" : "Cancelled"); tick(); } }, icon("stop"), state === "running" ? "Stop" : "Cancel"));
  if (["stopped", "failed", "cancelled"].includes(state))
    A.append(h("button", { class: "btn primary", type: "button", onclick: async () => { await api(`/api/runs/${R.id}/resume`, {}); toast("Queued"); tick(); } }, icon("play"), "Resume"));
  A.append(h("button", { class: "btn", type: "button", onclick: () => (location.hash = `#/new/${R.config.recipe}?from=${R.id}`) }, icon("copy"), "Duplicate"));
  if (state !== "running")
    A.append(h("button", { class: "btn danger", type: "button", onclick: async () => {
      if (!confirm(`Delete “${R.config.name}” and its checkpoints?`)) return;
      await api("/api/runs/" + R.id, {}, "DELETE");
      await refreshRuns();
      location.hash = "#/";
    } }, icon("trash"), "Delete"));
  // banner
  R.el.banner.replaceChildren();
  if (st.error) R.el.banner.append(h("div", { class: "banner critical" }, icon("alert"), h("div", { class: "txt" }, h("b", {}, "The run failed. "), st.error)));
  else if (st.message) R.el.banner.append(h("div", { class: "banner " + (/Paused/.test(st.message) ? "warn" : "info") }, icon(/Paused/.test(st.message) ? "therm" : "info"), h("div", { class: "txt" }, st.message)));
  else if (state === "queued") R.el.banner.append(h("div", { class: "banner info" }, icon("clock"), h("div", { class: "txt" }, "Waiting for the card: another run is training.")));
  // progress
  const total = st.total || +R.config.params.steps || 0, step = st.step || 0;
  const pct = total ? Math.min(100, (step / total) * 100) : 0;
  const rate = st.steps_per_s;
  const eta = state === "running" && rate ? (total - step) / rate : null;
  R.el.progress.replaceChildren(
    h("div", {}, h("b", {}, `Step ${step.toLocaleString()}`), h("span", { class: "muted" }, ` of ${total.toLocaleString()}`)),
    h("div", { class: "nums" }, state === "done" ? "Finished" : [rate ? `${fmt(rate)} steps/s` : "", eta ? ` · ${dur(eta)} left` : ""]),
    h("div", { class: "track" }, h("i", { style: `width:${pct}%` })));
  // the card
  const c = st.card || {};
  const sec = $("#card-sec");
  sec.hidden = state !== "running" || c.temperature_c === undefined;
  if (!sec.hidden) {
    const limit = +R.config.params.max_temp_c || 85, t = c.temperature_c;
    const lvl = t >= limit ? "critical" : t >= limit - 5 ? "warn" : "";
    const vramUsed = c.vram_bytes - c.vram_free_bytes;
    R.el.card.replaceChildren(
      h("div", { class: "tile" }, h("div", { class: "k" }, "Temperature", lvl ? h("span", { class: "flag " + lvl }, icon("alert"), lvl === "critical" ? "At the limit" : "Near the limit") : null),
        h("div", { class: "v" }, t, h("span", { class: "u" }, "°C")), h("div", { class: "meter " + lvl }, h("i", { style: `width:${Math.min(100, (t / 100) * 100)}%` })),
        h("div", { class: "c" }, `pauses at ${limit} °C`)),
      h("div", { class: "tile" }, h("div", { class: "k" }, "Engine clock"), h("div", { class: "v" }, c.engine_mhz > 0 ? c.engine_mhz : "–", h("span", { class: "u" }, "MHz")),
        h("div", { class: "c" }, `memory ${c.memory_mhz > 0 ? c.memory_mhz + " MHz" : "–"}${c.fan_percent >= 0 ? ` · fan ${c.fan_percent}%` : ""}`)),
      h("div", { class: "tile" }, h("div", { class: "k" }, "VRAM"), h("div", { class: "v" }, fmt(vramUsed / 2 ** 30, ".1f"), h("span", { class: "u" }, `of ${fmt(c.vram_bytes / 2 ** 30, ".0f")} GB`)),
        h("div", { class: "meter" }, h("i", { style: `width:${(vramUsed / c.vram_bytes) * 100}%` })),
        h("div", { class: "c" }, `tensors ${fmt(c.tensors_bytes / 2 ** 20, ",.0f")} MB now`)));
  }
}

function renderMetrics() {
  const R = S.run;
  // tiles: the recipe's headline numbers, with the best so far
  R.el.tiles.replaceChildren(...R.meta.tiles.map((t) => {
    const v = latest(t.key);
    const ser = series(t.key);
    let cap = t.help || "";
    if (t.best && ser.length > 1) {
      const b = ser.reduce((a, p) => ((t.best === "min" ? p[1] < a[1] : p[1] > a[1]) ? p : a));
      cap = `best ${fmt(b[1], t.fmt)} at step ${b[0].toLocaleString()}`;
    }
    const base = R.status.summary && R.status.summary["baseline_" + t.key];
    if (base !== undefined) cap = `before training ${fmt(base, t.fmt)}`;
    return h("div", { class: "tile" }, h("div", { class: "k" }, t.label),
      h("div", { class: "v" + (v === undefined ? " none" : "") }, v === undefined ? "–" : fmt(v, t.fmt), t.unit && v !== undefined ? h("span", { class: "u" }, t.unit) : null),
      h("div", { class: "c", title: cap }, cap || "\u00a0"));
  }));
  for (const { spec, chart } of R.charts) {
    const d = {};
    for (const sr of spec.series) d[sr.key] = series(sr.key);
    chart.update(d);
  }
}

function lastEvent(kind) {
  for (let i = S.run.events.length - 1; i >= 0; i--) if (S.run.events[i].kind === kind) return S.run.events[i];
  return null;
}

function renderEvents() {
  const R = S.run;
  R.el.info.replaceChildren(...R.events.filter((e) => e.kind === "info").map((e) => h("div", {}, e.text)));
  if (!R.el.info.childNodes.length) R.el.info.append(h("div", { class: "faint" }, "Nothing yet."));
  const key = R.meta.panels.map((p) => JSON.stringify(PANEL_KINDS[p].map((k) => (lastEvent(k) || {}).t))).join("|") + ":" + R.events.filter((e) => e.kind === "sample").length;
  if (key === R.panelsKey) return;
  R.panelsKey = key;
  R.el.panels.replaceChildren(...R.meta.panels.map((p) => PANELS[p]()).filter(Boolean));
}

const PANEL_KINDS = { samples: ["sample"], rollout: ["rollout"], planning: ["planning"], probes: ["probes"], reliability: ["reliability"], retrieval: ["retrieval"] };

function section(title, note, ...kids) {
  return h("div", { class: "section" }, h("h2", {}, title, note ? h("small", {}, note) : null), ...kids);
}

const PANELS = {
  samples() {
    const all = S.run.events.filter((e) => e.kind === "sample");
    if (!all.length) return section("Samples", "", h("div", { class: "panel faint" }, "The first sample appears after a few hundred steps."));
    let i = all.length - 1;
    const body = h("div", { class: "sample" }), pg = h("div", { class: "pager" });
    const show = () => {
      const e = all[i];
      body.replaceChildren(h("span", { class: "p" }, e.prompt), e.text);
      pg.replaceChildren(
        h("button", { class: "btn small", type: "button", disabled: i === 0, onclick: () => { i--; show(); } }, "Earlier"),
        h("span", {}, `Step ${(e.step + 1).toLocaleString()} · ${i + 1} of ${all.length}`),
        h("button", { class: "btn small", type: "button", disabled: i === all.length - 1, onclick: () => { i++; show(); } }, "Later"));
    };
    show();
    return section("Samples", "what the model writes, temperature 0.8", h("div", { class: "panel samples" }, h("div", { style: "margin-bottom:8px" }, pg), body));
  },

  rollout() {
    const e = lastEvent("rollout");
    if (!e) return section("Imagination", "", h("div", { class: "panel faint" }, "The first evaluation appears at the first evaluation step."));
    const left = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "Position error in an imagined rollout"), h("span", { class: "unit" }, "pixels, read out by a linear probe")));
    const hz = e.horizons;
    new LineChart(left, {
      series: [{ key: "a", label: "Agent" }, { key: "b", label: "Ball" }, { key: "fa", label: "Agent, encoded", dash: true, color: "--s1" }, { key: "fb", label: "Ball, encoded", dash: true, color: "--s2" }],
      height: 220, xLabel: "steps ahead", xName: "Steps ahead", fmt: ".2f",
    }).update({ a: hz.map((k, i) => [k, e.agent_px[i]]), b: hz.map((k, i) => [k, e.ball_px[i]]),
                fa: [[hz[0], e.floor_agent_px], [hz[hz.length - 1], e.floor_agent_px]], fb: [[hz[0], e.floor_ball_px], [hz[hz.length - 1], e.floor_ball_px]] });
    const lat = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "Latent prediction error"), h("span", { class: "unit" }, "vs. predicting no change")));
    new LineChart(lat, { series: [{ key: "p", label: "Predictor" }, { key: "c", label: "No change", color: "--s2" }], height: 160, xLabel: "steps ahead", xName: "Steps ahead", fmt: ".3f" })
      .update({ p: hz.map((k, i) => [k, e.latent_err[i]]), c: hz.map((k, i) => [k, e.copy_err[i]]) });
    const ex = e.example;
    const right = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "One episode, imagined from its first frame"), h("span", { class: "unit" }, `${hz.length} steps`)),
      h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--s1)" }), "Agent"), h("span", {}, h("i", { class: "dash", style: "color:var(--s1)" }), "Agent, imagined"),
        h("span", {}, h("i", { style: "background:var(--s2)" }), "Ball"), h("span", {}, h("i", { class: "dash", style: "color:var(--s2)" }), "Ball, imagined")),
      arena(e.arena, [
        { path: ex.true.map((p) => [p[0], p[1]]), color: "--s1" }, { path: [ex.true[0]].concat(ex.imagined).map((p) => [p[0], p[1]]), color: "--s1", dash: true },
        { path: ex.true.map((p) => [p[2], p[3]]), color: "--s2" }, { path: [ex.true[0]].concat(ex.imagined).map((p) => [p[2], p[3]]), color: "--s2", dash: true }]),
      h("div", { class: "faint", style: "margin-top:6px;font-size:11.5px" }, "Filled dot: start. Open dot: end. Actions: " + ex.actions.join(", ")));
    return section("Imagination", `evaluated at step ${(e.step + 1).toLocaleString()}; trained ${e.trained_horizon} steps ahead`, h("div", { class: "two" }, h("div", { style: "display:grid;gap:12px" }, left, lat), right));
  },

  planning() {
    const e = lastEvent("planning");
    if (!e) return null;
    const ex = e.example;
    return section("Planning", `random shooting over ${e.candidates} imagined ${e.plan_len}-step futures`,
      h("div", { class: "two" },
        h("div", { style: "display:grid;gap:12px;align-content:start" },
          h("div", { class: "tiles" },
            h("div", { class: "tile" }, h("div", { class: "k" }, "Reaches the goal"), h("div", { class: "v" }, fmt(e.success, ".0%")), h("div", { class: "c" }, `random actions: ${fmt(e.random_success, ".0%")}`)),
            h("div", { class: "tile" }, h("div", { class: "k" }, "Distance left"), h("div", { class: "v" }, fmt(e.mean_dist, ".1f"), h("span", { class: "u" }, "px")), h("div", { class: "c" }, `random actions: ${fmt(e.random_dist, ".1f")} px`))),
          h("div", { class: "panel muted", style: "font-size:12.5px" }, `${e.episodes} episodes: from a random start, the planner imagines ${e.candidates} action sequences in latent space, keeps the one the probe says ends nearest the goal, takes 4 steps of it in the real arena, and plans again. "Reaches" means within 3 px after 16 steps.`)),
        h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "One planned episode")),
          h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--s1)" }), "Agent's path"), h("span", {}, h("i", { class: "dash", style: "color:var(--s3)" }), "Goal (3 px)")),
          arena(e.arena, [{ path: ex.path, color: "--s1" }, { ring: ex.goal, color: "--s3" }]))));
  },

  probes() {
    const e = lastEvent("probes");
    if (!e) return section("Filters", "", h("div", { class: "panel faint" }, "Held-out results appear when training finishes."));
    const relBox = h("div", { class: "panel" });
    const showRel = (row) => {
      relBox.replaceChildren(h("div", { class: "chart-head" }, h("h3", {}, `Calibration: ${row.concept}`), h("span", { class: "unit" }, "predicted vs. observed")));
      const pts = row.reliability.filter((b) => b.count).map((b) => [b.confidence, b.accuracy]);
      new LineChart(relBox, { series: [{ key: "r", label: "Probe" }, { key: "d", label: "Perfect", dash: true, color: "--s2" }], height: 240, xDomain: [0, 1], yDomain: [0, 1],
        xLabel: "predicted", xName: "Predicted", fmt: ".2f", xfmt: (v) => fmt(v, ".2f") }).update({ r: pts, d: [[0, 0], [1, 1]] });
    };
    const tb = h("tbody");
    const maxAuc = 1;
    e.rows.forEach((r, i) => {
      const tr = h("tr", { class: "pick" + (i === 0 ? " sel" : ""), onclick: () => { tb.querySelectorAll("tr").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel"); showRel(r); } },
        h("td", {}, h("b", {}, r.concept), r.description ? h("div", { class: "faint", style: "font-size:11.5px" }, r.description) : null),
        h("td", { class: "n" }, r.positives.toLocaleString()),
        h("td", { class: "nw" }, h("span", { class: "hbar", style: `width:${(r.auc_probe || 0) / maxAuc * 70}px` }), fmt(r.auc_probe, ".3f")),
        h("td", { class: "nw" }, h("span", { class: "hbar b", style: `width:${(r.auc_cosine || 0) / maxAuc * 70}px` }), fmt(r.auc_cosine, ".3f")),
        h("td", { class: "n" }, fmt(r.f1_probe, ".3f")), h("td", { class: "n" }, fmt(r.f1_cosine, ".3f")), h("td", { class: "n" }, fmt(r.ece_probe, ".3f")));
      tb.append(tr);
    });
    showRel(e.rows[0]);
    return section("Filters", `held-out test split: ${e.split.test.toLocaleString()} examples from files never seen in training`,
      h("div", { style: "display:grid;gap:12px" },
        h("div", { class: "panel", style: "overflow:auto" }, h("table", { class: "t" },
          h("thead", {}, h("tr", {}, h("th", {}, "Concept"), h("th", { class: "n" }, "Positives"), h("th", {}, "AUC, probe"), h("th", {}, "AUC, cosine"),
            h("th", { class: "n" }, "F1, probe"), h("th", { class: "n" }, "F1, cosine"), h("th", { class: "n" }, "ECE"))), tb),
          h("div", { class: "faint", style: "font-size:11.5px;margin-top:8px" }, "Cosine: similarity to the mean embedding of the concept's examples, no training. Click a row for its calibration.")),
        h("div", { class: "two" }, relBox, h("div", { class: "panel muted", style: "font-size:12.5px" },
          "A filter fires when its probability passes the threshold that gave the best F1 on the validation split. AUC measures ranking whatever the threshold: 0.5 is chance, 1 perfect. ECE is how far the stated probabilities are from the observed frequencies, as in the chart: points on the diagonal are calibrated. The filters, prototypes and thresholds are saved as probes.npz in the run's folder."))));
  },

  reliability() {
    const e = lastEvent("reliability");
    if (!e) return section("Calibration", "", h("div", { class: "panel faint" }, "Calibration is measured when training finishes."));
    const box = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "Reliability"), h("span", { class: "unit" }, "stated confidence vs. how often it was right")));
    const pts = (bins) => bins.filter((b) => b.count).map((b) => [b.confidence, b.accuracy]);
    new LineChart(box, { series: [{ key: "a", label: "Calibrated" }, { key: "b", label: "Before", color: "--s2" }, { key: "d", label: "Perfect", dash: true, color: "--s3" }],
      height: 280, xDomain: [0, 1], yDomain: [0, 1], xName: "Confidence", xLabel: "confidence", fmt: ".2f", xfmt: (v) => fmt(v, ".2f") })
      .update({ a: pts(e.bins_after), b: pts(e.bins_before), d: [[0, 0], [1, 1]] });
    const rows = e.questions.map((q) => h("tr", {}, h("td", {}, h("span", { class: "badge" }, q.type)), h("td", {}, q.question),
      h("td", { class: "n" }, q.options), h("td", { class: "n" }, fmt(q.accuracy, ".1%")), h("td", { class: "n" }, fmt(q.confidence, ".2f")),
      h("td", { class: "n" }, fmt(q.ece, ".3f")), h("td", { class: "n" }, fmt(q.brier, ".3f")),
      h("td", { class: "n" }, q.temperature !== undefined ? fmt(q.temperature, ".2f") : "–"), h("td", { class: "n" }, q.n.toLocaleString())));
    return section("Calibration", e.temperatures ? "test split; a temperature per question, fit on the validation split" : `test split, temperature ${fmt(e.temperature, ".2f")} fit on the validation split`,
      h("div", { class: "two" },
        h("div", { style: "display:grid;gap:12px;align-content:start" },
          h("div", { class: "tiles" },
            h("div", { class: "tile" }, h("div", { class: "k" }, "ECE, calibrated"), h("div", { class: "v" }, fmt(e.after.ece, ".3f")),
              h("div", { class: "c" }, `before: ${fmt(e.before.ece, ".3f")}` + (e.global_t ? ` · one temperature: ${fmt(e.global_t.ece, ".3f")}` : ""))),
            h("div", { class: "tile" }, h("div", { class: "k" }, "NLL, calibrated"), h("div", { class: "v" }, fmt(e.after.nll, ".3f")), h("div", { class: "c" }, `before: ${fmt(e.before.nll, ".3f")}`)),
            h("div", { class: "tile" }, h("div", { class: "k" }, "Brier, calibrated"), h("div", { class: "v" }, fmt(e.after.brier, ".3f")), h("div", { class: "c" }, `before: ${fmt(e.before.brier, ".3f")}`)),
            h("div", { class: "tile" }, h("div", { class: "k" }, "Accuracy"), h("div", { class: "v" }, fmt(e.after.accuracy, ".1%")), h("div", { class: "c" }, `${e.after.n.toLocaleString()} test examples`))),
          h("div", { class: "panel muted", style: "font-size:12.5px" }, "Temperature scaling divides the logits by a number fit on the validation split, one per trained question (new questions get the global one): it changes how sure the critic is, never which option it picks. The diagonal is perfect calibration; points below it are overconfident.")),
        box),
      h("div", { class: "panel", style: "overflow:auto;margin-top:12px" }, h("table", { class: "t" },
        h("thead", {}, h("tr", {}, h("th", {}, "Type"), h("th", {}, "Question"), h("th", { class: "n" }, "Options"), h("th", { class: "n" }, "Accuracy"),
          h("th", { class: "n" }, "Confidence"), h("th", { class: "n" }, "ECE"), h("th", { class: "n" }, "Brier"), h("th", { class: "n" }, "Temperature"), h("th", { class: "n" }, "Examples"))), h("tbody", {}, rows)),
        h("div", { class: "faint", style: "font-size:11.5px;margin-top:8px" }, "Confidence is Jev's: (J × p_max − 1) / (J − 1) for J options, 0 for an even spread, 1 for certainty.")));
  },

  retrieval() {
    const all = S.run.events.filter((e) => e.kind === "retrieval");
    if (!all.length) return null;
    const first = all[0], e = all[all.length - 1];
    const hist = h("div", { class: "panel" }, h("div", { class: "chart-head" }, h("h3", {}, "Where the right document ranks"), h("span", { class: "unit" }, `among ${e.pool} held-out documents`)));
    barChart(hist, e.rank_hist.map((v, i) => ({ v, label: i === 20 ? "21+" : String(i + 1), title: i === 20 ? "rank 21 or worse" : `rank ${i + 1}` })), { height: 170, valueLabel: "queries" });
    const delta = (k) => `before training ${fmt(first[k], ".1%")}`;
    return section("Retrieval", `held-out pairs, evaluated at step ${e.step.toLocaleString()}`,
      h("div", { class: "two" },
        h("div", { style: "display:grid;gap:12px;align-content:start" },
          h("div", { class: "tiles" }, ["recall_1", "recall_5", "recall_10"].map((k) => h("div", { class: "tile" }, h("div", { class: "k" }, "Recall@" + k.split("_")[1]),
            h("div", { class: "v" }, fmt(e[k], ".1%")), h("div", { class: "c" }, delta(k))))),
          h("div", { class: "panel" }, h("h3", { style: "font-size:13px;margin-bottom:6px" }, "Hardest queries now"),
            h("div", { class: "info-lines" }, e.examples.map((x) => h("div", {}, h("span", { class: "badge" }, "rank " + x.rank), " ", x.query))))),
        hist));
  },
};

// ------------------------------------------------------------------------------------------------ playground

function buildPlayground() {
  const R = S.run, recipe = R.config.recipe, box = R.el.play;
  const out = h("div", { class: "play-out" });
  const go = h("button", { class: "btn primary", type: "button" }, icon("play"), "Run");
  const run = async (req, render) => {
    go.disabled = true;
    out.replaceChildren(h("div", { class: "faint" }, "Running on the card…"));
    try { render(await api(`/api/runs/${R.id}/try`, req)); } catch (e) { out.replaceChildren(h("div", { class: "banner critical" }, icon("alert"), h("div", { class: "txt" }, e.message))); }
    go.disabled = false;
  };
  const probRows = (items, top) => h("div", { class: "probs" }, items.map((it) => h("div", { class: "row" + (it.name === top ? " top" : "") },
    h("span", { class: "nm", title: it.name }, it.name), h("span", { class: "tr" }, h("i", { style: `width:${it.p * 100}%` })), h("span", { class: "pv" }, fmt(it.p, ".0%")))));
  if (recipe === "gpt") {
    const prompt = h("textarea", { rows: 3, spellcheck: "false" });
    prompt.value = R.config.params.prompt || "";
    const len = h("input", { type: "text", value: "400", inputmode: "numeric" });
    const temp = h("input", { type: "text", value: "0.8", inputmode: "decimal" });
    go.onclick = () => run({ prompt: prompt.value, length: +len.value, temperature: +temp.value },
      (r) => out.replaceChildren(h("div", { class: "samples" }, h("div", { class: "sample" }, h("span", { class: "p" }, prompt.value), r.text)),
        h("div", { class: "faint", style: "margin-top:6px;font-size:11.5px" }, `from the ${r.checkpoint} checkpoint`)));
    box.append(h("div", { class: "panel" }, h("div", { class: "form" }, h("div", { class: "field wide" }, h("label", {}, "Prompt"), prompt),
      h("div", { class: "field" }, h("label", {}, "Length ", h("span", { class: "unit" }, "(bytes)")), len), h("div", { class: "field" }, h("label", {}, "Temperature"), temp)),
      h("div", { class: "form-foot" }, go), out));
  } else if (recipe === "probe") {
    const text = h("textarea", { rows: 5, spellcheck: "false", placeholder: "Paste some code or text" });
    go.onclick = () => run({ text: text.value }, (r) => out.replaceChildren(h("table", { class: "t" },
      h("thead", {}, h("tr", {}, h("th", {}, "Concept"), h("th", {}, "Probability"), h("th", {}, "Fires"), h("th", { class: "n" }, "Cosine"))),
      h("tbody", {}, r.results.map((x) => h("tr", {}, h("td", {}, h("b", {}, x.concept)),
        h("td", {}, h("span", { class: "hbar", style: `width:${x.probability * 120}px` }), fmt(x.probability, ".1%")),
        h("td", {}, x.fires ? h("span", { class: "state done" }, icon("check"), "yes") : h("span", { class: "faint" }, "no")),
        h("td", { class: "n" }, fmt(x.cosine, ".3f"))))))));
    box.append(h("div", { class: "panel" }, h("div", { class: "field wide" }, h("label", {}, "Input"), text), h("div", { class: "form-foot" }, go), out));
  } else if (recipe === "critic") {
    const state = h("textarea", { rows: 5, spellcheck: "false", placeholder: "The state: code, a message, a JSON record…" });
    const e = lastEvent("reliability");
    const trained = (e && e.questions) || [];
    const pick = h("select", {}, trained.map((q) => h("option", { value: q.id || q.question }, q.question)), h("option", { value: "__custom" }, "A new question…"));
    const qText = h("input", { type: "text", placeholder: "Is this code safe to run?" });
    const qType = h("select", {}, ["choice", "score", "noul"].map((t) => h("option", { value: t }, t)));
    const opts = h("textarea", { rows: 4, spellcheck: "false", placeholder: "one option per line: name: description\nsafe: no side effects\nrisky: deletes or sends data" });
    const custom = h("div", { class: "form", hidden: true }, h("div", { class: "field wide" }, h("label", {}, "Question"), qText),
      h("div", { class: "field" }, h("label", {}, "Type"), qType), h("div", { class: "field wide" }, h("label", {}, "Options ", h("span", { class: "unit" }, "(not needed for noul: yes or no)")), opts));
    pick.onchange = () => (custom.hidden = pick.value !== "__custom");
    go.onclick = () => {
      let req;
      if (pick.value === "__custom") {
        const o = {};
        for (const line of opts.value.split("\n")) { const m = /^\s*([^:]+?)\s*(?::\s*(.*))?$/.exec(line); if (m && m[1]) o[m[1]] = m[2] || ""; }
        req = { state: state.value, question: qText.value, type: qType.value, options: qType.value === "noul" && !Object.keys(o).length ? null : o };
      } else req = { state: state.value, question_id: pick.value };
      run(req, (r) => out.replaceChildren(
        h("div", { class: "tiles", style: "margin-bottom:10px" },
          h("div", { class: "tile" }, h("div", { class: "k" }, r.type === "noul" ? "P(yes)" : "Choice"), h("div", { class: "v" }, r.type === "noul" ? fmt(r.probability_yes, ".0%") : r.choice)),
          h("div", { class: "tile" }, h("div", { class: "k" }, "Confidence"), h("div", { class: "v" }, fmt(r.confidence, ".2f"))),
          r.position !== undefined ? h("div", { class: "tile" }, h("div", { class: "k" }, "Score position"), h("div", { class: "v" }, fmt(r.position, ".2f"), h("span", { class: "u" }, `of ${r.options.length - 1}`))) : null),
        probRows(r.options.map((x) => ({ name: x.name, p: x.probability })), r.choice)));
    };
    box.append(h("div", { class: "panel" }, h("div", { class: "form" }, h("div", { class: "field wide" }, h("label", {}, "State"), state),
      h("div", { class: "field wide" }, h("label", {}, "Question"), pick)), custom, h("div", { class: "form-foot" }, go), out));
  }
}

boot();
