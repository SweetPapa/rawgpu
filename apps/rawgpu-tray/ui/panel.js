// The panel under the menu bar icon. All state comes from the app (src/main.rs) as "state" events.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function icon(paths, cls) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 16 16");
  s.setAttribute("aria-hidden", "true");
  if (cls) s.setAttribute("class", cls);
  for (const d of paths) {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    s.append(p);
  }
  return s;
}
const I = {
  eject: ["M8 3 13 9H3z", "M3 12.5h10"],
  check: ["m3.5 8.5 3 3 6-7"],
  warn: ["M8 2 14.5 13.5h-13z", "M8 6.5v3.2m0 2v.1"],
  plug: ["M6 2v3m4-3v3", "M4.5 5h7v2.5a3.5 3.5 0 0 1-7 0z", "M8 11v3"],
  search: ["M7 2.5a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9z", "m10.5 10.5 3 3"],
  details: ["M3 4h10M3 8h10M3 12h6"],
  power: ["M8 2v5", "M4.5 4.5a5 5 0 1 0 7 0"],
  db: ["M3 4c0-1.1 2.2-2 5-2s5 .9 5 2-2.2 2-5 2-5-.9-5-2z", "M3 4v8c0 1.1 2.2 2 5 2s5-.9 5-2V4", "M3 8c0 1.1 2.2 2 5 2s5-.9 5-2"],
  term: ["M2.5 3.5h11v9h-11z", "m5 6.5 2 1.5-2 1.5M8.5 10h2.5"],
  app: ["M3 3h10v10H3z", "M3 6h10"],
};
const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : String(n));
const ENGINE_MAX = 1545, MEMORY_MAX = 2000;

function ago(t) {
  const s = Math.max(0, Math.round(Date.now() / 1000 - t));
  if (s < 5) return "just now";
  if (s < 60) return `${s} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  return `${(s / 3600).toFixed(1)} h ago`;
}
function span(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}
const cap = (s) => (/^the /.test(s) ? "T" + s.slice(1) : s); // "the code indexer" starts a sentence; file names keep their case

let S = null;
let settingsOpen = false;

// ---------------------------------------------------------------- pieces

function pill(s) {
  const done = s.eject.state === "done";
  const map = {
    setup: ["warn", "Set up", I.warn], "no-card": ["", "No card"],
    idle: ["good", "Ready"], reading: ["", "Reading…"], busy: ["busy", "In use"], ejecting: ["busy", "Ejecting"],
    off: done ? ["good", "Safe to unplug", I.check] : ["", "Disconnected"],
    other: ["warn", "Not safe to unplug", I.warn], "not-posted": ["warn", "Needs bring-up", I.warn],
    "not-answering": ["critical", "Not answering", I.warn], "cli-error": ["critical", "CLI error", I.warn],
  };
  const [cls, text, ic] = map[s.mode] || ["", s.mode];
  const p = $("pill");
  p.className = "pill " + cls;
  p.replaceChildren(ic ? icon(ic) : h("span", "dot"), h("span", null, text));
}

function alertBox(s) {
  let kind = null, title = "", text = "";
  if (s.eject.state === "failed") { kind = "critical"; title = "Not safe to unplug"; text = s.eject.phase; }
  else if (s.eject.state === "cancelled") { kind = "info"; title = "Eject cancelled"; text = "Nothing changed; the card is still connected."; }
  else if (s.mode === "not-posted") { kind = "warn"; title = "The card was reset"; text = "It lost power or the Mac slept. Bring it up to use it again."; }
  else if (s.mode === "not-answering") { kind = "critical"; title = "The card is not answering"; text = (s.status_error || "") + " If this persists, eject and power-cycle the enclosure."; }
  else if (s.mode === "cli-error") { kind = "critical"; title = "The rawgpu CLI isn't working"; text = s.who_error || ""; }
  else if (s.mode === "other") { kind = "warn"; title = "Another TinyGPU server is running"; text = "Another program may have the card, so it isn't safe to unplug."; }
  const a = $("alert");
  a.hidden = !kind;
  a.className = "alert" + (kind === "critical" ? " critical" : "");
  a.style.background = kind === "info" ? "var(--hover)" : "";
  $("alert-title").textContent = title;
  $("alert-text").textContent = text;
}

// what the tiles show: the card's own status when idle, the indexer's log while it runs
function readings(s) {
  if (s.mode === "idle" && s.status) return { src: "card", v: s.status, at: s.status_at };
  if (s.mode === "busy" && s.indexer) return { src: "indexer", v: s.indexer, at: s.indexer.t };
  if (s.mode === "busy" && s.status && s.status.posted) return { src: "stale", v: s.status, at: s.status_at };
  return null;
}

function tile(label, value, unit, frac, cls, caption, flag, stale) {
  const t = h("div", "tile" + (stale ? " stale" : ""));
  const l = h("div", "tile-label");
  l.append(h("span", null, label));
  if (flag) { const f = h("span", "tile-flag " + flag[0]); f.append(icon(I.warn), h("span", null, flag[1])); l.append(f); }
  const v = h("div", "tile-value" + (value === null ? " none" : ""));
  v.append(document.createTextNode(value === null ? "—" : fmt(value)));
  if (value !== null && unit) v.append(h("span", "unit", unit));
  const m = h("div", "meter " + (value === null ? "none" : cls || ""));
  const i = h("i");
  i.style.width = (value === null || frac <= 0 ? 0 : Math.max(2, Math.min(100, frac * 100))) + "%";
  m.append(i);
  t.append(l, v, m, h("div", "tile-cap", caption));
  return t;
}

function renderTiles(s) {
  const r = readings(s);
  const box = $("tiles");
  box.replaceChildren();
  const paused = s.holder ? `paused while ${s.holder.name} has the card` : "no reading";
  const stale = r && r.src === "stale";
  const v = r ? r.v : {};
  const temp = r ? v.temperature_c : null;
  const alertC = s.settings.alert_c;
  let tcls = "", flag = null;
  if (temp !== null && temp >= alertC) { tcls = "critical"; flag = ["critical", "Hot"]; }
  else if (temp !== null && temp >= alertC - 8) { tcls = "warn"; flag = ["warn", "Warm"]; }
  const was = stale ? `before it took the card` : null;
  box.append(tile("Temperature", temp, "°C", (temp || 0) / 95, tcls, was || (r ? `alert at ${alertC} °C` : paused), stale ? null : flag, stale));
  const fan = r && v.fan_percent >= 0 ? v.fan_percent : null;
  box.append(tile("Fan", fan, "%", (fan || 0) / 100, "", was || (!r ? paused : fan === null ? "not under SMC control" : fan === 0 ? "stopped: quiet when cool" : "of full speed"), null, stale));
  const eng = r && v.engine_mhz >= 0 ? v.engine_mhz : null;
  box.append(tile("Engine clock", eng, "MHz", (eng || 0) / ENGINE_MAX, "", was || (r ? `of ${fmt(ENGINE_MAX)} MHz` : paused), null, stale));
  const mem = r && v.memory_mhz >= 0 ? v.memory_mhz : null;
  box.append(tile("Memory clock", mem, "MHz", (mem || 0) / MEMORY_MAX, "", was || (r ? `of ${fmt(MEMORY_MAX)} MHz` : paused), null, stale));
}

// ---------------------------------------------------------------- the temperature history

let chartKey = "";
let chartPts = [];
function renderChart(s, force) {
  const box = $("chart-box");
  const W = box.clientWidth, H = box.clientHeight;
  const now = Date.now() / 1000;
  const pts = (s.history || []).filter((p) => p.t >= now - 1800 && p.temp > 0 && p.temp < 200);
  const key = `${pts.length}:${pts.length ? pts[pts.length - 1].t : 0}:${W}:${s.settings.alert_c}`;
  if (!force && key === chartKey) return;
  chartKey = key;
  const svg = $("spark");
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  $("chart-empty").hidden = pts.length >= 2;
  chartPts = [];
  if (pts.length < 2) { $("chart-range").textContent = ""; return; }
  const t0 = Math.min(pts[0].t, now - 300), t1 = now; // at least the last 5 minutes
  const temps = pts.map((p) => p.temp);
  let lo = Math.min(...temps) - 3, hi = Math.max(...temps) + 3;
  const showLimit = s.settings.temp_alerts && s.settings.alert_c <= hi + 8;
  if (showLimit) hi = Math.max(hi, s.settings.alert_c + 1);
  const x = (t) => ((t - t0) / (t1 - t0)) * (W - 4) + 2;
  const y = (v) => H - 3 - ((v - lo) / (hi - lo)) * (H - 6);
  // break the line where readings stopped (another program had the card)
  const runs = [];
  let run = [];
  pts.forEach((p, i) => {
    const q = pts[i - 1];
    if (i && p.t - q.t > (p.src === "indexer" && q.src === "indexer" ? 150 : 40)) { runs.push(run); run = []; }
    run.push(p);
  });
  runs.push(run);
  const ns = "http://www.w3.org/2000/svg";
  for (const r of runs) {
    if (r.length < 2) continue;
    const line = r.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.temp).toFixed(1)}`).join("");
    const area = document.createElementNS(ns, "path");
    area.setAttribute("class", "area");
    area.setAttribute("d", `${line}L${x(r[r.length - 1].t).toFixed(1)},${H}L${x(r[0].t).toFixed(1)},${H}Z`);
    const path = document.createElementNS(ns, "path");
    path.setAttribute("class", "line");
    path.setAttribute("d", line);
    svg.append(area, path);
  }
  if (showLimit) {
    const l = document.createElementNS(ns, "line");
    l.setAttribute("class", "limit");
    l.setAttribute("x1", 0); l.setAttribute("x2", W);
    l.setAttribute("y1", y(s.settings.alert_c)); l.setAttribute("y2", y(s.settings.alert_c));
    svg.append(l);
  }
  chartPts = pts.map((p) => ({ ...p, x: x(p.t), y: y(p.temp) }));
  const minutes = Math.min(30, Math.round((t1 - t0) / 60));
  $("chart-range").textContent = `last ${minutes} min · ${Math.min(...temps)}–${Math.max(...temps)} °C`;
}

$("chart-box").addEventListener("mousemove", (e) => {
  if (chartPts.length < 2) return;
  const r = e.currentTarget.getBoundingClientRect();
  const mx = e.clientX - r.left;
  let best = chartPts[0];
  for (const p of chartPts) if (Math.abs(p.x - mx) < Math.abs(best.x - mx)) best = p;
  const svg = $("spark");
  svg.querySelector(".cross")?.remove();
  const ns = "http://www.w3.org/2000/svg";
  const c = document.createElementNS(ns, "line");
  c.setAttribute("class", "cross");
  c.setAttribute("x1", best.x); c.setAttribute("x2", best.x); c.setAttribute("y1", 0); c.setAttribute("y2", r.height);
  svg.append(c);
  let dot = e.currentTarget.querySelector(".dotmark");
  if (!dot) { dot = h("span", "dotmark"); e.currentTarget.append(dot); }
  dot.style.left = best.x + "px";
  dot.style.top = best.y + "px";
  const tip = $("chart-tip");
  tip.hidden = false;
  tip.replaceChildren();
  const when = new Date(best.t * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  tip.append(h("b", null, `${best.temp} °C`), document.createTextNode(` · ${when} · ${fmt(best.engine)} MHz${best.fan >= 0 ? ` · fan ${best.fan}%` : ""}`));
  tip.style.left = Math.max(70, Math.min(r.width - 70, best.x)) + "px";
});
$("chart-box").addEventListener("mouseleave", () => {
  $("chart-tip").hidden = true;
  $("spark").querySelector(".cross")?.remove();
  $("chart-box").querySelector(".dotmark")?.remove();
});

// ---------------------------------------------------------------- who has the card, eject and bring-up

function renderHolder(s) {
  const hd = s.holder;
  const show = !!hd && (s.mode === "busy" || s.mode === "ejecting");
  $("holder").hidden = !show;
  if (!show) return;
  const ic = { indexer: I.db, search: I.search, cli: I.term }[hd.kind] || I.app;
  $("holder-icon").replaceChildren(icon(ic));
  $("holder-name").textContent = cap(hd.name);
  $("holder-time").textContent = hd.started ? `for ${span(Date.now() / 1000 - hd.started)}` : "";
  $("holder-cmd").textContent = `pid ${hd.pid} · ${hd.command}`;
  $("holder-cmd").title = hd.command;
  let stats = "";
  const p = s.indexer;
  if (hd.kind === "indexer" && p) {
    stats = `${p.chunks_per_s.toFixed(1)} chunks/s · ${fmt(p.chunks_embedded)} chunks · ${fmt(p.files_read)} files · ${p.errors ? `${p.errors} errors` : "no errors"}`;
  } else if (hd.kind === "search") stats = "It lets go of the card 20 s after the last search.";
  $("holder-stats").textContent = stats;
  $("holder-stats").hidden = !stats;
}

function steps(list, at, title, started) {
  $("flow").hidden = false;
  $("flow-title").textContent = title;
  $("flow-time").textContent = started ? span(Date.now() / 1000 - started) : "";
  const ol = $("steps");
  ol.replaceChildren();
  list.forEach(([text, sub], i) => {
    const li = h("li", i < at ? "done" : i === at ? "now" : "");
    const mark = h("span", "mark");
    if (i < at) mark.append(icon(I.check));
    else if (i === at) mark.append(h("span", "spin"));
    li.append(mark, h("span", null, text));
    if (sub && i === at) li.append(h("span", "sub", sub));
    ol.append(li);
  });
}

let upStarted = 0;
function renderFlow(s) {
  $("flow").hidden = true;
  if (s.mode === "ejecting") {
    const waitFor = s.holder ? `Wait for ${s.holder.name} to finish` : "Check that nothing is using the card";
    steps([[waitFor], ["Stop the card's engines"], ["Stop the TinyGPU server"], ["Safe to unplug"]], s.eject.step, "Ejecting", s.eject.started);
  } else if (s.up_running) {
    if (!upStarted) upStarted = Date.now() / 1000;
    const at = { Connecting: 0, post: 1, "mc-load": 2, "dpm-enable": 3 }[s.up_step] ?? 0;
    steps([["Connect to the TinyGPU server"], ["Post the card", "VBIOS init"], ["Load the memory controller firmware"], ["Start power management"]],
      at, "Bringing up", upStarted);
  }
  if (!s.up_running) upStarted = 0;
}

function renderResult(s) {
  const r = $("result");
  r.hidden = s.mode !== "off";
  if (r.hidden) return;
  const done = s.eject.state === "done";
  const ic = $("result-icon");
  ic.className = "result-icon" + (done ? "" : " off");
  ic.replaceChildren(icon(done ? I.check : I.plug));
  $("result-title").textContent = done ? "Safe to unplug" : "Not connected";
  $("result-text").textContent = done
    ? "The card is closed and the TinyGPU server has stopped."
    : "Nothing is using the card, so it's safe to unplug. Connect when you want to use it.";
}

function act(label, ic, fn, { primary = false, disabled = false, title } = {}) {
  const b = h("button", "btn" + (primary ? " primary big" : ""));
  b.type = "button";
  if (ic) b.append(icon(ic));
  b.append(document.createTextNode(label));
  b.disabled = disabled;
  if (title) b.title = title;
  b.onclick = fn;
  return b;
}

function renderActions(s) {
  const box = $("actions");
  const sec = h("div", "secondary");
  let primary = null;
  const search = () => act("Code Search", I.search, () => invoke("open_code_search"));
  const details = () => act("Details", I.details, () => invoke("open_details"));
  const connect = (p) => act(s.mode === "not-posted" ? "Bring Up Card" : "Connect", I.power, () => invoke("up"), { primary: p, disabled: !s.up_enabled });
  const ejectBtn = (p, label) => act(label || "Eject", I.eject, () => invoke("eject"), { primary: p, disabled: !s.eject_enabled,
    title: "Waits for any program using the card, closes it and stops the TinyGPU server" });
  if (s.up_running) {
    primary = act("Bringing up…", null, () => {}, { primary: true, disabled: true });
  } else switch (s.mode) {
    case "ejecting":
      primary = s.eject.step === 0
        ? act("Cancel Eject", null, () => invoke("cancel"), { primary: false })
        : act("Finishing the eject…", null, () => {}, { disabled: true });
      primary.classList.add("big");
      break;
    case "busy":
      primary = ejectBtn(true, s.holder ? `Eject When ${s.holder.name.replace(/^the /, "")} Finishes` : "Eject When the Card Is Free");
      sec.append(search(), details());
      break;
    case "off":
      primary = connect(true);
      sec.append(search(), details());
      if (s.eject.state === "done") sec.prepend(act("Done", null, () => invoke("dismiss")));
      break;
    case "not-posted":
      primary = connect(true);
      sec.append(ejectBtn(false), details());
      break;
    case "setup": case "no-card":
      primary = act("Details", I.details, () => invoke("open_details"));
      primary.classList.add("big");
      break;
    case "other": case "cli-error":
      primary = act("Details", I.details, () => invoke("open_details"), { primary: true });
      break;
    default: // idle, reading, not-answering
      primary = ejectBtn(true, "Eject");
      sec.append(search(), details());
  }
  box.replaceChildren(primary, sec);
}

function renderSettings(s) {
  $("settings").hidden = !settingsOpen;
  $("b-settings").classList.toggle("on", settingsOpen);
  if (!settingsOpen) return;
  $("s-login").checked = s.launch_at_login;
  $("s-login").disabled = !s.installed;
  $("login-note").textContent = s.installed ? "" : "Install the app first: make install-apps";
  $("s-temp").checked = s.settings.menu_bar_temp;
  $("s-alerts").checked = s.settings.temp_alerts;
  $("s-alert-c").value = String(s.settings.alert_c);
  $("s-indexer").checked = s.settings.notify_indexer;
  $("cli-path").textContent = s.rawgpu;
  $("cli-path").title = s.rawgpu;
}

function renderFoot(s) {
  let t = "";
  const r = readings(s);
  if (r && r.src === "card") t = `Updated ${ago(r.at)}`;
  else if (r && r.src === "stale") t = `Readings paused · last read ${ago(r.at)}`;
  else if (r && r.src === "indexer") t = `From the indexer's log, ${ago(r.at)}`;
  else if (s.mode === "off") t = "TinyGPU server stopped";
  else if (s.mode === "busy") t = "Readings paused";
  $("updated").textContent = t;
}

// ---------------------------------------------------------------- setup: TinyGPU, the card, firmware

function renderSetup(s) {
  const d = s.doctor;
  const keys = ["tinygpu_installed", "tinygpu_enabled", "card_bound", "firmware_ok"];
  const show = !!d && keys.some((k) => d[k] === false);
  $("setup").hidden = !show;
  if (!show) return;
  const run = s.setup_running;
  const rows = [
    [d.tinygpu_installed, "TinyGPU driver", "tinygrad's signed driver, verified before it's installed",
      run === "install-tinygpu" ? "spin" : ["Install", () => invoke("setup", { action: "install-tinygpu" })]],
    [d.tinygpu_installed && d.tinygpu_enabled, "Driver approved", "System Settings › General › Login Items & Extensions › Driver Extensions",
      d.tinygpu_installed ? ["Open Settings", () => invoke("setup", { action: "open-settings" })] : null],
    [d.card_bound, "Card attached", "Plug in the enclosure with the card and switch it on", null],
    [d.firmware_ok, "AMD firmware", "Needed to bring the card up; downloaded from linux-firmware and checked",
      run === "fetch-firmware" ? "spin" : ["Download", () => invoke("setup", { action: "fetch-firmware" })]],
  ];
  const ol = $("checks");
  ol.replaceChildren();
  for (const [ok, title, hint, action] of rows) {
    const li = h("li", ok ? "ok" : "");
    const mark = h("span", "mark");
    if (ok) mark.append(icon(I.check));
    const text = h("span", "text");
    text.append(document.createTextNode(title));
    if (!ok) text.append(h("small", null, hint));
    li.append(mark, text);
    if (!ok && action === "spin") li.append(h("span", "spin"));
    else if (!ok && action) {
      const b = h("button", "btn", action[0]);
      b.type = "button";
      b.disabled = !!run;
      b.onclick = action[1];
      li.append(b);
    }
    ol.append(li);
  }
}

// ---------------------------------------------------------------- render

let fitH = 0;
function fit() {
  requestAnimationFrame(() => {
    const hgt = Math.ceil($("root").getBoundingClientRect().height);
    if (Math.abs(hgt - fitH) > 1) { fitH = hgt; invoke("fit_panel", { height: hgt }); }
  });
}

let cancelledTimer = null;
function render(s) {
  S = s;
  const v = (s.status && s.status.vram_mb) ? ` · ${fmt(s.status.vram_mb / 1024)} GB VRAM` : "";
  $("sub").textContent = `AMD Polaris 10${v}`;
  pill(s);
  alertBox(s);
  renderSetup(s);
  const main = ["idle", "reading", "busy", "not-posted", "not-answering"].includes(s.mode) && !s.up_running;
  $("main-view").hidden = !main;
  if (main) {
    renderTiles(s);
    $("chart").hidden = !["idle", "busy", "reading"].includes(s.mode);
    if (!$("chart").hidden) renderChart(s);
  }
  renderHolder(s);
  if (!$("holder").hidden && $("main-view").hidden) $("flow").before($("holder")); else $("main-view").append($("holder"));
  renderFlow(s);
  renderResult(s);
  renderActions(s);
  renderSettings(s);
  renderFoot(s);
  if (s.eject.state === "cancelled" && !cancelledTimer) cancelledTimer = setTimeout(() => { cancelledTimer = null; invoke("dismiss"); }, 4000);
  fit();
}

$("b-settings").onclick = () => { settingsOpen = !settingsOpen; if (S) render(S); };
$("quit").onclick = () => invoke("quit");
const setting = (key, value) => invoke("set_setting", { key, value }).catch((e) => { $("login-note").textContent = String(e); fit(); });
$("s-login").onchange = (e) => setting("launch_at_login", e.target.checked);
$("s-temp").onchange = (e) => setting("menu_bar_temp", e.target.checked);
$("s-alerts").onchange = (e) => setting("temp_alerts", e.target.checked);
$("s-alert-c").onchange = (e) => setting("alert_c", Number(e.target.value));
$("s-indexer").onchange = (e) => setting("notify_indexer", e.target.checked);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") invoke("close_panel");
  else if (e.metaKey && e.key === ",") { settingsOpen = !settingsOpen; if (S) render(S); }
});
window.addEventListener("resize", () => S && renderChart(S, true));
setInterval(() => { if (S) { renderFoot(S); renderHolder(S); if (!$("flow").hidden) renderFlow(S); } }, 1000);

listen("state", (e) => render(e.payload));
invoke("get_state").then(render);
