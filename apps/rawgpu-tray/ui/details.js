// The details window: activity log, the card's readings and full status, and the TinyGPU server.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n);
function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function grid(el, rows) {
  el.replaceChildren();
  for (const [k, v, mono] of rows) {
    if (v === undefined || v === null || v === "") continue;
    el.append(h("dt", null, k), h("dd", mono ? "mono" : null, v));
  }
}
const PILL = {
  idle: ["good", "Ready"], reading: ["", "Reading…"], busy: ["busy", "In use"], ejecting: ["busy", "Ejecting"], off: ["", "Disconnected"],
  other: ["warn", "Not safe to unplug"], "not-posted": ["warn", "Needs bring-up"], "not-answering": ["critical", "Not answering"],
  "cli-error": ["critical", "CLI error"],
};

let S = null;
let logLen = -1;
function render(s) {
  S = s;
  const [cls, text] = s.mode === "off" && s.eject.state === "done" ? ["good", "Safe to unplug"] : PILL[s.mode] || ["", s.mode];
  $("pill").className = "pill " + cls;
  $("pill-text").textContent = text;
  $("headline").textContent = s.detail || s.user || "";

  if (s.log.length !== logLen) {
    logLen = s.log.length;
    const log = $("log");
    const atBottom = log.parentElement.scrollTop + log.parentElement.clientHeight >= log.parentElement.scrollHeight - 8;
    log.replaceChildren();
    if (!s.log.length) log.append(h("div", "empty", "Ejects and bring-ups started from the menu bar show their output here."));
    for (const l of s.log) {
      const c = l.startsWith("$ ") ? "cmd" : /^\[(eject done|up done)\]/.test(l) ? "end-ok" : /^\[(eject|up) (failed|cancelled)\]/.test(l) ? "end-bad" : null;
      log.append(h("div", c, l));
    }
    $("log-count").textContent = s.log.length ? `${s.log.length} lines` : "";
    if (atBottom) log.parentElement.scrollTop = log.parentElement.scrollHeight;
  }

  const st = s.status;
  grid($("card"), st ? [
    ["Posted", st.posted ? "yes" : "no: the card needs bring-up"],
    ["VRAM", st.vram_mb ? `${fmt(st.vram_mb)} MB` : ""],
    ["SMC firmware", st.smc_firmware === undefined ? "" : st.smc_firmware ? "running" : "not running"],
    ["Power management", st.engine_dpm === undefined ? "" : `engine DPM ${st.engine_dpm ? "on" : "off"}, memory DPM ${st.memory_dpm ? "on" : "off"}`],
    ["Engine clock", st.engine_mhz >= 0 ? `${fmt(st.engine_mhz)} MHz` : ""],
    ["Memory clock", st.memory_mhz >= 0 ? `${fmt(st.memory_mhz)} MHz` : ""],
    ["Temperature", st.temperature_c !== undefined ? `${st.temperature_c} °C` : ""],
    ["Fan", st.fan_percent >= 0 ? `${st.fan_percent}%` : st.fan_percent === -1 ? "not under SMC control" : ""],
    ["Shader engines", st.busy === undefined ? "" : st.busy ? "busy" : "idle"],
  ] : [["Readings", s.status_error || (s.mode === "busy" ? "Paused while another program has the card." : "Not connected.")]]);
  $("card-age").textContent = s.status_at ? `Read at ${new Date(s.status_at * 1000).toLocaleTimeString()}` : "";

  const w = s.who || {};
  const hd = s.holder;
  grid($("server"), [
    ["Server", w.server_pid ? `running, pid ${w.server_pid}` : s.who_error ? "" : "not running"],
    ["Using the card", w.server_pid ? (hd ? `${hd.name}, pid ${hd.pid}` : w.client_count ? `${w.client_count} connection(s)` : "nothing") : ""],
    ["Since", hd && hd.started ? new Date(hd.started * 1000).toLocaleString() : ""],
    ["Command", hd ? hd.command : "", true],
    ["Socket", w.socket, true],
    ["Other TinyGPU", w.other_tinygpu ? `${w.other_tinygpu} process(es)` : ""],
    ["Error", s.who_error],
    ["rawgpu CLI", s.rawgpu, true],
  ]);
}

for (const tab of document.querySelectorAll(".tabs button")) {
  tab.onclick = () => {
    for (const t of document.querySelectorAll(".tabs button")) t.setAttribute("aria-selected", t === tab);
    for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== tab.dataset.tab;
  };
}
$("refresh").onclick = () => invoke("refresh_now");
$("logs").onclick = () => invoke("open_logs");
$("copy-log").onclick = () => S && navigator.clipboard.writeText(S.log.join("\n"));
$("full").onclick = async () => {
  const out = $("full-out");
  out.hidden = false;
  out.textContent = "Reading…";
  try { out.textContent = await invoke("full_status"); } catch (e) { out.textContent = String(e); }
};
listen("state", (e) => render(e.payload));
invoke("get_state").then(render);
