const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);

function grid(el, rows) {
  el.replaceChildren();
  for (const [k, v] of rows) {
    if (v === undefined || v === null || v === "") continue;
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    el.append(dt, dd);
  }
}

const n = (x) => (typeof x === "number" ? x.toLocaleString() : x);
const yes = (b) => (b ? "yes" : "no");

function render(s) {
  $("dot").className = "dot " + (s.level || "off");
  $("headline").textContent = s.headline || "RX 590";
  $("detail").textContent = s.detail || "";
  $("user").textContent = s.user || "";

  const eject = $("eject");
  const running = s.eject && s.eject.state === "running";
  eject.textContent = running ? "Cancel Eject" : "Eject";
  eject.disabled = !s.eject_enabled;
  eject.title = running ? "Stops waiting. Once the card is being closed, the eject finishes first." : "Waits for any program using the card, closes it, and stops the TinyGPU server";
  const up = $("up");
  up.textContent = s.up_running ? "Bringing Up…" : s.up_label || "Connect and Bring Up";
  up.disabled = !s.up_enabled;

  const st = s.status;
  if (st) {
    grid($("card"), [
      ["Posted", yes(st.posted)],
      ["VRAM", st.vram_mb ? `${n(st.vram_mb)} MB` : ""],
      ["SMC firmware", st.smc_firmware === undefined ? "" : yes(st.smc_firmware)],
      ["DPM", st.engine_dpm === undefined ? "" : `engine ${st.engine_dpm ? "on" : "off"}, memory ${st.memory_dpm ? "on" : "off"}`],
      ["Engine clock", st.engine_mhz >= 0 ? `${st.engine_mhz} MHz` : ""],
      ["Memory clock", st.memory_mhz >= 0 ? `${st.memory_mhz} MHz` : ""],
      ["Temperature", st.temperature_c !== undefined ? `${st.temperature_c} °C` : ""],
      ["Fan", st.fan_percent >= 0 ? `${st.fan_percent}%` : st.fan_percent === -1 ? "not under SMC control" : ""],
      ["Shader engines busy", st.busy === undefined ? "" : yes(st.busy)],
    ]);
    $("status-age").textContent = s.status_at ? `read ${Math.max(0, Math.round(Date.now() / 1000 - s.status_at))} s ago` : "";
  } else {
    grid($("card"), [["", s.status_error || (s.level === "busy" ? "Readings pause while another program has the card." : "Not connected.")]]);
    $("status-age").textContent = "";
  }

  const p = s.indexer;
  $("indexer-section").hidden = !p;
  if (p) {
    grid($("indexer"), [
      ["Rate", `${p.chunks_per_s} chunks/s, ${n(p.tokens_per_s)} tokens/s`],
      ["Embedded", `${n(p.chunks_embedded)} chunks from ${n(p.files_read)} files`],
      ["Card", `${p.engine_mhz} / ${p.memory_mhz} MHz, ${p.temperature_c} °C, fan ${p.fan_percent}%`],
      ["Host share", `${Math.round(p.host_share * 100)}%`],
      ["Errors", n(p.errors)],
      ["Running for", `${(p.elapsed_s / 3600).toFixed(2)} h`],
    ]);
  }

  const w = s.who || {};
  const clients = (w.clients || []).map((c) => `pid ${c.pid}: ${c.command.replace(/\s+/g, " ")}`).join("\n");
  grid($("server"), [
    ["Server", w.server_pid ? `pid ${w.server_pid}` : s.who_error ? "" : "not running"],
    ["Socket", w.socket],
    ["Using the card", w.server_pid ? clients || (w.client_count ? `${w.client_count} connection(s)` : "nothing") : ""],
    ["Other TinyGPU", w.other_tinygpu ? `${w.other_tinygpu} process(es)` : ""],
    ["Error", s.who_error],
    ["CLI", s.rawgpu],
  ]);

  const log = $("log");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = s.log && s.log.length ? s.log.join("\n") : "Nothing yet.";
  if (atBottom) log.scrollTop = log.scrollHeight;
}

$("eject").onclick = async () => {
  const s = await invoke("get_state");
  if (s.eject && s.eject.state === "running") await invoke("cancel");
  else await invoke("eject");
};
$("up").onclick = () => invoke("up");
$("refresh").onclick = () => invoke("refresh_now");
$("search").onclick = () => invoke("open_code_search");
$("logs").onclick = () => invoke("open_logs");
$("full").onclick = async () => {
  const out = $("full-out");
  out.hidden = false;
  out.textContent = "Reading…";
  try { out.textContent = await invoke("full_status"); } catch (e) { out.textContent = String(e); }
};

listen("state", (e) => render(e.payload));
invoke("get_state").then(render);
