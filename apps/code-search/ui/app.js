const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

let backend = store.get("backend", "rawgpu");
let results = [];
let sel = -1;
let busy = false;
let ready = false;
const NAMES = { rawgpu: "RX 590", mps: "Mac GPU", cpu: "Mac CPU" };

function setBackend(b) {
  backend = b;
  store.set("backend", b);
  for (const btn of document.querySelectorAll(".seg button")) btn.setAttribute("aria-checked", btn.dataset.backend === b);
  pollCard();
  if (b !== "rawgpu" && !warmed.has(b)) { warmed.add(b); invoke("warm", { backend: b }).catch(() => warmed.delete(b)); }
}
const warmed = new Set();
for (const btn of document.querySelectorAll(".seg button")) btn.onclick = () => setBackend(btn.dataset.backend);

function status(text) { $("status").textContent = text; }
function showError(text, { mac = false, restart = false } = {}) {
  $("error").hidden = !text;
  $("error-text").textContent = text || "";
  $("use-mac").hidden = !mac;
  $("restart").hidden = !restart;
}

function indexSummary(ix) {
  if (!ix) return;
  ready = true;
  if (pending) { pending = null; setTimeout(run, 0); }
  const gb = (ix.bytes / 1e9).toFixed(2);
  const line = `${ix.chunks.toLocaleString()} chunks from ${ix.files.toLocaleString()} files in ${ix.projects.length} projects (${gb} GB), under ${ix.root}`;
  $("index-line").textContent = line;
  status("Ready. " + line);
  const dl = $("projects");
  dl.replaceChildren();
  for (const [p, n] of ix.projects.slice().sort((a, b) => b[1] - a[1])) {
    const o = document.createElement("option");
    o.value = p;
    o.label = `${n.toLocaleString()} chunks`;
    dl.append(o);
  }
}

async function loadEditors() {
  const eds = await invoke("editors");
  const s = $("editor");
  for (const [id, name] of eds) { const o = document.createElement("option"); o.value = id; o.textContent = name; s.append(o); }
  s.value = store.get("editor", eds[0][0]);
  s.onchange = () => store.set("editor", s.value);
}

function short(cmd) {
  if (/index_code\.py\s+run/.test(cmd)) return "the code indexer";
  if (/search_server\.py/.test(cmd)) return "Code Search";
  const m = cmd.match(/([\w.-]+\.py)/);
  return m ? m[1] : cmd.split(/\s+/)[0].split("/").pop();
}

async function pollCard() {
  const el = $("card");
  if (backend !== "rawgpu") { el.textContent = `Queries are embedded with PyTorch on the ${backend === "mps" ? "Mac's GPU" : "Mac's CPU"}`; return; }
  try {
    const w = await invoke("card_state");
    if (!w.server_pid) el.textContent = "RX 590 not connected: the first search connects it";
    else if (!w.client_count) el.textContent = "RX 590 free";
    else {
      const c = (w.clients || [])[0];
      const who = c ? short(c.command) : "another program";
      el.textContent = who === "Code Search" ? "RX 590 held by this app (released after 20 s idle)" : `RX 590 in use by ${who}`;
    }
  } catch (e) { el.textContent = String(e); }
}
setInterval(pollCard, 4000);

function lines(text, start) {
  const pre = document.createElement("pre");
  pre.className = "code";
  text.split("\n").forEach((t, i) => {
    const row = document.createElement("span"); row.className = "l";
    const ln = document.createElement("span"); ln.className = "ln"; ln.textContent = start + i;
    row.append(ln, document.createTextNode(t));
    pre.append(row);
  });
  return pre;
}

function button(label, fn, title) {
  const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.onclick = fn; if (title) b.title = title; return b;
}

async function openSel(r) {
  try { await invoke("open_in_editor", { path: r.abs_path, line: r.start, editor: $("editor").value }); showError(""); }
  catch (e) { showError(String(e)); }
}

function preview(i) {
  const p = $("preview");
  p.replaceChildren();
  const r = results[i];
  if (!r) { p.append($("empty")); return; }
  const head = document.createElement("div"); head.className = "p-head";
  const left = document.createElement("div");
  const path = document.createElement("div"); path.className = "p-path"; path.textContent = `${r.path}:${r.start}-${r.end}`;
  const meta = document.createElement("div"); meta.className = "p-meta";
  meta.textContent = `${r.project} · score ${r.score.toFixed(4)} · ${r.tokens} tokens`;
  left.append(path, meta);
  const acts = document.createElement("div"); acts.className = "p-actions";
  acts.append(
    button("Open", () => openSel(r), "⌘↩"),
    button("Reveal in Finder", () => invoke("reveal", { path: r.abs_path })),
    button("Copy path", () => navigator.clipboard.writeText(`${r.abs_path}:${r.start}`)),
  );
  head.append(left, acts);
  p.append(head, lines(r.text, r.start));
  if (r.copies && r.copies.length) {
    const also = document.createElement("div"); also.className = "also";
    also.append(document.createTextNode(`The same text is also in ${r.copies.length} other place${r.copies.length > 1 ? "s" : ""}: `));
    r.copies.forEach((c, j) => { const code = document.createElement("code"); code.textContent = c; also.append(code); if (j < r.copies.length - 1) also.append(", "); });
    p.append(also);
  }
  p.scrollTop = 0;
}

function select(i) {
  if (!results.length) return;
  sel = Math.max(0, Math.min(results.length - 1, i));
  const items = $("results").children;
  for (let j = 0; j < items.length; j++) items[j].classList.toggle("sel", j === sel);
  items[sel].scrollIntoView({ block: "nearest" });
  preview(sel);
}

function renderResults() {
  const ol = $("results");
  ol.replaceChildren();
  results.forEach((r, i) => {
    const li = document.createElement("li");
    const top = document.createElement("div"); top.className = "r-top";
    const score = document.createElement("span"); score.className = "score"; score.textContent = r.score.toFixed(3);
    const cut = r.path.lastIndexOf("/");
    const name = document.createElement("span"); name.className = "name"; name.textContent = `${r.path.slice(cut + 1)}:${r.start}`;
    top.append(score, name);
    const dir = document.createElement("div"); dir.className = "dir"; dir.textContent = cut > 0 ? r.path.slice(0, cut) : r.project;
    const snip = document.createElement("div"); snip.className = "snippet"; snip.textContent = r.text.trim().slice(0, 400);
    li.append(top, dir, snip);
    if (r.copies && r.copies.length) {
      const c = document.createElement("div"); c.className = "copies"; c.textContent = `+ ${r.copies.length} identical cop${r.copies.length > 1 ? "ies" : "y"} elsewhere`; li.append(c);
    }
    li.onclick = () => select(i);
    li.ondblclick = () => openSel(r);
    ol.append(li);
  });
  if (results.length) select(0); else { preview(-1); }
}

async function run() {
  const q = $("q").value.trim();
  if (!q || busy) return;
  busy = true;
  $("go").disabled = true;
  showError("");
  status(`Searching on the ${NAMES[backend]}…`);
  const t0 = performance.now();
  try {
    const project = $("project").value.trim() || null;
    const r = await invoke("search", { query: q, backend, k: Number($("k").value), project, dedupe: $("dedupe").checked });
    results = r.results;
    const total = Math.round(performance.now() - t0);
    status(`${results.length} results${project ? ` in ${project}` : ""}. ${NAMES[r.backend]}: embedding ${r.embed_ms} ms, search ${r.search_ms} ms (${total} ms in all)`);
    renderResults();
  } catch (e) {
    const msg = String(e);
    status("");
    showError(msg, { mac: /in use by|RX 590/.test(msg) && backend === "rawgpu", restart: /sidecar|exited/.test(msg) });
  } finally {
    busy = false;
    $("go").disabled = false;
    pollCard();
  }
}

$("form").onsubmit = (e) => { e.preventDefault(); run(); };
$("use-mac").onclick = () => { setBackend("mps"); showError(""); run(); };
$("restart").onclick = async () => { showError(""); status("Restarting the search backend…"); try { await invoke("restart_sidecar"); } catch (e) { showError(String(e), { restart: true }); } };
$("log").onclick = () => invoke("open_log");
$("project").addEventListener("change", () => { if ($("q").value.trim()) run(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); select(sel + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); select(sel - 1); }
  else if (e.key === "Enter" && e.metaKey && results[sel]) { e.preventDefault(); openSel(results[sel]); }
  else if (e.key === "Escape") { $("q").select(); $("q").focus(); }
});

listen("sidecar", (e) => {
  const m = e.payload;
  if (m.event === "status") status(m.text);
  else if (m.event === "ready") indexSummary(m.index);
  else if (m.event === "card") pollCard();
  else if (m.event === "exit") { ready = false; showError("The search backend stopped.", { restart: true }); }
});

let pending = null; // a query from the command line, run once the index is loaded
async function start() {
  const a = await invoke("launch_args");
  setBackend(NAMES[a.backend] ? a.backend : backend);
  if (a.query) { $("q").value = a.query; pending = a.query; }
  loadEditors();
  try {
    const ix = await invoke("index_info");
    if (ix) indexSummary(ix); else status("Loading the index…");
  } catch (e) { showError(String(e), { restart: true }); }
}
start();
