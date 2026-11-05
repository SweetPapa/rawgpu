// Code Search: the window. The search runs in the app's Python sidecar (see src/main.rs); this file is the UI.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);
function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
function svg(paths, cls) {
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
const ICON = {
  open: ["M9 3h4v4", "M13 3 7.5 8.5", "M11 9.5V13H3V5h3.5"],
  finder: ["M2.5 4.5h4l1.2 1.5h5.8v6.5h-11z"],
  copy: ["M5.5 5.5h7v7h-7z", "M3.5 10.5v-7h7"],
  up: ["m4 10 4-4 4 4"],
  down: ["m4 6 4 4 4-4"],
  chevron: ["m5 6.5 3 3 3-3"],
  warn: ["M8 2 14.5 13.5h-13z", "M8 6.5v3.2m0 2v.1"],
};
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : String(n));
const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many || one + "s"}`;

const NAMES = { rawgpu: "RX 590", mps: "Mac GPU", cpu: "Mac CPU" };
const EXAMPLES = [
  "where do we retry failed network requests",
  "stream microphone audio over a websocket",
  "terraform for a Cloud Run service",
  "safely eject the GPU before unplugging",
];
const LANG_ALIAS = {
  py: "Python", python: "Python", c: "C", "c++": "C++", cpp: "C++", cc: "C++", ts: "TypeScript", tsx: "TypeScript",
  typescript: "TypeScript", js: "JavaScript", jsx: "JavaScript", javascript: "JavaScript", rs: "Rust", rust: "Rust",
  go: "Go", swift: "Swift", kt: "Kotlin", kotlin: "Kotlin", java: "Java", "c#": "C#", cs: "C#", csharp: "C#",
  md: "Markdown", markdown: "Markdown", sh: "Shell", bash: "Shell", shell: "Shell", zsh: "Shell", tf: "Terraform",
  terraform: "Terraform", yml: "YAML", yaml: "YAML", json: "JSON", html: "HTML", css: "CSS", vue: "Vue",
  objc: "Objective-C", "objective-c": "Objective-C", sql: "SQL", gd: "GDScript", gdscript: "GDScript",
  dart: "Dart", lua: "Lua", rb: "Ruby", ruby: "Ruby", php: "PHP", cl: "OpenCL", opencl: "OpenCL", xojo: "Xojo",
};
const SHORT = {
  Python: "PY", TypeScript: "TS", JavaScript: "JS", Markdown: "MD", "C++": "C++", "C#": "C#", "Objective-C": "OBJC",
  JSON: "{ }", YAML: "YML", Shell: "SH", Terraform: "TF", Kotlin: "KT", Swift: "SWFT", Rust: "RS", Other: "·",
  Text: "TXT", GDScript: "GD", Dockerfile: "DOCK", Protobuf: "PB", "Linker script": "LD", "Xcode project": "XC",
};
const short = (lang) => SHORT[lang] || lang.slice(0, 4).toUpperCase();
const HL_EXT = {
  py: "python", pyi: "python", gd: "python", c: "c", h: "c", cl: "c", glsl: "c", hlsl: "c", cc: "cpp", cpp: "cpp",
  cxx: "cpp", hpp: "cpp", hh: "cpp", ino: "cpp", metal: "cpp", m: "objectivec", mm: "objectivec", swift: "swift",
  rs: "rust", go: "go", java: "java", kt: "kotlin", kts: "kotlin", cs: "csharp", js: "javascript", mjs: "javascript",
  cjs: "javascript", jsx: "javascript", ts: "typescript", tsx: "typescript", mts: "typescript", vue: "xml",
  svelte: "xml", html: "xml", htm: "xml", xml: "xml", plist: "xml", xsd: "xml", storyboard: "xml", css: "css",
  scss: "scss", less: "less", md: "markdown", mdx: "markdown", json: "json", jsonc: "json", yaml: "yaml",
  yml: "yaml", toml: "ini", ini: "ini", cfg: "ini", sh: "bash", bash: "bash", zsh: "bash", sql: "sql", lua: "lua",
  rb: "ruby", php: "php", r: "r", dart: "dart", cmake: "cmake", mk: "makefile", proto: "protobuf",
  graphql: "graphql", gql: "graphql", gradle: "gradle", tex: "latex", sty: "latex", diff: "diff", patch: "diff",
};
const HL_NAME = { makefile: "makefile", gnumakefile: "makefile", dockerfile: "dockerfile", "cmakelists.txt": "cmake" };
function hlLang(path) {
  const base = path.split("/").pop().toLowerCase();
  const name = HL_NAME[base] || (base.startsWith("dockerfile") ? "dockerfile" : HL_EXT[base.includes(".") ? base.split(".").pop() : ""]);
  return name && window.hljs && hljs.getLanguage(name) ? name : null;
}

// ---------------------------------------------------------------- state

const S = {
  backend: store.get("backend", "rawgpu"),
  filters: { projects: [], langs: [], path: "" },
  group: store.get("group", true),
  dedupe: store.get("dedupe", true),
  k: store.get("k", 20),
  editor: store.get("editor", null),
  editors: [],
  index: null,
  groups: [],
  sel: -1,
  match: 0,
  seq: 0,
  lastKey: "",
  searching: false,
  last: null,         // the last result: {count, backend, embed_ms, search_ms}
  heldUntil: 0,       // when the sidecar will release the RX 590
  card: null,         // `rawgpu who` from the last poll
  warm: {},           // backend -> "loading" | "ready"
  recent: store.get("recent", []),
  sources: new Map(), // abs path -> {text, modified} | {error}
};

// ---------------------------------------------------------------- small UI pieces

let toastTimer;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 1400);
}

function banner(text, actions = []) {
  $("banner").hidden = !text;
  $("banner-text").textContent = text || "";
  const box = $("banner-actions");
  box.replaceChildren();
  for (const [label, fn] of actions) {
    const b = h("button", "btn", label);
    b.type = "button";
    b.onclick = fn;
    box.append(b);
  }
}

function button(label, icon, fn, title, cls = "btn") {
  const b = h("button", cls);
  b.type = "button";
  if (icon) b.append(svg(ICON[icon]));
  if (label) b.append(document.createTextNode(label));
  if (title) b.title = title;
  b.onclick = fn;
  return b;
}

// A popover anchored under an element. items: [{value, label, count}] or a custom body.
const pop = { el: null, anchor: null, onKey: null };
function closePopover() {
  $("popover").hidden = true;
  pop.anchor = null;
  pop.onKey = null;
}
function placePopover(anchor) {
  const p = $("popover");
  const r = anchor.getBoundingClientRect();
  p.hidden = false;
  const w = p.offsetWidth;
  p.style.left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8)) + "px";
  p.style.top = r.bottom + 6 + "px";
  pop.anchor = anchor;
}
function picker(anchor, { items, selected, placeholder, onChange }) {
  if (pop.anchor === anchor) return closePopover();
  const p = $("popover");
  p.replaceChildren();
  const input = h("input", "pop-search");
  input.placeholder = placeholder;
  input.spellcheck = false;
  const list = h("div", "pop-list");
  const foot = h("div", "pop-foot");
  const clear = h("button", "link-btn", "Clear");
  clear.type = "button";
  const hint = h("span", "pop-count", "↩ toggles · Esc closes");
  foot.append(hint, clear);
  p.append(input, list, foot);
  let shown = [];
  let active = 0;
  const sel = new Set(selected);
  function draw() {
    const q = input.value.trim().toLowerCase();
    shown = items.filter((it) => !q || it.label.toLowerCase().includes(q));
    shown.sort((a, b) => (sel.has(b.value) - sel.has(a.value)) || 0);
    active = Math.min(active, Math.max(0, shown.length - 1));
    list.replaceChildren();
    if (!shown.length) list.append(h("div", "pop-empty", "No matches"));
    shown.slice(0, 400).forEach((it, i) => {
      const row = h("div", "pop-item" + (i === active ? " active" : ""));
      row.append(h("span", "check", sel.has(it.value) ? "✓" : ""), h("span", "pop-name", it.label), h("span", "pop-count", fmt(it.count)));
      row.onmouseenter = () => { active = i; for (const [j, c] of [...list.children].entries()) c.classList.toggle("active", j === i); };
      row.onclick = () => toggle(it.value);
      list.append(row);
    });
    clear.hidden = !sel.size;
  }
  function toggle(v) {
    if (sel.has(v)) sel.delete(v); else sel.add(v);
    onChange([...sel]);
    draw();
  }
  clear.onclick = () => { sel.clear(); onChange([]); draw(); };
  input.oninput = () => { active = 0; draw(); };
  pop.onKey = (e) => {
    if (e.key === "ArrowDown") { active = Math.min(shown.length - 1, active + 1); draw(); list.children[active]?.scrollIntoView({ block: "nearest" }); }
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); draw(); list.children[active]?.scrollIntoView({ block: "nearest" }); }
    else if (e.key === "Enter" && shown[active]) toggle(shown[active].value);
    else return false;
    return true;
  };
  draw();
  placePopover(anchor);
  input.focus();
}

function pathPopover(anchor) {
  if (pop.anchor === anchor) return closePopover();
  const p = $("popover");
  p.replaceChildren();
  const input = h("input", "pop-search");
  input.placeholder = "e.g. src/ or firmware";
  input.value = S.filters.path;
  input.spellcheck = false;
  const apply = () => { setFilter("path", input.value.trim()); closePopover(); };
  input.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); apply(); } };
  const foot = h("div", "pop-foot");
  const clear = h("button", "link-btn", "Clear");
  clear.type = "button";
  clear.onclick = () => { setFilter("path", ""); closePopover(); };
  const ok = h("button", "btn primary", "Apply");
  ok.type = "button";
  ok.onclick = apply;
  foot.append(clear, ok);
  p.append(h("div", "pop-hint", "Only files whose path contains this text"), input, foot);
  placePopover(anchor);
  input.focus();
  input.select();
}

function editorMenu(anchor, r) {
  if (pop.anchor === anchor) return closePopover();
  const p = $("popover");
  p.replaceChildren();
  const list = h("div", "pop-list");
  for (const [id, name] of S.editors) {
    const row = h("div", "pop-item");
    row.append(h("span", "check", id === S.editor ? "✓" : ""), h("span", "pop-name", name));
    row.onmouseenter = () => row.classList.add("active");
    row.onmouseleave = () => row.classList.remove("active");
    row.onclick = () => { S.editor = id; store.set("editor", id); closePopover(); renderPreview(); openCurrent(); };
    list.append(row);
  }
  p.append(h("div", "pop-hint", "Open files in"), list);
  placePopover(anchor);
}

document.addEventListener("mousedown", (e) => {
  if (!$("popover").hidden && !$("popover").contains(e.target) && !(pop.anchor && pop.anchor.contains(e.target))) closePopover();
});

// ---------------------------------------------------------------- filters and query tokens

function setFilter(kind, value) {
  S.filters[kind] = value;
  renderFilters();
  schedule(true);
}

function renderFilters() {
  const f = S.filters;
  const set = (id, on, text) => {
    const c = $(id);
    c.classList.toggle("on", on);
    c.querySelector(".chip-v").textContent = text;
  };
  const many = (xs, noun) => (xs.length === 0 ? "Any" : xs.length <= 2 ? xs.join(", ") : `${xs.length} ${noun}`);
  set("f-project", f.projects.length > 0, many(f.projects, "projects"));
  set("f-lang", f.langs.length > 0, many(f.langs, "languages"));
  set("f-path", !!f.path, f.path || "Any");
  $("f-clear").hidden = !(f.projects.length || f.langs.length || f.path);
}

// `in:rawgpu`, `lang:py` and `path:src/` typed in the query become filters once followed by a space.
const TOKEN = /(^|\s)(in|project|lang|language|path):(\S+)(?=\s)/i;
function takeTokens(final) {
  const input = $("q");
  let v = input.value;
  const re = final ? /(^|\s)(in|project|lang|language|path):(\S+)(?=\s|$)/i : TOKEN;
  let m, changed = false;
  while ((m = re.exec(v))) {
    const [all, lead, kind, raw] = m;
    if (applyToken(kind.toLowerCase(), raw)) {
      v = (v.slice(0, m.index) + lead + v.slice(m.index + all.length)).replace(/\s{2,}/g, " ");
      changed = true;
    } else break;
  }
  if (changed) {
    input.value = v.replace(/^\s+/, "");
    renderFilters();
  }
  return changed;
}
function applyToken(kind, raw) {
  const ix = S.index;
  if (!ix) return false;
  if (kind === "path") { S.filters.path = raw; return true; }
  if (kind === "in" || kind === "project") {
    const names = ix.projects.map((p) => p[0]);
    const exact = names.find((n) => n.toLowerCase() === raw.toLowerCase());
    const pre = names.filter((n) => n.toLowerCase().startsWith(raw.toLowerCase()));
    const hit = exact || (pre.length === 1 ? pre[0] : null);
    if (!hit) { toast(pre.length ? `"${raw}" matches ${pre.length} projects` : `No project named "${raw}"`); return false; }
    if (!S.filters.projects.includes(hit)) S.filters.projects.push(hit);
    return true;
  }
  const names = ix.languages.map((l) => l[0]);
  const want = LANG_ALIAS[raw.toLowerCase()] || names.find((n) => n.toLowerCase() === raw.toLowerCase());
  if (!want || !names.includes(want)) { toast(`No language "${raw}" in the index`); return false; }
  if (!S.filters.langs.includes(want)) S.filters.langs.push(want);
  return true;
}

// ---------------------------------------------------------------- searching

function query() { return $("q").value.trim(); }
function searchKey() { return JSON.stringify([query(), S.backend, S.filters, S.group, S.dedupe, S.k]); }

let timer;
function schedule(now) {
  clearTimeout(timer);
  $("clear-q").hidden = !$("q").value;
  if (!query()) { S.groups = []; S.lastKey = ""; S.seq++; setSearching(false); renderList(); renderPreview(); return; }
  if (now) run(); else timer = setTimeout(run, 320);
}

function setSearching(on) {
  S.searching = on;
  $("spinner").hidden = !on;
}

async function run() {
  const q = query();
  if (!q) return;
  if (!S.index) { setStatusText("The index is still loading…"); return; }
  const key = searchKey();
  if (key === S.lastKey) return;
  const seq = ++S.seq;
  setSearching(true);
  if (!S.groups.length) renderSkeleton();
  const t0 = performance.now();
  try {
    const r = await invoke("search", {
      query: q, backend: S.backend, k: S.k, projects: S.filters.projects, langs: S.filters.langs,
      path: S.filters.path ? [S.filters.path] : [], dedupe: S.dedupe, group: S.group,
    });
    if (seq !== S.seq) return;
    S.lastKey = key;
    S.groups = r.groups || (r.results || []).map((x) => ({ ...x, file_chunks: null, hits: [x] }));
    S.sel = S.groups.length ? 0 : -1;
    S.match = 0;
    S.last = { files: S.groups.length, matches: S.groups.reduce((n, g) => n + g.hits.length, 0), backend: r.backend,
               embed_ms: r.embed_ms, search_ms: r.search_ms, total_ms: Math.round(performance.now() - t0) };
    if (r.backend === "rawgpu") S.heldUntil = Date.now() + 20000;
    S.warm[r.backend] = "ready";
    banner("");
    remember(q);
    renderList();
    renderPreview();
    renderStatus();
  } catch (e) {
    if (seq !== S.seq) return;
    S.lastKey = "";
    const msg = String(e);
    const acts = [];
    if (S.backend === "rawgpu" && /in use|RX 590|card/i.test(msg)) acts.push(["Use Mac GPU", () => { setBackend("mps"); }]);
    if (/sidecar|exited|gone/i.test(msg)) acts.push(["Restart", restartSidecar]);
    acts.push(["Log", () => invoke("open_log")]);
    banner(msg, acts);
    if (!S.groups.length) renderList();
  } finally {
    if (seq === S.seq) setSearching(false);
    pollCard();
  }
}

function remember(q) {
  S.recent = [q, ...S.recent.filter((x) => x !== q)].slice(0, 8);
  store.set("recent", S.recent);
}

// ---------------------------------------------------------------- result list

function renderSkeleton() {
  const ol = $("results");
  ol.replaceChildren();
  for (let i = 0; i < 7; i++) ol.append(h("li", "skel"));
}

function firstLine(text) {
  const lines = text.split("\n").map((l) => l.trim());
  return lines.find((l) => l.length > 3 && !/^[#/*\-=_{}()[\];,.\s]+$/.test(l)) || lines.find((l) => l) || "";
}

function renderList() {
  const ol = $("results");
  ol.replaceChildren();
  if (!query()) { ol.append(listHint()); return; }
  if (!S.groups.length) {
    if (S.searching) return renderSkeleton();
    const e = h("div", "empty");
    e.append(h("h2", null, "No matches"));
    const f = S.filters;
    if (f.projects.length || f.langs.length || f.path) {
      e.append(h("p", null, "Nothing passes these filters."));
      const b = h("button", "btn", "Clear filters");
      b.type = "button";
      b.onclick = clearFilters;
      e.append(b);
    }
    ol.append(e);
    return;
  }
  const scores = S.groups.map((g) => g.score);
  const hi = Math.max(...scores), lo = Math.min(...scores);
  S.groups.forEach((g, i) => {
    const li = h("li", "res" + (i === S.sel ? " sel" : ""));
    const cut = g.path.lastIndexOf("/");
    const file = g.path.slice(cut + 1);
    const dir = g.path.slice(g.project.length + 1, Math.max(cut, g.project.length + 1));
    const r1 = h("div", "r1");
    const ext = file.includes(".") ? file.split(".").pop() : "";
    const badge = h("span", "lang", g.lang === "Other" ? (ext ? ext.slice(0, 4).toUpperCase() : "FILE") : short(g.lang));
    badge.title = g.lang;
    const meter = h("span", "meter");
    const fill = h("i");
    fill.style.width = (hi > lo ? 22 + 78 * (g.score - lo) / (hi - lo) : 100) + "%";
    meter.append(fill);
    meter.title = `similarity ${g.score.toFixed(3)}`;
    r1.append(badge, h("span", "name", S.group ? file : `${file}:${g.hits[0].start}`), meter);
    const r2 = h("div", "r2");
    const d = h("span", "dir");
    d.append(h("bdi", null, dir || ""));
    r2.append(h("span", "proj", g.project), d);
    if (S.group) r2.append(h("span", "count", g.hits.length > 1 ? `${g.hits.length} matches` : `line ${g.hits[0].start}`));
    li.append(r1, r2, h("div", "snip", firstLine(g.hits[0].text)));
    const copies = g.copies || g.hits[0].copies || [];
    if (copies.length) li.append(h("div", "copies-note", `also in ${plural(copies.length, "other copy", "other copies")}`));
    li.onclick = () => select(i);
    li.ondblclick = () => openCurrent();
    ol.append(li);
  });
}

function listHint() {
  const e = h("div", "empty");
  e.append(h("p", null, "Results appear here as you type."));
  return e;
}

function select(i, match = 0) {
  if (!S.groups.length) return;
  S.sel = Math.max(0, Math.min(S.groups.length - 1, i));
  S.match = match;
  const items = $("results").children;
  for (let j = 0; j < items.length; j++) items[j].classList.toggle("sel", j === S.sel);
  items[S.sel]?.scrollIntoView({ block: "nearest" });
  renderPreview();
}

function clearFilters() {
  S.filters = { projects: [], langs: [], path: "" };
  renderFilters();
  schedule(true);
}

// ---------------------------------------------------------------- preview

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Split highlight.js output into lines, closing and reopening the spans that cross each newline.
function splitHtmlLines(html) {
  const out = [];
  const stack = [];
  let cur = "";
  const re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+)/g;
  let m;
  while ((m = re.exec(html))) {
    if (m[1]) { stack.push(m[1]); cur += m[1]; }
    else if (m[2]) { stack.pop(); cur += m[2]; }
    else if (m[3]) { cur += "</span>".repeat(stack.length); out.push(cur); cur = stack.join(""); }
    else cur += m[4];
  }
  out.push(cur + "</span>".repeat(stack.length));
  return out;
}

const norm = (s) => s.replace(/\s+/g, " ").trim();

// Where a chunk sits in the file now: its indexed lines, or where its text moved to.
function locate(lines, hit) {
  const want = norm(hit.text);
  const got = norm(lines.slice(hit.start - 1, hit.end).join("\n"));
  if (got && (got === want || got.includes(want) || want.includes(got))) return { start: hit.start, end: hit.end, moved: false };
  const first = hit.text.split("\n").map((l) => l.trim()).find((l) => l.length > 10);
  if (first) {
    let best = -1;
    lines.forEach((l, i) => { if (l.trim() === first && (best < 0 || Math.abs(i - hit.start) < Math.abs(best - hit.start))) best = i; });
    if (best >= 0) {
      const off = hit.text.split("\n").findIndex((l) => l.trim() === first);
      const start = best - off + 1;
      return { start, end: start + (hit.end - hit.start), moved: true };
    }
  }
  return null;
}

async function source(path) {
  if (S.sources.has(path)) return S.sources.get(path);
  let v;
  try { v = await invoke("read_source", { path }); } catch (e) { v = { error: String(e) }; }
  S.sources.set(path, v);
  if (S.sources.size > 24) S.sources.delete(S.sources.keys().next().value);
  return v;
}

let previewSeq = 0;
async function renderPreview() {
  const p = $("preview");
  const g = S.groups[S.sel];
  if (!g) { p.replaceChildren(startPanel()); return; }
  const mySeq = ++previewSeq;
  const src = g.path.endsWith(".ipynb") ? { error: "notebook" } : await source(g.abs_path);
  if (mySeq !== previewSeq) return;

  let lines = null;
  let ranges = g.hits.map(() => null);
  let lost = 0, moved = 0;
  if (!src.error) {
    lines = src.text.split("\n");
    ranges = g.hits.map((hit) => locate(lines, hit));
    lost = ranges.filter((r) => !r).length;
    moved = ranges.filter((r) => r && r.moved).length;
    if (lost === ranges.length) lines = null; // nothing found: show the indexed text instead
  }
  // matches the preview can show: in file mode, those still found in the file
  const navs = g.hits.map((_, k) => k).filter((k) => !lines || ranges[k]);
  S.navs = navs;
  if (!navs.includes(S.match)) S.match = navs.find((k) => k > S.match) ?? navs[0] ?? 0;

  // header
  const head = h("div", "p-head");
  const row1 = h("div", "p-row");
  const crumbs = h("div", "crumbs");
  const cut = g.path.lastIndexOf("/");
  const dir = g.path.slice(g.project.length + 1, Math.max(cut, g.project.length + 1));
  crumbs.append(h("span", "c-proj", g.project));
  if (dir) {
    crumbs.append(h("span", "c-sep", "›"));
    const d = h("span", "c-dir");
    d.append(h("bdi", null, dir));
    crumbs.append(d);
  }
  crumbs.append(h("span", "c-sep", "›"), h("span", "c-file", g.path.slice(cut + 1)));
  crumbs.title = g.abs_path;
  const acts = h("div", "p-actions");
  const ed = S.editors.find(([id]) => id === S.editor) || S.editors[0] || ["default", "Default app"];
  const grp = h("span", "btn-group");
  grp.append(button(`Open in ${ed[1]}`, "open", openCurrent, "⌘↩", "btn primary"));
  const more = button("", "chevron", (e) => editorMenu(e.currentTarget, g), "Choose the editor", "btn primary");
  grp.append(more);
  acts.append(grp,
    button("", "finder", () => invoke("reveal", { path: g.abs_path }), "Reveal in Finder"),
    button("", "copy", copyPath, "Copy path:line (⌘⇧C)"));
  row1.append(crumbs, acts);

  const row2 = h("div", "p-row");
  const meta = h("div", "p-meta");
  const hit = g.hits[S.match];
  const rng = ranges[S.match] || hit;
  const bits = [g.lang, `lines ${rng.start}–${rng.end}`, `similarity ${hit.score.toFixed(3)}`];
  if (g.file_chunks) bits.push(`${plural(g.hits.length, "match", "matches")} of ${g.file_chunks} chunks`);
  bits.forEach((b, i) => { if (i) meta.append(h("span", "dot-sep")); meta.append(h("span", null, b)); });
  row2.append(meta);
  if (navs.length > 1) {
    const nav = h("div", "nav");
    nav.append(button("", "up", () => step(-1), "Previous match (⌥↑)", "icon-btn"),
      h("span", null, `${navs.indexOf(S.match) + 1} of ${navs.length}`),
      button("", "down", () => step(1), "Next match (⌥↓)", "icon-btn"));
    row2.append(nav);
  }
  head.append(row1, row2);
  const note = (text) => { const n = h("div", "notice"); n.append(svg(ICON.warn), h("span", null, text)); head.append(n); };
  if (src.error && src.error !== "notebook") note(`${src.error} Showing the indexed text.`);
  else if (!lines && !src.error) note("This file changed since it was indexed, and these matches are no longer in it. Showing the indexed text.");
  else if (lost) note(`This file changed since it was indexed: ${plural(lost, "match", "matches")} could not be found in it.`);
  else if (moved) note("This file changed since it was indexed; the matches were found at new lines.");

  // code
  const wrap = h("div", "code-wrap");
  const code = h("div", "code");
  const rail = h("div", "rail");
  wrap.append(code, rail);
  p.replaceChildren(head, wrap);

  const lang = hlLang(g.path);
  if (lines) {
    let html;
    if (lang && src.text.length <= 400000) {
      try { html = splitHtmlLines(hljs.highlight(src.text, { language: lang, ignoreIllegals: true }).value); } catch { html = null; }
    }
    if (!html) html = lines.map(escapeHtml);
    const mark = new Array(lines.length).fill(-1);
    ranges.forEach((r, k) => { if (r) for (let i = Math.max(1, r.start); i <= Math.min(lines.length, r.end); i++) if (mark[i - 1] < 0) mark[i - 1] = k; });
    const parts = [];
    for (let i = 0; i < lines.length; i++) {
      const k = mark[i];
      const cls = k < 0 ? "row" : k === S.match ? "row m cur" : "row m";
      parts.push(`<div class="${cls}" data-k="${k}"><span class="ln">${i + 1}</span><span class="tx">${html[i] || " "}</span></div>`);
    }
    code.innerHTML = parts.join(""); // highlight.js escapes the text; the rest is ours
    ranges.forEach((r, k) => {
      if (!r) return;
      const m = h("i", k === S.match ? "cur" : "");
      m.style.top = (100 * (r.start - 1) / lines.length) + "%";
      m.style.height = Math.max(0.4, 100 * (r.end - r.start + 1) / lines.length) + "%";
      m.title = `match ${k + 1}, line ${r.start}`;
      m.onclick = () => { S.match = k; renderPreview(); };
      rail.append(m);
    });
    const first = code.querySelector(".row.cur");
    if (first) code.scrollTop = Math.max(0, first.offsetTop - code.clientHeight * 0.28);
  } else {
    // the indexed chunks themselves, in file order
    const order = g.hits.map((x, k) => [x, k]).sort((a, b) => a[0].start - b[0].start);
    const parts = [];
    order.forEach(([x, k], j) => {
      if (j) parts.push(`<div class="gap-row">⋯</div>`);
      let hl = null;
      if (lang) { try { hl = splitHtmlLines(hljs.highlight(x.text, { language: lang, ignoreIllegals: true }).value); } catch { hl = null; } }
      const ls = hl || x.text.split("\n").map(escapeHtml);
      ls.forEach((t, i) => parts.push(`<div class="row m${k === S.match ? " cur" : ""}"><span class="ln">${x.start + i}</span><span class="tx">${t || " "}</span></div>`));
    });
    code.innerHTML = parts.join("");
    const first = code.querySelector(".row.cur");
    if (first) code.scrollTop = Math.max(0, first.offsetTop - 40);
  }
}

function step(d) {
  const navs = S.navs || [];
  if (navs.length < 2) return;
  const i = navs.indexOf(S.match);
  S.match = navs[(i + d + navs.length) % navs.length];
  renderPreview();
}

function currentLine() {
  const g = S.groups[S.sel];
  if (!g) return 1;
  const row = document.querySelector(".code .row.cur .ln");
  return row ? Number(row.textContent) : g.hits[S.match].start;
}

async function openCurrent() {
  const g = S.groups[S.sel];
  if (!g) return;
  try {
    await invoke("open_in_editor", { path: g.abs_path, line: currentLine(), editor: S.editor || "default" });
  } catch (e) { banner(String(e)); }
}

function copyPath() {
  const g = S.groups[S.sel];
  if (!g) return;
  navigator.clipboard.writeText(`${g.abs_path}:${currentLine()}`).then(() => toast("Copied path"), () => toast("Could not copy"));
}

// the preview pane before a search: what this is, recent searches, examples and the filter syntax
function startPanel() {
  const e = h("div", "empty");
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("viewBox", "0 0 48 48");
  icon.setAttribute("class", "empty-icon");
  icon.innerHTML = '<path d="M14 15 6 24l8 9M34 15l8 9-8 9M27 11l-6 26"/>';
  e.append(icon);
  const ix = S.index;
  e.append(h("h2", null, ix ? `Search ${fmt(ix.chunks)} pieces of your code by meaning` : "Loading the index…"));
  if (ix) {
    const built = new Date(ix.built * 1000);
    e.append(h("p", "stats", `${fmt(ix.files)} files in ${ix.projects.length} projects under ${ix.root.replace(/^\/Users\/[^/]+/, "~")} · indexed ${built.toLocaleDateString(undefined, { month: "short", day: "numeric" })} at ${built.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`));
  }
  const pills = (label, list, removable) => {
    if (!list.length) return;
    e.append(h("div", "section-label", label));
    const box = h("div", "pills");
    for (const q of list) {
      const b = h("button", "pill");
      b.type = "button";
      b.append(h("span", null, q));
      b.title = q;
      b.onclick = () => { $("q").value = q; schedule(true); $("q").focus(); };
      if (removable) {
        const x = h("span", "x", "×");
        x.title = "Forget";
        x.onclick = (ev) => { ev.stopPropagation(); S.recent = S.recent.filter((r) => r !== q); store.set("recent", S.recent); renderPreview(); };
        b.append(x);
      }
      box.append(b);
    }
    e.append(box);
  };
  pills("Recent", S.recent, true);
  pills("Try", EXAMPLES.filter((x) => !S.recent.includes(x)), false);
  e.append(h("div", "section-label", "Narrow it down"));
  const syn = h("div", "syntax");
  for (const [c, d] of [["in:rawgpu", "one project"], ["lang:python", "one language"], ["path:src/", "paths containing this text"]]) {
    syn.append(h("code", null, c), h("span", null, d));
  }
  e.append(syn);
  return e;
}

// ---------------------------------------------------------------- backends and the status bar

function setBackend(b) {
  if (!NAMES[b]) return;
  S.backend = b;
  store.set("backend", b);
  for (const btn of document.querySelectorAll("#backends button")) btn.setAttribute("aria-checked", btn.dataset.backend === b);
  if (b !== "rawgpu" && !S.warm[b]) {
    S.warm[b] = "loading";
    renderStatus();
    invoke("warm", { backend: b }).then(() => { S.warm[b] = "ready"; renderStatus(); }, () => { delete S.warm[b]; renderStatus(); });
  }
  banner("");
  renderStatus();
  pollCard();
  if (query()) schedule(true);
}
for (const btn of document.querySelectorAll("#backends button")) btn.onclick = () => setBackend(btn.dataset.backend);

function holderName(w) {
  const c = (w.clients || [])[0];
  if (!c) return "another program";
  const cmd = c.command || "";
  if (/index_code\.py\s+run/.test(cmd)) return "the code indexer";
  if (/search_server\.py/.test(cmd)) return "Code Search";
  const m = cmd.match(/([\w.-]+\.py)\b/);
  return m ? m[1] : cmd.split(/\s+/)[0].split("/").pop();
}

function cardState() {
  const w = S.card;
  if (!w) return { cls: "", text: "RX 590" };
  if (!w.server_pid) return { cls: "", text: "RX 590 not connected · the first search connects it" };
  if (!w.client_count) return { cls: "free", text: "RX 590 free" };
  const who = holderName(w);
  if (who === "Code Search") return { cls: "free", text: "RX 590 held by Code Search", mine: true };
  return { cls: "busy", text: `RX 590 in use by ${who}`, other: who };
}

function setStatusText(t) { $("sb-timing").textContent = t; }

function renderStatus() {
  const c = cardState();
  // dots on the backend buttons
  for (const btn of document.querySelectorAll("#backends button")) {
    const b = btn.dataset.backend;
    const dot = btn.querySelector(".bdot");
    dot.className = "bdot " + (b === "rawgpu" ? c.cls : S.warm[b] || "");
  }
  const sb = $("sb-backend");
  sb.replaceChildren();
  const dot = h("span", "bdot");
  if (S.backend === "rawgpu") {
    dot.className = "bdot " + c.cls;
    sb.append(dot);
    let text = c.text;
    const left = Math.ceil((S.heldUntil - Date.now()) / 1000);
    if (c.mine && left > 0) text += ` · releases in ${left} s`;
    sb.append(document.createTextNode(text));
    if (c.mine) {
      const r = h("button", "link-btn", "Release");
      r.type = "button";
      r.onclick = async () => { try { await invoke("release_card"); S.heldUntil = 0; } catch (e) { banner(String(e)); } pollCard(); };
      sb.append(r);
    } else if (c.other) {
      const r = h("button", "link-btn", "Use Mac GPU");
      r.type = "button";
      r.onclick = () => setBackend("mps");
      sb.append(r);
    }
  } else {
    const st = S.warm[S.backend];
    dot.className = "bdot " + (st || "");
    sb.append(dot, document.createTextNode(`${NAMES[S.backend]} · ${st === "loading" ? "loading PyTorch and the model…" : st === "ready" ? "ready" : "loads on first search"}`));
  }
  const l = S.last;
  if (l) {
    const what = S.group ? `${plural(l.files, "file")}, ${plural(l.matches, "match", "matches")}` : plural(l.matches, "result");
    setStatusText(`${what} · embedded on ${NAMES[l.backend]} in ${Math.round(l.embed_ms)} ms · scored ${S.index ? fmt(S.index.chunks) + " chunks" : ""} in ${Math.round(l.search_ms)} ms`);
  }
}
setInterval(() => { if (S.heldUntil > Date.now() - 1000) renderStatus(); }, 1000);

async function pollCard() {
  try { S.card = await invoke("card_state"); } catch { S.card = null; }
  renderStatus();
}
setInterval(pollCard, 3000);

async function restartSidecar() {
  banner("");
  setStatusText("Restarting the search backend…");
  try { await invoke("restart_sidecar"); } catch (e) { banner(String(e), [["Log", () => invoke("open_log")]]); }
}

// ---------------------------------------------------------------- wiring

$("q").addEventListener("input", () => { takeTokens(false); schedule(false); });
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); takeTokens(true); S.lastKey = ""; schedule(true); }
});
$("clear-q").onclick = () => { $("q").value = ""; schedule(true); $("q").focus(); };
$("f-project").onclick = (e) => picker(e.currentTarget, {
  items: (S.index?.projects || []).map(([p, n]) => ({ value: p, label: p, count: n })).sort((a, b) => b.count - a.count),
  selected: S.filters.projects, placeholder: "Filter projects", onChange: (v) => setFilter("projects", v),
});
$("f-lang").onclick = (e) => picker(e.currentTarget, {
  items: (S.index?.languages || []).map(([l, n]) => ({ value: l, label: l, count: n })),
  selected: S.filters.langs, placeholder: "Filter languages", onChange: (v) => setFilter("langs", v),
});
$("f-path").onclick = (e) => pathPopover(e.currentTarget);
$("f-clear").onclick = clearFilters;
$("group").checked = S.group;
$("dedupe").checked = S.dedupe;
$("k").value = String(S.k);
$("group").onchange = (e) => { S.group = e.target.checked; store.set("group", S.group); schedule(true); };
$("dedupe").onchange = (e) => { S.dedupe = e.target.checked; store.set("dedupe", S.dedupe); schedule(true); };
$("k").onchange = (e) => { S.k = Number(e.target.value); store.set("k", S.k); schedule(true); };

document.addEventListener("keydown", (e) => {
  if (!$("popover").hidden && pop.onKey && pop.onKey(e)) { e.preventDefault(); return; }
  if (e.key === "Escape") {
    if (!$("popover").hidden) return closePopover();
    $("q").focus();
    $("q").select();
    return;
  }
  const inPopover = $("popover").contains(document.activeElement);
  if (inPopover) return;
  if (e.metaKey && (e.key === "k" || e.key === "f" || e.key === "l")) { e.preventDefault(); $("q").focus(); $("q").select(); }
  else if (e.metaKey && ["1", "2", "3"].includes(e.key)) { e.preventDefault(); setBackend(["rawgpu", "mps", "cpu"][Number(e.key) - 1]); }
  else if (e.metaKey && e.key === "Enter") { e.preventDefault(); openCurrent(); }
  else if (e.metaKey && e.shiftKey && (e.key === "c" || e.key === "C")) { e.preventDefault(); copyPath(); }
  else if (e.altKey && e.key === "ArrowDown") { e.preventDefault(); step(1); }
  else if (e.altKey && e.key === "ArrowUp") { e.preventDefault(); step(-1); }
  else if (e.key === "ArrowDown" && !e.metaKey) { e.preventDefault(); select(S.sel + 1); }
  else if (e.key === "ArrowUp" && !e.metaKey) { e.preventDefault(); select(S.sel - 1); }
});
window.addEventListener("focus", pollCard);
window.addEventListener("resize", closePopover);

listen("sidecar", (e) => {
  const m = e.payload;
  if (m.event === "status") setStatusText(m.text);
  else if (m.event === "ready") { S.index = m.index; onIndex(); }
  else if (m.event === "card") { S.heldUntil = 0; pollCard(); }
  else if (m.event === "exit") { S.index = null; banner("The search backend stopped.", [["Restart", restartSidecar], ["Log", () => invoke("open_log")]]); }
});

let pending = null; // a query from the command line, run once the index is loaded
function onIndex() {
  setStatusText(`Ready · ${fmt(S.index.chunks)} chunks`);
  renderFilters();
  if (!S.groups.length) renderPreview();
  if (pending) { $("q").value = pending; pending = null; }
  if (query()) { takeTokens(true); S.lastKey = ""; schedule(true); }
}

async function start() {
  const a = await invoke("launch_args");
  S.editors = await invoke("editors");
  if (!S.editors.some(([id]) => id === S.editor)) S.editor = S.editors[0][0];
  setBackend(NAMES[a.backend] ? a.backend : S.backend);
  if (a.query) pending = a.query;
  renderFilters();
  renderList();
  renderPreview();
  try {
    const ix = await invoke("index_info");
    if (ix) { S.index = ix; onIndex(); } else setStatusText("Loading the index…");
  } catch (e) { banner(String(e), [["Restart", restartSidecar], ["Log", () => invoke("open_log")]]); }
}
start();
