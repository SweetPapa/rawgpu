// rawgpu's menu bar app: the RX 590's state at a glance, `rawgpu eject` before unplugging, and bring-up.
//
// It drives the rawgpu CLI and is careful with the TinyGPU server, which serves one program at a time:
// - It polls `rawgpu who --json`, which only inspects processes and never connects.
// - It reads `rawgpu status --json` only while the server runs with no client, so it never queues behind
//   a long job and never starts a server by itself. After an eject, the card stays disconnected.
// - It never kills a rawgpu command that may be talking to the server: an abandoned request makes the
//   server die of SIGPIPE later. Cancelling an eject sends SIGTERM, which `rawgpu eject` honours only
//   while it is still waiting.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent, Wry};

/// The rawgpu checkout this app was built from; RAWGPU_BIN overrides the CLI.
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const POLL: Duration = Duration::from_secs(3);
const STATUS_EVERY: Duration = Duration::from_secs(10);

fn repo() -> PathBuf {
    Path::new(REPO).canonicalize().unwrap_or_else(|_| PathBuf::from(REPO))
}

fn rawgpu_bin() -> PathBuf {
    std::env::var_os("RAWGPU_BIN").map(PathBuf::from).unwrap_or_else(|| repo().join("build/rawgpu"))
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn run_text(args: &[&str]) -> Result<String, String> {
    let out = Command::new(rawgpu_bin())
        .args(args)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {}: {e}", rawgpu_bin().display()))?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    if out.status.success() {
        Ok(text)
    } else {
        let err = String::from_utf8_lossy(&out.stderr);
        Err(format!("{}{}", err.trim(), if text.trim().is_empty() { String::new() } else { format!("\n{}", text.trim()) }))
    }
}

fn run_json(args: &[&str]) -> Result<Value, String> {
    let text = run_text(args)?;
    let line = text.lines().rev().find(|l| l.trim_start().starts_with('{')).unwrap_or("");
    serde_json::from_str(line).map_err(|e| format!("rawgpu {}: unexpected output ({e}): {}", args.join(" "), text.trim()))
}

fn int(v: &Value, key: &str) -> i64 {
    v.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn thousands(n: i64) -> String {
    let s = n.abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// A short name for the program holding the card, from its command line.
fn program_name(cmd: &str) -> String {
    if cmd.contains("index_code.py") && cmd.contains(" run") {
        return "the code indexer".into();
    }
    if cmd.contains("search_server.py") {
        return "Code Search".into();
    }
    let words: Vec<&str> = cmd.split_whitespace().collect();
    let base = |w: &str| w.rsplit('/').next().unwrap_or(w).to_string();
    match words.first() {
        None => "another program".into(),
        Some(w0) => {
            let b0 = base(w0);
            if b0.starts_with("python") {
                if let Some(script) = words.iter().skip(1).find(|w| w.ends_with(".py")) {
                    return base(script);
                }
                if words.get(1) == Some(&"-c") {
                    return "a Python script".into();
                }
            }
            if b0 == "rawgpu" {
                return format!("rawgpu {}", words.get(1).copied().unwrap_or(""));
            }
            b0
        }
    }
}

/// The latest progress record of a running indexer (models/embeddinggemma2/index_code.py run).
fn indexer_progress() -> Option<Value> {
    let dir = home().join("rawgpu-index/logs");
    let newest = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())?;
    let mut f = std::fs::File::open(newest.path()).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(8192))).ok()?;
    let mut tail = String::new();
    f.read_to_string(&mut tail).ok()?;
    tail.lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v.get("kind").and_then(Value::as_str) == Some("progress"))
}

fn notify(title: &str, text: &str) {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = Command::new("osascript")
        .arg("-e")
        .arg(format!("display notification \"{}\" with title \"{}\"", esc(text), esc(title)))
        .spawn();
}

#[derive(Clone, Serialize, Default)]
struct EjectInfo {
    state: String, // "", "running", "done", "failed", "cancelled"
    phase: String,
    pid: i32,
}

#[derive(Clone, Serialize, Default)]
struct Snapshot {
    level: String, // idle, busy, off, warn
    headline: String,
    detail: String,
    user: String,
    eject_label: String,
    eject_enabled: bool,
    up_label: String,
    up_enabled: bool,
    who: Option<Value>,
    who_error: Option<String>,
    status: Option<Value>,
    status_error: Option<String>,
    status_at: u64,
    indexer: Option<Value>,
    eject: EjectInfo,
    up_running: bool,
    log: Vec<String>,
    rawgpu: String,
}

struct Items {
    headline: MenuItem<Wry>,
    detail: MenuItem<Wry>,
    user: MenuItem<Wry>,
    eject: MenuItem<Wry>,
    up: MenuItem<Wry>,
}

struct Shared {
    snap: Mutex<Snapshot>,
    status_busy: AtomicBool,
    last_status: Mutex<Option<Instant>>,
    items: Items,
    level: Mutex<String>,
}

type Sh = Arc<Shared>;

fn log_line(s: &mut Snapshot, line: impl Into<String>) {
    s.log.push(line.into());
    let n = s.log.len();
    if n > 300 {
        s.log.drain(..n - 300);
    }
}

/// Work out what the menu shows from the latest readings.
fn derive(s: &mut Snapshot) {
    let who = s.who.clone().unwrap_or(Value::Null);
    let server = int(&who, "server_pid");
    let clients = int(&who, "client_count");
    let others = int(&who, "other_tinygpu");
    let holder = who
        .get("clients")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .map(|c| (program_name(c.get("command").and_then(Value::as_str).unwrap_or("")), int(c, "pid")));
    let holder_text = match &holder {
        Some((name, pid)) => format!("{name} (pid {pid})"),
        None => "another program".into(),
    };
    s.eject_label = "Eject: Make Safe to Unplug".into();
    s.eject_enabled = server > 0;
    s.up_label = "Connect and Bring Up Card".into();
    s.up_enabled = !s.up_running && clients == 0;

    if s.eject.state == "running" {
        s.level = "busy".into();
        s.headline = "RX 590: ejecting".into();
        s.detail = s.eject.phase.clone();
        s.user = if clients > 0 { format!("Waiting for {holder_text} to finish") } else { "Closing the card".into() };
        s.eject_label = "Cancel Eject".into();
        s.eject_enabled = true;
        s.up_enabled = false;
        return;
    }
    if let Some(e) = &s.who_error {
        s.level = "warn".into();
        s.headline = "rawgpu CLI not working".into();
        s.detail = e.lines().next().unwrap_or("").chars().take(90).collect();
        s.user = format!("Expected at {}", s.rawgpu);
        s.eject_enabled = false;
        s.up_enabled = false;
        return;
    }
    if server == 0 {
        s.status = None;
        if others > 0 {
            s.level = "warn".into();
            s.headline = "RX 590: another TinyGPU server is running".into();
            s.detail = "Another program may have the card. Not safe to unplug.".into();
            s.user = "rawgpu's own server is not running".into();
        } else {
            s.level = "off".into();
            s.headline = "RX 590: not connected".into();
            s.detail = if s.eject.state == "done" {
                "Ejected. Safe to unplug.".into()
            } else {
                "No TinyGPU server. Safe to unplug.".into()
            };
            s.user = "Nothing is using the card".into();
        }
        s.eject_enabled = false;
        s.up_label = "Connect and Bring Up Card".into();
        return;
    }
    if clients > 0 {
        s.level = "busy".into();
        s.headline = "RX 590: in use".into();
        s.user = format!("In use by {holder_text}");
        s.detail = match &s.indexer {
            Some(p) => format!(
                "Indexing {:.1} chunks/s, {} °C, fan {}%, {} chunks",
                p.get("chunks_per_s").and_then(Value::as_f64).unwrap_or(0.0),
                int(p, "temperature_c"),
                int(p, "fan_percent"),
                thousands(int(p, "chunks_embedded"))
            ),
            None => "Card readings pause while another program has the card".into(),
        };
        s.eject_label = match &holder {
            Some((name, _)) => format!("Eject: Waits for {name}…"),
            None => "Eject: Waits for the Card to Be Free…".into(),
        };
        s.up_enabled = false;
        return;
    }
    s.user = "Nothing is using the card".into();
    match (&s.status, &s.status_error) {
        (_, Some(e)) => {
            s.level = "warn".into();
            s.headline = "RX 590: not answering".into();
            s.detail = e.lines().next().unwrap_or("").chars().take(90).collect();
        }
        (None, None) => {
            s.level = "idle".into();
            s.headline = "RX 590: reading the card…".into();
            s.detail = String::new();
        }
        (Some(st), None) => {
            let posted = st.get("posted").and_then(Value::as_bool).unwrap_or(false);
            if !posted {
                s.level = "warn".into();
                s.headline = "RX 590: needs bring-up".into();
                s.detail = "Not posted: the card lost power or the Mac slept".into();
                s.up_label = "Bring Up Card".into();
                return;
            }
            let dpm = st.get("engine_dpm").and_then(Value::as_bool).unwrap_or(false);
            let fan = int(st, "fan_percent");
            s.level = "idle".into();
            s.headline = format!(
                "RX 590: idle, {} °C{}",
                int(st, "temperature_c"),
                if fan >= 0 { format!(", fan {fan}%") } else { String::new() }
            );
            s.detail = format!(
                "Engine {} MHz, memory {} MHz, DPM {}",
                int(st, "engine_mhz"),
                int(st, "memory_mhz"),
                if dpm { "on" } else { "off (boot clocks)" }
            );
            s.up_label = "Card Is Up".into();
            s.up_enabled = false;
        }
    }
}

fn icon_for(level: &str) -> &'static [u8] {
    match level {
        "busy" => include_bytes!("../icons/tray-busy.png"),
        "off" => include_bytes!("../icons/tray-off.png"),
        "warn" => include_bytes!("../icons/tray-warn.png"),
        _ => include_bytes!("../icons/tray-idle.png"),
    }
}

/// Push the snapshot to the menu, the icon and the details window.
fn refresh(app: &AppHandle, sh: &Shared) {
    let snap = {
        let mut s = sh.snap.lock().unwrap();
        derive(&mut s);
        s.clone()
    };
    let it = &sh.items;
    let _ = it.headline.set_text(&snap.headline);
    let _ = it.detail.set_text(if snap.detail.is_empty() { " " } else { &snap.detail });
    let _ = it.user.set_text(&snap.user);
    let _ = it.eject.set_text(&snap.eject_label);
    let _ = it.eject.set_enabled(snap.eject_enabled);
    let _ = it.up.set_text(&snap.up_label);
    let _ = it.up.set_enabled(snap.up_enabled);
    if let Some(tray) = app.tray_by_id("main") {
        // No lock may be held across a tray or menu call: they run on the main thread, which may itself
        // be waiting in refresh().
        let changed = {
            let mut level = sh.level.lock().unwrap();
            let changed = *level != snap.level;
            *level = snap.level.clone();
            changed
        };
        if changed {
            if let Ok(img) = Image::from_bytes(icon_for(&snap.level)) {
                let _ = tray.set_icon(Some(img));
                let _ = tray.set_icon_as_template(true);
            }
        }
        let _ = tray.set_tooltip(Some(&snap.headline));
    }
    let _ = app.emit("state", &snap);
}

fn read_status(app: &AppHandle, sh: &Sh) {
    if sh.status_busy.swap(true, Ordering::SeqCst) {
        return;
    }
    let (app, sh) = (app.clone(), sh.clone());
    std::thread::spawn(move || {
        // Normally 20 ms. If a program took the card since `who` looked, this waits for it, as it must.
        let r = run_json(&["status", "--json"]);
        {
            let mut s = sh.snap.lock().unwrap();
            match r {
                Ok(v) => {
                    s.status = Some(v);
                    s.status_error = None;
                    s.status_at = now_secs();
                }
                Err(e) => s.status_error = Some(e),
            }
        }
        *sh.last_status.lock().unwrap() = Some(Instant::now());
        sh.status_busy.store(false, Ordering::SeqCst);
        refresh(&app, &sh);
    });
}

fn poll_once(app: &AppHandle, sh: &Sh) {
    let who = run_json(&["who", "--json"]);
    let need_status = {
        let mut s = sh.snap.lock().unwrap();
        match who {
            Ok(w) => {
                s.who = Some(w);
                s.who_error = None;
            }
            Err(e) => {
                s.who = None;
                s.who_error = Some(e);
            }
        }
        let w = s.who.clone().unwrap_or(Value::Null);
        let server = int(&w, "server_pid");
        let clients = int(&w, "client_count");
        let indexing = w
            .get("clients")
            .and_then(Value::as_array)
            .map(|c| c.iter().any(|c| c.get("command").and_then(Value::as_str).unwrap_or("").contains("index_code.py")))
            .unwrap_or(false);
        s.indexer = if indexing { indexer_progress() } else { None };
        if server == 0 {
            s.status_error = None;
        }
        let stale = sh.last_status.lock().unwrap().map_or(true, |t| t.elapsed() >= STATUS_EVERY);
        server > 0 && clients == 0 && s.eject.state != "running" && !s.up_running && stale
    };
    if need_status {
        read_status(app, sh);
    }
    refresh(app, sh);
}

fn start_eject(app: &AppHandle, sh: &Sh) {
    {
        let mut s = sh.snap.lock().unwrap();
        if s.eject.state == "running" {
            return;
        }
        s.eject = EjectInfo { state: "running".into(), phase: "Starting".into(), pid: 0 };
        log_line(&mut s, "$ rawgpu eject");
    }
    let child = Command::new(rawgpu_bin())
        .arg("eject")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            let mut s = sh.snap.lock().unwrap();
            s.eject = EjectInfo { state: "failed".into(), phase: format!("could not run rawgpu: {e}"), pid: 0 };
            drop(s);
            refresh(app, sh);
            return;
        }
    };
    sh.snap.lock().unwrap().eject.pid = child.id() as i32;
    refresh(app, sh);
    let stderr = child.stderr.take();
    {
        let (app, sh) = (app.clone(), sh.clone());
        std::thread::spawn(move || {
            if let Some(err) = stderr {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    log_line(&mut sh.snap.lock().unwrap(), line);
                    refresh(&app, &sh);
                }
            }
        });
    }
    let (app, sh) = (app.clone(), sh.clone());
    std::thread::spawn(move || {
        if let Some(out) = child.stdout.take() {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                let mut s = sh.snap.lock().unwrap();
                let phase = if line.starts_with("Another program is using the card") {
                    Some("Waiting for the card to be free")
                } else if line.starts_with("The card is free") || line.starts_with("TinyGPU server: pid") {
                    Some("Stopping the card's engines")
                } else if line.starts_with("TinyGPU server: stopped") {
                    Some("TinyGPU server stopped")
                } else if line.contains("still waiting") {
                    Some("Still waiting for the card")
                } else {
                    None
                };
                if let Some(p) = phase {
                    s.eject.phase = p.into();
                }
                log_line(&mut s, line);
                drop(s);
                refresh(&app, &sh);
            }
        }
        let status = child.wait();
        let mut s = sh.snap.lock().unwrap();
        let last = s.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        match status {
            Ok(st) if st.success() => {
                s.eject = EjectInfo { state: "done".into(), phase: "Safe to unplug".into(), pid: 0 };
                s.status = None;
                notify("RX 590 ejected", "Safe to unplug. The next rawgpu program reconnects.");
            }
            Ok(st) if st.signal().is_some() => {
                s.eject = EjectInfo { state: "cancelled".into(), phase: "Eject cancelled; nothing changed".into(), pid: 0 };
            }
            Ok(_) => {
                s.eject = EjectInfo { state: "failed".into(), phase: last.clone(), pid: 0 };
                notify("RX 590: not safe to unplug", &last);
            }
            Err(e) => s.eject = EjectInfo { state: "failed".into(), phase: e.to_string(), pid: 0 },
        }
        let done = format!("[eject {}]", s.eject.state);
        log_line(&mut s, done);
        drop(s);
        *sh.last_status.lock().unwrap() = None;
        refresh(&app, &sh);
    });
}

fn cancel_eject(sh: &Sh) {
    let pid = {
        let s = sh.snap.lock().unwrap();
        if s.eject.state != "running" { 0 } else { s.eject.pid }
    };
    if pid > 0 {
        // Safe at any point: while waiting it stops; once it is closing the card it finishes first.
        unsafe { libc::kill(pid, libc::SIGTERM) };
    }
}

fn bring_up(app: &AppHandle, sh: &Sh) {
    {
        let mut s = sh.snap.lock().unwrap();
        if s.up_running || s.eject.state == "running" {
            return;
        }
        s.up_running = true;
        log_line(&mut s, "$ rawgpu up");
    }
    refresh(app, sh);
    let (app, sh) = (app.clone(), sh.clone());
    std::thread::spawn(move || {
        let child = Command::new(rawgpu_bin())
            .arg("up")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let ok = match child {
            Ok(mut c) => {
                let err = c.stderr.take();
                let sh2 = sh.clone();
                let t = std::thread::spawn(move || {
                    if let Some(e) = err {
                        for l in BufReader::new(e).lines().map_while(Result::ok) {
                            log_line(&mut sh2.snap.lock().unwrap(), l);
                        }
                    }
                });
                if let Some(o) = c.stdout.take() {
                    for l in BufReader::new(o).lines().map_while(Result::ok) {
                        log_line(&mut sh.snap.lock().unwrap(), l);
                        refresh(&app, &sh);
                    }
                }
                let _ = t.join();
                c.wait().map(|s| s.success()).unwrap_or(false)
            }
            Err(e) => {
                log_line(&mut sh.snap.lock().unwrap(), format!("could not run rawgpu: {e}"));
                false
            }
        };
        {
            let mut s = sh.snap.lock().unwrap();
            s.up_running = false;
            if s.eject.state == "done" {
                s.eject = EjectInfo::default();
            }
            log_line(&mut s, if ok { "[up done]" } else { "[up failed]" });
        }
        if !ok {
            notify("RX 590: bring-up failed", "See the details window for rawgpu's messages.");
        }
        *sh.last_status.lock().unwrap() = None;
        poll_once(&app, &sh);
    });
}

fn show_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn open_search() {
    // the installed app, else the one built next to this one
    let ok = Command::new("open").args(["-b", "dev.rawgpu.code-search"]).status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        let dev = repo().join("apps/target/release/bundle/macos/Code Search.app");
        let _ = Command::new("open").arg(dev).status();
    }
}

#[tauri::command]
fn get_state(sh: State<'_, Sh>) -> Snapshot {
    let mut s = sh.snap.lock().unwrap();
    derive(&mut s);
    s.clone()
}

#[tauri::command]
fn eject(app: AppHandle, sh: State<'_, Sh>) {
    start_eject(&app, &sh);
}

#[tauri::command]
fn cancel(sh: State<'_, Sh>) {
    cancel_eject(&sh);
}

#[tauri::command]
fn up(app: AppHandle, sh: State<'_, Sh>) {
    bring_up(&app, &sh);
}

#[tauri::command]
fn refresh_now(app: AppHandle, sh: State<'_, Sh>) {
    *sh.last_status.lock().unwrap() = None;
    let (app, sh) = (app.clone(), sh.inner().clone());
    std::thread::spawn(move || poll_once(&app, &sh));
}

/// `rawgpu status` in full, only while the server runs and nothing has the card.
#[tauri::command]
async fn full_status() -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let w = run_json(&["who", "--json"])?;
        if int(&w, "server_pid") == 0 {
            return Err("The TinyGPU server is not running, so the card is not connected. \
                        Connect and Bring Up Card starts it."
                .into());
        }
        if int(&w, "client_count") > 0 {
            return Err("Another program has the card. Its full status can be read when it finishes.".into());
        }
        run_text(&["status"])
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn open_code_search() {
    open_search();
}

#[tauri::command]
fn open_logs() {
    let _ = Command::new("open").arg(home().join("rawgpu-index/logs")).status();
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let headline = MenuItem::with_id(app, "headline", "RX 590", false, None::<&str>)?;
            let detail = MenuItem::with_id(app, "detail", " ", false, None::<&str>)?;
            let user = MenuItem::with_id(app, "user", " ", false, None::<&str>)?;
            let eject = MenuItem::with_id(app, "eject", "Eject: Make Safe to Unplug", false, None::<&str>)?;
            let up = MenuItem::with_id(app, "up", "Connect and Bring Up Card", false, None::<&str>)?;
            let details = MenuItem::with_id(app, "details", "Show Details…", true, None::<&str>)?;
            let search = MenuItem::with_id(app, "search", "Open Code Search", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit rawgpu", true, Some("CmdOrCtrl+Q"))?;
            let menu = Menu::with_items(
                app,
                &[
                    &headline,
                    &detail,
                    &user,
                    &PredefinedMenuItem::separator(app)?,
                    &eject,
                    &up,
                    &PredefinedMenuItem::separator(app)?,
                    &details,
                    &search,
                    &PredefinedMenuItem::separator(app)?,
                    &quit,
                ],
            )?;

            let sh: Sh = Arc::new(Shared {
                snap: Mutex::new(Snapshot { rawgpu: rawgpu_bin().display().to_string(), ..Default::default() }),
                status_busy: AtomicBool::new(false),
                last_status: Mutex::new(None),
                items: Items { headline, detail, user, eject, up },
                level: Mutex::new(String::new()),
            });
            app.manage(sh.clone());

            let menu_sh = sh.clone();
            TrayIconBuilder::with_id("main")
                .icon(Image::from_bytes(icon_for("idle"))?)
                .icon_as_template(true)
                .tooltip("rawgpu")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "eject" => {
                        if menu_sh.snap.lock().unwrap().eject.state == "running" {
                            cancel_eject(&menu_sh)
                        } else {
                            start_eject(app, &menu_sh)
                        }
                    }
                    "up" => bring_up(app, &menu_sh),
                    "details" => show_window(app),
                    "search" => open_search(),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            #[cfg(debug_assertions)] // for testing without clicking: --test-eject-after SECONDS
            if let Some(secs) = std::env::args().skip_while(|a| a != "--test-eject-after").nth(1).and_then(|s| s.parse().ok()) {
                let (h, s2) = (app.handle().clone(), sh.clone());
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_secs(secs));
                    start_eject(&h, &s2);
                });
            }

            let handle = app.handle().clone();
            std::thread::spawn(move || loop {
                poll_once(&handle, &sh);
                std::thread::sleep(POLL);
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide(); // the app lives in the menu bar
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            eject,
            cancel,
            up,
            refresh_now,
            full_status,
            open_code_search,
            open_logs
        ])
        .build(tauri::generate_context!())
        .expect("error while starting the rawgpu menu bar app")
        .run(|app, event| match event {
            // opening the app again (Finder, Spotlight, `open -a rawgpu`) shows its window
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen { .. } => show_window(app),
            tauri::RunEvent::Ready if std::env::args().any(|a| a == "--show") => show_window(app),
            _ => {}
        });
}
