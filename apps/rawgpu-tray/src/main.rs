// rawgpu's menu bar app: the RX 590's state at a glance, `rawgpu eject` before unplugging, and bring-up.
//
// Clicking the icon opens a panel (the "panel" window, positioned under the icon, hidden when it loses
// focus); right-clicking opens a menu. "details" is a window with the log and the full status.
//
// It drives the rawgpu CLI and is careful with the TinyGPU server, which serves one program at a time:
// - It polls `rawgpu who --json`, which only inspects processes and never connects.
// - It reads `rawgpu status --json` only while the server runs with no client, so it never queues behind
//   a long job and never starts a server by itself. After an eject, the card stays disconnected.
// - It never kills a rawgpu command that may be talking to the server: an abandoned request makes the
//   server die of SIGPIPE later. Cancelling an eject sends SIGTERM, which `rawgpu eject` honours only
//   while it is still waiting.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, State, WindowEvent, Wry};
use tauri_plugin_notification::NotificationExt;

/// The rawgpu checkout this app was built from; RAWGPU_BIN overrides the CLI.
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const POLL: Duration = Duration::from_secs(3);
const POLL_OPEN: Duration = Duration::from_millis(1500);
const STATUS_EVERY: Duration = Duration::from_secs(10);
const STATUS_EVERY_OPEN: Duration = Duration::from_secs(3);
const SAMPLE_EVERY_S: f64 = 10.0;
const HISTORY: usize = 1080; // 3 hours at one sample per 10 s
const PANEL_WIDTH: f64 = 348.0;
const LAUNCH_AGENT: &str = "dev.rawgpu.tray";

fn repo() -> PathBuf {
    Path::new(REPO).canonicalize().unwrap_or_else(|_| PathBuf::from(REPO))
}

/// The CLI: $RAWGPU_BIN; else the one bundled beside this app's executable (a release, and `make apps`);
/// else the checkout's build/rawgpu.
fn rawgpu_bin() -> PathBuf {
    if let Some(p) = std::env::var_os("RAWGPU_BIN") {
        return PathBuf::from(p);
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let bundled = dir.join("rawgpu");
        if bundled.is_file() {
            return bundled;
        }
    }
    repo().join("build/rawgpu")
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn now_secs() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
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
    if cmd.contains("studio/runner.py") {
        return "Training Studio".into();
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

fn program_kind(cmd: &str) -> &'static str {
    if cmd.contains("index_code.py") {
        "indexer"
    } else if cmd.contains("search_server.py") {
        "search"
    } else if cmd.contains("studio/runner.py") {
        "studio"
    } else if cmd.split_whitespace().next().is_some_and(|w| w.ends_with("/rawgpu")) {
        "cli"
    } else {
        "program"
    }
}

/// The newest indexer log, ~/rawgpu-index/logs/run-*.jsonl.
fn indexer_log() -> Option<PathBuf> {
    std::fs::read_dir(home().join("rawgpu-index/logs"))
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("run-"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// The last record of the given kind ("progress", "end") in the newest indexer log.
fn indexer_record(kind: &str) -> Option<Value> {
    let mut f = std::fs::File::open(indexer_log()?).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(16384))).ok()?;
    let mut tail = String::new();
    f.read_to_string(&mut tail).ok()?;
    tail.lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v.get("kind").and_then(Value::as_str) == Some(kind))
}

// ---------------------------------------------------------------- settings

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    menu_bar_temp: bool,
    temp_alerts: bool,
    alert_c: i64,
    notify_indexer: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { menu_bar_temp: false, temp_alerts: true, alert_c: 80, notify_indexer: true }
    }
}

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("settings.json"))
}

fn load_settings(app: &AppHandle) -> Settings {
    settings_path(app)
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn save_settings(app: &AppHandle, s: &Settings) {
    if let Some(p) = settings_path(app) {
        let _ = std::fs::create_dir_all(p.parent().unwrap_or(Path::new("/")));
        let _ = std::fs::write(p, serde_json::to_vec_pretty(s).unwrap_or_default());
    }
}

/// The .app bundle this runs from, if it runs from one.
fn app_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().find(|p| p.extension().is_some_and(|e| e == "app")).map(Path::to_path_buf)
}

fn launch_agent_path() -> PathBuf {
    home().join(format!("Library/LaunchAgents/{LAUNCH_AGENT}.plist"))
}

fn launch_at_login() -> bool {
    launch_agent_path().exists()
}

/// Start at login through a LaunchAgent that opens the app bundle.
fn set_launch_at_login(on: bool) -> Result<(), String> {
    let path = launch_agent_path();
    if !on {
        return match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        };
    }
    let app = app_bundle().ok_or("Opening at login needs the installed app: run `make install-apps`.")?;
    let esc = |s: &str| s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCH_AGENT}</string>
  <key>ProgramArguments</key>
  <array><string>/usr/bin/open</string><string>-g</string><string>-a</string><string>{}</string></array>
  <key>RunAtLoad</key><true/>
</dict>
</plist>
"#,
        esc(&app.display().to_string())
    );
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&path, plist).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- state

#[derive(Clone, Serialize, Default)]
struct EjectInfo {
    state: String, // "", "running", "done", "failed", "cancelled"
    phase: String,
    step: u8, // 0 waiting, 1 stopping the engines, 2 stopping the server, 3 done
    pid: i32,
    started: f64,
}

#[derive(Clone, Serialize)]
struct Sample {
    t: f64,
    temp: i64,
    engine: i64,
    memory: i64,
    fan: i64,
    src: &'static str, // "card" (status) or "indexer" (its log)
}

#[derive(Clone, Serialize, Default)]
struct Holder {
    pid: i64,
    name: String,
    kind: String,
    command: String,
    started: i64,
}

#[derive(Clone, Serialize, Default)]
struct Snapshot {
    level: String, // idle, busy, off, warn
    mode: String,  // idle, busy, off, other, cli-error, not-posted, not-answering, reading, ejecting
    headline: String,
    detail: String,
    user: String,
    eject_label: String,
    eject_enabled: bool,
    up_label: String,
    up_enabled: bool,
    who: Option<Value>,
    who_error: Option<String>,
    holder: Option<Holder>,
    status: Option<Value>,
    status_error: Option<String>,
    status_at: f64,
    indexer: Option<Value>,
    eject: EjectInfo,
    up_running: bool,
    up_step: String,
    doctor: Option<Value>,
    setup_running: String, // the setup action running: "", "install-tinygpu", "fetch-firmware"
    log: Vec<String>,
    rawgpu: String,
    history: Vec<Sample>,
    settings: Settings,
    launch_at_login: bool,
    installed: bool,
    studio: bool, // this checkout has rawgpu Studio (tools/studio.sh)
    now: f64,
}

struct Items {
    headline: MenuItem<Wry>,
    detail: MenuItem<Wry>,
    eject: MenuItem<Wry>,
    up: MenuItem<Wry>,
    temp: CheckMenuItem<Wry>,
    login: CheckMenuItem<Wry>,
}

#[derive(Default)]
struct Watch {
    posted: Option<bool>,
    hot: bool,
    indexing: bool,
}

struct Shared {
    snap: Mutex<Snapshot>,
    history: Mutex<VecDeque<Sample>>,
    status_busy: AtomicBool,
    last_status: Mutex<Option<Instant>>,
    last_sample: Mutex<(f64, f64)>, // (time of the last card sample, time of the last indexer record)
    items: Items,
    level: Mutex<String>,
    title: Mutex<String>,
    panel_hidden_at: Mutex<Option<Instant>>,
    panel_open: AtomicBool,
    watch: Mutex<Watch>,
}

type Sh = Arc<Shared>;

fn log_line(s: &mut Snapshot, line: impl Into<String>) {
    s.log.push(line.into());
    let n = s.log.len();
    if n > 300 {
        s.log.drain(..n - 300);
    }
}

fn notify(app: &AppHandle, title: &str, body: &str) {
    if app.notification().builder().title(title).body(body).show().is_err() {
        let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let _ = Command::new("osascript")
            .arg("-e")
            .arg(format!("display notification \"{}\" with title \"{}\"", esc(body), esc(title)))
            .spawn();
    }
}

fn push_sample(sh: &Shared, smp: Sample) {
    let mut h = sh.history.lock().unwrap();
    h.push_back(smp);
    while h.len() > HISTORY {
        h.pop_front();
    }
}

/// Work out what the panel and the menu show from the latest readings.
fn derive(s: &mut Snapshot) {
    let who = s.who.clone().unwrap_or(Value::Null);
    let server = int(&who, "server_pid");
    let clients = int(&who, "client_count");
    let others = int(&who, "other_tinygpu");
    s.holder = who.get("clients").and_then(Value::as_array).and_then(|c| c.first()).map(|c| {
        let cmd = c.get("command").and_then(Value::as_str).unwrap_or("");
        Holder {
            pid: int(c, "pid"),
            name: program_name(cmd),
            kind: program_kind(cmd).into(),
            command: cmd.split_whitespace().collect::<Vec<_>>().join(" "),
            started: int(c, "started"),
        }
    });
    let holder_text = match &s.holder {
        Some(h) => format!("{} (pid {})", h.name, h.pid),
        None => "another program".into(),
    };
    s.eject_label = "Eject: Make Safe to Unplug".into();
    s.eject_enabled = server > 0;
    s.up_label = "Connect and Bring Up Card".into();
    s.up_enabled = !s.up_running && clients == 0;
    s.now = now_secs();

    if s.eject.state == "running" {
        s.level = "busy".into();
        s.mode = "ejecting".into();
        s.headline = "RX 590: ejecting".into();
        s.detail = s.eject.phase.clone();
        s.user = if clients > 0 { format!("Waiting for {holder_text} to finish") } else { "Closing the card".into() };
        s.eject_label = "Cancel Eject".into();
        s.eject_enabled = true;
        s.up_enabled = false;
        return;
    }
    // setup comes first: without TinyGPU nothing else can work
    if let Some(d) = &s.doctor {
        let ok = |k: &str| d.get(k).and_then(Value::as_bool).unwrap_or(true);
        if !ok("tinygpu_installed") || !ok("tinygpu_enabled") {
            s.level = "warn".into();
            s.mode = "setup".into();
            s.headline = "rawgpu: finish setting up".into();
            s.detail = if !ok("tinygpu_installed") {
                "Install the TinyGPU driver".into()
            } else {
                "Approve the TinyGPU driver in System Settings".into()
            };
            s.user = String::new();
            s.eject_enabled = false;
            s.up_enabled = false;
            return;
        }
        if !ok("card_bound") && server == 0 && s.eject.state != "done" {
            s.level = "off".into();
            s.mode = "no-card".into();
            s.headline = "RX 590: no card attached".into();
            s.detail = "Plug in the enclosure and switch it on".into();
            s.user = String::new();
            s.eject_enabled = false;
            s.up_enabled = false;
            return;
        }
    }
    if let Some(e) = &s.who_error {
        s.level = "warn".into();
        s.mode = "cli-error".into();
        s.headline = "rawgpu CLI not working".into();
        s.detail = e.lines().next().unwrap_or("").chars().take(90).collect();
        s.user = format!("Expected at {}", s.rawgpu);
        s.eject_enabled = false;
        s.up_enabled = false;
        return;
    }
    if server == 0 {
        s.status = None;
        s.user = "Nothing is using the card".into();
        s.eject_enabled = false;
        if others > 0 {
            s.level = "warn".into();
            s.mode = "other".into();
            s.headline = "RX 590: another TinyGPU server".into();
            s.detail = "Another program may have the card. Not safe to unplug.".into();
        } else {
            s.level = "off".into();
            s.mode = "off".into();
            s.headline = "RX 590: not connected".into();
            s.detail = if s.eject.state == "done" { "Ejected. Safe to unplug.".into() } else { "Safe to unplug.".into() };
        }
        return;
    }
    if clients > 0 {
        s.level = "busy".into();
        s.mode = "busy".into();
        s.headline = format!("RX 590: in use by {}", s.holder.as_ref().map_or("another program", |h| h.name.as_str()));
        s.user = format!("In use by {holder_text}");
        s.detail = match &s.indexer {
            Some(p) => format!(
                "Indexing {:.1} chunks/s, {} °C, fan {}%",
                p.get("chunks_per_s").and_then(Value::as_f64).unwrap_or(0.0),
                int(p, "temperature_c"),
                int(p, "fan_percent"),
            ),
            None => "Readings pause while another program has the card".into(),
        };
        s.eject_label = match &s.holder {
            Some(h) => format!("Eject: Wait for {}…", h.name),
            None => "Eject: Wait for the Card…".into(),
        };
        s.up_enabled = false;
        return;
    }
    s.user = "Nothing is using the card".into();
    match (&s.status, &s.status_error) {
        (_, Some(e)) => {
            s.level = "warn".into();
            s.mode = "not-answering".into();
            s.headline = "RX 590: not answering".into();
            s.detail = e.lines().next().unwrap_or("").chars().take(90).collect();
        }
        (None, None) => {
            s.level = "idle".into();
            s.mode = "reading".into();
            s.headline = "RX 590: reading the card…".into();
            s.detail = String::new();
        }
        (Some(st), None) => {
            if !st.get("posted").and_then(Value::as_bool).unwrap_or(false) {
                s.level = "warn".into();
                s.mode = "not-posted".into();
                s.headline = "RX 590: needs bring-up".into();
                s.detail = "The card was reset: it lost power or the Mac slept".into();
                s.up_label = "Bring Up Card".into();
                return;
            }
            let dpm = st.get("engine_dpm").and_then(Value::as_bool).unwrap_or(false);
            let fan = int(st, "fan_percent");
            s.level = "idle".into();
            s.mode = "idle".into();
            s.headline = format!(
                "RX 590: idle, {} °C{}",
                int(st, "temperature_c"),
                if fan >= 0 { format!(", fan {fan}%") } else { String::new() }
            );
            s.detail = format!(
                "Engine {} MHz, memory {} MHz{}",
                int(st, "engine_mhz"),
                int(st, "memory_mhz"),
                if dpm { "" } else { ", DPM off" }
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

/// The temperature beside the icon, when the setting is on.
fn tray_title(s: &Snapshot) -> String {
    if !s.settings.menu_bar_temp {
        return String::new();
    }
    let t = match (&s.status, &s.indexer, s.mode.as_str()) {
        (_, Some(p), "busy") => Some(int(p, "temperature_c")),
        (Some(st), _, "idle") => Some(int(st, "temperature_c")),
        _ => None,
    };
    t.map(|t| format!("{t}°")).unwrap_or_default()
}

/// Push the snapshot to the menu, the icon and the windows.
fn refresh(app: &AppHandle, sh: &Shared) {
    let snap = {
        let mut s = sh.snap.lock().unwrap();
        derive(&mut s);
        s.launch_at_login = launch_at_login();
        s.installed = app_bundle().is_some();
        s.studio = studio_script().is_some();
        s.history = sh.history.lock().unwrap().iter().cloned().collect();
        s.clone()
    };
    // No lock may be held across a tray or menu call: they run on the main thread, which may itself be
    // waiting in refresh().
    let it = &sh.items;
    let _ = it.headline.set_text(&snap.headline);
    let _ = it.detail.set_text(if snap.detail.is_empty() { &snap.user } else { &snap.detail });
    let _ = it.eject.set_text(&snap.eject_label);
    let _ = it.eject.set_enabled(snap.eject_enabled);
    let _ = it.up.set_text(&snap.up_label);
    let _ = it.up.set_enabled(snap.up_enabled);
    let _ = it.temp.set_checked(snap.settings.menu_bar_temp);
    let _ = it.login.set_checked(snap.launch_at_login);
    if let Some(tray) = app.tray_by_id("main") {
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
        let title = tray_title(&snap);
        let title_changed = {
            let mut t = sh.title.lock().unwrap();
            let c = *t != title;
            *t = title.clone();
            c
        };
        if title_changed {
            let _ = tray.set_title(if title.is_empty() { None } else { Some(title.as_str()) });
        }
        let _ = tray.set_tooltip(Some(&snap.headline));
    }
    let _ = app.emit("state", &snap);
}

/// Alerts that need the history of the readings: a card reset, high temperature, the indexer ending.
fn watch(app: &AppHandle, sh: &Shared) {
    let (posted, temp, indexing, settings) = {
        let s = sh.snap.lock().unwrap();
        let posted = s.status.as_ref().and_then(|st| st.get("posted")).and_then(Value::as_bool);
        let temp = match (&s.indexer, &s.status) {
            (Some(p), _) if s.mode == "busy" => Some(int(p, "temperature_c")),
            (_, Some(st)) if posted == Some(true) => Some(int(st, "temperature_c")),
            _ => None,
        };
        let indexing = s.holder.as_ref().is_some_and(|h| h.kind == "indexer");
        (posted, temp, indexing, s.settings.clone())
    };
    let mut alerts: Vec<(String, String)> = vec![];
    {
        let mut w = sh.watch.lock().unwrap();
        if let Some(p) = posted {
            if w.posted == Some(true) && !p {
                alerts.push((
                    "RX 590 was reset".into(),
                    "It lost power or the Mac slept. Choose Bring Up Card in the menu bar.".into(),
                ));
            }
            w.posted = Some(p);
        }
        if let Some(t) = temp {
            if settings.temp_alerts && !w.hot && t >= settings.alert_c {
                alerts.push((format!("RX 590 at {t} °C"), format!("The card reached your {} °C alert.", settings.alert_c)));
                w.hot = true;
            } else if t < settings.alert_c - 5 {
                w.hot = false;
            }
        }
        if w.indexing && !indexing && settings.notify_indexer {
            let end = indexer_record("end");
            let (title, body) = match end {
                Some(e) if e.get("ok").and_then(Value::as_bool) == Some(true) => (
                    "Indexing finished".to_string(),
                    format!(
                        "{} chunks embedded in {:.1} h; the index has {} chunks.",
                        thousands(int(&e, "chunks_embedded")),
                        e.get("wall_s").and_then(Value::as_f64).unwrap_or(0.0) / 3600.0,
                        thousands(int(&e, "index_chunks"))
                    ),
                ),
                Some(e) => (
                    "Indexing stopped".to_string(),
                    e.get("stop_reason").and_then(Value::as_str).unwrap_or("see its log").to_string(),
                ),
                None => ("The indexer exited".to_string(), "It did not finish. Run it again to resume.".to_string()),
            };
            alerts.push((title, body));
        }
        w.indexing = indexing;
    }
    for (t, b) in alerts {
        notify(app, &t, &b);
    }
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
                    let t = now_secs();
                    if v.get("posted").and_then(Value::as_bool) == Some(true) {
                        let mut last = sh.last_sample.lock().unwrap();
                        if t - last.0 >= SAMPLE_EVERY_S {
                            last.0 = t;
                            push_sample(&sh, Sample {
                                t,
                                temp: int(&v, "temperature_c"),
                                engine: int(&v, "engine_mhz"),
                                memory: int(&v, "memory_mhz"),
                                fan: int(&v, "fan_percent"),
                                src: "card",
                            });
                        }
                    }
                    s.status = Some(v);
                    s.status_error = None;
                    s.status_at = t;
                }
                Err(e) => s.status_error = Some(e),
            }
        }
        *sh.last_status.lock().unwrap() = Some(Instant::now());
        sh.status_busy.store(false, Ordering::SeqCst);
        refresh(&app, &sh);
        watch(&app, &sh);
    });
}

fn poll_once(app: &AppHandle, sh: &Sh) {
    let who = run_json(&["who", "--json"]);
    let open = sh.panel_open.load(Ordering::SeqCst);
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
        s.indexer = if indexing { indexer_record("progress") } else { None };
        if let Some(p) = &s.indexer {
            let t = p.get("t").and_then(Value::as_f64).unwrap_or(0.0);
            let mut last = sh.last_sample.lock().unwrap();
            if t > last.1 {
                last.1 = t;
                push_sample(sh, Sample {
                    t,
                    temp: int(p, "temperature_c"),
                    engine: int(p, "engine_mhz"),
                    memory: int(p, "memory_mhz"),
                    fan: int(p, "fan_percent"),
                    src: "indexer",
                });
            }
        }
        if server == 0 {
            s.status_error = None;
        }
        let every = if open { STATUS_EVERY_OPEN } else { STATUS_EVERY };
        let stale = sh.last_status.lock().unwrap().map_or(true, |t| t.elapsed() >= every);
        server > 0 && clients == 0 && s.eject.state != "running" && !s.up_running && stale
    };
    if need_status {
        read_status(app, sh);
    }
    refresh(app, sh);
    watch(app, sh);
}

/// `rawgpu doctor --json`: what is installed and attached. Looks only; cheap enough to run every minute.
fn run_doctor(app: &AppHandle, sh: &Sh) {
    let d = run_json(&["doctor", "--json"]).ok();
    sh.snap.lock().unwrap().doctor = d;
    refresh(app, sh);
}

/// A setup step: install TinyGPU, or download the AMD firmware, with the scripts that come with the CLI.
fn setup_action(app: &AppHandle, sh: &Sh, action: &str) -> Result<(), String> {
    if action == "open-settings" {
        let _ = Command::new("open").arg("x-apple.systempreferences:com.apple.LoginItems-Settings.extension").status();
        return Ok(());
    }
    let script = match action {
        "install-tinygpu" => "install-tinygpu.sh",
        "fetch-firmware" => "fetch-firmware.sh",
        _ => return Err(format!("unknown setup step {action}")),
    };
    let dir = {
        let s = sh.snap.lock().unwrap();
        if !s.setup_running.is_empty() {
            return Err("another setup step is running".into());
        }
        s.doctor.as_ref().and_then(|d| d.get("scripts")).and_then(Value::as_str).unwrap_or("").to_string()
    };
    let path = Path::new(&dir).join(script);
    if !path.is_file() {
        return Err(format!("{} is missing", path.display()));
    }
    {
        let mut s = sh.snap.lock().unwrap();
        s.setup_running = action.to_string();
        log_line(&mut s, format!("$ {}", path.display()));
    }
    refresh(app, sh);
    let (app, sh) = (app.clone(), sh.clone());
    std::thread::spawn(move || {
        let out = Command::new(if script.ends_with(".sh") && script.starts_with("install") { "/bin/bash" } else { "/bin/sh" })
            .arg(&path)
            .stdin(Stdio::null())
            .output();
        let ok = {
            let mut s = sh.snap.lock().unwrap();
            let ok = match out {
                Ok(o) => {
                    for l in String::from_utf8_lossy(&o.stdout).lines().chain(String::from_utf8_lossy(&o.stderr).lines()) {
                        log_line(&mut s, l.to_string());
                    }
                    o.status.success()
                }
                Err(e) => {
                    log_line(&mut s, format!("could not run {}: {e}", path.display()));
                    false
                }
            };
            let end = format!("[{} {}]", s.setup_running, if ok { "done" } else { "failed" });
            log_line(&mut s, end);
            s.setup_running.clear();
            ok
        };
        if !ok {
            notify(&app, "rawgpu setup", "A setup step failed. Open Details from the menu bar for its output.");
        }
        run_doctor(&app, &sh);
    });
    Ok(())
}

fn start_eject(app: &AppHandle, sh: &Sh) {
    {
        let mut s = sh.snap.lock().unwrap();
        if s.eject.state == "running" {
            return;
        }
        s.eject = EjectInfo { state: "running".into(), phase: "Starting".into(), step: 0, pid: 0, started: now_secs() };
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
            s.eject = EjectInfo { state: "failed".into(), phase: format!("could not run rawgpu: {e}"), ..Default::default() };
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
                let phase = if line.starts_with("Another program is using the card") || line.contains("still waiting") {
                    Some(("Waiting for the card to be free", 0))
                } else if line.starts_with("The card is free") || line.starts_with("TinyGPU server: pid") {
                    Some(("Stopping the card's engines", 1))
                } else if line.starts_with("GPU:") {
                    Some(("Stopping the TinyGPU server", 2))
                } else if line.starts_with("TinyGPU server: stopped") {
                    Some(("TinyGPU server stopped", 3))
                } else {
                    None
                };
                if let Some((p, step)) = phase {
                    s.eject.phase = p.into();
                    s.eject.step = step;
                }
                log_line(&mut s, line);
                drop(s);
                refresh(&app, &sh);
            }
        }
        let status = child.wait();
        let mut s = sh.snap.lock().unwrap();
        let last = s.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        let started = s.eject.started;
        let notice = match status {
            Ok(st) if st.success() => {
                s.eject = EjectInfo { state: "done".into(), phase: "Safe to unplug".into(), step: 3, pid: 0, started };
                s.status = None;
                Some(("RX 590 ejected".to_string(), "Safe to unplug. Connect again from the menu bar.".to_string()))
            }
            Ok(st) if st.signal().is_some() => {
                s.eject = EjectInfo { state: "cancelled".into(), phase: "Eject cancelled; nothing changed".into(), ..Default::default() };
                None
            }
            Ok(_) => {
                s.eject = EjectInfo { state: "failed".into(), phase: last.clone(), ..Default::default() };
                Some(("RX 590: not safe to unplug".to_string(), last.clone()))
            }
            Err(e) => {
                s.eject = EjectInfo { state: "failed".into(), phase: e.to_string(), ..Default::default() };
                None
            }
        };
        let done = format!("[eject {}]", s.eject.state);
        log_line(&mut s, done);
        drop(s);
        *sh.last_status.lock().unwrap() = None;
        sh.watch.lock().unwrap().posted = None; // an eject is not a reset
        if let Some((t, b)) = notice {
            notify(&app, &t, &b);
        }
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
        s.up_step = "Connecting".into();
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
                        {
                            let mut s = sh.snap.lock().unwrap();
                            if let Some(step) = l.strip_prefix("rawgpu: bring-up: ") {
                                s.up_step = step.trim().to_string();
                            }
                            log_line(&mut s, l);
                        }
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
            s.up_step.clear();
            if s.eject.state == "done" {
                s.eject = EjectInfo::default();
            }
            log_line(&mut s, if ok { "[up done]" } else { "[up failed]" });
        }
        if !ok {
            notify(&app, "RX 590: bring-up failed", "Open Details from the menu bar for rawgpu's messages.");
        }
        *sh.last_status.lock().unwrap() = None;
        poll_once(&app, &sh);
    });
}

// ---------------------------------------------------------------- windows

fn show_details(app: &AppHandle) {
    hide_panel(app);
    if let Some(w) = app.get_webview_window("details") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

fn hide_panel(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("panel") {
        if w.is_visible().unwrap_or(false) {
            let _ = w.hide();
        }
    }
}

/// Open the panel under the tray icon (or close it). `rect` is the icon's rectangle on screen.
fn toggle_panel(app: &AppHandle, rect: Option<tauri::Rect>) {
    let Some(w) = app.get_webview_window("panel") else { return };
    let sh = app.state::<Sh>().inner().clone();
    if w.is_visible().unwrap_or(false) {
        let _ = w.hide();
        return;
    }
    // A click on the icon while the panel is open first takes its focus away, which hides it; don't
    // reopen it straight away.
    if sh.panel_hidden_at.lock().unwrap().is_some_and(|t| t.elapsed() < Duration::from_millis(300)) {
        return;
    }
    if let Some(r) = rect {
        // The icon's rectangle is in its display's pixels. Work in points (macOS's global window
        // coordinates), with the scale of the display the icon is on, which may differ from the panel's.
        let guess = w.scale_factor().unwrap_or(1.0);
        let p0 = r.position.to_physical::<f64>(guess);
        let mon = app.monitor_from_point(p0.x, p0.y).ok().flatten();
        let scale = mon.as_ref().map_or(guess, |m| m.scale_factor());
        let pos = r.position.to_logical::<f64>(scale);
        let size = r.size.to_logical::<f64>(scale);
        let mut x = pos.x + size.width / 2.0 - PANEL_WIDTH / 2.0;
        if let Some(m) = mon {
            let left = m.position().x as f64 / scale + 8.0;
            let right = (m.position().x as f64 + m.size().width as f64) / scale - PANEL_WIDTH - 8.0;
            x = x.clamp(left, right.max(left));
        }
        let _ = w.set_position(tauri::LogicalPosition::new(x, pos.y + size.height + 4.0));
    }
    sh.panel_open.store(true, Ordering::SeqCst);
    let _ = w.show();
    let _ = w.set_focus();
    let (app2, sh2) = (app.clone(), sh.clone());
    std::thread::spawn(move || {
        *sh2.last_status.lock().unwrap() = None; // fresh readings for the open panel
        poll_once(&app2, &sh2);
        run_doctor(&app2, &sh2);
    });
}

/// Debug builds only: RAWGPU_PIN_PANEL=1 keeps the panel open when it loses focus (for screenshots).
fn pinned() -> bool {
    cfg!(debug_assertions) && std::env::var_os("RAWGPU_PIN_PANEL").is_some()
}

fn open_search() {
    // the installed app, else the one built next to this one
    let ok = Command::new("open").args(["-b", "dev.rawgpu.code-search"]).status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        let dev = repo().join("apps/target/release/bundle/macos/Code Search.app");
        let _ = Command::new("open").arg(dev).status();
    }
}

const STUDIO_PORT: u16 = 8765;

/// tools/studio.sh in the checkout this app was built from (a release has no Studio)
fn studio_script() -> Option<PathBuf> {
    let p = repo().join("tools/studio.sh");
    p.is_file().then_some(p)
}

/// rawgpu Studio, the training web app: open it, starting its server first if nothing answers on its port
/// (the script opens the browser once the server is listening).
fn open_studio_page() {
    let url = format!("http://127.0.0.1:{STUDIO_PORT}");
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], STUDIO_PORT));
    if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok() {
        let _ = Command::new("open").arg(&url).status();
        return;
    }
    let Some(script) = studio_script() else { return };
    let log = std::fs::OpenOptions::new().create(true).append(true).open(repo().join("local/studio.log"));
    let mut cmd = Command::new(script);
    cmd.current_dir(repo()).stdin(Stdio::null());
    if let Ok(f) = log {
        if let Ok(f2) = f.try_clone() {
            cmd.stdout(f).stderr(f2);
        }
    }
    let _ = cmd.spawn();
}

// ---------------------------------------------------------------- commands

#[tauri::command]
fn get_state(app: AppHandle, sh: State<'_, Sh>) -> Snapshot {
    let mut s = sh.snap.lock().unwrap();
    derive(&mut s);
    s.launch_at_login = launch_at_login();
    s.installed = app_bundle().is_some();
    s.studio = studio_script().is_some();
    s.history = sh.history.lock().unwrap().iter().cloned().collect();
    let _ = app;
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

#[tauri::command]
fn dismiss(app: AppHandle, sh: State<'_, Sh>) {
    // forget a finished, failed or cancelled eject
    {
        let mut s = sh.snap.lock().unwrap();
        if s.eject.state != "running" {
            s.eject = EjectInfo::default();
        }
    }
    refresh(&app, &sh);
}

#[tauri::command]
fn set_setting(app: AppHandle, sh: State<'_, Sh>, key: String, value: Value) -> Result<(), String> {
    if key == "launch_at_login" {
        set_launch_at_login(value.as_bool().unwrap_or(false))?;
    } else {
        let settings = {
            let mut s = sh.snap.lock().unwrap();
            match key.as_str() {
                "menu_bar_temp" => s.settings.menu_bar_temp = value.as_bool().unwrap_or(false),
                "temp_alerts" => s.settings.temp_alerts = value.as_bool().unwrap_or(true),
                "notify_indexer" => s.settings.notify_indexer = value.as_bool().unwrap_or(true),
                "alert_c" => s.settings.alert_c = value.as_i64().unwrap_or(80).clamp(50, 95),
                _ => return Err(format!("unknown setting {key}")),
            }
            s.settings.clone()
        };
        save_settings(&app, &settings);
    }
    refresh(&app, &sh);
    Ok(())
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
fn setup(app: AppHandle, sh: State<'_, Sh>, action: String) -> Result<(), String> {
    setup_action(&app, &sh, &action)
}

#[tauri::command]
fn fit_panel(app: AppHandle, height: f64) {
    if let Some(w) = app.get_webview_window("panel") {
        let _ = w.set_size(LogicalSize::new(PANEL_WIDTH, height.clamp(200.0, 900.0)));
    }
}

#[tauri::command]
fn close_panel(app: AppHandle) {
    hide_panel(&app);
}

#[tauri::command]
fn open_details(app: AppHandle) {
    show_details(&app);
}

#[tauri::command]
fn open_code_search(app: AppHandle) {
    hide_panel(&app);
    open_search();
}

#[tauri::command]
fn open_studio(app: AppHandle) {
    hide_panel(&app);
    open_studio_page();
}

#[tauri::command]
fn open_logs(app: AppHandle) {
    hide_panel(&app);
    let _ = Command::new("open").arg(home().join("rawgpu-index/logs")).status();
}

#[tauri::command]
fn quit(app: AppHandle) {
    app.exit(0);
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            let settings = load_settings(app.handle());

            let headline = MenuItem::with_id(app, "headline", "RX 590", false, None::<&str>)?;
            let detail = MenuItem::with_id(app, "detail", " ", false, None::<&str>)?;
            let eject = MenuItem::with_id(app, "eject", "Eject: Make Safe to Unplug", false, None::<&str>)?;
            let up = MenuItem::with_id(app, "up", "Connect and Bring Up Card", false, None::<&str>)?;
            let details = MenuItem::with_id(app, "details", "Details…", true, None::<&str>)?;
            let search = MenuItem::with_id(app, "search", "Code Search", true, None::<&str>)?;
            let studio = MenuItem::with_id(app, "studio", "Training Studio", studio_script().is_some(), None::<&str>)?;
            let temp = CheckMenuItem::with_id(app, "temp", "Show Temperature in Menu Bar", true, settings.menu_bar_temp, None::<&str>)?;
            let login = CheckMenuItem::with_id(app, "login", "Open at Login", true, launch_at_login(), None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit rawgpu", true, Some("CmdOrCtrl+Q"))?;
            let menu = Menu::with_items(
                app,
                &[
                    &headline,
                    &detail,
                    &PredefinedMenuItem::separator(app)?,
                    &eject,
                    &up,
                    &PredefinedMenuItem::separator(app)?,
                    &search,
                    &studio,
                    &details,
                    &PredefinedMenuItem::separator(app)?,
                    &temp,
                    &login,
                    &PredefinedMenuItem::separator(app)?,
                    &quit,
                ],
            )?;

            let sh: Sh = Arc::new(Shared {
                snap: Mutex::new(Snapshot {
                    rawgpu: rawgpu_bin().display().to_string(),
                    settings,
                    ..Default::default()
                }),
                history: Mutex::new(VecDeque::new()),
                status_busy: AtomicBool::new(false),
                last_status: Mutex::new(None),
                last_sample: Mutex::new((0.0, 0.0)),
                items: Items { headline, detail, eject, up, temp, login },
                level: Mutex::new(String::new()),
                title: Mutex::new(String::new()),
                panel_hidden_at: Mutex::new(None),
                panel_open: AtomicBool::new(false),
                watch: Mutex::new(Watch::default()),
            });
            app.manage(sh.clone());

            let menu_sh = sh.clone();
            TrayIconBuilder::with_id("main")
                .icon(Image::from_bytes(icon_for("idle"))?)
                .icon_as_template(true)
                .tooltip("rawgpu")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, rect, .. } = event {
                        toggle_panel(tray.app_handle(), Some(rect));
                    }
                })
                .on_menu_event(move |app, event| match event.id.as_ref() {
                    "eject" => {
                        if menu_sh.snap.lock().unwrap().eject.state == "running" {
                            cancel_eject(&menu_sh)
                        } else {
                            start_eject(app, &menu_sh)
                        }
                    }
                    "up" => bring_up(app, &menu_sh),
                    "details" => show_details(app),
                    "search" => open_search(),
                    "studio" => open_studio_page(),
                    "temp" => {
                        let on = !menu_sh.snap.lock().unwrap().settings.menu_bar_temp;
                        let _ = set_setting(app.clone(), app.state::<Sh>(), "menu_bar_temp".into(), Value::Bool(on));
                    }
                    "login" => {
                        if let Err(e) = set_setting(app.clone(), app.state::<Sh>(), "launch_at_login".into(), Value::Bool(!launch_at_login())) {
                            notify(app, "Open at Login", &e);
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            #[cfg(debug_assertions)] // for testing without clicking: --test-eject-after SECONDS, --show-panel
            {
                if let Some(secs) = std::env::args().skip_while(|a| a != "--test-eject-after").nth(1).and_then(|s| s.parse().ok()) {
                    let (h, s2) = (app.handle().clone(), sh.clone());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_secs(secs));
                        start_eject(&h, &s2);
                    });
                }
            }

            let (dh, dsh) = (app.handle().clone(), sh.clone());
            std::thread::spawn(move || loop {
                // the setup checks: every 20 s until everything is in place, then every 2 minutes
                run_doctor(&dh, &dsh);
                let ready = dsh.snap.lock().unwrap().doctor.as_ref().is_some_and(|d| {
                    ["tinygpu_installed", "tinygpu_enabled", "card_bound", "firmware_ok"]
                        .iter()
                        .all(|k| d.get(*k).and_then(Value::as_bool).unwrap_or(false))
                });
                std::thread::sleep(Duration::from_secs(if ready { 120 } else { 20 }));
            });

            let handle = app.handle().clone();
            std::thread::spawn(move || loop {
                poll_once(&handle, &sh);
                let open = sh.panel_open.load(Ordering::SeqCst);
                std::thread::sleep(if open { POLL_OPEN } else { POLL });
            });
            Ok(())
        })
        .on_window_event(|window, event| match event {
            WindowEvent::CloseRequested { api, .. } => {
                let _ = window.hide(); // the app lives in the menu bar
                api.prevent_close();
            }
            WindowEvent::Focused(false) if window.label() == "panel" && !pinned() => {
                // like a menu: the panel goes away when you click elsewhere
                let app = window.app_handle();
                let sh = app.state::<Sh>();
                *sh.panel_hidden_at.lock().unwrap() = Some(Instant::now());
                sh.panel_open.store(false, Ordering::SeqCst);
                let _ = window.hide();
            }
            _ => {}
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            eject,
            cancel,
            up,
            refresh_now,
            dismiss,
            set_setting,
            setup,
            full_status,
            fit_panel,
            close_panel,
            open_details,
            open_code_search,
            open_studio,
            open_logs,
            quit
        ])
        .build(tauri::generate_context!())
        .expect("error while starting the rawgpu menu bar app")
        .run(|app, event| match event {
            // opening the app again (Finder, Spotlight, `open -a rawgpu`) shows the details window
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen { .. } => show_details(app),
            tauri::RunEvent::Ready => {
                if std::env::args().any(|a| a == "--show") {
                    show_details(app);
                }
                if std::env::args().any(|a| a == "--show-panel") {
                    toggle_panel(app, None);
                }
                #[cfg(debug_assertions)] // --show-panel-at X,Y,W,H: as if the icon at that physical rectangle were clicked
                if let Some(r) = std::env::args().skip_while(|a| a != "--show-panel-at").nth(1) {
                    let v: Vec<f64> = r.split(',').filter_map(|x| x.parse().ok()).collect();
                    if v.len() == 4 {
                        toggle_panel(app, Some(tauri::Rect {
                            position: tauri::Position::Physical(PhysicalPosition::new(v[0] as i32, v[1] as i32)),
                            size: tauri::Size::Physical(tauri::PhysicalSize::new(v[2] as u32, v[3] as u32)),
                        }));
                    }
                }
            }
            _ => {}
        });
}
