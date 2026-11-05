// Code Search: semantic search over the EmbeddingGemma 2 index of ~/code (models/embeddinggemma2/index_code.py).
//
// The search runs in a Python sidecar (models/embeddinggemma2/search_server.py) that keeps the model
// loaded and speaks JSON lines over stdin/stdout. It embeds each query on the backend picked here:
// the RX 590 through rawgpu, or the Mac's GPU (MPS) or CPU through PyTorch. It holds the RX 590 only
// while queries come in and releases it after 20 idle seconds, so `rawgpu eject` and the indexer can
// have it. The sidecar stops when its stdin closes: when this app quits, or if it crashes.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const RELEASE_AFTER_S: &str = "20";

fn repo() -> PathBuf {
    std::env::var_os("RAWGPU_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(REPO).canonicalize().unwrap_or_else(|_| PathBuf::from(REPO)))
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

struct Sidecar {
    stdin: Mutex<ChildStdin>,
    pending: Mutex<HashMap<u64, mpsc::Sender<Value>>>,
    next: AtomicU64,
    pid: u32,
}

#[derive(Default)]
struct App {
    sidecar: Mutex<Option<Arc<Sidecar>>>,
    ready: Mutex<Option<Value>>, // the index summary from the sidecar's "ready" event
    last_error: Mutex<Option<String>>,
}

fn log_path() -> PathBuf {
    home().join("Library/Logs/rawgpu-code-search.log")
}

fn spawn_sidecar(app: &AppHandle, st: &App) -> Result<Arc<Sidecar>, String> {
    let root = repo();
    let python = root.join("local/eg2-venv/bin/python");
    let script = root.join("models/embeddinggemma2/search_server.py");
    if !python.exists() {
        return Err(format!("{} is missing: see models/embeddinggemma2/README.md to set up the venv", python.display()));
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(log_path()).map_err(|e| e.to_string())?;
    let mut child = Command::new(&python)
        .arg(&script)
        .args(["--release-after", RELEASE_AFTER_S])
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| format!("cannot start the search sidecar: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let sc = Arc::new(Sidecar {
        stdin: Mutex::new(child.stdin.take().ok_or("no stdin")?),
        pending: Mutex::new(HashMap::new()),
        next: AtomicU64::new(1),
        pid: child.id(),
    });
    *st.ready.lock().unwrap() = None;
    let (app2, sc2) = (app.clone(), sc.clone());
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if let Some(id) = v.get("id").and_then(Value::as_u64) {
                if let Some(tx) = sc2.pending.lock().unwrap().remove(&id) {
                    let _ = tx.send(v);
                }
                continue;
            }
            if v.get("event").and_then(Value::as_str) == Some("ready") {
                let st = app2.state::<App>();
                *st.ready.lock().unwrap() = v.get("index").cloned();
            }
            let _ = app2.emit("sidecar", &v);
        }
        // stdout closed: the sidecar has exited. Fail whatever was waiting on it.
        let _ = child.wait();
        for (_, tx) in sc2.pending.lock().unwrap().drain() {
            let _ = tx.send(json!({"ok": false, "error": "the search sidecar exited; see ~/Library/Logs/rawgpu-code-search.log"}));
        }
        let st = app2.state::<App>();
        let mut cur = st.sidecar.lock().unwrap();
        if cur.as_ref().map(|c| c.pid) == Some(sc2.pid) {
            *cur = None;
        }
        let _ = app2.emit("sidecar", json!({"event": "exit"}));
    });
    Ok(sc)
}

fn sidecar(app: &AppHandle) -> Result<Arc<Sidecar>, String> {
    let st = app.state::<App>();
    let mut cur = st.sidecar.lock().unwrap();
    if let Some(sc) = cur.as_ref() {
        return Ok(sc.clone());
    }
    let sc = spawn_sidecar(app, &st)?;
    *cur = Some(sc.clone());
    Ok(sc)
}

async fn request(app: AppHandle, mut req: Value) -> Result<Value, String> {
    let sc = sidecar(&app)?;
    let id = sc.next.fetch_add(1, Ordering::SeqCst);
    req["id"] = json!(id);
    let (tx, rx) = mpsc::channel();
    sc.pending.lock().unwrap().insert(id, tx);
    {
        let mut w = sc.stdin.lock().unwrap();
        writeln!(w, "{req}").and_then(|_| w.flush()).map_err(|e| format!("the search sidecar is gone: {e}"))?;
    }
    // no timeout: the first query on a backend loads the model (seconds), and the card may be bringing up
    let v = tauri::async_runtime::spawn_blocking(move || rx.recv()).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    if v.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(v)
    } else {
        Err(v.get("error").and_then(Value::as_str).unwrap_or("search failed").to_string())
    }
}

#[tauri::command]
async fn index_info(app: AppHandle) -> Result<Value, String> {
    if let Some(v) = app.state::<App>().ready.lock().unwrap().clone() {
        return Ok(v);
    }
    let _ = sidecar(&app)?; // started; its "ready" event follows
    Ok(Value::Null)
}

#[tauri::command]
async fn search(app: AppHandle, query: String, backend: String, k: u32, project: Option<String>, dedupe: bool) -> Result<Value, String> {
    let r = request(app.clone(), json!({"op": "search", "query": query, "backend": backend, "k": k,
                                         "project": project, "dedupe": dedupe, "task": "code"}))
        .await;
    *app.state::<App>().last_error.lock().unwrap() = r.as_ref().err().cloned();
    r
}

#[tauri::command]
async fn warm(app: AppHandle, backend: String) -> Result<Value, String> {
    request(app, json!({"op": "warm", "backend": backend})).await
}

#[tauri::command]
async fn release_card(app: AppHandle) -> Result<Value, String> {
    request(app, json!({"op": "release"})).await
}

#[tauri::command]
fn restart_sidecar(app: AppHandle, st: State<'_, App>) -> Result<(), String> {
    if st.sidecar.lock().unwrap().is_some() {
        return Ok(()); // still running
    }
    sidecar(&app).map(|_| ())
}

/// Who has the RX 590, from `rawgpu who --json` (which never connects to the card).
#[tauri::command]
async fn card_state() -> Result<Value, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let out = Command::new(repo().join("build/rawgpu")).args(["who", "--json"]).stdin(Stdio::null()).output()
            .map_err(|e| e.to_string())?;
        serde_json::from_slice::<Value>(&out.stdout).map_err(|e| format!("rawgpu who: {e}"))
    })
    .await
    .map_err(|e| e.to_string())?
}

fn editor_cli(app: &str) -> Option<PathBuf> {
    let p = match app {
        "vscode" => "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        "zed" => "/Applications/Zed.app/Contents/MacOS/cli",
        _ => return None,
    };
    Path::new(p).exists().then(|| PathBuf::from(p))
}

#[tauri::command]
fn editors() -> Vec<(String, String)> {
    let mut v = vec![];
    if editor_cli("vscode").is_some() {
        v.push(("vscode".into(), "VS Code".into()));
    }
    if editor_cli("zed").is_some() {
        v.push(("zed".into(), "Zed".into()));
    }
    if Path::new("/Applications/Xcode.app").exists() {
        v.push(("xcode".into(), "Xcode".into()));
    }
    v.push(("default".into(), "Default app".into()));
    v
}

#[tauri::command]
fn open_in_editor(path: String, line: u32, editor: String) -> Result<(), String> {
    if !Path::new(&path).exists() {
        return Err(format!("{path} no longer exists"));
    }
    let status = match editor.as_str() {
        "vscode" | "zed" => {
            let cli = editor_cli(&editor).ok_or("editor not found")?;
            let mut c = Command::new(cli);
            if editor == "vscode" {
                c.arg("-g");
            }
            c.arg(format!("{path}:{line}")).status()
        }
        "xcode" => Command::new("/usr/bin/xed").args(["-l", &line.to_string(), &path]).status(),
        _ => Command::new("open").arg(&path).status(),
    };
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("the editor exited with {s}")),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    Command::new("open").args(["-R", &path]).status().map(|_| ()).map_err(|e| e.to_string())
}

/// `--query TEXT` and `--backend rawgpu|mps|cpu` on the command line (`open -a "Code Search" --args --query ...`)
#[tauri::command]
fn launch_args() -> Value {
    let args: Vec<String> = std::env::args().collect();
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    json!({"query": after("--query"), "backend": after("--backend")})
}

#[tauri::command]
fn open_log() {
    let _ = Command::new("open").args(["-a", "Console"]).arg(log_path()).status();
}

fn main() {
    tauri::Builder::default()
        .manage(App::default())
        .setup(|app| {
            let handle = app.handle().clone();
            if let Err(e) = sidecar(&handle) {
                *app.state::<App>().last_error.lock().unwrap() = Some(e);
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            index_info,
            search,
            release_card,
            warm,
            restart_sidecar,
            card_state,
            editors,
            open_in_editor,
            reveal,
            launch_args,
            open_log
        ])
        .run(tauri::generate_context!())
        .expect("error while running Code Search");
}
