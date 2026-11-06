#!/usr/bin/env python3
"""List the Rust crates an app links (name, version, license, source) as Markdown, for a release.

  python3 tools/rust-licenses.py apps/rawgpu-tray > THIRD_PARTY_RUST.md
"""
import json
import subprocess
import sys

app = sys.argv[1] if len(sys.argv) > 1 else "apps/rawgpu-tray"
meta = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--format-version", "1", "--locked", "--filter-platform", "aarch64-apple-darwin",
     "--manifest-path", f"{app}/Cargo.toml"]))
pkgs = {p["id"]: p for p in meta["packages"]}
nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
root = meta["resolve"]["root"]
seen, todo = set(), [root]
while todo:  # everything the app depends on, for this platform, without dev-dependencies
    i = todo.pop()
    if i in seen:
        continue
    seen.add(i)
    for d in nodes[i]["deps"]:
        if any(k["kind"] in (None, "normal") for k in d["dep_kinds"]):
            todo.append(d["pkg"])
seen.discard(root)
rows = sorted((pkgs[i]["name"], pkgs[i]["version"], pkgs[i].get("license") or "see source",
               pkgs[i].get("repository") or "") for i in seen if pkgs[i]["source"])
print(f"# Rust crates in {pkgs[root]['name']}\n")
print("Each crate's license text is in its source, at the repository listed.\n")
print("| crate | version | license | source |\n|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
