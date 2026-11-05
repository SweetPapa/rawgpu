# Apps

Two small [Tauri](https://tauri.app) apps for the Mac. Each has a Rust side
and a page of plain HTML and JavaScript, with no bundler and no Node
packages.

```sh
make install-apps     # builds both into ~/Applications (needs cargo, and npx for Tauri's CLI)
make apps             # builds them only, into apps/target/release/bundle/macos/
```

Both find this checkout through the path they were built from: the CLI in
`build/rawgpu` and the Python venv in `local/eg2-venv`. Rebuild them if the
checkout moves. For the CLI, `RAWGPU_BIN` overrides the path, and for the
venv, `RAWGPU_REPO`.

## rawgpu (menu bar)

A chip icon in the menu bar. It's filled while a program uses the card, faded
when nothing is connected, and shows a "!" when the card needs attention.
Click it for the panel, or right-click for a menu.

**The panel:**
- **Status:** a status pill (Ready, In use, Ejecting, Disconnected, Needs
  bring-up, Not answering), with tiles for temperature, fan, engine clock and
  memory clock. Each tile has a meter, and temperature turns amber, then red,
  as it nears your alert.
- **Temperature history:** the last 30 minutes, from the card's readings and,
  while the indexer runs, from its log. Hover for the time, temperature, clock
  and fan.
- **In use by:** the program holding the card, for how long, and its command
  line. For the indexer, also its rate, chunks, files and errors. Meanwhile
  the tiles show the last reading, faded.
- **Eject:** one button, labelled "Eject When the Indexer Finishes" while
  something has the card. It then shows its steps as they happen: wait for the
  program, stop the engines, stop the TinyGPU server, safe to unplug. You can
  cancel while it waits, and a notification says when it's safe.
- **Connect / Bring Up:** runs `rawgpu up`, with its steps: post, memory
  firmware, power management.
- **Settings (⌘, or the gear):**
  - open at login;
  - the temperature beside the icon;
  - an alert at a chosen temperature;
  - a notification when indexing finishes.

**Notifications** say when the card was reset (it lost power or the Mac
slept), when it reaches your alert temperature, when the indexer finishes or
stops, and when an eject is done.

**Details** (from the panel, the menu, or opening the app again) has the
activity log of ejects and bring-ups, the card's readings and full
`rawgpu status`, and the server and its client.

Because the TinyGPU server takes one program at a time, the app is careful:
- It polls `rawgpu who`, which looks at processes and never connects.
- It reads the card only while the server runs with nothing connected. While
  another program has the card, the readings pause instead of queueing
  behind it.
- It never starts the server on its own. After an eject, the card stays
  disconnected until you choose Connect, or another program opens it.
- It never kills a rawgpu command mid-request. Cancelling an eject sends
  SIGTERM, which `eject` honours only while it is still waiting.

## Code Search

A search box over the index that `models/embeddinggemma2/index_code.py`
builds in `~/rawgpu-index`. Results update as you type.

- **Where queries are embedded:** pick RX 590 (rawgpu), Mac GPU (PyTorch
  MPS) or Mac CPU (⌘1–3). All three run the same model in float32 and find
  the same results. A dot on each button shows its state: free, busy, or
  loading. The status bar shows who has the card, and while Code Search
  holds it, a countdown to its release and a Release button.
- **Results** are grouped by file, with each file's best matches:
  - a language badge, a relative score meter, and the match count;
  - **Fold copies** collapses identical chunks from copies of a project.
- **The preview** shows the whole file, with syntax highlighting
  (highlight.js, bundled) and its matches shaded.
  - **Navigation:** marks on the scroll rail and ⌥↑/⌥↓ move between matches.
  - **Changed files:** if a file changed since it was indexed, the matches
    are found again by their text, and a note says so.
  - **Actions:** open in VS Code, Zed, Xcode or the default app at the
    match's line (⌘↩), reveal in Finder, or copy `path:line` (⌘⇧C).
- **Filters:** project, language and path, from the chips or typed in the
  query: `in:rawgpu`, `lang:python` or `lang:ts`, `path:src/`. They become
  chips once you type a space after them.
- **Start page:** recent searches, examples, the filter syntax, and when
  the index was built.
- **Speed:**
  - **Embedding:** after the first query, about 50 ms on the RX 590 and
    about 40 ms on the Mac.
  - **Scoring:** 435k chunks in 25–50 ms.
  - **First query:** on the RX 590 it loads the model onto the card (about
    1.5 s). On the Mac, a backend starts loading PyTorch as soon as you pick
    it.
- **The card:** it's held only while you search, and released after 20 idle
  seconds, so an eject or the indexer gets it. If something else has the
  card, Code Search says what, and offers the Mac GPU.
- **From scripts:** `open -a "Code Search" --args --query "..." --backend mps`.

The search runs in `models/embeddinggemma2/search_server.py`, which answers
JSON lines on stdin and exits when the app quits. Its log is
`~/Library/Logs/rawgpu-code-search.log`.
