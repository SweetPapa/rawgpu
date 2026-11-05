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

- **The menu** shows the card's state, temperature, fan and clocks, and which
  program has the card. While the code indexer runs, it shows the indexer's
  rate, temperature and fan from its log.
- **Eject: Make Safe to Unplug** runs `rawgpu eject`. If a program has the
  card, it waits for it, and the item becomes Cancel Eject. When the eject
  finishes, a notification says it's safe to unplug.
- **Connect and Bring Up Card** runs `rawgpu up`, which starts the TinyGPU
  server and brings the card up if it lost power or the Mac slept.
- **Show Details** opens a window with every reading, the server and its
  client, the full `rawgpu status`, and the output of the last eject or
  bring-up. Opening the app again from Finder or Spotlight shows it too.

Because the TinyGPU server takes one program at a time, the app is careful:
- It polls `rawgpu who`, which looks at processes and never connects.
- It reads the card only while the server runs with nothing connected. While
  another program has the card, the readings pause instead of queueing
  behind it.
- It never starts the server on its own. After an eject, the card stays
  disconnected until you choose Connect, or another program opens it.
- It never kills a rawgpu command mid-request. Cancelling an eject sends
  SIGTERM, which `eject` honours only while it is still waiting.

To start it at login, add it in System Settings → General → Login Items.

## Code Search

A search box over the index that `models/embeddinggemma2/index_code.py`
builds in `~/rawgpu-index`.

- **Where queries are embedded:** pick RX 590 (rawgpu), Mac GPU (PyTorch
  MPS) or Mac CPU. All three run the same model in float32, and their
  vectors agree to cosine 1.000000, so they find the same results.
- **Speed:** after the first query, embedding takes about 50 ms on the RX 590
  and about 40 ms on the Mac. Scoring all 435k chunks takes 25-50 ms. The
  first query on the RX 590 loads the model onto the card (about 1.5 s), and
  the first on the Mac loads PyTorch (5-8 s). A Mac model starts loading as
  soon as you pick it.
- **The card:** it's held only while you search, and released after 20 idle
  seconds, so an eject or the indexer gets it. If something else has the
  card, Code Search says what, and offers the Mac GPU.
- **Results:**
  - Filter by project.
  - **Fold copies** groups identical chunks from copies of a project.
  - The preview shows the chunk with line numbers.
  - Open in VS Code, Zed, Xcode or the default app, at the chunk's line.
- **Keys:** ↑ and ↓ move through the results, ⌘↩ opens one, and Esc returns
  to the search box.
- **From scripts:** `open -a "Code Search" --args --query "..." --backend mps`.

The search itself runs in `models/embeddinggemma2/search_server.py`, which
answers JSON lines on stdin and exits when the app quits. Its log is
`~/Library/Logs/rawgpu-code-search.log`.
