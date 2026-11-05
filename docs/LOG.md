# Bring-up log

Dated, factual notes from real hardware. Newest first.

## 2026-11-05: the apps, second pass

**The menu bar app:**
- **Panel:** clicking the icon opens a panel in the macOS popover material,
  under the icon, closing when it loses focus. Right-click gives the menu.
- **Readings:**
  - stat tiles with meters; temperature turns amber at 8 °C below the
    alert and red at it;
  - a 30-minute temperature history, sampled every 10 s from
    `status --json` while idle and from the indexer's log while it runs.
    The line breaks where nothing could be read.
- **Who has the card:** shown with its run time, from a new `started`
  field in `rawgpu who --json`.
- **Steps:** the eject and the bring-up show their steps as they happen.
- **Notifications:** for a card reset (posted became not posted), the alert
  temperature, indexing finishing or stopping (from the log's end record),
  and an eject.
- **Settings:** open at login (a LaunchAgent that runs
  `open -g -a rawgpu.app`), and the temperature in the menu bar.
- **Positioning:** the panel is placed in points, using the icon's own
  display scale. This main display is 2560 × 1080 at scale 1. Using the
  panel's own scale would misplace it on mixed-DPI setups.
- **Tested** with an SGEMM load (45 → 73 °C) holding the card, then an
  eject while it held it:
  - **busy:** last readings faded, the holder card;
  - **ejecting:** waiting on the holder, then the steps;
  - **ejected:** safe to unplug.

**Code Search:**
- **Layout:** an overlay title bar with the search field, over the sidebar
  material. Search as you type (320 ms debounce), with replies to stale
  queries dropped.
- **Results:** grouped by file in the sidecar (a pool of 600 candidate
  chunks gathered into the top k files, up to 8 hits each), with language
  badges and relative score meters.
- **Filters:** project, language and path, as chips or `in:`, `lang:` and
  `path:` tokens. The language comes from the extension, with names for the
  Xojo, gettext, LaTeX and Xcode files that showed up as "Other".
- **Preview:** the whole file, read only from under the index root, with
  highlight.js (bundled, checked against two CDNs) and the matches shaded
  on a scroll rail. A match whose lines changed is found again by its text;
  one that is gone is left out of the navigation, with a note.
- **Speed:** a filtered, grouped search scores 435k chunks in about 30 ms
  once the vectors are in memory. The sidecar reads them once at startup.

## 2026-11-05: a menu bar app, Code Search, and a second-server bug

**CLI for programs:**
- **`rawgpu who [--json]`** names the program using the card from its
  process: the server's connected socket's peer is the client socket with
  the same kernel control block, which is how lsof matches them. It takes
  about 30 ms, and never connects or starts a server.
- **`rawgpu status --json`** prints one line with the state, clocks,
  temperature and fan.
- **`rawgpu up`** is the library's bring-up as a command.

**Bug: a full backlog started a second TinyGPU server.** The server
accepts one client, with one more connection waiting. A third program's
connect is refused, and `rg_open_tinygpu` took a refusal to mean no server,
so it spawned another for the same card. Reproduced with one holder, one
waiting `status` and a third `status`. Now it checks the processes for a
server on the socket, and if there is one, it waits its turn. The test
showed one server throughout, with the third command waiting and then
succeeding.

**Bug: Ctrl-C could abandon an eject mid-request.** `rawgpu eject` now
blocks SIGINT, SIGTERM and SIGHUP from the moment it attaches until the
server is stopped. Before that, a signal still stops it (tested: a SIGTERM
while it waited for a busy card exited at once, and the holder carried on).
After it, the eject finishes first and says a stop request came (tested:
SIGTERM at 50 and 200 ms into an eject of an idle card).

**Apps (`apps/`, Tauri 2):**
- **rawgpu**, in the menu bar: the card's state, who has it, the indexer's
  progress while it runs, Eject (cancellable while it waits), and bring-up.
  It never starts the server and never queues behind a client. Tested
  through busy, ejecting and ejected, with a program holding the card for
  22 s. The eject waited 10 s, then finished, and the server stayed down
  afterwards.
- **Code Search:** queries on the RX 590, the Mac's GPU or its CPU, through
  `search_server.py`. The three give the same top 5, to four digits of the
  score. A warm query on the RX 590 embeds in 49 ms and scores the 435k
  chunks in 25 ms. The card is released after 20 idle seconds.

## 2026-11-05: `rawgpu eject`, the fan readout, and ~/code indexed on the card

**`rawgpu eject`** makes unplugging the enclosure safe.
- **Waiting:** while another program holds the card, it waits, checking
  every 5 seconds. It finds the server by its arguments and counts its
  connected sockets (`proc_pidinfo`), because it must not connect. TinyGPU
  takes one client, a second connection waits in the backlog, and further
  ones are refused.
- **Never abandon a request:** an earlier version sent a probe, timed out
  and closed. The server later answered the closed socket and died of
  SIGPIPE. Attaching now sets `SO_NOSIGPIPE` and waits for the reply.
- **Closing:** it waits for GUI_ACTIVE to clear, retires the compute queue,
  halts both SDMA engines, closes the card, stops the server (SIGTERM, then
  SIGKILL after 5 s) and removes its socket.
- **Tested:** on an idle card (the next command starts a new server, and
  the card stays posted with DPM on); with a program holding the card (it
  waited 15 s, then finished); and against the indexing run below with
  `--timeout 20` (it gave up with "NOT safe to unplug", and the run went
  on undisturbed). Not tested against a program killed mid-launch.

**Fan:** `rawgpu_info.fan_percent` is the SMC's PWM duty
(CG_THERMAL_STATUS.FDO_PWM_DUTY over CG_FDO_CTRL1.FMAX_DUTY100), -1 before
`dpm-enable`. `bench.py` now logs the memory clock and the fan too.

**Indexing `~/code`** with `models/embeddinggemma2/index_code.py`:
- **Other people's code:** the first dry run came to 548M tokens and about
  15 hours. Two ESP-IDF copies were 46% of it. Skipping clones, vendored
  folders and known open-source projects cut it to 161M tokens. It found 58
  such folders, among them llama.cpp, whisper.cpp, LVGL, OpenSSL, Espressif
  components and rawgpu's own `upstream/` Linux headers. Before that,
  Linux's headers outranked rawgpu's code in searches.
- **The full run:** 4.59 hours for 429,940 chunks, at 26.0 chunks/s and
  9.7k tokens/s. 48,706 files, 2.00 GB. No errors.
- **Canary:** all 28 checks matched the CPU reference to cosine 1.0000001.
- **Card:** 49-75 C, fan up to 57%.
- **Memory clock:** samples read 400, 1000 or 2000 MHz. The SMC's memory
  DPM switches levels, and throughput stayed flat through all of them.
- **Host time** was 4%, against a 10% threshold for rewriting the host
  side in Rust. The host side is file reading, chunking, tokenizing and
  the embedding lookup. The split needed care: a launch blocks when the
  command ring is full, so the first measurement counted GPU time as host
  time and showed 86%.

## 2026-11-05: EmbeddingGemma 2 1.56x faster: padded rows, windowed attention, 8K inputs

A per-kernel profile of a 2048-token batch showed most of the products far
below the SGEMM's speed. The sliding layers' score product ran at 0.9
TFLOPS and the MLP down-projection at 1.4.

**The cause was the row strides.** Every operand's leading dimension was a
power of two (512, 1024, 2048 floats). Measured alone, with M = 8192:

| product | unpadded | rows padded by 32 floats |
|---|---|---|
| `nt`, N 512, K 2048 | 0.79 TFLOPS | 4.62 TFLOPS |
| `nt`, N 4096, K 512 | 2.74 | 4.27 |
| scores, `nt`, 2048 × 2048 × 256 | 0.63 | 3.25 |
| P·V, `nn`, 2048 × 256 × 2048 | 1.57 | 2.97 |

- **Which operands matter:** padding A and B fixes it, and C hardly
  matters.
- **How much padding:** 4 floats is enough.
- **The fix:** `eg2.py` pads the rows of every weight and of every
  activation a product reads, including the attention scores.

**Then sliding-window attention.** The batched kernels take a `window`
argument.
- **Scores:** the score product skips 128 × 128 tiles with no |q - k| below
  the window.
- **Softmax:** it touches only the keys its query's tile can see (the
  shared `band_keys()`).
- **P·V:** the product multiplies over just those keys.

At 2048 tokens that's 9 of 16 tiles. The skipped terms were exact zeros, so
nothing changes numerically.

**Three smaller changes:**
- **Softmax:** it finds the maximum and the sum in a single online pass.
- **Norms:** the RMSNorm after each residual add is fused into the add.
- **Pooling:** it runs per column. One work-group had been pooling a whole
  8192-token sequence, at 30 ms against 1 ms now.

**Results:**
- **Tokens per second, per forward pass:** 6.5k to 11.6k at 128 tokens, and
  3.3k to 8.1k at 2048.
- **Context:** `max_seq` is now 8192, with the scores buffer at 1.1 GB, and
  one 8192-token sequence runs at 5.2k tokens/s.
- **Against the reference:** `compare.py` now includes a 5998-token text.
  Every embedding still agrees with the reference to cosine 1.000000.
- **MuSiQue, 500 questions:** the RX 590 took 98 s, down from 152 s, with
  rates of 89 / 277 / 3.8 per second. The M4 Pro's MPS took about 245 s
  (36 / 92 / 1.5 per second), so the RX 590 ran about 2.5x as fast.
  Recall is unchanged.
- **The library:** `rawgpu_sgemm()` is unchanged at 4.83 TFLOPS at 3072,
  and `docs/LIBRARY.md` now has a note on padding.

## 2026-11-04: EmbeddingGemma 2 on the card, and a stale fence after an unclean exit

`models/embeddinggemma2/` runs Google's EmbeddingGemma 2 text encoder end to
end on the RX 590. It's 24 layers with per-layer embeddings, sliding-window
and full attention, a 768-dimensional output, and float32 throughout. The
matrix products use the register-blocked SGEMM, plus new batched NN/NT
variants for attention (heads and grouped KV heads by strides). Fourteen
small kernels do the rest.

- **Against the reference** (sentence-transformers, fp32, CPU), across 8
  texts, two prompts and up to 1209 tokens: identical token ids, and cosine
  1.000000 for every embedding, with a largest coordinate difference of
  2.4e-7.
- **MuSiQue retrieval:** 7029 paragraphs, 500 questions, and 64 documents of
  about 1926 tokens.
  - RX 590: 62 / 184 / 1.8 per second.
  - M4 Pro GPU (MPS, fp32): 35 / 160 / 1.5 per second.
  - Totals: 152 s against 246 s, so 1.61x. Recall@5 is 0.668 on both,
    and the two devices' embeddings agree to cosine 1.000000.

**The bug the first wrong answers came from:** OpenCL's `sincos(x, &c)`
returns the sine and stores the cosine, and the RoPE kernel had them
swapped.

**A library bug found on the way:**
- **Symptom:** a process opening the card right after another exited
  without `rawgpu_close()` had its first few launches report done before
  they ran, so reads returned stale data. The trigger was any process with
  torch loaded, where `__del__` does not run at exit.
- **Cause:** `vi_cq_start()` cleared the writeback page (now host memory,
  at the same GART address in every process) and only then drained the
  queue the earlier process had left active. The drain rewrote that
  queue's last fence (600 in the repro) into the new page. Waits for
  sequence numbers below it returned at once until the GPU wrote its first
  real fence.
- **Fix:**
  - Retire the old queue first, then clear the memory.
  - After a dequeue, reset `CP_HQD_DEQUEUE_REQUEST`, `PQ_RPTR` and
    `PQ_WPTR` to 0, as `gfx_v8_0_deactivate_hqd()` does.
  - The Python binding closes open devices from an `atexit` hook.
- **Result:** the repro (a hard `os._exit()`, then launch-and-read
  rounds) went from 4 of 4 failing to 0 of 8.

**Also:** the weight download hung for an hour on dead CDN connections
(CLOSE_WAIT) with one 48 MiB Xet chunk never written. Fetching that range
with an HTTP Range request and checking the SHA-256 recovered it.

## 2026-11-04: host load of the demos, and fences in host memory

A windowed stress suite: each scene for 60 s, then a 65536-star galaxy.
- **The card:** 49.9 / 71.0 / 59.7 / 16.8 fps, with a peak of 76 C.
  - In the 65K-star galaxy the engine fell to 1208-1373 MHz while doing its
    most work of the run (144 G interactions/s, >= 2.9 TFLOPS).
  - At 75 C that is likely the SMC's power limit, not heat.
- **The Mac:** the demo took 42-64% of one core and about 175 MB, and the
  TinyGPU server 42-62% of one core. The dext and WindowServer did not
  change.
- **Why:** the compute fence was in VRAM, so every look at it was a
  register-sized BAR read, a socket round trip through the server, in a
  tight loop.

**Fix:**
- The context's compute-queue writeback (read pointer and fences) now
  lives in its GART-mapped control buffer. The CPU reads its own memory.
- Waits on host writeback spin for 30 us, then sleep with a backoff to
  250 us. The SDMA wait does the same.
- Same windowed suite, 25 s per scene: the TinyGPU server went from about
  50% to 2.0% of a core, and the demo to 36-48%, which is its own Python
  and pygame drawing.
- Frame rates are unchanged: 48.7 / 73.7 / 60.0 fps. Launch latency in
  `compute-test` is still 96 us, and every suite passes.

## 2026-11-04: visual demos, and a launch-ordering bug they found

`demos/rawgpu_demo.py` runs three scenes, rendered and simulated on the
RX 590, in a pygame window. Frames go straight into GART-mapped host
memory, two of them alternating.

**The first ball-scene frames were corrupt** (flat dark halves, a block of
garbage). Consecutive dispatches on the compute queue overlapped:
- The end-of-pipe `RELEASE_MEM` after each dispatch only reports
  completion; it does not hold back the next dispatch.
- So the 8 physics substeps and the render ran on top of each other.
- The fix: `vi_cq_dispatch()` now emits `EVENT_WRITE CS_PARTIAL_FLUSH`
  before its cache invalidate, as amdgpu's own dispatches do
  (`gfx_v8_0_do_edc_gpr_workarounds()`).
- No earlier test had chained dependent launches without a wait.
  `python/test_rawgpu.py` now runs 100 of them on 1M floats, exact.
- Speed is unchanged (SGEMM 4.80 TFLOPS at 4096).

**Rendering fixes:**
- NaN pixels came from `powr()` of negative colours, through an AO term
  that went below zero where balls overlap. Both are clamped now.
- Mirror balls aliased badly at one ray per pixel. They now take 4 more
  reflected rays across the pixel, from re-intersecting the same ball, and
  matte balls reflect only the sky.

**`--bench` at 1280x720:**

| scene | fps | engine | max temp | work |
|---|---|---|---|---|
| bouncing balls | 46.8-47.5 | 1191-1545 MHz | 72 C | 512 balls, 98-100 M collision checks/s |
| valley flyover | 74-79 | 1544-1545 MHz | 70 C | 68-72 Mpixel/s ray-marched |
| galaxy collision | 60.1-60.6 | 1430-1545 MHz | 72 C | 32768 stars, 129-130 G interactions/s, >= 2.58 TFLOPS |

## 2026-11-04: the library, on a power-cycled card

- **`build/examples/saxpy` brought the cold card up through the library**
  (`RAWGPU_OPEN_BRINGUP`: post, mc-load, dpm-enable with AVFS). It then
  compiled its OpenCL source at runtime and got 1M results right.
- **`rawgpu kernel-test`: every check passed.** That covers work-items in
  3-D, the AQL packet, relocated constants, dynamic LDS, vectors and
  conversions, double precision, and 29 ocml functions within their
  OpenCL 1.2 limits.
  - **Scratch:** 1 KB per work-item ran at 1152 waves in 73 MB. 8 KB per
    work-item got 508 waves in 254 MB, and the SPI throttled to that, with
    no corruption.
  - **Atomics:** global add, max, xor and min, plus LDS atomics, all right
    under MTYPE NC. That confirms the UC diagnosis.
- **MTYPE NC also turned on L2 caching for every kernel.** The old 16x16
  SGEMM went from 435 to 800 GFLOPS with no change to its code, and VRAM
  copy from 172 to 186 GB/s.
- **The register-blocked SGEMM (`sgemm_nn`)** is exact against the CPU at
  every size. It ran at 2987 GFLOPS at 1024, 3911 at 2048 and 4838 at
  4096: 68% of peak. In the `KUNROLL` sweep at 4096, unroll 1/2/4 gave
  4582/4835/4653, and a full unroll at one wave per SIMD gave 3300.
- **`python/test_rawgpu.py` passed:**
  - runtime compile, 73-270 ms, then cached;
  - saxpy and a `__local` reduction;
  - GELU through ocml `erf`, within 2.7e-7;
  - host buffers;
  - compiler errors and argument errors as exceptions;
  - all 28 exact SGEMM cases (4 transposes x 7 shapes, alpha and beta);
  - a random 1000x700x900 matmul to 1.3e-6.

## 2026-11-04: kernel ABI for ordinary OpenCL C

Work toward a library: kernels that use OpenCL's built-ins, scratch and
program-scope constants.

- **Device libraries.** Clang links AMD's ROCm-Device-Libs bitcode into each
  kernel at compile time, so every OpenCL built-in is there (work-item
  functions, ocml math, vload/vstore, conversions, atomics) and no linker is
  needed. `tools/fetch-device-libs.sh` pins Debian's rocm-device-libs-17
  package (ROCm 6.0, built with upstream LLVM 17); Homebrew clang 20 reads it.
- **Loader** (`src/amd_kernel.c`). It is now a small linker:
  - It lays out every allocated section and resolves relocations for the
    load address. That makes `__constant` tables work. It can build a GOT,
    and it names any undefined symbols.
  - It parses the msgpack metadata note: argument layout, hidden arguments,
    work-group limits.
  - `rawgpu kernel-info FILE` prints this without a GPU.
- **Dispatch.** Every user SGPR except the queue pointer:
  - scratch as ROCr sizes it (swizzled buffer descriptor, `COMPUTE_TMPRING_SIZE`,
    flat-scratch init, `SH_HIDDEN_PRIVATE_BASE_VMID` = base >> 16);
  - an AQL dispatch packet for the dispatch pointer;
  - the code object v5 hidden arguments, from the metadata;
  - dynamic `__local` arguments.
- **First run of `rawgpu kernel-test`:**
  - Work-item functions in a 3-D launch passed, as did the dispatch packet
    and relocated constants.
  - **Global atomics on VRAM hung the GPU.** 1440 waves sat at
    `s_waitcnt vmcnt(0)` after their `flat_atomic_*`.
  - Cause: rawgpu runs everything in VMID 0 and had copied amdgpu's VMID 0
    `SH_MEM_CONFIG`, `DEFAULT_MTYPE = UC`, which bypasses L2, where GCN
    executes atomics. amdgpu gives its user-shader VMIDs MTYPE NC, and so does
    KFD for compute (`kfd_device_queue_manager_vi.c`). rawgpu now does too.
  - `gfx-reset` (now also forcing the HQD inactive) cleared the waves. The
    next queue still stalled on its first cache-flushing fence, so the stuck
    atomics outlive a GRBM soft reset: power cycle.

## 2026-11-04: stage 5, the RX 590 at its rated clocks

Sixth attempt, on a freshly power-cycled card. The Mac stayed up through all
the power cycles since the morning's restart.

**Sequence:** `post`, `mc-load`, then `dpm-enable`.

- `dpm-enable` started the SMC through amdgpu's `polaris10_start_smu()`,
  with AVFS (`FEATURE_STATUS` `0x0001e003`).
- It ran `hwmgr_hw_init()` and entered RLC safe mode around the DIDT
  programming, as amdgpu does on Polaris 10 (two `RLC_SAFE_MODE` writes in
  the trace).
- It then brought GFX up once, as amdgpu's IP order does.

**Checks:**

- `vram-test` passed (BAR writes at 18.7 MB/s).
- `compute-test` passed every phase, including `vadd` storing into host
  memory, the step that hung in the fifth attempt.

**`bench-compute`**, every result checked against the CPU:

| | boot clocks (earlier today) | with DPM |
|---|---|---|
| engine clock, measured with `s_memtime` | 299.3 MHz | **1540.1 MHz** |
| copy 256 MB VRAM -> VRAM | 35.4 GB/s | **171.7 GB/s** |
| FP32 `v_mad_f32` | 1363 GFLOPS | **6972 GFLOPS** |
| FP32 fused `v_fma_f32` | 343 GFLOPS | 1763 GFLOPS |
| SGEMM 1024, 16x16 LDS tiles | 175 GFLOPS | 441 GFLOPS |

- MAD reached 97.9% of the 7119 GFLOPS that 36 CUs x 128 FLOPs x 1545 MHz
  allow.
- After the run the SMC reported engine level 7 (1545 MHz) and memory level 2
  (2000 MHz), with VDDC at 1150 mV. Memory clock switching works.
- The card peaked at 50 C, with no VM faults and no PCIe errors.

`sdma-test 64` at these clocks:

- host -> VRAM 2.83 GB/s and VRAM -> host 3.06 GB/s, unchanged: the
  Thunderbolt link is the limit.
- VRAM -> VRAM on one SDMA engine: 20.4 GB/s, up from 6.7 GB/s.

**What it took, compared with amdgpu:**

- **AVFS on.** amdgpu's own SMC start must run, not rawgpu's `smc-load`.
  Without AVFS, idle VDDC dropped to 750 mV and host access to VRAM died.
- **RLC safe mode** around powerplay's GFX register programming.
- **GFX brought up once per power cycle.** No RLC reset in later processes.
- **The MC is programmed once.** A live GART's MC setup is left alone.
- **Still different from amdgpu:** PCIe DPM stays off (the Thunderbolt
  link), and no display is configured.

## 2026-11-04: stage 5, amdgpu's power management runs; memory DPM wedged the card

Same setup, posted and with `smc-load` done.

**`rawgpu dpm-info`** (`src/pp/`, amdgpu's powerplay code compiled unmodified,
docs/DPM.md) parsed the VBIOS PowerPlay table and ran the EVV voltage lookups:

```
card:         a kicker part, SMC firmware 0x00170400, reference clock 25.00 MHz, PCIe Gen3 x4 (GPU: Gen3), 256-bit GDDR5, 36 CUs
engine clock levels (vdd_dep_on_sclk):
  [0]  300 MHz  VDDC 0.750 V    [1]  909 MHz  0.800 V    [2] 1134 MHz  0.837 V    [3] 1266 MHz  0.850 V
  [4] 1365 MHz  VDDC 0.900 V    [5] 1432 MHz  0.950 V    [6] 1490 MHz  1.000 V    [7] 1545 MHz  1.018 V
memory clock levels (vdd_dep_on_mclk):  400 MHz, 1000 MHz, 2000 MHz (VDDC 0.750 / 1.150 / 1.150 V)
EVV leakage voltages (VBIOS GetVoltageInfo): 0xff02=0.837 V 0xff03=1.018 V 0xff04=1.162 V 0xff05..08=1.150 V
limits on AC: engine 1545 MHz, memory 2000 MHz, VDDC 1.162 V; TDP 185 W, max power delivery 175 W
fan: present; curve 40 C -> 20%, 65 C -> 40%, 85 C -> 60%
```

(The output is condensed here; `dpm-info` prints one level per line.)

**`rawgpu dpm-enable`** ran `hwmgr_hw_init()` in 0.28 s.

- `FEATURE_STATUS` read `0x0000e003`: engine DPM, memory DPM, the voltage
  controller, the TDC limit and GPU CAC were on.
- Feature mask: amdgpu's default without PCIe DPM and AVFS.
- One warning: "Failed to populate edc leakage registers!" The SMC refused
  the EDC enable/disable message.
- Idle, the SMC reported 285 MHz engine, 400 MHz memory, 32 C, 3.1 W.
  `status` then showed 300 / 400 MHz.

**Then the card wedged.** The next command was `bench-compute`.

- Its context setup ran `vi_gmc_init()`: the MC programming, then 512 KB of
  GART table zeroed through BAR0.
- Its SDMA microcode check then found SDMA0's RAM differing from the image at
  dword 0. It had matched before DPM was on.
- After that, config reads got no answer (`no device answered config reads`),
  as in the earlier wedge. The `display@0` device and TinyGPU's binding stayed
  registered.

**What amdgpu does that rawgpu had not:**

- `gmc_v8_0_hw_init()` loads the driver's MC sequencer firmware on every
  Polaris (`gmc_v8_0_polaris_mc_load_microcode()`, `polaris10_k_mc.bin` for P30)
  and retrains memory, before power management starts. rawgpu had left the
  VBIOS's MC program in place and then let the SMC switch memory clocks.
- amdgpu programs the MC once, at hw_init. rawgpu re-ran `mc_program()` in a
  new process with DPM live.

At the time, this looked like the likely sequence: memory traffic made the
SMC switch the memory clock, and the switch failed without the driver's MC
firmware. It was not confirmed (no trace ran during `bench-compute`), and the
second attempt below rules it out.

**Changes:**

- New `rawgpu mc-load`, a port of `gmc_v8_0_polaris_mc_load_microcode()`.
  It refuses to run while memory DPM is on.
- `dpm-enable` refuses unless the MC firmware version register (IO debug
  index `0x9F`, as `smu7_get_mc_microcode_version()` reads it) matches the
  driver's file.
- `vi_gmc_init()` leaves a live GART's MC setup alone. Each process only
  rewrites the table and the fault page.
- `dpm-enable --no-mclk` turns on engine DPM alone, for a staged test.

Order from power-on: `post`, `mc-load`, `smc-load`, `dpm-enable`.

**Second attempt (same day, after unplugging and power-cycling the enclosure;
no Mac restart).**

- The teardown completed this time, and the card re-enumerated with TinyGPU
  bound. The old dext instance stayed registered but idle.
- `post` and `mc-load` ran. The MC firmware version register read
  `0x01b4df80` **before** the load, which is the driver file's version. So the
  VBIOS had already installed it, and a missing MC firmware was not the cause.
- `smc-load`, `sdma-test` and `compute-test` all passed. The second
  `vi_gmc_init()` took the new live-GART path.
- `dpm-enable --no-mclk`: `FEATURE_STATUS` `0x0000e001` (engine DPM,
  voltage controller, TDC, CAC; memory DPM off). The SMC reported 17 MHz
  engine at idle (deep sleep), 400 MHz memory, 2.1 W.
- `bench-compute` with `RAWGPU_TRACE`: the same wedge.
  - The trace shows the SMC and VM reads fine, then 516 KB written into VRAM
    through BAR0 (the GART table and scratch).
  - The very next register read (`HDP_MEM_COHERENCY_FLUSH_CNTL`) returned
    `0xffffffff`, and every read after it did too.
  - So memory DPM was not the cause either. The card dies on a burst of
    host BAR writes once DPM runs, while register reads alone (`status`) keep
    working.
- The DPM table rawgpu wrote, rebuilt from the `dpm-enable` traces, looks
  sane.
  - Levels: 300, 909, 1134, 1266, 1365, 1432, 1490, 1545 MHz.
  - The PLL settings follow the formulas of amdgpu's reference table
    (`avfs_graphics_level_polaris10`), and level 0 matches its PLL settings
    exactly, apart from spread spectrum, which this VBIOS enables.
  - Minimum voltages: 0.75-1.16 V.
  - Deep-sleep dividers: 3 at level 0, 5 above.
- **Working theory.** At idle the SMC drops the engine clock into deep sleep,
  and probably ULV, and the host data path runs on that clock.
  - A register read queued behind a burst of posted BAR writes waits for them
    to drain.
  - If Apple's PCIe controller times out that completion sooner than an x86
    host would, the device is lost.
  - Next test: `dpm-enable --no-mclk --no-deep-sleep --no-ulv`, then
    `vram-test`.

**Third attempt (after another unplug and power cycle; no Mac restart).**

- The teardown completed again and the card came back. The bring-up passed:
  `post`, `mc-load` (training 3.9 ms), `smc-load`.
- `dpm-enable --no-mclk --no-deep-sleep --no-ulv`: `FEATURE_STATUS`
  `0x0000e001`.
  - The rebuilt table confirms `DeepSleepDivId` = 0 at every level.
  - Even so, the SMC reported 13 MHz, then 60 MHz, at idle.
- `vram-test` (traced): the first two 64 KB BAR0 writes drained, slowly
  (1.8 MB/s against 18 MB/s before DPM). The card died during the third.
  - **So deep sleep and ULV are not the cause either.**
- Two things in the traces, which amdgpu also does:
  - Every DPM run clears `SWRST_COMMAND_1.RESETLC` in the GPU's PCIe space.
    It reads `0x40` and is written `0`; `smu7_start_dpm()` calls this
    "prepare for PCIE DPM" and runs it even with PCIe DPM off.
  - `MC_ARB_CG` switches to timing set F1, as `smu7_initial_switch_from_arbf0_to_f1()`
    does. The arbiter timings the VBIOS computed per level look sane, and the
    first equals the boot timing.
- **A pattern in the SMC's idle engine-clock report:**
  - Run 1, memory DPM on: 285-300 MHz.
  - Runs 2 and 3, memory DPM off: 13-60 MHz.
  - If those are real, the host data path ran at a tenth of its clock. That
    matches the 10x slower BAR writes and a stall long enough to lose the
    device.
  - Run 1 died on the MC reprogramming under DPM, which is now fixed. So
    amdgpu's own configuration, memory DPM on, has not had a clean test.
- `status` now reads the SMC's current and target DPM indices
  (`TARGET_AND_CURRENT_PROFILE_INDEX`, `_1`) and the regulator's live
  voltages (`PWR_SVI2_STATUS`, converted as `convert_to_vddc()` does).
- Next: `dpm-enable` with amdgpu's features (less PCIe DPM and AVFS), with
  `status` before and after, then `vram-test`.

**Fourth attempt (card power-cycled; the Mac stayed up).** Bring-up as
before. `dpm-enable` with amdgpu's features less PCIe DPM and AVFS:
`FEATURE_STATUS` `0x0000e003` after 0.29 s.

| | before DPM | after DPM (and 3 s later) |
|---|---|---|
| SVI2 plane 1 (VDDC) | VID 104 = 900 mV | VID 128 = **750 mV** |
| SVI2 plane 2 | VID 104 = 900 mV | VID 104 = 900 mV |
| DPM indices | SCLK 0, MCLK 0, PCIe 0 | SCLK 0 -> 0, MCLK 0 -> 0, PCIe 0 -> 0 |
| SMC clock report | 135 / 150 MHz (bogus) | 288-300 MHz engine, 400 MHz memory |

`vram-test` (traced) failed the same way. BAR writes ran at 1.7 MB/s and the
card died within 192 KB.

- **So the engine clock was normal this time, and memory DPM was on.**
- What all four attempts share: the SMC takes VDDC from the VBIOS boot
  900 mV down to level 0's 750 mV. Then host writes into VRAM stall and the
  device is lost, while register reads keep working until then.
- amdgpu's `polaris10_populate_vr_config()` puts VDDC on SVI2 plane 1, so
  plane 1 is the GPU core rail.
- Level 0's 750 mV is a fixed entry in the VBIOS lookup table, not a leakage
  ID (EVV resolves IDs 0xff02-0xff08 only).
- **Hypothesis.** On Linux this part runs with AVFS on (closed-loop voltage,
  calibrated by the BTC in `polaris10_start_smu()`), and rawgpu turned AVFS
  off. With AVFS off, the SMC applies the table's 750 mV floor at idle, and
  this chip's memory/host path may not hold at that voltage.
  - Not tested yet. The next step is to run amdgpu's own `start_smu()` path,
    including the AVFS BTC, instead of rawgpu's smc-load. That removes the
    biggest remaining difference from Linux.

**Fifth attempt: amdgpu's own SMC start. The wedge is gone.** After `post`
and `mc-load`, `dpm-enable` started the SMC itself, through
`polaris10_start_smu()`: the protected-mode upload, AVFS staging, then the
microcode load through rawgpu's `cgs_get_firmware_info()` and
`amdgpu_ucode_init_bo()`.

- `start_smu`: 0.10 s, UcodeLoadStatus `0x5fe`, both SDMA RAMs match.
  (`avfs_btc_param` is never set in amdgpu, so the BTC message itself is
  never sent.)
- `FEATURE_STATUS` `0x0001e003`: bit 16, **AVS_ON**, is new.
- Idle: 750 mV, 288-300 MHz, 400 MHz, 5.0 W.
- **`vram-test` passed under DPM.** BAR writes ran at 18 MB/s.
  - Afterwards the SMC was at engine level 6 (1296 MHz) with VDDC at
    1100 mV: DPM followed the load.
  - So running with AVFS off was what broke host access to VRAM.
- `sdma-test --vram-only` and `sdma-test 1` passed: host writes, reads, round
  trip and fill. VRAM -> VRAM ran at 8.5 GB/s.
- **New problem: shader stores to host memory never complete.**
  - `bench-compute`'s clock probe kernel never fenced.
  - `rawgpu waves` (new) found its wave at `s_endpgm` with `IB_STS` VM_CNT
    = 2: its two stores to the host buffer were still outstanding. SDMA's
    host writes work, so the GART and PCIe are fine; the shader path through
    the texture units is not.
  - The SMC sat at engine level 7 (1545 MHz, 1156 mV). The card reached
    57-58 C while stuck.
- DPM had programmed the SQ/DB/TD/TCP di/dt limiters (880 DIDT writes).
  - Clearing `DIDT_CTRL_EN` (`rawgpu didt off`, new) did not release the wave.
  - `rawgpu gfx-reset` (new, `gfx_v8_0_soft_reset()`) cleared the waves, but
    the HQD would not deactivate afterwards, so compute needs a power cycle.
- **Two differences from amdgpu, both now fixed:**
  - amdgpu programs DIDT inside RLC safe mode, because Polaris 10's
    `cg_flags` include GFX CGCG/MGCG. rawgpu's glue had made safe mode a
    no-op. It now runs `gfx_v8_0_set_safe_mode()`.
  - amdgpu brings GFX up once, right after powerplay. rawgpu reset the RLC in
    every process. Now `dpm-enable` brings GFX up after `hwmgr_hw_init()`,
    and a marker in VRAM (`VI_VRAM_STATE`) stops later processes from
    resetting the RLC.
  - `dpm-enable --no-didt` exists, in case the stores still hang.

## 2026-11-04: stage 5 begins: host API, LDS, and what boot clocks give

Same setup. `rawgpu bench-compute` (`src/amd_vi_bench.c`) runs entirely on
the new host API (`src/amd_vi_ctx.h`): a context that owns the GART, SDMA0
and the compute queue, plus buffers, uploads by SDMA, kernel loading and
launches. Every result is checked against the CPU before its time is printed.

```
card:         36 compute units active; temperature 32 C
clock:        one wave, 4194304 loop iterations in 728.8 ms: engine (s_memtime) 299.3 MHz, reference (s_memrealtime) 25.0 MHz
copy:         256 MB VRAM -> VRAM: 15.19 ms, 35.4 GB/s (read + write); first 16 MB and last 1 MB intact
fp32 mad:     1048576 work-items x 4096 x 8 MADs (v_mad_f32: full rate): 50.40 ms, 1363 GFLOPS; 4096 samples match the CPU
fp32 fma:     1048576 work-items x 4096 x 8 FMAs (fused: quarter rate on Polaris): 200.28 ms, 343 GFLOPS; 4096 samples match the CPU
sgemm:        1024x1024, 16x16 LDS tiles: 12.24 ms, 175 GFLOPS; 64 rows match the CPU
```

- **36 CUs are active.** That count is read from
  `CC_GC_SHADER_ARRAY_CONFIG | GC_USER_SHADER_ARRAY_CONFIG` per shader
  engine, as `gfx_v8_0_get_cu_active_bitmap()` does.
- **The engine clock is 300 MHz under load.** It was measured on the GPU:
  `s_memtime` ticks over the host's wall time for a 0.73 s launch.
  `s_memrealtime` counts at 25.0 MHz, the reference clock.
- **The SMC's own clock report is wrong here.** `PPSMC_MSG_API_GetSclkFrequency`
  and `GetMclkFrequency` answer 135 MHz and 150 MHz.
  - That cannot be the memory clock: 150 MHz GDDR5 on 256 bits peaks at
    19 GB/s, and the copy moves 35 GB/s.
  - rawgpu has not given the SMC any DPM tables, so `status` labels the
    report as unreliable.
- **Full-rate FP32 is `v_mad_f32`.** It reached 1363 GFLOPS, 98.6% of the
  1382 GFLOPS that 36 CUs × 64 lanes × 2 at 300 MHz allow.
  - Fused `v_fma_f32` runs at exactly a quarter of that: 343 GFLOPS. LLVM's
    gfx803 target lacks `FastFMAF32`, for this reason.
  - OpenCL's `a * b + c` becomes `v_mac`/`v_mad`. `__builtin_fmaf` stays fused.
- **Copy:** 35.4 GB/s is 69% of the 51.2 GB/s a 256-bit GDDR5 bus gives at
  the 400 MHz boot memory clock.
- **SGEMM** is the first kernel with LDS. The descriptor asks for 2048 bytes,
  and rawgpu sets `COMPUTE_PGM_RSRC2.LDS_SIZE` in 512-byte units.
  - It took 12.2 ms at 1024 (175 GFLOPS) and 93 ms at 2048 (184 GFLOPS).
  - The speed did not change when its inner product went from fused to MAD,
    so it is bound by LDS traffic, not ALU. Its inputs are multiples of 1/16,
    so every partial sum is exact and the CPU check is bit for bit.
- After the runs: no VM faults, no PCI/PCIe/AER errors, 33 C.

At the RX 590's rated clocks the same measurements should scale about 5×:
roughly 7 TFLOPS and 200+ GB/s. That makes clocks the next thing to bring up.

## 2026-11-04: stage 4, a compute queue and the first kernel

Same host, enclosure and card, posted and with `smc-load` done. The commands
below ran in this order, several times each.

**`rawgpu compute-test`** (`src/amd_vi_gfx.c`):

```
GFX:          golden registers, VMID0 SH_MEM, RLC resumed (RLC_CNTL 0x00000001), doorbells on (BAR2 2 MB)
queue:        ME1 pipe 0 queue 0, VMID0, doorbell 16; ring 64 KB, MQD and EOP in VRAM; HQD active
[1] ring:         SET_UCONFIG_REG landed in SCRATCH_REG0
[2] VRAM write:   WRITE_DATA + RELEASE_MEM fence 1; VRAM reads 0x600df00d (intact)
[3] host write:   WRITE_DATA through the GART; host memory reads 0x5eed5eed (intact)
[4] latency:      submit -> fence seen, best of 20: 117 us
kernel:       vadd from build/kernels/vadd.o: 136 bytes of gfx803 code, 28-byte kernargs, RSRC1 0x00ac0041 RSRC2 0x0000008c
[5] vadd host: 1048576 floats in 16384 groups of 64: matches the CPU; 5.86 ms submit-to-fence (2.1 GB/s of a, b, c)
[6] vadd VRAM: 1048576 floats in 16384 groups of 64: matches the CPU; 0.46 ms submit-to-fence (27.2 GB/s of a, b, c)
```

- **The queue** is ME1 pipe 0 queue 0, in VMID0.
  - It is loaded the way amdgpu loads its KIQ: `gfx_v8_0_mqd_init()`, then
    `gfx_v8_0_mqd_commit()` with `cp_hqd_active` = 1. No KIQ or
    `MAP_QUEUES` is involved.
  - The write pointer goes to doorbell 16 (`AMDGPU_DOORBELL_MEC_RING0`) in
    BAR2, an MMIO write through the socket like any other.
  - A second run finds the queue drained by the first and reloads it.
- **Latency:** a `WRITE_DATA` plus a fence makes the round trip in 96–117 µs
  through the socket.
- **The kernel** is `kernels/vadd.cl`, built by Homebrew clang 20.1.7
  (`-target amdgcn-amd-amdhsa -mcpu=gfx803`, no device libraries, no linker).
  - rawgpu reads `.text` and `vadd.kd` straight out of the relocatable object.
    The descriptor's one relocation, its entry offset, is not needed, because
    `COMPUTE_PGM_LO/HI` points at the code directly.
  - The descriptor enables the private-segment-buffer user SGPRs (s0–s3, given
    zeros) and the kernarg pointer (s4–s5).
  - `RSRC2` asks for 6 user SGPRs plus the workgroup ID X.
  - Global pointers compile to `flat_*` instructions. VMID0's
    `SH_MEM_BASES` = 0 leaves the LDS and scratch apertures in the low 8 GB,
    below every MC address rawgpu uses.
- **Inputs** are exact in float, and the results are compared bit for bit with
  the CPU. The VRAM output is pre-filled with `0xFF` bytes each run, so the
  previous run's results cannot pass.
- **Host-memory `vadd` is limited by the link:** 12 MB moves over
  Thunderbolt. The VRAM run includes the ~100 µs submit overhead, at the boot
  clocks (engine 300 MHz, memory 400 MHz).
- `sdma-test` still passes after `compute-test`, at the same bandwidth.
- After every run: no VM faults, no PCI/PCIe/AER errors, `GRBM_STATUS` idle.
  The card was at 33 C.

## 2026-11-04: stage 3, SMC firmware loaded, SDMA moves data

Same host, enclosure and card. After the restart (below) the card was cold, and
`rawgpu post` posted it as in stage 2.

**Sleep resets the card.** About 20 minutes after posting, with nothing using
the card, `CONFIG_MEMSIZE`, `MC_VM_FB_LOCATION` and all 16 BIOS scratch
registers read 0. That is a full ASIC reset. The Mac had gone to sleep.

- The `display@0` IOPCIDevice (same registry ID), TinyGPU's binding and our
  TinyGPU server process all survived.
- Config space was restored, and PCI/PCIe/AER showed no errors.

Nothing on the Mac side reports the reset. Run `post` and `smc-load` again
after any sleep.

**`rawgpu smc-load`**, which ports `polaris10_start_smu()` and
`smu7_request_smu_load_fw()`:

```
temperature:  25 C
SMC firmware: polaris10_k2_smc.bin v0x170400, 129972 bytes; SMU in protected mode, hard key
  uploaded 129972 bytes to SMC RAM at 0x20000; start and end read back intact
  boot ROM: Test message answered 0x1; SMU_STATUS 0x00000003 (signature accepted)
  firmware up after 0.08 s: PC 0x0002021c, FIRMWARE_FLAGS 0x00000001
microcode:    9 TOC entries; images at VRAM +26 MB (368 KB), TOC at +24 MB, SMU buffer at +25 MB
LoadUcodes:   answered 0x1; UcodeLoadStatus 0x000005fe after 2 ms: SDMA0 SDMA1 CE PFP ME MEC MEC_JT1 MEC_JT2 RLC
```

- The TOC, the images and the SMU's 800 KB work buffer are all in VRAM.
  amdgpu puts its firmware buffer in GTT on bare metal, and in VRAM only with
  `debug_use_vram_fw_buf` or SR-IOV.
- The VBIOS leaves `MC_VM_MX_L1_TLB_CNTL.SYSTEM_ACCESS_MODE` = 0 (physical
  addressing). The SMU therefore read VRAM by MC address with no GART set up,
  and nothing touched host memory.
- **The SDMA microcode RAM now holds the image.** The readback matched 3104 of
  3109 dwords on both engines.
  - The 5 that differ are the last 20 bytes of each file
    (`SMU_DIGEST_SIZE_BYTES`). They are high-entropy in the file, and the RAM
    reads `0x00f00449` there, the value of the last code dword.
  - So the SMU checks the digest and does not store it. rawgpu now compares the
    3104 code dwords.
- The SMU left `SDMA0/1_UCODE_ADDR` = `0x3a`: the firmware version (58), the
  same value amdgpu's direct load writes there.
- **Side effects of the load:**
  - The TOC marks RLC and MEC "unhalt". Afterwards `RLC_CNTL` went from 0 to 1,
    and `CP_MEC_CNTL` from `0x50000000` (both MEs halted) to 0.
  - `SMU_FIRMWARE` went from `0x00030006` to `0x0003001e`: the SRAM read and
    write block bits are set. Reading the soft registers still works.
- A second `smc-load` sees the firmware running and the images loaded, and only
  re-verifies.

**`rawgpu sdma-test`: all six phases pass.**

- Phases: VRAM write, VRAM copy, host write, host read, round trip and fill.
- Every round trip was checked word for word, plus 64 VRAM spot checks through
  `MM_INDEX`.

| Run | host → VRAM | VRAM → host | VRAM → VRAM | fill → host |
|---|---|---|---|---|
| SDMA0, ring in VRAM, 64 MB | 2.77 GB/s | 3.06 GB/s | 6.71 GB/s | 3.06 GB/s |
| SDMA1, ring in VRAM, 64 MB | 2.78 GB/s | 3.06 GB/s | 7.33 GB/s | 3.06 GB/s |
| SDMA0, ring in VRAM, 127 MB | 2.78 GB/s | 3.06 GB/s | 6.73 GB/s | 3.06 GB/s |
| SDMA0, ring in host memory, 64 MB | 2.88 GB/s | 3.18 GB/s | 7.28 GB/s | 3.18 GB/s |

- Each figure is the best of 3, and includes submission and fence polling over
  the socket.
- The PCIe link is Gen3 x4 (3.94 GB/s raw).
- VRAM → VRAM ran at the VBIOS boot clocks (memory 400 MHz), so it says
  nothing about the card's real memory bandwidth.
- The host-ring runs poll their fence in local memory. Polling a VRAM ring's
  fence costs an HDP invalidate and a BAR0 read per poll.
- After every run: no PCI/PCIe/AER errors, no VM faults, both engines halted,
  `SRBM_STATUS` idle. Temperature peaked at 30 C.

**Fill packet.** amdgpu's `sdma_v3_0_emit_fill_buffer()` leaves the header's
`FILLSIZE` at 0. With that, a fill of `0xA5A55A5A` wrote `0x5a5a5a5a` across
all 16 MB: it fills with the value's low byte. amdgpu only ever fills with
zero, so it never sees this. `FILLSIZE` = 2 fills whole dwords; rawgpu now
sets it.

**Limits found:** source, destination and control pages share the 256 MB
GART, so a test buffer is at most 127 MB. `sdma-test 256` now refuses
cleanly.

## 2026-11-04: stage 3, first SDMA attempts (not working yet)

**Firmware.** `tools/fetch-firmware.sh` pulls linux-firmware
`703a4d95f86606b14b8433e7197663874ef89eed`; every file matches its pinned
sha256. `polaris10_sdma.bin` is v58 (feature 31), 3109 dwords.

**State before any writes** (`rawgpu status`):

- **SMC:** on its boot ROM (PC `0x2c8c`, no driver firmware).
  `SMU_FIRMWARE` = `0x00030006`, i.e. protected mode with `SMU_SEL` = 1.
- **SDMA0 and SDMA1:** halted and idle.
- **GART:** off.
- **RLC:** off. **MEC:** halted.
- **DMA buffers:** a 64 MB buffer maps as one segment at bus `0x80000000`.

**Runs 1–2: `sdma-test`, ring in host memory.** The engine never fetched a
packet. Afterwards:

- PCI status showed *received target abort*.
- PCIe device status showed *transactions pending*.
- `SRBM_STATUS` had `MCB_BUSY` stuck, and SDMA0 stayed busy.

Cause: `sdma-test` never mapped BAR0. The TinyGPU server silently drops MMIO
writes to an unmapped BAR (writes get no response), so the GART table was
never written. The GPU translated through stale VRAM.

Fix: the backend maps BAR0 on open, and the transport now refuses MMIO to an
unmapped or out-of-range BAR.

**Runs 4–5: ring in VRAM, nothing touching host memory.**

- PTE 0 reads back `0x0000000080000077` (bus `0x80000000`, valid, system,
  snooped, R/W/X).
- The ring in VRAM holds the right packet: `00000002 02010020 000000f4
  00000001 deadbeef`, followed by NOPs. BAR0 and `MM_INDEX` read back the same.
- Neither SDMA0 nor SDMA1 advanced its read pointer.
- SDMA0 took a VMID0 range fault reading MC address 0 (client `0x53444d30`,
  "SDM0").

**Microcode readback.** I halted each engine, set `SDMA0_UCODE_ADDR` = 0 and
read `SDMA0_UCODE_DATA` 16 times:

```
readback: ff77d9f9 bea5bef7 957bfb73 e969ef21 fb7afbab 7fa7fffe 5fec6b63 fbffdff5 ...
file:     00000000 c819007c 00000000 1d940010 994008f4 00000000 840007ad 00000000 ...
```

- The values are identical across runs, and different per engine. That is
  uninitialised SRAM.
- The host's microcode writes never landed: **Polaris ignores
  `SDMA*_UCODE_DATA` writes from the host**. The engines were running
  garbage. Firmware has to go through the SMC (`PPSMC_MSG_LoadUcodes`).

**Resets tried:**

- `rawgpu reset` (TinyGPU: FLR, else hot reset): reported success, changed
  nothing.
- `rawgpu asic-reset` (amdgpu's PCI-config reset, `0x39d5e86b` to config
  `0x7c`): no effect.
- `rawgpu soft-reset`:
  - It soft-reset SDMA, then the MC (SRBM bit `0x800`), and `MCB_BUSY`
    cleared.
  - PCIe *transactions pending* stayed set.
  - The next `vram-test` failed its `MM_INDEX` checks.
  - After that the card answered config reads with all-ones, and TinyGPU's
    reset did not bring it back.
  - **Needs a power cycle.**

**After the power cycle,** macOS never re-enumerated the card:

- The old `display@0` IOPCIDevice, and its HDMI-audio sibling, sat at
  *inactive, busy 1* for over 7 minutes. TinyGPU's per-device `IOUserServer`
  for the old instance stayed active.
- The Thunderbolt port above them stayed busy, so the new enclosure never
  showed up.
- `ps`, `pgrep -f` and `system_profiler` hung on the stuck teardown. Only
  `ioreg` still answered.
- Killing rawgpu's TinyGPU server did not release it.

The enclosure's Thunderbolt cable was then unplugged and replugged, and the
Mac froze and was restarted. After the restart the card enumerated normally and
TinyGPU bound it.

Lesson: power-cycling the enclosure while the dext holds a wedged card can
leave the teardown stuck, and only a restart of the Mac cleared it.

## 2026-11-04: stage 2, card posted from userspace

Same host, enclosure and card as stage 1. The card was cold and had never been
posted since power-on.

`rawgpu atom-info`:

- amdgpu's `amdgpu_atom_parse()` accepts the VBIOS. Its own banner reads
  `113-C94441_100, build: 441714, ver: 015.050.003.000.000000, 2019/01/17`.
- 61 of 81 command tables are present. `ASIC_Init` is at 0xad30 (149 bytes, v1.2).
- `FirmwareInfo` boot clocks, passed to ASIC_Init: engine 300 MHz, memory 400 MHz.
- `VRAM_UsageByFirmware` = 0 KB, so the ATOM FB scratch falls back to amdgpu's
  20 KB default.

`rawgpu post --trace local/post-001.trace --trace-ops`:

```
ASIC_Init returned OK (0): 0.68 s, 12242 reads, 2174 writes, 0 indirect, 3525 delays (349.2 ms requested)
posted:            yes (CONFIG_MEMSIZE = 8192 MB)
  MC_VM_FB_LOCATION          0xf5fff400   VRAM at MC 0xf400000000-0xf5ffffffff
  MC_VM_FB_OFFSET            0x00000000
  MC_SHARED_CHMAP            0x00433210
  MC_ARB_RAMCFG              0x000060a2
  MC_SEQ_MISC0               0x50601ff2   memory type field (MT) = 5
  MC_SEQ_SUP_CNTL            0xe9800001
  MC_SEQ_TRAIN_WAKEUP_CNTL   0xc00000e0
  HDP_NONSURFACE_BASE        0xf4000000
  HDP_NONSURFACE_INFO        0x00020080
  HDP_NONSURFACE_SIZE        0x7ffffbff
  BIF_FB_EN                  0x00000003   host access to VRAM: read on, write on
```

- The interpreter printed no "not implemented" or "unknown opcode" messages,
  and no register fell outside the BAR (0 indirect).
- The VBIOS left HDP pointed at the FB base (`0xf4000000 << 8` =
  `0xf400000000`) and host FB access enabled. `--enable-fb` was not needed.
- MT = 5 matches the "GDDR5" in the VBIOS name.
- Sub-tables called, counted from the opcode trace:

  | Table | Calls |
  |---|---|
  | ProcessI2cChannelTransaction | 35 |
  | AdjustMemoryController | 27 |
  | GetSCLKOverMCLKRatio | 18 |
  | SetUniphyInstance | 18 |
  | VRAM_GetCurrentInfoBlock | 16 |
  | DynamicMemorySettings | 16 |
  | MC_Synchronization | 14 |
  | GetVoltageInfo | 11 |
  | SetVoltage | 7 |
  | MC_SEQ_Control | 5 |
  | MemoryPLLInit | 3 |
  | MemoryTraining | 3 |
  | VRAM_BlockDetectionByStrap | 2 |
  | SetMemoryClock | 2 |

  This is real voltage-regulator programming over I2C and real memory training.

`rawgpu vram-test`:

```
  VRAM +   1 MB: OK
  VRAM +  64 MB: OK
  VRAM + 192 MB: OK
  through BAR0 (CPU MMIO over the socket, not DMA): write 17.8 MB/s, read 1.8 MB/s
  VRAM + 1024 MB (MM_INDEX): OK
  VRAM + 4096 MB (MM_INDEX): OK
  VRAM + 8191 MB (MM_INDEX): OK
  VRAM +    1 MB (MM_INDEX, re-reading the BAR-written region): OK
```

- The BAR regions are each written before any is read back, which catches
  address aliasing.
- The MM_INDEX re-read of the BAR region proves BAR0 offset 0 = VRAM offset 0.
- A second `rawgpu post` is a no-op ("already posted").

## 2026-11-04: stage 1 on a PowerColor RX 590

**Host:** MacBook Pro (Mac16,7, M4 Pro, 24 GB), macOS 15.5, SIP on.

**Enclosure:** AKiTiO Node Titan (TB3, firmware 50.1) on Thunderbolt bus 0,
directly on the Mac (not through the dock).

**TinyGPU:** release `c0d024f` (zip sha256 `0c47285e…a21f`). Developer ID
tinygrad, Corp. (9YG3G8543N), notarized. The dext's PCI entitlement is
vendor-wide for `0x1002` and `0x10de`, and its personality matches class
`0x03000000`. It bound the card about 10 s after the user approved it in System Settings.

```
device:            1002:67df rev e1  (AMD/ATI)
  class:           030000 VGA-compatible display controller
  board:           148c:2391  (PowerColor (TUL))
  family:          AMD GCN3/GCN4 (Volcanic Islands: Tonga/Fiji/Polaris) / POLARIS10
  PCIe link:       Gen3 x4 (card supports Gen3 x16)
  BAR0:            mem 64-bit prefetchable, 256 MB      <- VRAM window
  BAR2:            mem 64-bit prefetchable, 2 MB        <- doorbells
  BAR4:            I/O (not usable: Apple Silicon has no I/O port space)
  BAR5:            mem, 256 KB                          <- registers
  posted:          NO (CONFIG_MEMSIZE = 0x00000000)
  VBIOS:           59 KB image, checksum OK, ATOM BIOS
  part number:     113-C94441_100
  name:            C94441 POLARIS 30 XT A1 GDDR5 P10FF K0115OAX.MLC 2019
  version:         015.050.003.000.000000
  build date:      2019/01/17
  ATOM header:     @0x022e rev 1.1, subsystem 148c:2391
  command tables:  @0x9760   data tables: @0x9806
```

- **Latency:** config read 27.6 µs per op; register read 9.8 µs per op. Both are
  dominated by the socket round trip to the TinyGPU server.
- **DMA:** a 4 MB buffer mapped as one contiguous segment at bus address
  `0x80000000`. CPU write and read-back OK.
- The expansion-ROM BAR (config 0x30) is unassigned. The VBIOS is read through
  the SMC `ROM_INDEX`/`ROM_DATA` port instead, as amdgpu's `vi_read_bios_from_rom()` does.
- Polaris 30 = "P30" in amdgpu's ASIC tables, so the SMC firmware is
  `polaris10_k2_smc.bin`.

**Open question:** does it still bind and perform behind the CalDigit TS3 Plus?
Replug the enclosure into the dock's downstream TB3 port and rerun
`rawgpu probe` and `rawgpu bench`. Real bandwidth numbers wait for SDMA (stage 3).
