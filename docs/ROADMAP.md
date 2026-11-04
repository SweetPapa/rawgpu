# Roadmap

The target is compute on the RX 590 (Polaris 30) from an Apple Silicon Mac. No
display output and no Metal: our own submission API, the way tinygrad does it.
Each stage has a check that proves it is done.

## Stage 1: talk to it ✅ (2026-11-04)

- TinyGPU binds the card, with SIP on.
- Config space, BAR mapping, register reads, VBIOS read, DMA buffer mapping.
- **Check:** `rawgpu probe` prints the ATOM header, and `rawgpu dma-test` gets bus addresses.

## Stage 2: post the card ✅ (2026-11-04)

Nothing on a Mac runs the card's VBIOS init, so a cold card has an untrained
memory controller and `CONFIG_MEMSIZE` reads 0.

- Linux amdgpu's ATOM interpreter (`atom.c`) is vendored **unmodified** and
  compiled against kernel shims; see `src/atom/UPSTREAM.md`.
- `rawgpu post` runs `ASIC_Init` with the VBIOS boot clocks, as
  `amdgpu_atom_asic_init()` does. On the RX 590 it took 0.68 s, with
  12,242 register reads, 2,174 writes and 3,525 delays.
- I/O space was not needed. amdgpu's interpreter does all register traffic
  through MMIO.
- **Check, passed:**
  - `CONFIG_MEMSIZE` = 8192 MB.
  - `rawgpu vram-test` patterns 3 × 64 KB through BAR0. It also checks 1 GB,
    4 GB and 8 GB − 1 MB through `MM_INDEX`, and re-reads the BAR-written
    region the same way. All intact.

What we learned:

- **Host access to VRAM is slow.** BAR0 runs at about 18 MB/s write and
  1.8 MB/s read, because every read is a PCIe round trip through the socket
  server. Bulk data must move by DMA (SDMA, stage 3), never through the BAR.
- **Posting should not survive a power cycle.** Memory training lives in the
  card, so this is expected but not yet tested. Run `rawgpu post` after every
  power-on; it is a no-op on an already-posted card.

## Stage 3: firmware, memory, queues

The reference is Linux amdgpu `vi.c`, `gmc_v8_0.c`, `gfx_v8_0.c` and
`sdma_v3_0.c`, plus `pm/powerplay/smumgr/polaris10_smumgr.c`.

- **SMC firmware.** Polaris 30 (the RX 590) uses `polaris10_k2_smc.bin`: the
  "kicker" variant, chosen by device ID and revision in `amdgpu_cgs.c`. The
  blobs come from `linux-firmware`. Do not commit them here.
- **GMC v8.** Set up a VM page table so the GPU can reach DMA buffers. Bus
  addresses come from `rg_dma.seg[]`, 40-bit, 16 KB-page DART underneath.
- **GFX microcode:** RLC, MEC, then one compute queue (an HQD programmed from
  an MQD).
- **SDMA engine.** Gives real DMA bandwidth numbers, which settles the
  "behind the CalDigit dock" question.
- No interrupts: TinyGPU exposes no MSI path. Poll fences in DMA memory.
- **Check:** SDMA copies a buffer host → VRAM → host intact. Bandwidth is
  logged in `docs/LOG.md`.

## Stage 4: first dispatch

- Compile a kernel for `gfx803`. Homebrew LLVM has the target:
  `clang -target amdgcn-amd-amdhsa -mcpu=gfx803`.
- Load it into VRAM and submit `DISPATCH_DIRECT` through PM4 on the compute queue.
- **Check:** a vector add matches the CPU result.

## Stage 5: workloads

Pick a narrow workload that fits 8 GB and Polaris's FP32 strength (6–7 TFLOPS,
256 GB/s, no fast FP16), and that can run as a background job from the Mac:
embeddings, batch inference of a small model, image or audio preprocessing.
Measure it against the M4 Pro's own GPU before calling it a win.

## Side quests (good first issues)

- **Linux VFIO transport.** The same backend running on a Linux PC lets us
  diff register state against amdgpu's known-good bring-up.
- **`rawgpu regs`** dump/diff of named register ranges.
- **More backends.** SI/CIK (GCN1/2) are close cousins of amd-vi.
