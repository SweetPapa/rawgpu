# Roadmap

The target is compute on the RX 590 (Polaris 30) from an Apple Silicon Mac. No
display output and no Metal: our own submission API, the way tinygrad does it.
Each stage has a check that proves it is done.

## Stage 1: talk to it ✅ (2026-11-04)

- TinyGPU binds the card, with SIP on.
- Config space, BAR mapping, register reads, VBIOS read, DMA buffer mapping.
- **Check:** `rawgpu probe` prints the ATOM header, and `rawgpu dma-test` gets bus addresses.

## Stage 2: post the card

Nothing on a Mac runs the card's VBIOS init, so the memory controller is
untrained and `CONFIG_MEMSIZE` reads 0.

- Port the ATOM interpreter from Linux `drivers/gpu/drm/amd/amdgpu/atom.c`
  (MIT; ~1,600 lines of portable C) onto `rg_dev` register access. It needs
  `reg_read/write`, `mc_read/write` and `pll_read/write` callbacks, plus a
  scratch area.
- Run `ASIC_Init` (master command table index 0) with the default engine and
  memory clocks from the `FirmwareInfo` data table. This is what
  `amdgpu_atom_asic_init()` does on an unposted card.
- I/O space is not needed. amdgpu's interpreter does all register traffic
  through MMIO and has no I/O-port path at all, which is why ARM hosts can post
  AMD cards.
- **Check:** `CONFIG_MEMSIZE` reads 8192. A pattern written through BAR0 (the
  256 MB VRAM window) reads back.

Things to watch:

- Interpreter speed. Each register read is a ~10 µs socket round trip, while
  writes are pipelined. ASIC_Init should still finish in well under a second.
  If not, add a "batch of register ops" command to a fork of the TinyGPU
  server, or run the interpreter inside it.
- Unaligned access. Apple Silicon faults on unaligned device-memory access.
  The TinyGPU server already uses aligned 32-bit copies; keep it that way.

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
