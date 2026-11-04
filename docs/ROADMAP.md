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

## Stage 3: firmware, memory, queues (in progress)

The reference is Linux amdgpu `vi.c`, `gmc_v8_0.c`, `gfx_v8_0.c` and
`sdma_v3_0.c`, plus `pm/powerplay/smumgr/{smu7,polaris10}_smumgr.c`.

Done:

- `tools/fetch-firmware.sh` fetches pinned, hash-checked blobs from
  `linux-firmware` into `local/`. They are never committed.
- `tools/gen_vi_regs.py` generates `src/amd_vi_regs.h`: the registers and
  fields rawgpu uses, copied by script from amdgpu's headers.
- **GMC v8 GART** (`src/amd_vi_gmc.c`), as `gmc_v8_0_mc_program()` and
  `gart_enable()` do it. It has a 256 MB VMID0 GART at MC `0xff00000000`, with
  the page table in VRAM. PTEs carry TinyGPU's bus addresses; PTE readback is
  verified.
- **SDMA v3 ring code** (`src/amd_vi_sdma.c`), as `sdma_v3_0_gfx_resume()`
  does it. The ring and fences can live in VRAM or host memory.
  `rawgpu sdma-test` runs gated phases: engine, VRAM copy, host write, host
  read, then round trip and bandwidth.
- Diagnostics: `status`, `regs PATTERN`, `cfg` (PCI/PCIe/AER errors),
  `soft-reset`.

What we learned (see `docs/LOG.md`, 2026-11-04):

- **Polaris will not take engine microcode from the host.** After a direct
  load, the SDMA microcode RAM still holds its power-on contents. Writes to
  `SDMA0_UCODE_DATA` are ignored, as the same readback shows on both engines.
  amdgpu dropped direct loading for VI for this reason. The SMC loads the
  SDMA, RLC and MEC images.

Next:

- **SMC firmware.** Polaris 30 uses `polaris10_k2_smc.bin`. The SMU here runs
  in protected mode (`SMU_FIRMWARE`: `SMU_MODE` = 1, `SMU_SEL` = 1), so port
  `polaris10_start_smu_in_protection_mode()`.
- **SMU-loaded microcode.** `smu7_request_smu_load_fw()`: a TOC in VRAM lists
  each image's MC address, then `PPSMC_MSG_LoadUcodes` and a poll of
  `UcodeLoadStatus`. Images sit in VRAM, so loading needs no host reads.
- Rerun `sdma-test`. Check that the microcode RAM reads back as the image.
- **GFX:** RLC, MEC, then one compute queue (an HQD programmed from an MQD).
- No interrupts: TinyGPU exposes no MSI path. Poll fences.
- **Check:** SDMA copies a buffer host → VRAM → host intact. Bandwidth is
  logged in `docs/LOG.md`.

Recovery notes:

- A wedged card cannot be reset from macOS. TinyGPU's reset (FLR, then hot
  reset) returns success and changes nothing. amdgpu's PCI-config reset
  (write `0x39d5e86b` to config `0x7c`) also has no effect through DriverKit.
- A GMC soft reset cleared a stuck MC busy bit. Shortly afterwards the card
  stopped answering config reads, and only a power cycle recovers that.

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
