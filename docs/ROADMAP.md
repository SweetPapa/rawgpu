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

## Stage 3: firmware, memory, DMA ✅ (2026-11-04)

The reference is Linux amdgpu `vi.c`, `gmc_v8_0.c` and `sdma_v3_0.c`, plus
`pm/powerplay/smumgr/{smu7,polaris10}_smumgr.c`.

- `tools/fetch-firmware.sh` fetches pinned, hash-checked blobs from
  `linux-firmware` into `local/`. They are never committed.
- `tools/gen_vi_regs.py` generates `src/amd_vi_regs.h`: the registers and
  fields rawgpu uses, copied by script from amdgpu's headers.
- **GMC v8 GART** (`src/amd_vi_gmc.c`), as `gmc_v8_0_mc_program()` and
  `gart_enable()` do it. It has a 256 MB VMID0 GART at MC `0xff00000000`, with
  the page table in VRAM. PTEs carry TinyGPU's bus addresses.
- **SMC firmware and SMU-loaded microcode** (`src/amd_vi_smu.c`,
  `rawgpu smc-load`):
  - It follows `polaris10_start_smu_in_protection_mode()`: upload
    `polaris10_k2_smc.bin`, after which the boot ROM checks its signature and
    runs it.
  - Then `smu7_request_smu_load_fw()`: a TOC in VRAM lists the SDMA, RLC, CE,
    PFP, ME and MEC images, and `PPSMC_MSG_LoadUcodes` loads them all in 2 ms.
  - It reads the SDMA microcode RAM back to prove the load.
- **SDMA v3 rings** (`src/amd_vi_sdma.c`), as `sdma_v3_0_gfx_resume()` does
  it. The ring and fences can live in VRAM or host memory. `rawgpu sdma-test`
  runs gated phases, and refuses to unhalt an engine whose microcode RAM does
  not hold the image.
- Diagnostics: `status` (now with SMU load state and temperature), `regs
  PATTERN`, `cfg` (PCI/PCIe/AER errors), `soft-reset`.
- **Check, passed:** SDMA copies host → VRAM → VRAM → host intact, on both
  engines, up to 127 MB.
  - host → VRAM: 2.8 GB/s.
  - VRAM → host: 3.1 GB/s.
  - Numbers are in `docs/LOG.md`.

What we learned (see `docs/LOG.md`, 2026-11-04):

- **Polaris will not take engine microcode from the host.**
  - Writes to `SDMA0_UCODE_DATA` are ignored: after a direct load, the RAM
    still holds its power-on contents.
  - The SMC loads the images instead. The SMU checks the 20-byte digest at the
    end of each image and does not store it.
- **No GART is needed for the SMU.** The VBIOS leaves the memory controller in
  physical addressing mode. The TOC and images sit in VRAM and the SMU reads
  them by MC address, so loading firmware needs no host DMA.
- **The load unhalts the RLC and MEC.** Their TOC entries carry the "unhalt"
  flag, so they are running, with no queues, from `smc-load` on.
- **amdgpu's SDMA fill packet fills bytes.** `FILLSIZE` = 0 repeats the
  value's low byte, and amdgpu only fills with zero. rawgpu sets `FILLSIZE` = 2.
- **Mac sleep resets the card.** The device node and TinyGPU binding survive,
  so nothing reports it. Run `post` and `smc-load` after any sleep, and keep
  the Mac awake while a job uses the card.

Recovery notes:

- A wedged card cannot be reset from macOS. TinyGPU's reset (FLR, then hot
  reset) returns success and changes nothing. amdgpu's PCI-config reset
  (write `0x39d5e86b` to config `0x7c`) also has no effect through DriverKit.
- A GMC soft reset cleared a stuck MC busy bit. Shortly afterwards the card
  stopped answering config reads, and only a power cycle recovers that.
- Power-cycling the enclosure while TinyGPU holds a wedged card left the old
  device stuck mid-teardown. Unplugging the cable then froze the Mac, and only
  a restart cleared it. Prefer restarting the Mac with the enclosure on.

## Stage 4: compute queue and first dispatch ✅ (2026-11-04)

The reference is `gfx_v8_0.c`, plus amdkfd's `kgd_hqd_load()`.

- **GFX init for compute** (`src/amd_vi_gfx.c`):
  - The Polaris 10 golden registers, then VMID0's shader memory config.
  - `gfx_v8_0_rlc_resume()` stops, soft-resets and restarts the RLC the SMU
    left running. Doorbells are on, and the GUI-idle interrupts stay off.
- **One compute queue:** ME1 pipe 0 queue 0 in VMID0.
  - The MQD is built as `gfx_v8_0_mqd_init()` does, and the HQD registers
    are written from it as `gfx_v8_0_mqd_commit()` does. No KIQ is involved.
  - The write pointer goes to a doorbell in BAR2. Fences are `RELEASE_MEM`,
    polled; there are no interrupts.
- **Kernels:** `make kernels` builds `kernels/*.cl` for gfx803 with Homebrew
  LLVM. `src/amd_kernel.c` reads the code and kernel descriptor out of the
  object, and `vi_cq_dispatch()` sets the `COMPUTE_*` state from it, then
  `DISPATCH_DIRECT`.
- **Check 1, passed:** a PM4 `WRITE_DATA` on the compute queue lands in VRAM
  and in host memory. A fenced packet takes about 100 µs to round-trip.
- **Check 2, passed:** `vadd` over 1M floats matches the CPU bit for bit.
  - With buffers in Mac memory: 5.9 ms.
  - With buffers in VRAM: 0.46 ms.

Not supported yet, and refused by `vi_kernel_check()`: scratch (private
segment), LDS (group segment), dispatch or queue pointers, and code that
needs a linker.

## Stage 5: workloads

Done (2026-11-04):

- **Host API** (`src/amd_vi_ctx.h`): a context with the GART, SDMA0 and a
  compute queue; VRAM and host buffers; reads and writes by SDMA through a
  staging buffer; kernel loading; synchronous launches with 3D grids.
- **LDS** sized from the kernel descriptor.
- **`rawgpu bench-compute`:** copy bandwidth, FP32 MAD and FMA rates, and an
  LDS-tiled SGEMM, all checked on the CPU. At the boot clocks the card
  delivers what the silicon allows:
  - MAD: 1363 of a possible 1382 GFLOPS.
  - Copy: 35 GB/s, against 51 GB/s for the bus at that clock.

Next:

- **Clocks ✅ (docs/DPM.md).** amdgpu's powerplay code runs unmodified in
  rawgpu (`src/pp/`). `post`, `mc-load`, `dpm-enable` bring the card up
  under SMC power management.
  - The engine runs at 1540 MHz (measured on the GPU), memory at 2000 MHz.
  - FP32 MAD: 6972 GFLOPS (98% of peak). VRAM copy: 172 GB/s.
  - Getting there took AVFS (amdgpu's own SMC start), RLC safe mode, and
    bringing GFX up once.
  - Clocks are raised the way amdgpu does: SMU7 DPM. The VBIOS PowerPlay
    table (`process_pptables_v1_0.c`) gives the clock and voltage levels.
    `polaris10_init_smc_table()` writes them into the SMC's DPM table, then
    the enable messages follow.
  - Thermal and fan control come first. Voltages come only from the VBIOS
    tables, through the same code amdgpu uses.
- **Kernel features ✅.** Ordinary OpenCL C runs, with AMD's device
  libraries linked in at compile time.
  - The loader relocates whole code objects and reads their metadata.
  - Dispatch provides scratch, the AQL packet, the hidden arguments and
    dynamic `__local` memory.
  - `rawgpu kernel-test` checks it all on the card. It covers atomics, and
    29 ocml functions, every one within its OpenCL 1.2 ulp limit.
  - Atomics needed VMID 0 at MTYPE NC instead of amdgpu's UC. That also
    turned on L2 caching for every kernel.
- **A library ✅** (`include/rawgpu.h`, docs/LIBRARY.md).
  - It provides `librawgpu.a`/`.dylib` and a Python binding with numpy.
  - Bring-up runs automatically when the card needs it.
  - OpenCL C is compiled at runtime and cached.
  - Launches are asynchronous.
  - Register-blocked SGEMM: 4.8 TFLOPS at 4096 (68% of peak), exact
    against the CPU.
- **Not yet:** printf, images, OpenCL 2.0 generic-to-local/private casts
  (they need the queue pointer on gfx8), a dynamic stack, several queues,
  and GPU timestamps.

Pick a narrow workload that fits 8 GB and Polaris's FP32 strength (6–7 TFLOPS,
256 GB/s, no fast FP16), and that can run as a background job from the Mac:
embeddings, batch inference of a small model, image or audio preprocessing.
Measure it against the M4 Pro's own GPU before calling it a win.

## Side quests (good first issues)

- **Linux VFIO transport.** The same backend running on a Linux PC lets us
  diff register state against amdgpu's known-good bring-up.
- **`rawgpu regs`** dump/diff of named register ranges.
- **More backends.** SI/CIK (GCN1/2) are close cousins of amd-vi.
