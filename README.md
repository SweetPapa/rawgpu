# rawgpu

Drive a PCIe GPU from userspace on an Apple Silicon Mac, over Thunderbolt/USB4,
for compute. No kernel extension, no disabled SIP, no Metal.

macOS on Apple Silicon has no eGPU support and no AMD drivers. The card
enumerates, but nothing binds to it. rawgpu builds on tinygrad's **TinyGPU**: an
Apple-signed DriverKit extension that hands raw PCIe access (config space, BAR
MMIO, DMA buffers) to userspace. On top of that, rawgpu is a small C codebase
that identifies the card and, family by family, brings it up.

**Status:** stages 1–3 of 5 done; stage 4 (compute) is next. The first card
is a PowerColor RX 590 (Polaris 30).

- **Posting works.** rawgpu posts the card from userspace: Linux amdgpu's own
  ATOM interpreter, unmodified, runs the VBIOS `ASIC_Init`. That brings up the
  voltage regulators and trains the GDDR5, and all 8 GB of VRAM then reads and
  writes correctly.
- **The SMC runs AMD's firmware and loads the engines' microcode.** Polaris
  ignores microcode written by the host. rawgpu ports amdgpu's SMU loader:
  the boot ROM accepts the signed `polaris10_k2_smc.bin`, and the firmware
  loads the SDMA, RLC and CP images from VRAM.
- **DMA works.** Both SDMA engines copy host → VRAM at 2.8 GB/s and
  VRAM → host at 3.1 GB/s through the GART, verified word for word.

See [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/LOG.md](docs/LOG.md).

## Quick start

```sh
tools/install-tinygpu.sh      # verified install; then approve it in System Settings
tools/fetch-firmware.sh       # pinned, hash-checked AMD firmware into local/ (never committed)
make
./build/rawgpu probe
./build/rawgpu post           # after each power-on of the card, and after the Mac sleeps
./build/rawgpu vram-test
./build/rawgpu smc-load       # SMC firmware + engine microcode, once per post
./build/rawgpu sdma-test      # DMA host <-> VRAM, with bandwidth
```

**Keep the Mac awake while the card holds state.** Sleep resets the card (the
memory controller comes back untrained), and macOS does not report it. Use
`caffeinate -i` around long jobs. After a sleep, run `post` and `smc-load`
again.

Approve the driver at System Settings → General → Login Items & Extensions →
Driver Extensions → TinyGPU. Check that it bound with
`ioreg -r -c IOPCIDevice | grep -A1 display`: a `tinygpu` child should appear.

| Command | What it does | Touches the GPU |
|---|---|---|
| `rawgpu probe` | Identity, family, PCIe link, BARs, family probe | Reads only (plus the ROM index write below) |
| `rawgpu cfg` | Hex dump of config space | Config reads |
| `rawgpu vbios -o FILE` | Saves the VBIOS image | Writes the SMC ROM-index register, as amdgpu does |
| `rawgpu bench [N]` | Round-trip latency of config and register reads | Reads one scratch register |
| `rawgpu dma-test [KB]` | Allocates a DMA buffer and prints its bus addresses | None (IOMMU mapping only) |
| `rawgpu atom-info` | Lists the VBIOS command tables and boot clocks | VBIOS read only |
| `rawgpu post [--trace F] [--trace-ops]` | Runs the VBIOS `ASIC_Init` | **Writes**: clocks, voltages, memory training |
| `rawgpu vram-test [--enable-fb]` | Pattern test through BAR0 and the indirect window | **Writes** a few KB of VRAM |
| `rawgpu status` | Engine, VM, SMC and microcode state; temperature | Reads only |
| `rawgpu regs PATTERN...` | Reads registers by name, e.g. `'SDMA0_*'` | Reads only; skips data ports unless named exactly |
| `rawgpu cfg [--clear-errors]` | Config space, plus PCI/PCIe/AER error status | Config reads (clearing writes the status bits) |
| `rawgpu smc-load [--force]` | Starts the SMC firmware; it loads the SDMA, RLC and CP microcode from VRAM | **Writes**: SMC RAM, 8 MB of reserved VRAM. Lasts until the card loses power |
| `rawgpu sdma-test [MB] ...` | GART + SDMA bring-up, copy and fill tests with timing (MB ≤ 127) | **Writes**: VM setup, VRAM, host buffers. Needs `smc-load` first |
| `rawgpu soft-reset` | amdgpu's SDMA/MC soft reset for blocks that report busy | **Writes**; see the recovery notes in the roadmap |
| `rawgpu reset`, `rawgpu asic-reset` | PCIe reset / amdgpu's PCI-config reset | **Writes**; neither had any effect on this card |

Nothing touches the VRAM window until the card is posted. `post` refuses to
re-run on a posted card unless you pass `--force`. If the card stops answering
(`no device answered config reads`), power-cycle the enclosure and run `post`
again: nothing from macOS has been able to reset it.

## Swapping cards

Identification is data-driven. The AMD table is generated from Linux amdgpu's
PCI ID list (`tools/gen_amd_ids.py`), and each chip family maps to a backend or
to a note saying what can drive it. `rawgpu probe` on any card tells you where
it lands:

| Card | Today |
|---|---|
| Any Polaris: RX 460/470/480/550/560/570/580/590, Radeon Pro WX 2100–7100 | `amd-vi` backend. Same code path; the firmware file differs by chip |
| Tonga / Fiji (R9 285/380/Fury/Nano) | `amd-vi` probe works; bring-up not tested |
| AMD RDNA3+ (RX 7000/9000), NVIDIA Ampere+ (RTX 30+) | Use tinygrad directly; it drives these over the same TinyGPU transport |
| GCN1/2, Vega, RDNA1/2 | Transport and probe work; needs a backend |
| Intel Arc and other vendors | TinyGPU will not bind: its signed entitlement covers only AMD and NVIDIA |

Adding a family means adding `src/<family>.c` with a `rg_backend` and one line
in `src/family.c`.

## Architecture

```
src/transport_tinygpu.c  raw PCIe over TinyGPU's socket server (one client at a time)
src/pci.c                generic config-space decoding
src/family.c             PCI ID -> family -> backend   (+ generated src/amd_ids.inc)
src/amd_vi.c             AMD GCN3/4 backend: register helpers, probe, VBIOS, post, commands
src/amd_vi_gmc.c         GMC v8: VRAM placement and the VMID0 GART
src/amd_vi_sdma.c        SDMA v3 rings, copies, fences; sdma-test
src/amd_vi_smu.c         SMU7: SMC firmware start and SMU-loaded engine microcode; smc-load
src/amd_vi_regs.h        generated register subset (tools/gen_vi_regs.py)
src/amd_fw.c             AMD firmware file loading and header checks
src/amd_atom.c           ATOM BIOS header parsing
src/atom/upstream/       Linux amdgpu's ATOM interpreter, byte for byte (see src/atom/UPSTREAM.md)
src/atom/shim/           kernel-API shims so upstream compiles unmodified
src/atom/rg_atom.c       register/delay callbacks from the interpreter to the card
```

TinyGPU's extension can only be opened by an app carrying tinygrad's
user-client entitlement, so `TinyGPU.app` runs a Unix-socket server
(`TinyGPU server <path>`) that rawgpu starts on demand. DMA buffers are shared
memory passed over the socket, so they are zero-copy. Register reads cost a
socket round trip (about 10 µs); writes are pipelined. The server takes one
client at a time, so rawgpu and tinygrad can't share the GPU at once.

## Notes

- **Bandwidth.** The card runs at PCIe Gen3 x4 through a Thunderbolt 3
  enclosure (~32 Gb/s raw, less in practice). That matters for loading data,
  not for work that stays in the card's VRAM.
- **Firmware and VBIOS images are not committed.** `local/` and `*.rom` are
  gitignored.
- **Credits.** TinyGPU and the server protocol are from
  [tinygrad](https://github.com/tinygrad/tinygrad) (MIT). Register definitions,
  ATOM layouts and the vendored ATOM interpreter are from Linux amdgpu (MIT).
- **License.** TBD before this goes public. MIT would match both upstreams.
