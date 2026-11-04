# rawgpu

Drive a PCIe GPU from userspace on an Apple Silicon Mac, over Thunderbolt/USB4,
for compute. No kernel extension, no disabled SIP, no Metal.

macOS on Apple Silicon has no eGPU support and no AMD drivers. The card
enumerates, but nothing binds to it. rawgpu builds on tinygrad's **TinyGPU**: an
Apple-signed DriverKit extension that hands raw PCIe access (config space, BAR
MMIO, DMA buffers) to userspace. On top of that, rawgpu is a small C codebase
that identifies the card and, family by family, brings it up.

**Status:** stage 1 of 5. The first card is a PowerColor RX 590 (Polaris 30).
TinyGPU binds it, and rawgpu reads its config space, registers and VBIOS, and maps
DMA buffers for it. The card is not posted yet, so its VRAM is not usable. See
[docs/ROADMAP.md](docs/ROADMAP.md) and [docs/LOG.md](docs/LOG.md).

## Quick start

```sh
tools/install-tinygpu.sh      # verified install; then approve it in System Settings
make
./build/rawgpu probe
```

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

Nothing here resets the card or touches the VRAM window before the card is posted.

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
src/amd_vi.c             AMD GCN3/4 backend: probe, VBIOS
src/amd_atom.c           ATOM BIOS parsing (the interpreter comes in stage 2)
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
  [tinygrad](https://github.com/tinygrad/tinygrad) (MIT). Register definitions
  and ATOM layouts are from Linux amdgpu (MIT).
- **License.** TBD before this goes public. MIT would match both upstreams.
