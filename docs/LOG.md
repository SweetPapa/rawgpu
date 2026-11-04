# Bring-up log

Dated, factual notes from real hardware. Newest first.

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
