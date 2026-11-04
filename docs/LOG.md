# Bring-up log

Dated, factual notes from real hardware. Newest first.

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
