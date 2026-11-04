# Bring-up log

Dated, factual notes from real hardware. Newest first.

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
