# Vendored: amdgpu ATOM BIOS interpreter

`upstream/` holds files from Linux `drivers/gpu/drm/amd/`, **byte for byte**.
They were fetched from torvalds/linux `master` on 2026-11-04; master was at
`05441d697f7f9a927264f4058cbbc6a03dfcac9d` at the first fetch. They are
MIT-licensed by AMD and Red Hat; see each file's header.

| File | sha256 (first 16) |
|---|---|
| `atom-bits.h` | `d885526a0ebf5f47` |
| `atom-names.h` | `e197bec3d065d70d` |
| `atom-types.h` | `ca242c70b270fa2f` |
| `atom.c` | `f53d0c31ac52f051` |
| `atom.h` | `f67723a000d98959` |
| `atombios.h` | `2f543d5518d0a2bf` |
| `atomfirmware.h` | `b389f8aed3dc383b` |
| `atomfirmwareid.h` | `9e6bd22e466b4b0d` |
| `pptable.h` | `a3f907e51ad17ce7` |

`atom.c` and `atom.h` come from `amdgpu/`; the rest come from `include/`.

## How it compiles unmodified

- `shim/` provides the kernel headers `atom.c` includes (`linux/*.h`,
  `drm/drm_util.h`, `amdgpu.h`, `ObjectID.h`). Each of them just includes
  `shim/rg_kcompat.h`.
- `rg_kcompat.h` covers the API the file uses: allocation, logging, `do_div`,
  `jiffies`, delays, and no-op mutexes.
- Delays and the 20 s jump-loop watchdog call back into `rg_atom.c`. Every delay
  first does one register read, which drains register writes still queued in
  the TinyGPU socket, so the hardware sees the delay the VBIOS asked for.

`rg_atom.c` supplies the `struct card_info` register callbacks, as
`amdgpu_atombios.c` does in the kernel.
- Registers past the 256 KB BAR use the `MM_INDEX`/`MM_DATA` window.
- MC and PLL spaces fail loudly, as they `BUG_ON` in amdgpu.

## Updating

Copy the same files from a newer kernel, rebuild, and rerun `rawgpu atom-info`
and `rawgpu post --force` against a card. Update the table above. If the
upstream API changes, fix `rg_kcompat.h`; never edit `upstream/`.
