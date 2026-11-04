# Vendored: amdgpu structure headers

Files here come from Linux `drivers/gpu/drm/amd/`, **byte for byte**, at
torvalds/linux `05441d697f7f9a927264f4058cbbc6a03dfcac9d`, the same commit as
the generated register header (`src/amd_vi_regs.h`). They are MIT-licensed by
AMD; see each file's header. Include `<stdint.h>` before them.

| File | From | sha256 |
|---|---|---|
| `vi_structs.h` | `include/vi_structs.h` | `b76155a25e98f3788e12a784824c04aecfc654543c09bc07345e2c9e111d569a` |

`vi_structs.h` holds `struct vi_mqd`, the memory queue descriptor a compute
queue is loaded from. Its fields from `cp_mqd_base_addr_lo` on mirror the HQD
registers from `mmCP_MQD_BASE_ADDR` on, which `src/amd_vi_gfx.c` relies on
(and checks with static asserts).
