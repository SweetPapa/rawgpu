#!/usr/bin/env python3
"""Generate src/amd_vi_regs.h: the VI (GCN3/4) registers rawgpu uses, with their fields.

Linux amdgpu's register headers (drivers/gpu/drm/amd/include/asic_reg/) are
several MB; this pulls out only the registers listed below, so every offset and
field mask in rawgpu is copied by a script, not by hand.

usage: gen_vi_regs.py <dir with the upstream headers> <linux commit sha> [names.inc]

With a third argument, also writes a {name, dword index} table of the mmFOO
registers for `rawgpu regs`.

The directory holds the files named in HEADERS (flat or in their asic_reg/
subdirectories). Names in REGS may end in '*' to take every register with that
prefix. mmFOO is a dword index into the register BAR; ixFOO is an SMC
indirect address.
"""
import fnmatch
import os
import re
import sys

HEADERS = [
    "bif/bif_5_0", "gmc/gmc_8_1", "oss/oss_3_0", "gca/gfx_8_0", "smu/smu_7_1_3",
]

REGS = [
    # BIF: indirect window, memsize, FB access, scratch, doorbells, HDP flush
    "MM_INDEX", "MM_DATA", "MM_INDEX_HI", "CONFIG_MEMSIZE", "BIF_FB_EN", "BIOS_SCRATCH_*",
    "BIF_DOORBELL_APER_EN", "GPU_HDP_FLUSH_REQ", "GPU_HDP_FLUSH_DONE",
    # GMC: memory controller and VM
    "MC_SHARED_CHMAP", "MC_ARB_RAMCFG", "MC_SEQ_SUP_CNTL", "MC_SEQ_TRAIN_WAKEUP_CNTL", "MC_SEQ_MISC0",
    "MC_ARB_WTM_GRPWT_RD", "MC_SHARED_BLACKOUT_CNTL", "MC_VM_FB_LOCATION", "MC_VM_FB_OFFSET", "MC_VM_AGP_*",
    "MC_VM_SYSTEM_APERTURE_*", "MC_VM_MX_L1_TLB_CNTL", "VM_L2_CNTL", "VM_L2_CNTL2", "VM_L2_CNTL3",
    "VM_L2_CNTL4", "VM_L2_STATUS", "VM_CONTEXT0_*", "VM_CONTEXT1_CNTL", "VM_CONTEXT1_CNTL2",
    "VM_CONTEXT1_PAGE_TABLE_START_ADDR", "VM_CONTEXT1_PAGE_TABLE_END_ADDR",
    "VM_CONTEXT1_PROTECTION_FAULT_*", "VM_L2_CONTEXT1_IDENTITY_APERTURE_*",
    "VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", "VM_INVALIDATE_REQUEST", "VM_INVALIDATE_RESPONSE",
    "VM_PRT_APERTURE*", "VM_PRT_CNTL",
    # OSS: HDP, SRBM, SDMA
    "HDP_NONSURFACE_BASE", "HDP_NONSURFACE_INFO", "HDP_NONSURFACE_SIZE", "HDP_MEM_COHERENCY_FLUSH_CNTL",
    "HDP_REG_COHERENCY_FLUSH_CNTL", "HDP_DEBUG0", "HDP_MISC_CNTL", "HDP_HOST_PATH_CNTL",
    "SRBM_STATUS", "SRBM_STATUS2", "SRBM_GFX_CNTL", "SRBM_SOFT_RESET",
    "SDMA0_UCODE_ADDR", "SDMA0_UCODE_DATA", "SDMA0_F32_CNTL", "SDMA0_CNTL", "SDMA0_CHICKEN_BITS",
    "SDMA0_CLK_CTRL", "SDMA0_POWER_CNTL", "SDMA0_STATUS_REG", "SDMA0_STATUS1_REG",
    "SDMA0_TILING_CONFIG", "SDMA0_SEM_WAIT_FAIL_TIMER_CNTL", "SDMA0_PHASE0_QUANTUM", "SDMA0_GFX_*",
    "SDMA0_RLC0_IB_CNTL", "SDMA0_RLC1_IB_CNTL", "SDMA0_RLC0_RB_CNTL", "SDMA0_RLC1_RB_CNTL",
    "SDMA0_FREEZE", "SDMA0_VERSION",
    "SDMA1_UCODE_ADDR", "SDMA1_F32_CNTL", "SDMA1_STATUS_REG",
    # GFX: status now; RLC/MEC/HQD for the compute queue
    "GRBM_STATUS", "GRBM_STATUS2", "GRBM_SOFT_RESET", "GRBM_GFX_INDEX", "GB_ADDR_CONFIG",
    "RLC_CNTL", "RLC_STAT", "RLC_GPM_STAT", "RLC_GPM_UCODE_ADDR", "RLC_GPM_UCODE_DATA",
    "RLC_PG_CNTL", "RLC_CGCG_CGLS_CTRL", "RLC_LB_CNTL", "RLC_SRM_CNTL", "RLC_CGTT_MGCG_OVERRIDE",
    "RLC_SAFE_MODE", "RLC_CP_SCHEDULERS", "RLC_SERDES_CU_MASTER_BUSY", "RLC_SERDES_NONCU_MASTER_BUSY",
    "CP_ME_CNTL", "CP_MEC_CNTL", "CP_MEC_ME1_UCODE_*", "CP_MEC_ME2_UCODE_*", "CP_STAT",
    "CP_CPC_STATUS", "CP_CPC_BUSY_STAT", "CP_CPC_STALLED_STAT1", "CP_CPF_STATUS", "CP_HQD_*",
    "CP_MQD_*", "CP_PQ_WPTR_POLL_CNTL", "CP_PQ_WPTR_POLL_CNTL1", "CP_PQ_STATUS", "CP_INT_CNTL_RING0",
    "CP_ME1_PIPE0_INT_CNTL", "CPC_INT_CNTL", "CP_MEC_DOORBELL_RANGE_*", "CP_INT_STATUS",
    "SH_MEM_CONFIG", "SH_MEM_BASES", "SH_MEM_APE1_BASE", "SH_MEM_APE1_LIMIT", "SCRATCH_REG0",
    "COMPUTE_*",
    # SMU: indirect window, SMC state and the mailbox
    "SMC_IND_INDEX_11", "SMC_IND_DATA_11", "SMC_IND_ACCESS_CNTL", "SMC_MESSAGE_0", "SMC_RESP_0",
    "SMC_MSG_ARG_0", "SMC_SYSCON_CLOCK_CNTL_0", "SMC_SYSCON_RESET_CNTL", "SMC_PC_C",
    "FIRMWARE_FLAGS", "SMU_FIRMWARE", "SMU_STATUS", "SMU_INPUT_DATA", "ROM_INDEX", "ROM_DATA",
    "CG_SPLL_FUNC_CNTL*", "CG_CLKPIN_CNTL", "CG_CLKPIN_CNTL_2", "SMU_SMC_IND_INDEX", "SMU_SMC_IND_DATA",
    "RCU_UC_EVENTS", "SMC_MSG_ARG_1", "SMC_RESP_1", "SMC_MESSAGE_1",
    # thermal: smu7_thermal_get_temperature()
    "CG_MULT_THERMAL_STATUS",
    # GFX v8 bring-up for compute: gfx_v8_0 golden settings (polaris10), constants, RLC, MEC queues
    "ATC_MISC_CG", "CB_HW_CONTROL", "CB_HW_CONTROL_2", "CB_HW_CONTROL_3", "DB_DEBUG2", "PA_SC_ENHANCE",
    "PA_SC_LINE_STIPPLE_STATE", "PA_SC_RASTER_CONFIG", "PA_SC_RASTER_CONFIG_1", "RLC_CGCG_CGLS_CTRL_3D",
    "SQ_CONFIG", "TA_CNTL_AUX", "TCC_CTRL", "TCP_ADDR_CONFIG", "TCP_CHAN_STEER_HI", "VGT_RESET_DEBUG",
    "SPI_RESOURCE_RESERVE_CU_0", "SPI_RESOURCE_RESERVE_CU_1", "SPI_RESOURCE_RESERVE_EN_CU_0",
    "SPI_RESOURCE_RESERVE_EN_CU_1", "CG_ACLK_CNTL", "GRBM_CNTL", "SH_STATIC_MEM_CONFIG",
]

DEFINE = re.compile(r"^#define\s+(\S+)\s+(0x[0-9a-fA-F]+|\d+)[uUlL]*\s*$")


def find(root, base):
    for cand in (os.path.join(root, base), os.path.join(root, os.path.basename(base)),
                 os.path.join(root, base.replace("/", "_")),
                 os.path.join(root, "include_asic_reg_" + base.replace("/", "_"))):
        if os.path.exists(cand):
            return cand
    sys.exit(f"missing {base} under {root}")


def parse(path):
    out = {}
    with open(path) as f:
        for line in f:
            m = DEFINE.match(line.strip())
            if m:
                out.setdefault(m.group(1), int(m.group(2), 0))
    return out


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    root, sha = sys.argv[1], sys.argv[2]
    offsets, fields = {}, {}  # name -> (value, header)
    for h in HEADERS:
        for name, v in parse(find(root, h + "_d.h")).items():
            if name.startswith(("mm", "ix")):
                if name in offsets and offsets[name][0] != v:
                    sys.exit(f"{name}: {offsets[name][1]} and {h} disagree")
                offsets.setdefault(name, (v, h))
        for name, v in parse(find(root, h + "_sh_mask.h")).items():
            fields.setdefault(name, (v, h))

    regs, missing = {}, []  # bare register name -> list of (prefixed name, value, header)
    for pat in REGS:
        hits = [n for n in offsets if fnmatch.fnmatchcase(n[2:], pat)]
        missing += [] if hits else [pat]
        for n in hits:
            entry = (n, *offsets[n])
            if entry not in regs.setdefault(n[2:], []):  # a register matched by two patterns
                regs[n[2:]].append(entry)
    if missing:
        sys.exit("no register matches: " + " ".join(missing))

    print("/* Generated by tools/gen_vi_regs.py from Linux amdgpu (MIT) at")
    print(f" * {sha}, drivers/gpu/drm/amd/include/asic_reg/{{{','.join(HEADERS)}}}_{{d,sh_mask}}.h.")
    print(" * Do not edit; add names to REGS in the script and regenerate.")
    print(" * mmFOO = dword index into the register BAR (byte offset = index * 4); ixFOO = SMC address. */")
    print("#ifndef RAWGPU_AMD_VI_REGS_H\n#define RAWGPU_AMD_VI_REGS_H")
    for bare in sorted(regs):
        print()
        for n, v, h in sorted(regs[bare]):
            print(f"#define {n:<52} 0x{v:x}  /* {h} */")
        pre = bare + "__"
        for f in sorted(k for k in fields if k.startswith(pre)):
            print(f"#define {f:<52} 0x{fields[f][0]:x}")
    print("\n#endif")
    if len(sys.argv) == 4:
        with open(sys.argv[3], "w") as f:
            f.write(f"/* Generated by tools/gen_vi_regs.py from Linux amdgpu (MIT) at {sha}. Do not edit. */\n")
            for bare in sorted(regs):
                for n, v, h in sorted(regs[bare]):
                    if n.startswith("mm"):
                        f.write(f'{{"{bare}", 0x{v:x}}},\n')


if __name__ == "__main__":
    main()
