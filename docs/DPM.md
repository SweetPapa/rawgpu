# Power management (SMU7 DPM) bring-up plan

Goal: run the RX 590 at its rated clocks under SMC control, the way Linux
amdgpu does. The SMC firmware then picks engine/memory clocks and voltages from
the card's own VBIOS PowerPlay table, and runs fan and thermal control.

The user chose this route on 2026-11-04 over setting clocks with VBIOS command
tables, because Polaris derives voltages per chip (EVV/leakage IDs) and a wrong
voltage means hangs, which are hard to recover from on this Mac.

## Method: amdgpu's powerplay code, unmodified, against shims

The same method as stage 2's ATOM interpreter (`src/atom/UPSTREAM.md`):

- `src/pp/upstream/` holds amdgpu's SMU7/Polaris powerplay files byte for byte
  (Linux `05441d697f7f`): `hwmgr/{hwmgr,hardwaremanager,pp_psm,smu7_*,
  process_pptables_v1_0,ppatomctrl,pppcielanes,smu_helper,common_baco,
  polaris_baco}.c`, `smumgr/{smumgr,smu7_smumgr,polaris10_smumgr}.c`, their
  headers, `pm/powerplay/inc/`, and the `amd/include` headers they need.
- `src/pp/asic_reg/` is generated (tools/gen_pp_regs.py): the upstream register
  headers at their include paths, trimmed to the registers the vendored code
  names. The full headers are several MB.
- `src/pp/shim/` supplies the kernel API and a minimal `amdgpu.h`. The real
  `struct amdgpu_device` is replaced by one with only the fields the vendored
  code reads (PCI IDs, asic_type, the ATOM context, pm/gmc/gfx bits).
- `src/pp/rg_pp.c` is rawgpu's glue: the CGS register callbacks (to vi_dev),
  the ATOM context (rg_atom, already used by `post`), VRAM for
  `amdgpu_bo_create_kernel()`, stubs for other chips' function tables, and
  the driver flow.

## Driver flow (what amdgpu does, in order)

1. `hwmgr_early_init()`: function tables for Polaris10, pptable v1 parser, caps.
2. `hwmgr_sw_init()`: `polaris10_smu_init()` (buffers), `smu7` backend data.
3. SMC start and microcode load: already done by `rawgpu smc-load`; not repeated.
4. `hwmgr_hw_init()`: pptable init, `smu7_hwmgr_backend_init()` (EVV voltages
   via ATOM GetVoltageInfo), power-state table, `phm_setup_asic()`,
   `phm_enable_dynamic_state_management()` (= `smu7_enable_dpm_tasks()`: DPM
   table into SMC RAM, enable SCLK/MCLK DPM, ULV, power containment...),
   `phm_start_thermal_controller()` (fan control), performance states.
5. `psm_adjust_power_state_dynamic()` to leave the boot state; optionally
   `force_dpm_level(HIGH)` for benchmarks.

## Milestones (each checked on hardware before the next)

1. **Compiles.** All vendored files build unmodified with no warnings that
   matter; stubs for unused chips link.
2. **`rawgpu dpm-info` (read-only).** Parse the PowerPlay table and print the
   SCLK/MCLK/VDDC levels, fan table and limits. Compare with the RX 590's
   published clocks (engine up to ~1545 MHz, memory 2000 MHz).
3. **EVV voltages.** `smu7_get_evv_voltages()` runs VBIOS GetVoltageInfo, which
   reads fuses. Print the resolved voltage per level; they must be plausible
   (about 0.8-1.2 V) before anything is written to the SMC.
4. **`rawgpu dpm-enable`.** The full `hwmgr_hw_init()`. Watch temperature, then
   `bench-compute` with the s_memtime clock probe. Thermal/fan first.
5. **Docs and limits.** Log measured clocks, power, temperature under load.

## Status

- 2026-11-04: plan written; upstream files fetched into the scratchpad;
  include graph and adev usage surveyed (small: see the plan above).
- Milestone 1 done: 17 upstream files build unmodified (`src/pp/upstream`),
  no warnings with the kernel's flags; trimmed register headers
  (`tools/gen_pp_regs.py`, 128 KB) give byte-identical objects to the full ones.
- Milestones 2-3 done: `rawgpu dpm-info` prints the PowerPlay table.
  - Engine levels: 300, 909, 1134, 1266, 1365, 1432, 1490, 1545 MHz at
    0.750-1.018 V. Memory: 400, 1000, 2000 MHz.
  - EVV leakage IDs resolve to 0.837-1.162 V.
  - Limits: TDP 185 W, max power delivery 175 W; fan curve 40 C/20%,
    65 C/40%, 85 C/60%.
- Feature mask: amdgpu's 0xfff7bfff without PCIe DPM (Thunderbolt link) and
  AVFS. AVFS needs the BTC that polaris10_start_smu() runs, and smc-load skips
  it.
- Milestone 4, first try: `dpm-enable` worked (FEATURE_STATUS 0xe003, idle
  285/400 MHz, 3.1 W). The card wedged on the next run, which was the first
  memory traffic under memory DPM.
  - Fix: `rawgpu mc-load` (the driver's MC firmware, which amdgpu loads before
    DPM) and no MC reprogramming under a live GART. See docs/LOG.md.
- Next, after a power cycle: `post`, `mc-load`, `smc-load`, then
  `dpm-enable --no-mclk` (engine DPM alone) and a benchmark, then memory DPM.
