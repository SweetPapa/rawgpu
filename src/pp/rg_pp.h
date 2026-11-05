/* rawgpu's interface to amdgpu's powerplay code (src/pp/upstream), for Polaris.
 *
 * This header uses only plain C types, so both sides can include it: the glue
 * in rg_pp.c is compiled against the kernel shims, and the backend
 * (amd_vi_dpm.c) against rawgpu's own headers. The two never share a
 * translation unit, so their register and macro definitions cannot clash.
 */
#ifndef RAWGPU_RG_PP_H
#define RAWGPU_RG_PP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Indirect register spaces, numbered as cgs_common.h's enum cgs_ind_reg */
enum { RG_PP_IND_PCIE = 0, RG_PP_IND_SMC = 1, RG_PP_IND_UVD_CTX = 2, RG_PP_IND_DIDT = 3, RG_PP_IND_GC_CAC = 4,
       RG_PP_IND_SE_CAC = 5, RG_PP_IND_AUDIO = 6 };

/* An engine image as amdgpu_ucode_init_bo() leaves it: in VRAM, with the
 * header fields amdgpu_cgs_get_firmware_info() reports. */
enum { RG_PP_UC_SDMA0, RG_PP_UC_SDMA1, RG_PP_UC_CE, RG_PP_UC_PFP, RG_PP_UC_ME, RG_PP_UC_MEC, RG_PP_UC_RLC, RG_PP_UC_COUNT };
typedef struct {
  uint64_t mc;               /* MC address of the image; the MEC jump table follows at mc + ALIGN(ucode bytes, 4 KB) */
  uint32_t ucode_bytes;      /* common header ucode_size_bytes */
  uint32_t ucode_version, feature_version;
  uint32_t jt_offset, jt_size; /* gfx_firmware_header_v1_0, dwords (MEC) */
  const void *image;
} rg_pp_ucode;

typedef struct {
  void *ctx;
  uint32_t (*rreg)(void *ctx, uint32_t idx); /* dword index, as amdgpu's RREG32 */
  void (*wreg)(void *ctx, uint32_t idx, uint32_t v);
  uint32_t (*rind)(void *ctx, int space, uint32_t addr);
  void (*wind)(void *ctx, int space, uint32_t addr, uint32_t v);
  void (*delay_us)(void *ctx, unsigned long us); /* drains queued register writes first */
  int (*err)(void *ctx);                       /* sticky transport error, 0 if none */
  /* VRAM write by MC address, then an HDP flush (memcpy_toio() into an SMU buffer) */
  int (*vram_write)(void *ctx, uint64_t mc, const void *src, size_t len);
  /* amdgpu_ucode_init_bo(): put the engine images in VRAM and describe them */
  int (*ucode_init_bo)(void *ctx, rg_pp_ucode out[RG_PP_UC_COUNT]);
  /* gfx_v8_0_set_safe_mode() / unset_safe_mode(): the RLC holds GFX clocks
   * and power on while powerplay programs GFX registers (DIDT, CAC) */
  void (*rlc_safe_mode)(void *ctx, bool enter);
  bool no_didt;                  /* leave the SQ/DB/TD/TCP di/dt limiters off (rawgpu option) */
  /* the SMC firmware, for smu7_upload_smu_firmware_image() */
  const void *smc_image;
  uint32_t smc_bytes, smc_start_addr;

  void *atom_context;            /* struct atom_context * (rg_atom_context()) */
  const uint8_t *bios;
  uint32_t bios_size;
  uint16_t device, subsys_vendor, subsys_device;
  uint8_t revision;
  bool is_kicker;                /* amdgpu learns this while uploading SMC firmware (smu7_upload_smu_firmware_image) */
  uint32_t smc_version;          /* the SMC firmware's ucode_version (hwmgr->smu_version) */
  uint32_t xclk;                 /* amdgpu_asic_get_xclk(): reference clock, 10 kHz units */
  uint64_t smu_toc_mc, smu_buf_mc; /* the VRAM `rawgpu smc-load` gave the SMU (header_buffer, smu_buffer) */
  uint32_t feature_mask;         /* amdgpu's ppfeaturemask */
  int link_gen, link_width;      /* the PCIe link as trained (the platform's limit through the enclosure) */
  int asic_max_gen;              /* what the GPU itself supports */
  uint32_t vram_width;           /* bits */
  unsigned cu_count, shader_engines;
  int verbosity;                 /* 0 = errors and warnings, 1 = + info, 2 = + debug */
} rg_pp_host;

typedef struct rg_pp rg_pp;

/* amd_powerplay_create() + hwmgr_early_init() + hwmgr_sw_init(). Touches no hardware. */
int rg_pp_open(rg_pp **out, const rg_pp_host *h);
void rg_pp_close(rg_pp *p);

/* polaris10_start_smu(), as amdgpu runs it from pp_dpm_load_fw(): start the
 * SMC firmware in protected mode, stage AVFS (polaris10_avfs_event_mgr()), and
 * have the SMU load the engine microcode (smu7_request_smu_load_fw()). For a
 * card whose SMC is still on its boot ROM. */
int rg_pp_start_smu(rg_pp *p);

/* The first half of hwmgr_hw_init(): parse the PowerPlay table and run
 * smu7_hwmgr_backend_init(), which resolves EVV voltages through VBIOS
 * GetVoltageInfo. No SMC or clock writes. For `rawgpu dpm-info`. */
int rg_pp_read_tables(rg_pp *p);
void rg_pp_print_tables(rg_pp *p, FILE *f);

/* All of hwmgr_hw_init() (after a fresh rg_pp_open), then pp_late_init()'s
 * AMD_PP_TASK_COMPLETE_INIT: DPM tables into the SMC, DPM, thermal and fan
 * control on. The SMC must already run its firmware (`rawgpu smc-load`). */
int rg_pp_hw_init(rg_pp *p);

/* force_dpm_level: 0 auto, 1 low, 2 high (enum amd_dpm_forced_level values are mapped inside) */
int rg_pp_force_level(rg_pp *p, int level);
/* read_sensor: sclk/mclk in MHz, GPU load in %, temperature in C, power in W. -1 if unavailable. */
typedef struct { int sclk_mhz, mclk_mhz, gpu_load, temp_c; double power_w; } rg_pp_sensors;
void rg_pp_read_sensors(rg_pp *p, rg_pp_sensors *s);

#endif
