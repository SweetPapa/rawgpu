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

typedef struct {
  void *ctx;
  uint32_t (*rreg)(void *ctx, uint32_t idx); /* dword index, as amdgpu's RREG32 */
  void (*wreg)(void *ctx, uint32_t idx, uint32_t v);
  uint32_t (*rind)(void *ctx, int space, uint32_t addr);
  void (*wind)(void *ctx, int space, uint32_t addr, uint32_t v);
  void (*delay_us)(void *ctx, unsigned long us); /* drains queued register writes first */
  int (*err)(void *ctx);                       /* sticky transport error, 0 if none */

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
