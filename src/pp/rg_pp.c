/* Glue between rawgpu and amdgpu's powerplay code (src/pp/upstream).
 *
 * Compiled against the kernel shims (src/pp/shim), like the vendored files. It
 * stands in for what amdgpu's IP-block layer provides around powerplay:
 * amd_powerplay.c's create/early/sw/hw/late init, the CGS register device
 * (amdgpu_cgs.c), and the few amdgpu_* helpers the code calls. Hardware access
 * goes through the rg_pp_host callbacks (rg_pp.h).
 */
#include "rg_pp.h"

#include "amdgpu.h"
#include "atom.h"
#include "cgs_common.h"
#include "hwmgr.h"
#include "smu7_hwmgr.h"
#include "smumgr.h"
#include "amd_pcie.h"

#include <time.h>

struct rg_cgs {
  struct cgs_device base; /* must stay first, as in amdgpu_cgs.c */
  struct rg_pp *p;
};

struct rg_pp {
  rg_pp_host h;
  struct amdgpu_device adev;
  struct pci_dev pdev;
  struct device dev;
  struct rg_cgs cgs;
  struct amd_pp_display_configuration display; /* no displays: rawgpu drives none */
  struct pp_hwmgr *hwmgr;
  bool tables_read, hw_init;
};

static struct rg_pp *g_pp; /* for the kernel-style free functions (logging, delays) */
int amdgpu_dpm = 1;
const char *const amdgpu_pp_profile_name[] = { /* amdgpu_pm.c */
  "BOOTUP_DEFAULT", "3D_FULL_SCREEN", "POWER_SAVING", "VIDEO", "VR", "COMPUTE", "CUSTOM", "WINDOW_3D", "CAPPED",
  "UNCAPPED",
};

/* ---- kernel services ---- */

void rg_pp_log(const char *level, const char *fmt, ...) {
  int v = g_pp ? g_pp->h.verbosity : 0;
  int rank = !strcmp(level, "debug") ? 2 : !strcmp(level, "info") ? 1 : 0;
  if (rank > v) return;
  fprintf(stderr, "powerplay %s: ", level);
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

void rg_pp_udelay(unsigned long us) {
  if (g_pp) g_pp->h.delay_us(g_pp->h.ctx, us);
}

unsigned long rg_pp_jiffies(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (unsigned long)t.tv_sec * 1000ul + (unsigned long)(t.tv_nsec / 1000000);
}

void rg_pp_poweroff(const char *why) {
  /* smu_helper.c calls orderly_poweroff() on a critical temperature. rawgpu
   * cannot power anything off; stopping here leaves the SMC's own thermal
   * protection in charge and makes the event impossible to miss. */
  fprintf(stderr, "powerplay EMERGENCY: %s requested by the thermal code; rawgpu is exiting\n", why);
  exit(3);
}

uint32_t rg_pp_rreg(struct amdgpu_device *adev, uint32_t reg) {
  struct rg_pp *p = adev->rg;
  return p->h.rreg(p->h.ctx, reg);
}

void rg_pp_wreg(struct amdgpu_device *adev, uint32_t reg, uint32_t v) {
  struct rg_pp *p = adev->rg;
  p->h.wreg(p->h.ctx, reg, v);
}

/* ---- CGS (amdgpu_cgs.c) ---- */

static struct rg_pp *cgs_pp(struct cgs_device *d) { return ((struct rg_cgs *)d)->p; }

static uint32_t cgs_rreg(struct cgs_device *d, unsigned offset) {
  struct rg_pp *p = cgs_pp(d);
  return p->h.rreg(p->h.ctx, offset);
}
static void cgs_wreg(struct cgs_device *d, unsigned offset, uint32_t v) {
  struct rg_pp *p = cgs_pp(d);
  p->h.wreg(p->h.ctx, offset, v);
}
static uint32_t cgs_rind(struct cgs_device *d, enum cgs_ind_reg space, unsigned index) {
  struct rg_pp *p = cgs_pp(d);
  return p->h.rind(p->h.ctx, (int)space, index);
}
static void cgs_wind(struct cgs_device *d, enum cgs_ind_reg space, unsigned index, uint32_t v) {
  struct rg_pp *p = cgs_pp(d);
  p->h.wind(p->h.ctx, (int)space, index, v);
}
/* Only the SMU-load path asks for firmware (smu7_request_smu_load_fw()), and
 * rawgpu loads the SMC and the engine images itself (`rawgpu smc-load`). */
static int cgs_fw_info(struct cgs_device *d, enum cgs_ucode_id type, struct cgs_firmware_info *info) {
  rg_pp_log("error", "cgs_get_firmware_info(%d): firmware loading belongs to `rawgpu smc-load`\n", (int)type);
  return -EINVAL;
}
static const struct cgs_ops rg_cgs_ops = {
  .read_register = cgs_rreg,
  .write_register = cgs_wreg,
  .read_ind_register = cgs_rind,
  .write_ind_register = cgs_wind,
  .get_firmware_info = cgs_fw_info,
};

/* ---- amdgpu helpers the powerplay code calls ---- */

uint32_t amdgpu_asic_get_xclk(struct amdgpu_device *adev) { return ((struct rg_pp *)adev->rg)->h.xclk; }

/* smu7_init() asks for two VRAM buffers: the TOC header and the SMU's work
 * buffer. `rawgpu smc-load` already placed both and told the SMU where they
 * are, so hand back those addresses. The CPU pointer is a shadow: only
 * smu7_request_smu_load_fw() writes through it, and rawgpu never calls that. */
int amdgpu_bo_create_kernel(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
                            struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr) {
  struct rg_pp *p = adev->rg;
  uint64_t mc = size <= 4096 ? p->h.smu_toc_mc : size <= (1ul << 20) ? p->h.smu_buf_mc : 0;
  if (!mc || domain != AMDGPU_GEM_DOMAIN_VRAM) {
    rg_pp_log("error", "unexpected buffer request: %lu bytes in domain 0x%x\n", size, domain);
    return -ENOMEM;
  }
  void *shadow = calloc(1, size);
  if (!shadow) return -ENOMEM;
  *bo_ptr = (struct amdgpu_bo *)shadow;
  if (gpu_addr) *gpu_addr = mc;
  if (cpu_addr) *cpu_addr = shadow;
  return 0;
}

void amdgpu_bo_free_kernel(struct amdgpu_bo **bo, u64 *gpu_addr, void **cpu_addr) {
  if (bo && *bo) { free(*bo); *bo = NULL; }
  if (gpu_addr) *gpu_addr = 0;
  if (cpu_addr) *cpu_addr = NULL;
}

/* No interrupts reach rawgpu (TinyGPU has no MSI path): thermal and CTF IRQ
 * sources are registered with nobody. The SMC's own protection still acts. */
int amdgpu_irq_add_id(struct amdgpu_device *adev, unsigned client_id, unsigned src_id, struct amdgpu_irq_src *source) {
  return 0;
}

/* amdgpu_gfx_rlc_enter_safe_mode() does nothing unless GFX clock gating is on
 * (adev->cg_flags), and rawgpu never enables it. */
void amdgpu_gfx_rlc_enter_safe_mode(struct amdgpu_device *adev, int xcc_id) {}
void amdgpu_gfx_rlc_exit_safe_mode(struct amdgpu_device *adev, int xcc_id) {}

/* UVD/VCE gating: rawgpu brings up neither block. */
int amdgpu_device_ip_set_powergating_state(void *dev, enum amd_ip_block_type t, enum amd_powergating_state s) {
  return 0;
}
int amdgpu_device_ip_set_clockgating_state(void *dev, enum amd_ip_block_type t, enum amd_clockgating_state s) {
  return 0;
}
struct amdgpu_ip_block *amdgpu_device_ip_get_ip_block(struct amdgpu_device *adev, enum amd_ip_block_type type) {
  return NULL;
}
int amdgpu_ucode_init_bo(struct amdgpu_device *adev) {
  rg_pp_log("error", "amdgpu_ucode_init_bo(): firmware loading belongs to `rawgpu smc-load`\n");
  return -EINVAL;
}
/* No ACPI on a Mac: the PCIe performance request (ATCS) is not available. */
int amdgpu_acpi_pcie_performance_request(struct amdgpu_device *adev, u8 perf_req, bool advertise) { return -EINVAL; }
bool amdgpu_acpi_is_pcie_performance_request_supported(struct amdgpu_device *adev) { return false; }

/* ---- the driver flow ---- */

int rg_pp_open(rg_pp **out, const rg_pp_host *h) {
  struct rg_pp *p = calloc(1, sizeof *p);
  if (!p) return -ENOMEM;
  p->h = *h;
  g_pp = p;
  struct amdgpu_device *adev = &p->adev;
  p->pdev = (struct pci_dev){.vendor = 0x1002, .device = h->device, .subsystem_vendor = h->subsys_vendor,
                             .subsystem_device = h->subsys_device, .revision = h->revision};
  adev->dev = &p->dev;
  adev->pdev = &p->pdev;
  adev->asic_type = CHIP_POLARIS10;
  adev->bios = (uint8_t *)h->bios;
  adev->bios_size = h->bios_size;
  adev->mode_info.atom_context = h->atom_context;
  adev->pm.pp_feature = h->feature_mask;
  /* amdgpu_device_get_pcie_info(): ASIC speeds | platform speeds, platform widths */
  static const uint32_t asic_gen[] = {CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN1, CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN2,
                                      CAIL_ASIC_PCIE_LINK_SPEED_SUPPORT_GEN3};
  static const uint32_t plat_gen[] = {CAIL_PCIE_LINK_SPEED_SUPPORT_GEN1, CAIL_PCIE_LINK_SPEED_SUPPORT_GEN2,
                                      CAIL_PCIE_LINK_SPEED_SUPPORT_GEN3};
  for (int g = 0; g < 3; g++) {
    if (g < h->asic_max_gen) adev->pm.pcie_gen_mask |= asic_gen[g];
    if (g < h->link_gen) adev->pm.pcie_gen_mask |= plat_gen[g];
  }
  static const struct { int w; uint32_t m; } widths[] = {
    {1, CAIL_PCIE_LINK_WIDTH_SUPPORT_X1}, {2, CAIL_PCIE_LINK_WIDTH_SUPPORT_X2}, {4, CAIL_PCIE_LINK_WIDTH_SUPPORT_X4},
    {8, CAIL_PCIE_LINK_WIDTH_SUPPORT_X8}, {12, CAIL_PCIE_LINK_WIDTH_SUPPORT_X12}, {16, CAIL_PCIE_LINK_WIDTH_SUPPORT_X16},
  };
  for (size_t i = 0; i < ARRAY_SIZE(widths); i++)
    if (widths[i].w <= h->link_width) adev->pm.pcie_mlw_mask |= widths[i].m;
  adev->pm.ac_power = true;
  adev->gmc.vram_width = h->vram_width;
  adev->gmc.vram_type = AMDGPU_VRAM_TYPE_GDDR5;
  adev->gfx.cu_info.number = h->cu_count;
  adev->gfx.config.max_shader_engines = h->shader_engines;
  adev->rg = p;
  p->cgs = (struct rg_cgs){.base = {.ops = &rg_cgs_ops}, .p = p};

  /* amd_powerplay_create() */
  struct pp_hwmgr *hwmgr = calloc(1, sizeof *hwmgr);
  if (!hwmgr) { free(p); g_pp = NULL; return -ENOMEM; }
  hwmgr->adev = adev;
  hwmgr->not_vf = true;
  hwmgr->device = &p->cgs.base;
  hwmgr->chip_family = AMDGPU_FAMILY_VI;
  hwmgr->chip_id = adev->asic_type;
  hwmgr->feature_mask = adev->pm.pp_feature;
  hwmgr->display_config = &p->display;
  adev->powerplay.pp_handle = hwmgr;
  p->hwmgr = hwmgr;

  int r = hwmgr_early_init(hwmgr); /* pp_early_init() */
  if (!r) r = hwmgr_sw_init(hwmgr);  /* pp_sw_init(): polaris10_smu_init() */
  /* What smu7_upload_smu_firmware_image() records while it uploads the SMC
   * firmware; `rawgpu smc-load` did the upload. */
  hwmgr->is_kicker = h->is_kicker;
  hwmgr->smu_version = h->smc_version;
  if (r) { rg_pp_log("error", "powerplay init failed (%d)\n", r); rg_pp_close(p); return r; }
  *out = p;
  return 0;
}

void rg_pp_close(rg_pp *p) {
  if (!p) return;
  struct pp_hwmgr *hwmgr = p->hwmgr;
  if (hwmgr) {
    if (p->tables_read) {
      if (hwmgr->hwmgr_func && hwmgr->hwmgr_func->backend_fini) hwmgr->hwmgr_func->backend_fini(hwmgr);
      if (hwmgr->pptable_func && hwmgr->pptable_func->pptable_fini) hwmgr->pptable_func->pptable_fini(hwmgr);
    }
    if (hwmgr->smumgr_funcs && hwmgr->smumgr_funcs->smu_fini) hwmgr->smumgr_funcs->smu_fini(hwmgr);
    free(hwmgr);
  }
  if (g_pp == p) g_pp = NULL;
  free(p);
}

int rg_pp_read_tables(rg_pp *p) {
  struct pp_hwmgr *hwmgr = p->hwmgr;
  int r = hwmgr->pptable_func->pptable_init(hwmgr);
  if (r) { rg_pp_log("error", "PowerPlay table parse failed (%d)\n", r); return r; }
  r = hwmgr->hwmgr_func->backend_init(hwmgr);
  if (r) {
    rg_pp_log("error", "smu7 backend init failed (%d)\n", r);
    hwmgr->pptable_func->pptable_fini(hwmgr);
    return r;
  }
  p->tables_read = true;
  return p->h.err(p->h.ctx);
}

static double mv(uint16_t x) { return x / 1000.0; }

void rg_pp_print_tables(rg_pp *p, FILE *f) {
  struct pp_hwmgr *hwmgr = p->hwmgr;
  struct phm_ppt_v1_information *t = hwmgr->pptable;
  struct smu7_hwmgr *data = hwmgr->backend;
  fprintf(f, "PowerPlay table v%d, SMC firmware 0x%08x, feature mask 0x%08x%s\n", hwmgr->pp_table_version + 0,
          hwmgr->smu_version, hwmgr->feature_mask, hwmgr->is_kicker ? ", kicker part" : "");
  if (!t) return;
  const struct phm_ppt_v1_voltage_lookup_table *lk = t->vddc_lookup_table;
  if (t->vdd_dep_on_sclk) {
    fprintf(f, "engine clock levels (vdd_dep_on_sclk):\n");
    for (uint32_t i = 0; i < t->vdd_dep_on_sclk->count; i++) {
      const struct phm_ppt_v1_clock_voltage_dependency_record *e = &t->vdd_dep_on_sclk->entries[i];
      uint16_t v = lk && e->vddInd < lk->count ? lk->entries[e->vddInd].us_vdd : 0;
      fprintf(f, "  [%u] %4u MHz  VDDC %.3f V (lookup %u)  CKS %u/%u\n", i, e->clk / 100, mv(v), e->vddInd,
              e->cks_enable, e->cks_voffset);
    }
  }
  if (t->vdd_dep_on_mclk) {
    fprintf(f, "memory clock levels (vdd_dep_on_mclk):\n");
    for (uint32_t i = 0; i < t->vdd_dep_on_mclk->count; i++) {
      const struct phm_ppt_v1_clock_voltage_dependency_record *e = &t->vdd_dep_on_mclk->entries[i];
      uint16_t v = lk && e->vddInd < lk->count ? lk->entries[e->vddInd].us_vdd : 0;
      fprintf(f, "  [%u] %4u MHz  VDDC %.3f V  VDDCI %.3f V  MVDD %.3f V\n", i, e->clk / 100, mv(v), mv(e->vddci),
              mv(e->mvdd));
    }
  }
  if (lk) {
    fprintf(f, "VDDC lookup table:");
    for (uint32_t i = 0; i < lk->count; i++) fprintf(f, " %.3f", mv(lk->entries[i].us_vdd));
    fprintf(f, " V\n");
  }
  if (data && data->vddc_leakage.count) {
    fprintf(f, "EVV leakage voltages (VBIOS GetVoltageInfo):");
    for (uint16_t i = 0; i < data->vddc_leakage.count; i++)
      fprintf(f, " 0x%04x=%.3f V", data->vddc_leakage.leakage_id[i], mv(data->vddc_leakage.actual_voltage[i]));
    fprintf(f, "\n");
  }
  fprintf(f, "limits on AC: engine %u MHz, memory %u MHz, VDDC %.3f V, VDDCI %.3f V\n",
          t->max_clock_voltage_on_ac.sclk / 100, t->max_clock_voltage_on_ac.mclk / 100,
          mv(t->max_clock_voltage_on_ac.vddc), mv(t->max_clock_voltage_on_ac.vddci));
  if (t->cac_dtp_table)
    fprintf(f, "power: TDP %u W, TDC %u A, max power delivery %u W\n", t->cac_dtp_table->usTDP,
            t->cac_dtp_table->usTDC, t->cac_dtp_table->usMaximumPowerDeliveryLimit);
  const struct pp_advance_fan_control_parameters *fan = &hwmgr->thermal_controller.advanceFanControlParameters;
  fprintf(f, "fan: %s; curve %.0f C -> %.0f%%, %.0f C -> %.0f%%, %.0f C -> %.0f%%; max %.0f C\n",
          hwmgr->thermal_controller.fanInfo.bNoFan ? "none" : "present", fan->usTMin / 100.0, fan->usPWMMin / 100.0,
          fan->usTMed / 100.0, fan->usPWMMed / 100.0, fan->usTHigh / 100.0, fan->usPWMHigh / 100.0,
          fan->usTMax / 100.0);
}

int rg_pp_hw_init(rg_pp *p) {
  if (p->tables_read) return -EBUSY; /* hwmgr_hw_init() parses the tables itself: use a fresh rg_pp */
  struct pp_hwmgr *hwmgr = p->hwmgr;
  int r = hwmgr_hw_init(hwmgr);     /* pp_hw_init() */
  if (r) { rg_pp_log("error", "hwmgr_hw_init failed (%d)\n", r); return r; }
  p->tables_read = false; /* hwmgr_hw_fini() would own the teardown; rawgpu leaves DPM running */
  p->hw_init = true;
  if (hwmgr->pm_en) hwmgr_handle_task(hwmgr, AMD_PP_TASK_COMPLETE_INIT, NULL); /* pp_late_init() */
  return p->h.err(p->h.ctx);
}

int rg_pp_force_level(rg_pp *p, int level) {
  static const enum amd_dpm_forced_level map[] = {AMD_DPM_FORCED_LEVEL_AUTO, AMD_DPM_FORCED_LEVEL_LOW,
                                                  AMD_DPM_FORCED_LEVEL_HIGH};
  if (level < 0 || level > 2 || !p->hw_init) return -EINVAL;
  struct pp_hwmgr *hwmgr = p->hwmgr;
  /* pp_dpm_force_performance_level() for the non-profiling levels */
  if (hwmgr->dpm_level == map[level]) return 0;
  hwmgr->request_dpm_level = map[level];
  int r = hwmgr_handle_task(hwmgr, AMD_PP_TASK_READJUST_POWER_STATE, NULL);
  return r ? r : p->h.err(p->h.ctx);
}

static int sensor(struct pp_hwmgr *hwmgr, int idx, uint32_t *v) {
  int size = 4;
  if (!hwmgr->hwmgr_func->read_sensor) return -EINVAL;
  return hwmgr->hwmgr_func->read_sensor(hwmgr, idx, v, &size);
}

void rg_pp_read_sensors(rg_pp *p, rg_pp_sensors *s) {
  struct pp_hwmgr *hwmgr = p->hwmgr;
  uint32_t v;
  *s = (rg_pp_sensors){-1, -1, -1, -1, -1.0};
  if (!sensor(hwmgr, AMDGPU_PP_SENSOR_GFX_SCLK, &v)) s->sclk_mhz = (int)(v / 100);
  if (!sensor(hwmgr, AMDGPU_PP_SENSOR_GFX_MCLK, &v)) s->mclk_mhz = (int)(v / 100);
  if (!sensor(hwmgr, AMDGPU_PP_SENSOR_GPU_LOAD, &v)) s->gpu_load = (int)v;
  if (!sensor(hwmgr, AMDGPU_PP_SENSOR_GPU_TEMP, &v)) s->temp_c = (int)v / 1000;
  if (!sensor(hwmgr, AMDGPU_PP_SENSOR_GPU_INPUT_POWER, &v)) s->power_w = v / 256.0; /* 8.8 fixed point */
}
