/* Polaris power management through amdgpu's powerplay code (src/pp, docs/DPM.md):
 * `rawgpu dpm-info` (read-only) and `rawgpu dpm-enable`.
 *
 * This file is the host side of src/pp/rg_pp.h: it gives the vendored code
 * rawgpu's register access, the VBIOS interpreter, and the facts amdgpu would
 * have gathered about the card by the time powerplay starts.
 */
#include "amd_vi.h"
#include "atom/rg_atom.h"
#include "pp/rg_pp.h"

#include <errno.h>
#include <string.h>

/* amdgpu_drv.c amdgpu_pp_feature_mask, and amd_shared.h PP_FEATURE_MASK bits */
#define PP_FEATURE_MASK_DEFAULT 0xfff7bfffu
#define PP_MCLK_DPM_MASK 0x2u
#define PP_SCLK_DEEP_SLEEP_MASK 0x8u
#define PP_ULV_MASK 0x100u
#define PP_PCIE_DPM_MASK 0x4u
#define PP_AVFS_MASK 0x40000u

/* ---- host callbacks ---- */

static uint32_t h_rreg(void *c, uint32_t idx) { return vi_rreg(c, idx); }
static void h_wreg(void *c, uint32_t idx, uint32_t val) { vi_wreg(c, idx, val); }
static int h_err(void *c) { return ((vi_dev *)c)->err; }

/* vi.c: vi_smc_rreg(), vi_pcie_rreg(), vi_didt_rreg(), vi_gc_cac_rreg() */
static uint32_t h_rind(void *c, int space, uint32_t a) {
  vi_dev *v = c;
  switch (space) {
  case RG_PP_IND_SMC: return vi_rsmc(v, a);
  case RG_PP_IND_PCIE:
    vi_wreg(v, mmPCIE_INDEX, a);
    (void)vi_rreg(v, mmPCIE_INDEX);
    return vi_rreg(v, mmPCIE_DATA);
  case RG_PP_IND_DIDT: vi_wreg(v, mmDIDT_IND_INDEX, a); return vi_rreg(v, mmDIDT_IND_DATA);
  case RG_PP_IND_GC_CAC: vi_wreg(v, mmGC_CAC_IND_INDEX, a); return vi_rreg(v, mmGC_CAC_IND_DATA);
  }
  fprintf(stderr, "rawgpu: powerplay read from indirect space %d (0x%x) is not wired up\n", space, a);
  return 0;
}

static void h_wind(void *c, int space, uint32_t a, uint32_t val) {
  vi_dev *v = c;
  switch (space) {
  case RG_PP_IND_SMC: vi_wsmc(v, a, val); return;
  case RG_PP_IND_PCIE: /* vi_pcie_wreg() */
    vi_wreg(v, mmPCIE_INDEX, a);
    (void)vi_rreg(v, mmPCIE_INDEX);
    vi_wreg(v, mmPCIE_DATA, val);
    (void)vi_rreg(v, mmPCIE_DATA);
    return;
  case RG_PP_IND_DIDT: vi_wreg(v, mmDIDT_IND_INDEX, a); vi_wreg(v, mmDIDT_IND_DATA, val); return;
  case RG_PP_IND_GC_CAC: vi_wreg(v, mmGC_CAC_IND_INDEX, a); vi_wreg(v, mmGC_CAC_IND_DATA, val); return;
  }
  fprintf(stderr, "rawgpu: powerplay write to indirect space %d (0x%x) is not wired up\n", space, a);
}

/* Every delay first drains queued register writes (a read is a round trip), so
 * the hardware sees at least the requested gap. The round trip alone is about
 * 10 us, so short delays need no sleep on top. */
static void h_delay(void *c, unsigned long us) {
  vi_dev *v = c;
  (void)vi_rreg(v, mmBIOS_SCRATCH_0);
  if (us > 10) vi_udelay((unsigned)us);
}

/* ---- what amdgpu knows about the card before powerplay starts ---- */

/* vi_get_xclk() */
static uint32_t xclk(vi_dev *v, uint32_t ref) {
  if (REG_GET_FIELD(vi_rsmc(v, ixCG_CLKPIN_CNTL_2), CG_CLKPIN_CNTL_2, MUX_TCLK_TO_XCLK)) return 1000;
  if (REG_GET_FIELD(vi_rsmc(v, ixCG_CLKPIN_CNTL), CG_CLKPIN_CNTL, XTALIN_DIVIDE)) return ref / 4;
  return ref;
}

/* gmc_v8_0_mc_init(): channels x channel width */
static uint32_t vram_width(vi_dev *v) {
  static const int numchan[] = {1, 2, 4, 8, 3, 6, 10, 12, 16};
  int chansize = REG_GET_FIELD(vi_rreg(v, mmMC_ARB_RAMCFG), MC_ARB_RAMCFG, CHANSIZE) ? 64 : 32;
  uint32_t n = REG_GET_FIELD(vi_rreg(v, mmMC_SHARED_CHMAP), MC_SHARED_CHMAP, NOOFCHAN);
  return (uint32_t)(chansize * (n < 9 ? numchan[n] : 1));
}

typedef struct {
  vi_dev v;
  rg_atom *atom;
  rg_pp *pp;
} dpm_ctx;

static int dpm_open(dpm_ctx *c, rg_dev *d, const rg_ident *id, int verbosity, bool need_smc, uint32_t drop_features) {
  memset(c, 0, sizeof *c);
  if (!id->chip || strcmp(id->chip, "POLARIS10")) {
    fprintf(stderr, "rawgpu: power management is only wired up for POLARIS10 (this is %s)\n", id->chip ? id->chip : "?");
    return -ENOTSUP;
  }
  vi_dev *v = &c->v;
  int r = vi_open(v, d);
  if (r) return r;
  uint32_t memsize, status = 0;
  if (!vi_posted(v, &memsize)) {
    if (!v->err) fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post`, then `rawgpu smc-load`\n");
    return v->err ? v->err : -EIO;
  }
  if (need_smc && (vi_smu_ucode_status(v, &status) || !status)) {
    if (!v->err) fprintf(stderr, "rawgpu: the SMC is not running its firmware; run `rawgpu smc-load` first\n");
    return v->err ? v->err : -EIO;
  }
  if (need_smc) { /* amdgpu loads the MC firmware before power management starts (gmc_v8_0_hw_init()) */
    amd_fw mc;
    if ((r = amd_fw_load(&mc, NULL, vi_mc_fw_name(d)))) return r;
    uint32_t want = mc.ucode_version, have = vi_mc_ucode_version(v);
    amd_fw_free(&mc);
    if (have != want) {
      fprintf(stderr, "rawgpu: the MC sequencer runs firmware 0x%08x, not the driver's 0x%08x. Memory clock\n"
                      "        switching needs the driver's: run `rawgpu mc-load` after `post`, before `smc-load`.\n",
              have, want);
      return v->err ? v->err : -EIO;
    }
  }
  if (!(c->atom = vi_atom_open(d, NULL, false))) return -EIO;

  rg_pp_host h = {.ctx = v, .rreg = h_rreg, .wreg = h_wreg, .rind = h_rind, .wind = h_wind, .delay_us = h_delay,
                  .err = h_err, .verbosity = verbosity};
  size_t bios_len;
  h.atom_context = rg_atom_context(c->atom);
  h.bios = rg_atom_bios(c->atom, &bios_len);
  h.bios_size = (uint32_t)bios_len;
  h.device = d->device;
  h.revision = d->revision;
  h.subsys_vendor = d->subsys_vendor;
  h.subsys_device = d->subsys_device;
  h.is_kicker = vi_smc_is_kicker(d);
  bool hard = REG_GET_FIELD(vi_rsmc(v, ixSMU_FIRMWARE), SMU_FIRMWARE, SMU_SEL);
  amd_fw smc;
  if ((r = amd_fw_load(&smc, NULL, vi_smc_fw_name(d, hard)))) return r;
  h.smc_version = smc.ucode_version;
  amd_fw_free(&smc);
  uint32_t ref = 0;
  if (rg_atom_ref_clock(c->atom, &ref) || !ref) { fprintf(stderr, "rawgpu: no reference clock in FirmwareInfo\n"); return -EIO; }
  h.xclk = xclk(v, ref);
  uint64_t vram_mc = (uint64_t)(vi_rreg(v, mmMC_VM_FB_LOCATION) & 0xFFFF) << 24;
  h.smu_toc_mc = vram_mc + VI_VRAM_SMU_TOC;
  h.smu_buf_mc = vram_mc + VI_VRAM_SMU_BUF;
  /* amdgpu's defaults, less PCIe DPM (the link runs through a Thunderbolt
   * tunnel; retraining it is not worth the risk) and AVFS (it needs the
   * boot-time calibration in polaris10_start_smu(), which `rawgpu smc-load`
   * does not run). Without AVFS the SMC uses the VBIOS's EVV voltages. */
  h.feature_mask = PP_FEATURE_MASK_DEFAULT & ~(PP_PCIE_DPM_MASK | PP_AVFS_MASK | drop_features);
  int cur_gen = 0, cur_width = 0, max_gen = 0, max_width = 0;
  rg_pci_link(d, &cur_gen, &cur_width, &max_gen, &max_width);
  h.link_gen = cur_gen;
  h.link_width = cur_width;
  h.asic_max_gen = max_gen;
  h.vram_width = vram_width(v);
  int cus = vi_gfx_cu_count(v);
  h.cu_count = cus > 0 ? (unsigned)cus : 0;
  h.shader_engines = 4;
  if (v->err) return v->err;
  printf("card:         %s kicker part, SMC firmware 0x%08x, reference clock %.2f MHz, PCIe Gen%d x%d (GPU: Gen%d), "
         "%u-bit GDDR5, %u CUs\n", h.is_kicker ? "a" : "not a", h.smc_version, h.xclk / 100.0, cur_gen, cur_width,
         max_gen, h.vram_width, h.cu_count);
  printf("features:     ppfeaturemask 0x%08x (amdgpu's 0x%08x without PCIe DPM and AVFS%s%s%s)\n", h.feature_mask,
         PP_FEATURE_MASK_DEFAULT, drop_features & PP_MCLK_DPM_MASK ? ", memory DPM" : "",
         drop_features & PP_SCLK_DEEP_SLEEP_MASK ? ", deep sleep" : "", drop_features & PP_ULV_MASK ? ", ULV" : "");
  return rg_pp_open(&c->pp, &h);
}

static void dpm_close(dpm_ctx *c) {
  rg_pp_close(c->pp);
  if (c->atom) rg_atom_close(c->atom);
}

static int parse_v(int argc, char **argv, int *verbosity) {
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) (*verbosity)++;
    else if (!strcmp(argv[i], "-vv")) *verbosity += 2;
    else return -EINVAL;
  }
  return 0;
}

/* `rawgpu dpm-info`: the PowerPlay table and the EVV voltages, as amdgpu reads
 * them. Runs VBIOS GetVoltageInfo; writes nothing to the SMC or the clocks. */
int vi_cmd_dpm_info(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  int verbosity = 0, r = parse_v(argc, argv, &verbosity);
  if (r) return r;
  dpm_ctx c;
  r = dpm_open(&c, d, id, verbosity, false, 0);
  if (!r) r = rg_pp_read_tables(c.pp);
  if (!r) rg_pp_print_tables(c.pp, stdout);
  printf("temperature:  %d C\n", vi_temp_c(&c.v));
  dpm_close(&c);
  if (r) fprintf(stderr, "rawgpu: dpm-info failed (%d)\n", r);
  return r;
}

static void print_sensors(rg_pp *pp, const char *when) {
  rg_pp_sensors s;
  rg_pp_read_sensors(pp, &s);
  printf("%-13s engine %d MHz, memory %d MHz, GPU load %d%%, %d C, %.1f W\n", when, s.sclk_mhz, s.mclk_mhz, s.gpu_load,
         s.temp_c, s.power_w);
}

/* `rawgpu dpm-enable`: amdgpu's hwmgr_hw_init() and its late init task. The
 * SMC then runs DPM, fan control and power containment on its own; the state
 * stays in the SMC after rawgpu exits, until the card loses power. */
int vi_cmd_dpm_enable(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  int verbosity = 0, level = -1;
  uint32_t drop = 0;
  const char *trace_path = NULL;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "-v")) verbosity++;
    else if (!strcmp(argv[i], "--no-mclk")) drop |= PP_MCLK_DPM_MASK; /* engine DPM only: memory stays at boot clock */
    else if (!strcmp(argv[i], "--no-deep-sleep")) drop |= PP_SCLK_DEEP_SLEEP_MASK; /* no SCLK divider at idle */
    else if (!strcmp(argv[i], "--no-ulv")) drop |= PP_ULV_MASK;                     /* no ultra-low voltage at idle */
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!strcmp(argv[i], "--level") && i + 1 < argc) {
      const char *l = argv[++i];
      level = !strcmp(l, "auto") ? 0 : !strcmp(l, "low") ? 1 : !strcmp(l, "high") ? 2 : -2;
      if (level < -1) return -EINVAL;
    } else return -EINVAL;
  }
  dpm_ctx c;
  int r = dpm_open(&c, d, id, verbosity, true, drop);
  if (!r && trace_path && !(c.v.trace = fopen(trace_path, "w"))) { perror(trace_path); r = -EIO; }
  if (!r) {
    printf("before:       temperature %d C\n", vi_temp_c(&c.v));
    fflush(stdout);
    double t0 = vi_now();
    r = rg_pp_hw_init(c.pp);
    printf("hwmgr_hw_init: %s after %.2f s (FEATURE_STATUS 0x%08x)\n", r ? "FAILED" : "done", vi_now() - t0,
           vi_rsmc(&c.v, ixFEATURE_STATUS));
  }
  if (!r) {
    print_sensors(c.pp, "DPM on:");
    if (level >= 0) {
      r = rg_pp_force_level(c.pp, level);
      printf("level:        %s%s\n", level == 0 ? "auto" : level == 1 ? "low" : "high", r ? " (FAILED)" : "");
      if (!r) print_sensors(c.pp, "now:");
    }
  }
  if (c.v.trace) { fclose(c.v.trace); c.v.trace = NULL; printf("trace: %s\n", trace_path); }
  dpm_close(&c);
  if (r) fprintf(stderr, "rawgpu: dpm-enable failed (%d)\n", r);
  return r;
}
