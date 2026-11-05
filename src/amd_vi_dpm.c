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
#define RG_DROP_DIDT 0x80000000u /* not a ppfeaturemask bit: rawgpu's --no-didt */
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
  vi_dev v;          /* must stay first: the callbacks get a dpm_ctx * as their vi_dev * */
  rg_atom *atom;
  rg_pp *pp;
  amd_fw smc;        /* the SMC image, for smu7_upload_smu_firmware_image() */
  vi_ucode_set uc;   /* the engine images, once powerplay asks for them */
  uint64_t vram_mc;  /* MC address of VRAM offset 0 */
  bool smc_was_up;   /* the SMC already ran its firmware (`rawgpu smc-load`) */
} dpm_ctx;

/* memcpy_toio() into an SMU buffer: MC address -> BAR0 offset, then HDP flush */
static void h_rlc_safe_mode(void *ctx, bool enter) { vi_rlc_safe_mode(ctx, enter); }

static int h_vram_write(void *ctx, uint64_t mc, const void *src, size_t len) {
  dpm_ctx *c = ctx;
  if (mc < c->vram_mc || mc - c->vram_mc + len > (256ull << 20)) return -EINVAL;
  int r = vi_vram_write(&c->v, mc - c->vram_mc, src, len);
  vi_hdp_flush(&c->v);
  return r ? r : c->v.err;
}

/* amdgpu_ucode_init_bo(): the same layout `rawgpu smc-load` uses */
static int h_ucode_init_bo(void *ctx, rg_pp_ucode out[RG_PP_UC_COUNT]) {
  dpm_ctx *c = ctx;
  _Static_assert(RG_PP_UC_COUNT == VI_UCODE_IMAGES, "image lists");
  int r = vi_ucode_place(&c->v, &c->uc, NULL);
  if (r) return r;
  vi_hdp_flush(&c->v);
  for (int i = 0; i < VI_UCODE_IMAGES; i++) {
    const amd_fw *f = &c->uc.fw[i];
    out[i] = (rg_pp_ucode){.mc = c->vram_mc + c->uc.off[i], .ucode_bytes = f->ucode_dw * 4,
                           .ucode_version = f->ucode_version, .feature_version = amd_fw_hdr32(f, 32),
                           .jt_offset = amd_fw_hdr32(f, 36), .jt_size = amd_fw_hdr32(f, 40), .image = f->ucode};
  }
  printf("microcode:    7 images placed at VRAM +%llu MB for the SMU (amdgpu_ucode_init_bo())\n",
         (unsigned long long)(VI_VRAM_UCODE >> 20));
  return c->v.err;
}

static int dpm_open(dpm_ctx *c, rg_dev *d, const rg_ident *id, int verbosity, bool for_hw, uint32_t drop_features) {
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
  c->smc_was_up = !vi_smu_ucode_status(v, &status) && status;
  if (for_hw) { /* amdgpu loads the MC firmware before power management starts (gmc_v8_0_hw_init()) */
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

  rg_pp_host h = {.ctx = c, .rreg = h_rreg, .wreg = h_wreg, .rind = h_rind, .wind = h_wind, .delay_us = h_delay,
                  .err = h_err, .vram_write = h_vram_write, .ucode_init_bo = h_ucode_init_bo,
                  .rlc_safe_mode = h_rlc_safe_mode, .no_didt = (drop_features & RG_DROP_DIDT) != 0, .verbosity = verbosity};
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
  if ((r = amd_fw_load(&c->smc, NULL, vi_smc_fw_name(d, hard)))) return r;
  h.smc_version = c->smc.ucode_version;
  h.smc_image = c->smc.ucode;
  h.smc_bytes = c->smc.ucode_dw * 4;
  h.smc_start_addr = amd_fw_hdr32(&c->smc, 32); /* smc_firmware_header_v1_0 ucode_start_addr */
  uint32_t ref = 0;
  if (rg_atom_ref_clock(c->atom, &ref) || !ref) { fprintf(stderr, "rawgpu: no reference clock in FirmwareInfo\n"); return -EIO; }
  h.xclk = xclk(v, ref);
  uint64_t vram_mc = (uint64_t)(vi_rreg(v, mmMC_VM_FB_LOCATION) & 0xFFFF) << 24;
  c->vram_mc = vram_mc;
  h.smu_toc_mc = vram_mc + VI_VRAM_SMU_TOC;
  h.smu_buf_mc = vram_mc + VI_VRAM_SMU_BUF;
  /* amdgpu's defaults, less PCIe DPM: the link runs through a Thunderbolt
   * tunnel, and retraining it is not worth the risk. AVFS stays on when
   * amdgpu's own polaris10_start_smu() starts the SMC (it stages AVFS there);
   * an SMC that `rawgpu smc-load` started never had that, so AVFS goes off. */
  if (c->smc_was_up) drop_features |= PP_AVFS_MASK;
  h.feature_mask = PP_FEATURE_MASK_DEFAULT & ~(PP_PCIE_DPM_MASK | (drop_features & ~RG_DROP_DIDT));
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
  printf("features:     ppfeaturemask 0x%08x (amdgpu's 0x%08x without PCIe DPM%s%s%s%s)\n", h.feature_mask,
         PP_FEATURE_MASK_DEFAULT, drop_features & PP_AVFS_MASK ? ", AVFS" : "",
         drop_features & PP_MCLK_DPM_MASK ? ", memory DPM" : "", drop_features & PP_SCLK_DEEP_SLEEP_MASK ? ", deep sleep" : "",
         drop_features & PP_ULV_MASK ? ", ULV" : "");
  if (drop_features & RG_DROP_DIDT) printf("              di/dt limiters (SQ/DB/TD/TCP ramping) left off\n");
  return rg_pp_open(&c->pp, &h);
}

static void dpm_close(dpm_ctx *c) {
  rg_pp_close(c->pp);
  if (c->atom) rg_atom_close(c->atom);
  vi_ucode_free(&c->uc);
  amd_fw_free(&c->smc);
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
    else if (!strcmp(argv[i], "--no-didt")) drop |= RG_DROP_DIDT;
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
    if (!c.smc_was_up) { /* pp_dpm_load_fw(): amdgpu starts the SMC and loads the engine images itself */
      double t1 = vi_now();
      r = rg_pp_start_smu(c.pp);
      uint32_t st = 0;
      vi_smu_ucode_status(&c.v, &st);
      printf("start_smu:    %s after %.2f s: SMC PC 0x%08x, UcodeLoadStatus 0x%08x\n", r ? "FAILED" : "done",
             vi_now() - t1, vi_rsmc(&c.v, ixSMC_PC_C), st);
      for (int me = 0; !r && me < 2; me++) { /* the same check smc-load makes */
        uint32_t first = 0;
        int bad = vi_sdma_check_ucode(&c.v, me, &c.uc.fw[me], &first);
        printf("SDMA%d ucode:  %s\n", me, bad == 0 ? "microcode RAM matches the image" : "MISMATCH");
        if (bad) r = bad < 0 ? bad : -EIO;
      }
    } else {
      printf("start_smu:    skipped: the SMC already runs (smc-load), so AVFS stays off\n");
    }
    if (r) goto done;
    double t0 = vi_now();
    r = rg_pp_hw_init(c.pp);
    printf("hwmgr_hw_init: %s after %.2f s (FEATURE_STATUS 0x%08x)\n", r ? "FAILED" : "done", vi_now() - t0,
           vi_rsmc(&c.v, ixFEATURE_STATUS));
  }
  if (!r) { /* amdgpu's IP order: gfx_v8_0_hw_init() right after powerplay's */
    r = vi_gfx_compute_init(&c.v);
    printf("GFX:          %s (RLC_CNTL 0x%08x)\n", r ? "FAILED" : "up", vi_rreg(&c.v, mmRLC_CNTL));
  }
  if (!r) {
    print_sensors(c.pp, "DPM on:");
    if (level >= 0) {
      r = rg_pp_force_level(c.pp, level);
      printf("level:        %s%s\n", level == 0 ? "auto" : level == 1 ? "low" : "high", r ? " (FAILED)" : "");
      if (!r) print_sensors(c.pp, "now:");
    }
  }
done:
  if (c.v.trace) { fclose(c.v.trace); c.v.trace = NULL; printf("trace: %s\n", trace_path); }
  dpm_close(&c);
  if (r) fprintf(stderr, "rawgpu: dpm-enable failed (%d)\n", r);
  return r;
}

/* `rawgpu didt [off]`: the di/dt current limiters smu7_enable_didt_config()
 * turns on (SQ, DB, TD, TCP), GRBM busy state, and the host API's compute
 * fence in VRAM. `off` clears DIDT_CTRL_EN, as smu7_enable_didt(false) does. */
int vi_cmd_didt(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  bool off = argc == 1 && !strcmp(argv[0], "off");
  if (argc && !off) return -EINVAL;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  static const struct { const char *name; uint32_t ix; } blk[] = {
    {"SQ", ixDIDT_SQ_CTRL0}, {"DB", ixDIDT_DB_CTRL0}, {"TD", ixDIDT_TD_CTRL0}, {"TCP", ixDIDT_TCP_CTRL0},
  };
  for (int pass = 0; pass < (off ? 2 : 1); pass++) {
    if (pass) {
      for (size_t i = 0; i < 4; i++) {
        vi_wreg(&v, mmDIDT_IND_INDEX, blk[i].ix);
        uint32_t x = vi_rreg(&v, mmDIDT_IND_DATA);
        vi_wreg(&v, mmDIDT_IND_INDEX, blk[i].ix);
        vi_wreg(&v, mmDIDT_IND_DATA, x & ~DIDT_SQ_CTRL0__DIDT_CTRL_EN_MASK);
      }
      vi_udelay(100000);
      printf("cleared DIDT_CTRL_EN; 100 ms later:\n");
    }
    printf("DIDT CTRL0:  ");
    for (size_t i = 0; i < 4; i++) {
      vi_wreg(&v, mmDIDT_IND_INDEX, blk[i].ix);
      uint32_t x = vi_rreg(&v, mmDIDT_IND_DATA);
      printf(" %s 0x%08x (%s)", blk[i].name, x, x & DIDT_SQ_CTRL0__DIDT_CTRL_EN_MASK ? "on" : "off");
    }
    printf("\nGRBM_STATUS 0x%08x\n", vi_rreg(&v, mmGRBM_STATUS)); /* the host API's fences are in its own memory now */
  }
  return v.err;
}
