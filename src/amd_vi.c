/* Backend: AMD GCN3/GCN4 "Volcanic Islands" dGPUs (Tonga, Fiji, Polaris 10/11/12, VegaM).
 *
 * Register offsets come from Linux amdgpu via tools/gen_vi_regs.py (src/amd_vi_regs.h);
 * they are dword indices, as in the kernel. This file has the register helpers,
 * probe, VBIOS, post and the VRAM test; amd_vi_gmc.c and amd_vi_sdma.c bring up
 * the memory controller's GART and the DMA engines.
 */
#include "amd_vi.h"
#include "atom/rg_atom.h"

#include <errno.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* atombios.h: what amdgpu_atombios_scratch_regs_init() adjusts before posting */
#define ATOM_S2_DEVICE_DPMS_STATE        0x00010000u
#define ATOM_S2_VRI_BRIGHT_ENABLE        0x20000000u
#define ATOM_S6_ACC_BLOCK_DISPLAY_SWITCH 0x20000000u
#define ATOM_S3_ASIC_GUI_ENGINE_HUNG     0x20000000u
#define AMDGPU_ASIC_RESET_DATA           0x39d5e86bu /* amdgpu.h */

#define VI_ROM_MAX (256u << 10)

/* ---- register helpers ---- */

uint32_t vi_rreg(vi_dev *v, uint32_t idx) {
  uint32_t val = 0xFFFFFFFFu;
  if (v->err) return val;
  int r = rg_rreg32(v->d, VI_REG_BAR, idx * 4, &val);
  if (r) { v->err = r; return 0xFFFFFFFFu; }
  if (v->trace) fprintf(v->trace, "R %05x %08x\n", idx, val);
  return val;
}

void vi_wreg(vi_dev *v, uint32_t idx, uint32_t val) {
  if (v->err) return;
  int r = rg_wreg32(v->d, VI_REG_BAR, idx * 4, val);
  if (r) v->err = r;
  if (v->trace) fprintf(v->trace, "W %05x %08x\n", idx, val);
}

uint32_t vi_rsmc(vi_dev *v, uint32_t addr) {
  vi_wreg(v, mmSMC_IND_INDEX_11, addr);
  return vi_rreg(v, mmSMC_IND_DATA_11);
}

void vi_wsmc(vi_dev *v, uint32_t addr, uint32_t val) {
  vi_wreg(v, mmSMC_IND_INDEX_11, addr);
  vi_wreg(v, mmSMC_IND_DATA_11, val);
}

double vi_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec / 1e9;
}

void vi_udelay(unsigned us) {
  struct timespec ts = {us / 1000000u, (long)(us % 1000000u) * 1000};
  nanosleep(&ts, NULL);
}

/* Each read is a socket round trip (~10 us), so this polls about that often
 * without sleeping. */
int vi_wait(vi_dev *v, uint32_t idx, uint32_t mask, uint32_t want, unsigned timeout_us) {
  double end = vi_now() + timeout_us / 1e6;
  do {
    uint32_t x = vi_rreg(v, idx);
    if (v->err) return v->err;
    if ((x & mask) == want) return 0;
  } while (vi_now() < end);
  return -ETIMEDOUT;
}

void vi_program_sequence(vi_dev *v, const uint32_t *seq, size_t n) {
  for (size_t i = 0; i + 2 < n; i += 3) {
    uint32_t reg = seq[i], and_mask = seq[i + 1], or_mask = seq[i + 2], tmp;
    if (and_mask == 0xFFFFFFFFu) tmp = or_mask;
    else tmp = (vi_rreg(v, reg) & ~and_mask) | or_mask; /* pre-AI semantics */
    vi_wreg(v, reg, tmp);
  }
}

void vi_srbm_select(vi_dev *v, uint32_t me, uint32_t pipe, uint32_t queue, uint32_t vmid) {
  uint32_t x = 0;
  x = REG_SET_FIELD(x, SRBM_GFX_CNTL, PIPEID, pipe);
  x = REG_SET_FIELD(x, SRBM_GFX_CNTL, MEID, me);
  x = REG_SET_FIELD(x, SRBM_GFX_CNTL, VMID, vmid);
  x = REG_SET_FIELD(x, SRBM_GFX_CNTL, QUEUEID, queue);
  vi_wreg(v, mmSRBM_GFX_CNTL, x);
}

int vi_open(vi_dev *v, rg_dev *d) {
  memset(v, 0, sizeof *v);
  v->d = d;
  uint64_t sz = 0;
  int r = rg_bar_map(d, VI_REG_BAR, &sz);
  if (r) { fprintf(stderr, "rawgpu: could not map register BAR%d\n", VI_REG_BAR); return r; }
  if (sz < 256u << 10) { fprintf(stderr, "rawgpu: register BAR too small (%llu)\n", (unsigned long long)sz); return -EINVAL; }
  /* The VRAM window. Everything rawgpu keeps in VRAM sits inside it (see VI_VRAM_*). */
  if ((r = rg_bar_map(d, VI_VRAM_BAR, &sz))) { fprintf(stderr, "rawgpu: could not map VRAM BAR%d\n", VI_VRAM_BAR); return r; }
  if (sz < (128ull << 20) + (4u << 20)) { fprintf(stderr, "rawgpu: VRAM BAR is only %llu MB\n", (unsigned long long)(sz >> 20)); return -EINVAL; }
  return 0;
}

bool vi_posted(vi_dev *v, uint32_t *memsize_mb) {
  uint32_t m = vi_rreg(v, mmCONFIG_MEMSIZE);
  if (memsize_mb) *memsize_mb = m;
  return !v->err && m != 0 && m != 0xFFFFFFFFu; /* amdgpu_device_need_post() */
}

/* ---- VBIOS ---- */

/* Streams `n` dwords of the VBIOS starting at byte 0 through the auto-incrementing ROM_DATA port,
 * as vi_read_bios_from_rom() does. */
static int rom_read(vi_dev *v, uint32_t *dst, size_t n) {
  vi_wreg(v, mmSMC_IND_INDEX_11, ixROM_INDEX);
  vi_wreg(v, mmSMC_IND_DATA_11, 0);
  vi_wreg(v, mmSMC_IND_INDEX_11, ixROM_DATA);
  for (size_t i = 0; i < n && !v->err; i++) dst[i] = vi_rreg(v, mmSMC_IND_DATA_11);
  return v->err;
}

static int vi_vbios_read(rg_dev *d, uint8_t **out, size_t *len) {
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t hdr[128];
  if ((r = rom_read(&v, hdr, 128))) return r;
  const uint8_t *h = (const uint8_t *)hdr;
  if (h[0] != 0x55 || h[1] != 0xAA) {
    fprintf(stderr, "rawgpu: no option-ROM signature (got %02x %02x)\n", h[0], h[1]);
    return -ENOENT;
  }
  size_t size = (size_t)h[2] * 512;
  if (size == 0 || size > VI_ROM_MAX) size = 64u << 10;
  uint8_t *buf = malloc(size);
  if (!buf) return -ENOMEM;
  if ((r = rom_read(&v, (uint32_t *)buf, size / 4))) { free(buf); return r; }
  *out = buf;
  *len = size;
  return 0;
}

/* ---- probe and status ---- */

/* Memory-controller state after posting. Only read once CONFIG_MEMSIZE says
 * the card is posted: on a cold card these blocks are not initialised. */
static void vi_report_mc(vi_dev *v) {
  struct { const char *name; uint32_t idx; } regs[] = {
    {"MC_VM_FB_LOCATION", mmMC_VM_FB_LOCATION}, {"MC_VM_FB_OFFSET", mmMC_VM_FB_OFFSET},
    {"MC_SHARED_CHMAP", mmMC_SHARED_CHMAP}, {"MC_ARB_RAMCFG", mmMC_ARB_RAMCFG},
    {"MC_SEQ_MISC0", mmMC_SEQ_MISC0}, {"MC_SEQ_SUP_CNTL", mmMC_SEQ_SUP_CNTL},
    {"MC_SEQ_TRAIN_WAKEUP_CNTL", mmMC_SEQ_TRAIN_WAKEUP_CNTL}, {"HDP_NONSURFACE_BASE", mmHDP_NONSURFACE_BASE},
    {"HDP_NONSURFACE_INFO", mmHDP_NONSURFACE_INFO}, {"HDP_NONSURFACE_SIZE", mmHDP_NONSURFACE_SIZE},
    {"BIF_FB_EN", mmBIF_FB_EN},
  };
  for (size_t i = 0; i < sizeof regs / sizeof regs[0]; i++) {
    uint32_t x = vi_rreg(v, regs[i].idx);
    if (v->err) return;
    printf("  %-26s 0x%08x", regs[i].name, x);
    if (regs[i].idx == mmMC_VM_FB_LOCATION)
      printf("   VRAM at MC 0x%010llx-0x%010llx", (unsigned long long)(x & 0xFFFF) << 24,
             ((unsigned long long)(x >> 16) << 24) | 0xFFFFFF);
    if (regs[i].idx == mmMC_SEQ_MISC0) printf("   memory type field (MT) = %u", x >> 28);
    if (regs[i].idx == mmBIF_FB_EN)
      printf("   host access to VRAM: read %s, write %s", x & BIF_FB_EN__FB_READ_EN_MASK ? "on" : "off",
             x & BIF_FB_EN__FB_WRITE_EN_MASK ? "on" : "off");
    printf("\n");
  }
}

static int vi_probe(rg_dev *d, const rg_ident *id) {
  (void)id;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;

  uint32_t memsize = 0;
  bool posted = vi_posted(&v, &memsize);
  if (v.err) return v.err;
  printf("  posted:          %s (CONFIG_MEMSIZE = 0x%08x%s)\n", posted ? "yes" : "NO",
         memsize, posted ? "" : "; nothing has run the VBIOS init, so VRAM is not usable yet");
  if (posted) { printf("  VRAM:            %u MB\n", memsize); vi_report_mc(&v); }

  printf("  BIOS scratch:   ");
  for (uint32_t i = 0; i < 8; i++) printf(" %08x", vi_rreg(&v, mmBIOS_SCRATCH_0 + i));
  printf("\n");
  if (v.err) return v.err;

  uint8_t *bios = NULL;
  size_t len = 0;
  if ((r = vi_vbios_read(d, &bios, &len))) {
    printf("  VBIOS:           unreadable (%d)\n", r);
    return 0;
  }
  rg_atom_report(bios, len);
  free(bios);
  return 0;
}

/* `rawgpu status`: read-only snapshot of the engines rawgpu brings up. */
static int cmd_status(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id; (void)argc; (void)argv;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize;
  bool posted = vi_posted(&v, &memsize);
  printf("posted:                %s (CONFIG_MEMSIZE = %u MB)\n", posted ? "yes" : "NO", memsize);
  if (!posted) return v.err;

  uint32_t srbm = vi_rreg(&v, mmSRBM_STATUS), srbm2 = vi_rreg(&v, mmSRBM_STATUS2);
  uint32_t grbm = vi_rreg(&v, mmGRBM_STATUS);
  printf("SRBM_STATUS            0x%08x   SRBM_STATUS2 0x%08x   GRBM_STATUS 0x%08x\n", srbm, srbm2, grbm);
  printf("GB_ADDR_CONFIG         0x%08x\n", vi_rreg(&v, mmGB_ADDR_CONFIG));

  vi_smu_report(&v);

  printf("RLC_CNTL               0x%08x   RLC_STAT 0x%08x   CP_MEC_CNTL 0x%08x   CP_ME_CNTL 0x%08x\n",
         vi_rreg(&v, mmRLC_CNTL), vi_rreg(&v, mmRLC_STAT), vi_rreg(&v, mmCP_MEC_CNTL), vi_rreg(&v, mmCP_ME_CNTL));
  vi_gmc_report(&v);
  vi_sdma_report(&v);
  return v.err;
}

/* ---- post ---- */

static rg_atom *open_atom(rg_dev *d, FILE *trace, bool trace_ops) {
  uint8_t *bios = NULL;
  size_t len = 0;
  if (vi_vbios_read(d, &bios, &len)) return NULL;
  rg_atom_cfg cfg = {.reg_bar = VI_REG_BAR, .mm_index = mmMM_INDEX * 4, .mm_data = mmMM_DATA * 4,
                     .flush_reg = mmBIOS_SCRATCH_0 * 4, .trace = trace, .trace_ops = trace_ops};
  rg_atom *a = rg_atom_open(d, bios, len, &cfg);
  free(bios);
  return a;
}

static int cmd_atom_info(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id; (void)argc; (void)argv;
  rg_atom *a = open_atom(d, NULL, false);
  if (!a) return -EIO;
  rg_atom_print_info(a);
  rg_atom_close(a);
  return 0;
}

static int cmd_post(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id;
  const char *trace_path = NULL;
  bool trace_ops = false, force = false;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!strcmp(argv[i], "--trace-ops")) trace_ops = true;
    else if (!strcmp(argv[i], "--force")) force = true;
    else return -EINVAL;
  }

  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize;
  bool posted = vi_posted(&v, &memsize);
  if (v.err) return v.err;
  if (posted && !force) {
    printf("already posted (CONFIG_MEMSIZE = %u MB); nothing to do. --force re-runs ASIC_Init.\n", memsize);
    return 0;
  }

  FILE *trace = NULL;
  if (trace_path && !(trace = fopen(trace_path, "w"))) { perror(trace_path); return -EIO; }
  rg_atom *a = open_atom(d, trace, trace_ops);
  if (!a) { if (trace) fclose(trace); return -EIO; }
  rg_atom_print_info(a);

  /* amdgpu_atombios_scratch_regs_init(): BIOS controls backlight, no mode switching, clear DPMS state */
  uint32_t s2 = vi_rreg(&v, mmBIOS_SCRATCH_2), s6 = vi_rreg(&v, mmBIOS_SCRATCH_6);
  vi_wreg(&v, mmBIOS_SCRATCH_2, s2 & ~(ATOM_S2_VRI_BRIGHT_ENABLE | ATOM_S2_DEVICE_DPMS_STATE));
  vi_wreg(&v, mmBIOS_SCRATCH_6, s6 | ATOM_S6_ACC_BLOCK_DISPLAY_SWITCH);
  if ((r = v.err)) goto out;

  printf("running ASIC_Init...\n");
  fflush(stdout);
  rg_atom_stats st;
  r = rg_atom_asic_init(a, &st);
  printf("ASIC_Init %s (%d): %.2f s, %llu reads, %llu writes, %llu indirect, %llu delays (%.1f ms requested)\n",
         r == 0 ? "returned OK" : "FAILED", r, st.seconds, (unsigned long long)st.reads,
         (unsigned long long)st.writes, (unsigned long long)st.indirect, (unsigned long long)st.delays,
         st.delay_us / 1000.0);
  if (r) goto out;

  posted = vi_posted(&v, &memsize);
  printf("posted:            %s (CONFIG_MEMSIZE = %u MB)\n", posted ? "yes" : "NO", memsize);
  if (posted) vi_report_mc(&v);
  else r = v.err ? v.err : -EIO;
out:
  rg_atom_close(a);
  if (trace) { fclose(trace); printf("trace: %s\n", trace_path); }
  return r;
}

/* ---- vram-test ---- */

/* Fills/validates `n` dwords with a pattern unique to (seed, index). */
static uint32_t pat(uint32_t seed, size_t i) {
  uint32_t x = seed * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA77u;
  x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12;
  return x;
}

/* One dword of VRAM through the indirect window: MM_INDEX bit 31 selects the FB aperture. */
static uint32_t vram_mm_access(vi_dev *v, uint64_t pos, uint32_t wval, bool write) {
  vi_wreg(v, mmMM_INDEX, (uint32_t)pos | 0x80000000u);
  vi_wreg(v, mmMM_INDEX_HI, (uint32_t)(pos >> 31));
  if (write) { vi_wreg(v, mmMM_DATA, wval); return 0; }
  return vi_rreg(v, mmMM_DATA);
}

static int cmd_vram_test(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id;
  bool enable_fb = argc > 0 && !strcmp(argv[0], "--enable-fb");
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize;
  if (!vi_posted(&v, &memsize)) {
    if (v.err) return v.err;
    fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post` first\n");
    return -EIO;
  }
  uint32_t fb_en = vi_rreg(&v, mmBIF_FB_EN);
  uint32_t want = BIF_FB_EN__FB_READ_EN_MASK | BIF_FB_EN__FB_WRITE_EN_MASK;
  if ((fb_en & want) != want) {
    if (!enable_fb) {
      fprintf(stderr, "rawgpu: BIF_FB_EN = 0x%x: host access to VRAM is off. --enable-fb turns it on, as\n"
                      "        amdgpu's gmc_v8_0_mc_program() does.\n", fb_en);
      return -EACCES;
    }
    vi_wreg(&v, mmBIF_FB_EN, fb_en | want);
    printf("BIF_FB_EN: 0x%x -> 0x%x\n", fb_en, fb_en | want);
  }
  if (v.err) return v.err;

  uint64_t bar = 0;
  if ((r = rg_bar_map(d, VI_VRAM_BAR, &bar))) return r;
  enum { N = 16384 }; /* 64 KB per region */
  const uint64_t offs[] = {1ull << 20, 64ull << 20, 192ull << 20};
  static uint32_t buf[N], got[N];
  double wt = 0, rt = 0;
  int bad = 0;

  for (size_t k = 0; k < 3; k++) {
    if (offs[k] + sizeof buf > bar) continue;
    for (size_t i = 0; i < N; i++) buf[i] = pat((uint32_t)k + 1, i);
    double t0 = vi_now();
    if ((r = d->ops->mmio_write(d, VI_VRAM_BAR, offs[k], buf, sizeof buf))) return r;
    vi_rreg(&v, mmBIOS_SCRATCH_0); /* drain writes */
    wt += vi_now() - t0;
  }
  for (size_t k = 0; k < 3; k++) { /* read back only after all writes: catches aliasing */
    if (offs[k] + sizeof got > bar) continue;
    double t0 = vi_now();
    if ((r = d->ops->mmio_read(d, VI_VRAM_BAR, offs[k], got, sizeof got))) return r;
    rt += vi_now() - t0;
    int mism = 0;
    for (size_t i = 0; i < N; i++) mism += got[i] != pat((uint32_t)k + 1, i);
    printf("  VRAM +%4llu MB: %s", (unsigned long long)(offs[k] >> 20), mism ? "MISMATCH" : "OK");
    if (mism) printf(" (%d of %d dwords wrong; first dword read 0x%08x, wanted 0x%08x)", mism, N, got[0], pat((uint32_t)k + 1, 0));
    printf("\n");
    bad += mism != 0;
  }
  double mb = 3.0 * sizeof buf / 1e6;
  printf("  through BAR0 (CPU MMIO over the socket, not DMA): write %.1f MB/s, read %.1f MB/s\n", mb / wt, mb / rt);

  /* The rest of VRAM, past the 256 MB BAR, through the MM_INDEX window exactly as
   * amdgpu_device_mm_access() does. Reading back the BAR-written region this way
   * also proves BAR0 offset 0 is VRAM offset 0. */
  uint64_t vram = (uint64_t)memsize << 20;
  const uint64_t far[] = {1ull << 30, 4ull << 30, vram - (1ull << 20)};
  enum { M = 256 };
  for (size_t k = 0; k < 3 && far[k] + 4 * M <= vram; k++)
    for (size_t i = 0; i < M; i++) vram_mm_access(&v, far[k] + 4 * i, pat((uint32_t)k + 11, i), true);
  for (size_t k = 0; k < 4; k++) {
    uint64_t base = k < 3 ? far[k] : offs[0];
    uint32_t seed = k < 3 ? (uint32_t)k + 11 : 1;
    if (base + 4 * M > vram) continue;
    int mism = 0;
    for (size_t i = 0; i < M; i++) mism += vram_mm_access(&v, base + 4 * i, 0, false) != pat(seed, i);
    if (v.err) return v.err;
    printf("  VRAM +%5llu MB (MM_INDEX%s): %s\n", (unsigned long long)(base >> 20),
           k == 3 ? ", re-reading the BAR-written region" : "", mism ? "MISMATCH" : "OK");
    bad += mism != 0;
  }
  return bad ? -EIO : v.err;
}

/* `rawgpu regs PATTERN...`: read registers by name (shell-style patterns). Data
 * ports and ucode windows advance on every read, so a pattern only reaches
 * them when it names them exactly. */
static const struct { const char *name; uint32_t idx; } vi_regnames[] = {
#include "amd_vi_regnames.inc"
};

static bool side_effect_read(const char *name) {
  return strstr(name, "_DATA") || strstr(name, "UCODE") || !strncmp(name, "MM_", 3) || !strncmp(name, "SMC_IND_", 8);
}

static int cmd_regs(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id;
  if (argc < 1) return -EINVAL;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  int shown = 0;
  for (size_t i = 0; i < sizeof vi_regnames / sizeof vi_regnames[0]; i++) {
    for (int k = 0; k < argc; k++) {
      const char *n = vi_regnames[i].name;
      bool exact = !strcmp(argv[k], n);
      if (!exact && (fnmatch(argv[k], n, 0) || side_effect_read(n))) continue;
      printf("%-44s [0x%05x] 0x%08x\n", n, vi_regnames[i].idx, vi_rreg(&v, vi_regnames[i].idx));
      shown++;
      break;
    }
  }
  if (!shown) fprintf(stderr, "rawgpu: no register in src/amd_vi_regnames.inc matches\n");
  return v.err;
}

/* `rawgpu asic-reset`: vi_asic_pci_config_reset(), amdgpu's full ("legacy")
 * reset for VI when BACO is unavailable (BACO needs SMC firmware). Bus mastering
 * is off during the reset, so nothing the engines had in flight can land
 * afterwards. The card comes back unposted. */
static int cmd_asic_reset(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id; (void)argc; (void)argv;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t cmd;
  if ((r = rg_cfg_read(d, 0x04, 2, &cmd))) return r;
  vi_wreg(&v, mmBIOS_SCRATCH_3, vi_rreg(&v, mmBIOS_SCRATCH_3) | ATOM_S3_ASIC_GUI_ENGINE_HUNG);
  if (v.err) return v.err;
  if ((r = rg_cfg_write(d, 0x04, 2, cmd & ~0x4u))) return r; /* pci_clear_master() */
  if ((r = rg_cfg_write(d, 0x7c, 4, AMDGPU_ASIC_RESET_DATA))) return r;
  vi_udelay(100);
  double t0 = vi_now();
  uint32_t m = 0xFFFFFFFFu;
  while ((m = vi_rreg(&v, mmCONFIG_MEMSIZE)) == 0xFFFFFFFFu && !v.err && vi_now() - t0 < 1.0) {}
  double dt = vi_now() - t0;
  r = rg_cfg_write(d, 0x04, 2, cmd | 0x4u); /* pci_set_master() */
  if (v.err || r) return v.err ? v.err : r;
  if (m == 0xFFFFFFFFu) { fprintf(stderr, "rawgpu: card did not come back from reset within 1 s\n"); return -ETIMEDOUT; }
  vi_wreg(&v, mmBIOS_SCRATCH_3, vi_rreg(&v, mmBIOS_SCRATCH_3) & ~ATOM_S3_ASIC_GUI_ENGINE_HUNG);
  printf("ASIC reset: registers back after %.1f ms; CONFIG_MEMSIZE = %u (%s)\n", dt * 1e3, m,
         m ? "still posted" : "unposted: run `rawgpu post`");
  rg_pci_err e;
  if (!rg_pci_errors(d, &e)) rg_pci_print_errors(&e);
  return v.err;
}

/* `rawgpu soft-reset`: amdgpu's per-block soft resets for the blocks rawgpu
 * drives, applied only to the ones that report busy. */
static int cmd_soft_reset(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  (void)id; (void)argc; (void)argv;
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  printf("before: SRBM_STATUS 0x%08x SRBM_STATUS2 0x%08x\n", vi_rreg(&v, mmSRBM_STATUS), vi_rreg(&v, mmSRBM_STATUS2));
  vi_sdma_stop(&v, 0);
  vi_sdma_stop(&v, 1);
  if (vi_rreg(&v, mmSRBM_STATUS2) & (SRBM_STATUS2__SDMA_BUSY_MASK | SRBM_STATUS2__SDMA1_BUSY_MASK)) {
    vi_sdma_soft_reset(&v);
    printf("SDMA soft reset\n");
  }
  uint32_t bits = vi_gmc_soft_reset(&v);
  if (bits) printf("GMC soft reset (SRBM_SOFT_RESET bits 0x%08x)\n", bits);
  vi_wreg(&v, mmVM_CONTEXT0_CNTL2, VM_CONTEXT0_CNTL2__CLEAR_PROTECTION_FAULT_STATUS_ADDR_MASK); /* clear latched fault */
  vi_wreg(&v, mmVM_CONTEXT0_CNTL2, 0);
  printf("after:  SRBM_STATUS 0x%08x SRBM_STATUS2 0x%08x VM0 fault 0x%08x\n", vi_rreg(&v, mmSRBM_STATUS),
         vi_rreg(&v, mmSRBM_STATUS2), vi_rreg(&v, mmVM_CONTEXT0_PROTECTION_FAULT_STATUS));
  rg_pci_err e;
  if (!rg_pci_errors(d, &e)) rg_pci_print_errors(&e);
  return v.err;
}

int vi_cmd_sdma_test(rg_dev *d, const rg_ident *id, int argc, char **argv); /* amd_vi_sdma.c */
int vi_cmd_smc_load(rg_dev *d, const rg_ident *id, int argc, char **argv);  /* amd_vi_smu.c */
int vi_cmd_compute_test(rg_dev *d, const rg_ident *id, int argc, char **argv); /* amd_vi_gfx.c */

static const rg_cmd vi_cmds[] = {
  {"status", "", "engine, VM and SMC state (read-only)", cmd_status},
  {"regs", "PATTERN...", "read registers by name, e.g. 'SDMA0_GFX_*' (read-only)", cmd_regs},
  {"soft-reset", "", "WRITES: amdgpu's soft reset of SDMA and the memory controller, if busy", cmd_soft_reset},
  {"asic-reset", "", "WRITES: full ASIC reset through PCI config (amdgpu's legacy VI reset)", cmd_asic_reset},
  {"atom-info", "", "list the VBIOS command tables and ASIC_Init parameters (no writes)", cmd_atom_info},
  {"post", "[--trace FILE] [--trace-ops] [--force]", "WRITES: run the VBIOS ASIC_Init so the card's memory works",
   cmd_post},
  {"vram-test", "[--enable-fb]", "WRITES VRAM: pattern test through BAR0 (needs a posted card)", cmd_vram_test},
  {"smc-load", "[--fw DIR] [--force] [--trace FILE]",
   "WRITES: start the SMC firmware and have it load the SDMA, RLC and CP microcode", vi_cmd_smc_load},
  {"sdma-test", "[MB] [--engine 0|1] [--vram-only] [--host-ring] [--fw DIR] [--trace FILE]",
   "WRITES: GART + SDMA bring-up, then DMA copies host<->VRAM with timing", vi_cmd_sdma_test},
  {"compute-test", "[--trace FILE]", "WRITES: RLC + MEC bring-up, one compute queue, PM4 writes and fences",
   vi_cmd_compute_test},
  {0},
};

const rg_backend rg_backend_amd_vi = {
  .name = "amd-vi",
  .reg_bar = VI_REG_BAR,
  .safe_reg = mmBIOS_SCRATCH_0 * 4,
  .probe = vi_probe,
  .vbios_read = vi_vbios_read,
  .cmds = vi_cmds,
};
