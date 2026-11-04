/* Backend: AMD GCN3/GCN4 "Volcanic Islands" dGPUs (Tonga, Fiji, Polaris 10/11/12, VegaM).
 *
 * Register offsets come from Linux amdgpu (drivers/gpu/drm/amd/include/asic_reg/,
 * MIT-licensed); the Linux headers give dword indices, so byte offset = index * 4.
 *
 * Everything here is read-only except the ROM read, which writes the SMC
 * index register exactly as amdgpu's vi_read_bios_from_rom() does before the
 * card is posted.
 */
#include "rg.h"
#include "atom/rg_atom.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VI_REG_BAR 5
#define VI_VRAM_BAR 0

/* bif_5_0_d.h */
#define mmMM_INDEX         (0x0U * 4)
#define mmMM_DATA          (0x1U * 4)
#define mmCONFIG_MEMSIZE   (0x150aU * 4)
#define mmBIF_FB_EN        (0x1524U * 4)
#define mmBIOS_SCRATCH_0   (0x5c9U * 4)
#define BIF_FB_EN__FB_READ_EN_MASK  0x1
#define BIF_FB_EN__FB_WRITE_EN_MASK 0x2
/* gmc_8_1_d.h */
#define mmMC_SHARED_CHMAP          (0x801U * 4)
#define mmMC_VM_FB_LOCATION        (0x809U * 4)
#define mmMC_VM_FB_OFFSET          (0x81aU * 4)
#define mmMC_ARB_RAMCFG            (0x9d8U * 4)
#define mmMC_SEQ_SUP_CNTL          (0xa32U * 4)
#define mmMC_SEQ_TRAIN_WAKEUP_CNTL (0xa3aU * 4)
#define mmMC_SEQ_MISC0             (0xa80U * 4)
/* oss_3_0_d.h */
#define mmHDP_NONSURFACE_BASE      (0xb01U * 4)
#define mmHDP_NONSURFACE_INFO      (0xb02U * 4)
#define mmHDP_NONSURFACE_SIZE      (0xb03U * 4)
/* atombios.h: what amdgpu_atombios_scratch_regs_init() adjusts before posting */
#define ATOM_S2_DEVICE_DPMS_STATE        0x00010000u
#define ATOM_S2_VRI_BRIGHT_ENABLE        0x20000000u
#define ATOM_S6_ACC_BLOCK_DISPLAY_SWITCH 0x20000000u
/* smu_7_1_1_d.h / smu_7_1_3_d.h (same values on Polaris) */
#define mmSMC_IND_INDEX_11 (0x1ACU * 4)
#define mmSMC_IND_DATA_11  (0x1ADU * 4)
#define ixROM_INDEX        0xc0600010U
#define ixROM_DATA         0xc0600014U

#define VI_ROM_MAX (256u << 10)

static void vi_report_mc(rg_dev *d);

static int map_regs(rg_dev *d) {
  uint64_t sz = 0;
  int r = rg_bar_map(d, VI_REG_BAR, &sz);
  if (r) { fprintf(stderr, "rawgpu: could not map register BAR%d\n", VI_REG_BAR); return r; }
  if (sz < mmCONFIG_MEMSIZE + 4) { fprintf(stderr, "rawgpu: register BAR too small (%llu)\n", (unsigned long long)sz); return -EINVAL; }
  return 0;
}

/* Streams `n` dwords of the VBIOS starting at byte 0 through the auto-incrementing ROM_DATA port. */
static int rom_read(rg_dev *d, uint32_t *dst, size_t n) {
  int r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_INDEX_11, ixROM_INDEX))) return r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_DATA_11, 0))) return r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_INDEX_11, ixROM_DATA))) return r;
  for (size_t i = 0; i < n; i++)
    if ((r = rg_rreg32(d, VI_REG_BAR, mmSMC_IND_DATA_11, &dst[i]))) return r;
  return 0;
}

static int vi_vbios_read(rg_dev *d, uint8_t **out, size_t *len) {
  int r = map_regs(d);
  if (r) return r;
  uint32_t hdr[128];
  if ((r = rom_read(d, hdr, 128))) return r;
  const uint8_t *h = (const uint8_t *)hdr;
  if (h[0] != 0x55 || h[1] != 0xAA) {
    fprintf(stderr, "rawgpu: no option-ROM signature (got %02x %02x)\n", h[0], h[1]);
    return -ENOENT;
  }
  size_t size = (size_t)h[2] * 512;
  if (size == 0 || size > VI_ROM_MAX) size = 64u << 10;
  uint8_t *buf = malloc(size);
  if (!buf) return -ENOMEM;
  if ((r = rom_read(d, (uint32_t *)buf, size / 4))) { free(buf); return r; }
  *out = buf;
  *len = size;
  return 0;
}

static int vi_probe(rg_dev *d, const rg_ident *id) {
  (void)id;
  int r = map_regs(d);
  if (r) return r;

  uint32_t memsize = 0;
  if ((r = rg_rreg32(d, VI_REG_BAR, mmCONFIG_MEMSIZE, &memsize))) return r;
  bool posted = memsize != 0 && memsize != 0xFFFFFFFFu; /* amdgpu_device_need_post() */
  printf("  posted:          %s (CONFIG_MEMSIZE = 0x%08x%s)\n", posted ? "yes" : "NO",
         memsize, posted ? "" : "; nothing has run the VBIOS init, so VRAM is not usable yet");
  if (posted) { printf("  VRAM:            %u MB\n", memsize); vi_report_mc(d); }

  printf("  BIOS scratch:   ");
  for (int i = 0; i < 8; i++) {
    uint32_t v = 0;
    if ((r = rg_rreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 4u * (uint32_t)i, &v))) return r;
    printf(" %08x", v);
  }
  printf("\n");

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

static int read_memsize(rg_dev *d, uint32_t *memsize) {
  int r = map_regs(d);
  return r ? r : rg_rreg32(d, VI_REG_BAR, mmCONFIG_MEMSIZE, memsize);
}

static rg_atom *open_atom(rg_dev *d, FILE *trace, bool trace_ops) {
  uint8_t *bios = NULL;
  size_t len = 0;
  if (vi_vbios_read(d, &bios, &len)) return NULL;
  rg_atom_cfg cfg = {.reg_bar = VI_REG_BAR, .mm_index = mmMM_INDEX, .mm_data = mmMM_DATA,
                     .flush_reg = mmBIOS_SCRATCH_0, .trace = trace, .trace_ops = trace_ops};
  rg_atom *a = rg_atom_open(d, bios, len, &cfg);
  free(bios);
  return a;
}

static int vi_atom_info(rg_dev *d) {
  rg_atom *a = open_atom(d, NULL, false);
  if (!a) return -EIO;
  rg_atom_print_info(a);
  rg_atom_close(a);
  return 0;
}

/* Memory-controller state after posting. Only read once CONFIG_MEMSIZE says
 * the card is posted: on a cold card these blocks are not initialised. */
static void vi_report_mc(rg_dev *d) {
  struct { const char *name; uint32_t off; } regs[] = {
    {"MC_VM_FB_LOCATION", mmMC_VM_FB_LOCATION}, {"MC_VM_FB_OFFSET", mmMC_VM_FB_OFFSET},
    {"MC_SHARED_CHMAP", mmMC_SHARED_CHMAP}, {"MC_ARB_RAMCFG", mmMC_ARB_RAMCFG},
    {"MC_SEQ_MISC0", mmMC_SEQ_MISC0}, {"MC_SEQ_SUP_CNTL", mmMC_SEQ_SUP_CNTL},
    {"MC_SEQ_TRAIN_WAKEUP_CNTL", mmMC_SEQ_TRAIN_WAKEUP_CNTL}, {"HDP_NONSURFACE_BASE", mmHDP_NONSURFACE_BASE},
    {"HDP_NONSURFACE_INFO", mmHDP_NONSURFACE_INFO}, {"HDP_NONSURFACE_SIZE", mmHDP_NONSURFACE_SIZE},
    {"BIF_FB_EN", mmBIF_FB_EN},
  };
  for (size_t i = 0; i < sizeof regs / sizeof regs[0]; i++) {
    uint32_t v = 0;
    if (rg_rreg32(d, VI_REG_BAR, regs[i].off, &v)) return;
    printf("  %-26s 0x%08x", regs[i].name, v);
    if (regs[i].off == mmMC_VM_FB_LOCATION)
      printf("   VRAM at MC 0x%010llx-0x%010llx", (unsigned long long)(v & 0xFFFF) << 24,
             ((unsigned long long)(v >> 16) << 24) | 0xFFFFFF);
    if (regs[i].off == mmMC_SEQ_MISC0) printf("   memory type field (MT) = %u", v >> 28);
    if (regs[i].off == mmBIF_FB_EN)
      printf("   host access to VRAM: read %s, write %s", v & BIF_FB_EN__FB_READ_EN_MASK ? "on" : "off",
             v & BIF_FB_EN__FB_WRITE_EN_MASK ? "on" : "off");
    printf("\n");
  }
}

static int vi_post(rg_dev *d, const rg_post_opts *o) {
  uint32_t memsize;
  int r = read_memsize(d, &memsize);
  if (r) return r;
  bool posted = memsize != 0 && memsize != 0xFFFFFFFFu;
  if (posted && !o->force) {
    printf("already posted (CONFIG_MEMSIZE = %u MB); nothing to do. --force re-runs ASIC_Init.\n", memsize);
    return 0;
  }

  FILE *trace = NULL;
  if (o->trace_path && !(trace = fopen(o->trace_path, "w"))) { perror(o->trace_path); return -EIO; }
  rg_atom *a = open_atom(d, trace, o->trace_ops);
  if (!a) { if (trace) fclose(trace); return -EIO; }
  rg_atom_print_info(a);

  /* amdgpu_atombios_scratch_regs_init(): BIOS controls backlight, no mode switching, clear DPMS state */
  uint32_t s2, s6;
  if ((r = rg_rreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 8, &s2)) ||
      (r = rg_rreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 24, &s6))) goto out;
  s2 &= ~(ATOM_S2_VRI_BRIGHT_ENABLE | ATOM_S2_DEVICE_DPMS_STATE);
  s6 |= ATOM_S6_ACC_BLOCK_DISPLAY_SWITCH;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 8, s2)) ||
      (r = rg_wreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 24, s6))) goto out;

  printf("running ASIC_Init...\n");
  fflush(stdout);
  rg_atom_stats st;
  r = rg_atom_asic_init(a, &st);
  printf("ASIC_Init %s (%d): %.2f s, %llu reads, %llu writes, %llu indirect, %llu delays (%.1f ms requested)\n",
         r == 0 ? "returned OK" : "FAILED", r, st.seconds, (unsigned long long)st.reads,
         (unsigned long long)st.writes, (unsigned long long)st.indirect, (unsigned long long)st.delays,
         st.delay_us / 1000.0);
  if (r) goto out;

  if ((r = read_memsize(d, &memsize))) goto out;
  posted = memsize != 0 && memsize != 0xFFFFFFFFu;
  printf("posted:            %s (CONFIG_MEMSIZE = %u MB)\n", posted ? "yes" : "NO", memsize);
  if (posted) vi_report_mc(d);
  else r = -EIO;
out:
  rg_atom_close(a);
  if (trace) { fclose(trace); printf("trace: %s\n", o->trace_path); }
  return r;
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec / 1e9;
}

/* Fills/validates `n` dwords with a pattern unique to (seed, index). */
static uint32_t pat(uint32_t seed, size_t i) {
  uint32_t x = seed * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA77u;
  x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12;
  return x;
}

#define mmMM_INDEX_HI (0x6U * 4)

/* One dword of VRAM through the indirect window: MM_INDEX bit 31 selects the FB aperture. */
static int vram_mm_access(rg_dev *d, uint64_t pos, uint32_t wval, bool write, uint32_t *rval) {
  int r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmMM_INDEX, (uint32_t)pos | 0x80000000u))) return r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmMM_INDEX_HI, (uint32_t)(pos >> 31)))) return r;
  return write ? rg_wreg32(d, VI_REG_BAR, mmMM_DATA, wval) : rg_rreg32(d, VI_REG_BAR, mmMM_DATA, rval);
}

static int vi_vram_test(rg_dev *d, bool enable_fb) {
  uint32_t memsize, fb_en;
  int r = read_memsize(d, &memsize);
  if (r) return r;
  if (memsize == 0 || memsize == 0xFFFFFFFFu) { fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post` first\n"); return -EIO; }
  if ((r = rg_rreg32(d, VI_REG_BAR, mmBIF_FB_EN, &fb_en))) return r;
  uint32_t want = BIF_FB_EN__FB_READ_EN_MASK | BIF_FB_EN__FB_WRITE_EN_MASK;
  if ((fb_en & want) != want) {
    if (!enable_fb) {
      fprintf(stderr, "rawgpu: BIF_FB_EN = 0x%x: host access to VRAM is off. --enable-fb turns it on, as\n"
                      "        amdgpu's gmc_v8_0_mc_program() does.\n", fb_en);
      return -EACCES;
    }
    if ((r = rg_wreg32(d, VI_REG_BAR, mmBIF_FB_EN, fb_en | want))) return r;
    printf("BIF_FB_EN: 0x%x -> 0x%x\n", fb_en, fb_en | want);
  }

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
    double t0 = now_s();
    if ((r = d->ops->mmio_write(d, VI_VRAM_BAR, offs[k], buf, sizeof buf))) return r;
    uint32_t sync;
    if ((r = rg_rreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0, &sync))) return r; /* drain writes */
    wt += now_s() - t0;
  }
  for (size_t k = 0; k < 3; k++) { /* read back only after all writes: catches aliasing */
    if (offs[k] + sizeof got > bar) continue;
    double t0 = now_s();
    if ((r = d->ops->mmio_read(d, VI_VRAM_BAR, offs[k], got, sizeof got))) return r;
    rt += now_s() - t0;
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
    for (size_t i = 0; i < M; i++)
      if ((r = vram_mm_access(d, far[k] + 4 * i, pat((uint32_t)k + 11, i), true, NULL))) return r;
  for (size_t k = 0; k < 4; k++) {
    uint64_t base = k < 3 ? far[k] : offs[0];
    uint32_t seed = k < 3 ? (uint32_t)k + 11 : 1;
    if (base + 4 * M > vram) continue;
    int mism = 0;
    for (size_t i = 0; i < M; i++) {
      uint32_t v;
      if ((r = vram_mm_access(d, base + 4 * i, 0, false, &v))) return r;
      mism += v != pat(seed, i);
    }
    printf("  VRAM +%5llu MB (MM_INDEX%s): %s\n", (unsigned long long)(base >> 20),
           k == 3 ? ", re-reading the BAR-written region" : "", mism ? "MISMATCH" : "OK");
    bad += mism != 0;
  }
  return bad ? -EIO : 0;
}

const rg_backend rg_backend_amd_vi = {
  .name = "amd-vi",
  .reg_bar = VI_REG_BAR,
  .safe_reg = mmBIOS_SCRATCH_0,
  .probe = vi_probe,
  .vbios_read = vi_vbios_read,
  .atom_info = vi_atom_info,
  .post = vi_post,
  .vram_test = vi_vram_test,
};
