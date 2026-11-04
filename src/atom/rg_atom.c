/* Glue between amdgpu's ATOM interpreter and an rg_dev.
 *
 * amdgpu hands the interpreter a `struct card_info` of register callbacks
 * (amdgpu_atombios.c: cail_reg_read/write). We provide the same callbacks
 * over the rawgpu transport, plus the delay and clock hooks rg_kcompat.h routes
 * to us.
 */
#include "atom.h" /* upstream, through the shim include path */
#include "rg_atom.h"

#include <time.h>

#include "rg_atom_names.inc"

struct rg_atom {
  struct card_info card; /* must stay first: callbacks cast card_info* back to rg_atom* */
  rg_dev *dev;
  rg_atom_cfg cfg;
  uint64_t bar_size;
  struct atom_context *ctx;
  uint8_t *bios;
  size_t bios_len;
  bool failed;
  rg_atom_stats st;
};

FILE *rg_atom_logf;
static rg_atom *g_active; /* the context whose delays must flush its device */

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + t.tv_nsec / 1e9;
}

unsigned long rg_atom_jiffies(void) { return (unsigned long)(now_s() * 1000.0); }

static void fail(rg_atom *a, const char *what) {
  if (!a->failed) fprintf(stderr, "rawgpu: ATOM %s failed; stopping all further hardware access\n", what);
  a->failed = true;
}

void rg_atom_udelay(unsigned long us) {
  rg_atom *a = g_active;
  if (a && !a->failed) {
    uint32_t v;
    if (rg_rreg32(a->dev, a->cfg.reg_bar, a->cfg.flush_reg, &v)) fail(a, "flush before delay");
    a->st.delays++;
    a->st.delay_us += us;
    if (a->cfg.trace) fprintf(a->cfg.trace, "D %lu\n", us);
  }
  struct timespec ts = {(time_t)(us / 1000000ul), (long)(us % 1000000ul) * 1000L};
  nanosleep(&ts, NULL);
}

static uint32_t cb_reg_read(struct card_info *info, uint32_t reg) {
  rg_atom *a = (rg_atom *)info;
  uint32_t v = 0xFFFFFFFFu;
  if (a->failed) return v;
  uint64_t off = (uint64_t)reg * 4;
  int r;
  if (off + 4 <= a->bar_size) {
    r = rg_rreg32(a->dev, a->cfg.reg_bar, (uint32_t)off, &v);
  } else { /* past the BAR: legacy MM_INDEX/MM_DATA window (bit 31 clear = register space) */
    a->st.indirect++;
    r = rg_wreg32(a->dev, a->cfg.reg_bar, a->cfg.mm_index, (uint32_t)off);
    if (!r) r = rg_rreg32(a->dev, a->cfg.reg_bar, a->cfg.mm_data, &v);
  }
  if (r) { fail(a, "register read"); return 0xFFFFFFFFu; }
  a->st.reads++;
  if (a->cfg.trace) fprintf(a->cfg.trace, "R %05x %08x\n", reg, v);
  return v;
}

static void cb_reg_write(struct card_info *info, uint32_t reg, uint32_t val) {
  rg_atom *a = (rg_atom *)info;
  if (a->failed) return;
  uint64_t off = (uint64_t)reg * 4;
  int r;
  if (off + 4 <= a->bar_size) {
    r = rg_wreg32(a->dev, a->cfg.reg_bar, (uint32_t)off, val);
  } else {
    a->st.indirect++;
    r = rg_wreg32(a->dev, a->cfg.reg_bar, a->cfg.mm_index, (uint32_t)off);
    if (!r) r = rg_wreg32(a->dev, a->cfg.reg_bar, a->cfg.mm_data, val);
  }
  if (r) { fail(a, "register write"); return; }
  a->st.writes++;
  if (a->cfg.trace) fprintf(a->cfg.trace, "W %05x %08x\n", reg, val);
}

/* amdgpu treats MC and PLL I/O spaces as a bug (cail_mc_read et al. BUG_ON). */
static uint32_t cb_unsupported_read(struct card_info *info, uint32_t reg) {
  fprintf(stderr, "rawgpu: ATOM table accessed MC/PLL space (reg 0x%x); amdgpu does not support this either\n", reg);
  fail((rg_atom *)info, "MC/PLL access");
  return 0;
}
static void cb_unsupported_write(struct card_info *info, uint32_t reg, uint32_t val) {
  (void)val;
  cb_unsupported_read(info, reg);
}

static uint16_t rd16(const rg_atom *a, size_t off) {
  return off + 2 <= a->bios_len ? (uint16_t)(a->bios[off] | a->bios[off + 1] << 8) : 0;
}
static uint32_t rd32(const rg_atom *a, size_t off) {
  return off + 4 <= a->bios_len ? (uint32_t)rd16(a, off) | (uint32_t)rd16(a, off + 2) << 16 : 0;
}

rg_atom *rg_atom_open(rg_dev *d, const uint8_t *bios, size_t len, const rg_atom_cfg *cfg) {
  rg_atom *a = calloc(1, sizeof *a);
  if (!a) return NULL;
  a->dev = d;
  a->cfg = *cfg;
  a->bios = malloc(len);
  if (!a->bios) { free(a); return NULL; }
  memcpy(a->bios, bios, len);
  a->bios_len = len;
  if (rg_bar_map(d, cfg->reg_bar, &a->bar_size)) { rg_atom_close(a); return NULL; }

  a->card.reg_read = cb_reg_read;
  a->card.reg_write = cb_reg_write;
  a->card.mc_read = cb_unsupported_read;
  a->card.mc_write = cb_unsupported_write;
  a->card.pll_read = cb_unsupported_read;
  a->card.pll_write = cb_unsupported_write;

  a->ctx = amdgpu_atom_parse(&a->card, a->bios, (uint32_t)len);
  if (!a->ctx) { fprintf(stderr, "rawgpu: amdgpu_atom_parse rejected this VBIOS\n"); rg_atom_close(a); return NULL; }
  return a;
}

void rg_atom_close(rg_atom *a) {
  if (!a) return;
  if (a->ctx) {
    free(a->ctx->scratch);
    amdgpu_atom_destroy(a->ctx);
  }
  free(a->bios);
  free(a);
}

int rg_atom_default_clocks(rg_atom *a, uint32_t *sclk, uint32_t *mclk) {
  uint16_t fwi = rd16(a, a->ctx->data_table + ATOM_DATA_FWI_PTR);
  if (!fwi) return -1;
  *sclk = rd32(a, fwi + ATOM_FWI_DEFSCLK_PTR);
  *mclk = rd32(a, fwi + ATOM_FWI_DEFMCLK_PTR);
  return 0;
}

int rg_atom_fw_vram_reserve(rg_atom *a, uint32_t *start, uint32_t *kb) {
  uint16_t off;
  if (!amdgpu_atom_parse_data_header(a->ctx, GetIndexIntoMasterTable(DATA, VRAM_UsageByFirmware),
                                     NULL, NULL, NULL, &off))
    return -1;
  const ATOM_VRAM_USAGE_BY_FIRMWARE *u = (const void *)(a->bios + off);
  *start = u->asFirmwareVramReserveInfo[0].ulStartAddrUsedByFirmware;
  *kb = u->asFirmwareVramReserveInfo[0].usFirmwareUseInKb;
  return 0;
}

void rg_atom_print_info(rg_atom *a) {
  struct atom_context *c = a->ctx;
  printf("  ATOM BIOS:       %s, %s\n", (char *)c->name, (char *)c->vbios_ver_str);
  printf("  firmware rev:    0x%08x\n", c->version);
  uint32_t s, m;
  if (!rg_atom_default_clocks(a, &s, &m))
    printf("  default clocks:  engine %u MHz, memory %u MHz (ASIC_Init parameters)\n", s / 100, m / 100);
  uint32_t start, kb;
  if (!rg_atom_fw_vram_reserve(a, &start, &kb))
    printf("  VBIOS VRAM use:  %u KB at 0x%08x\n", kb, start);

  int ncmd = (rd16(a, c->cmd_table) - 4) / 2, present = 0;
  for (int i = 0; i < ncmd; i++) present += rd16(a, c->cmd_table + 4 + 2u * (unsigned)i) != 0;
  printf("  command tables:  %d of %d present\n", present, ncmd);
  for (int i = 0; i < ncmd; i++) {
    uint16_t off = rd16(a, c->cmd_table + 4 + 2u * (unsigned)i);
    if (!off) continue;
    uint8_t frev = 0, crev = 0;
    amdgpu_atom_parse_cmd_header(c, i, &frev, &crev);
    const char *nm = i < (int)ARRAY_SIZE(rg_atom_cmd_names) ? rg_atom_cmd_names[i] : "?";
    printf("    [%2d] %-36s @0x%04x  %5u bytes  v%u.%u\n", i, nm, off, rd16(a, off), frev, crev);
  }
}

int rg_atom_asic_init(rg_atom *a, rg_atom_stats *st) {
  /* ATOM_ARG_FB scratch, sized like amdgpu_atombios_allocate_fb_scratch() */
  uint32_t start, kb;
  size_t bytes = 20 * 1024;
  if (!rg_atom_fw_vram_reserve(a, &start, &kb) && kb) bytes = (size_t)kb * 1024;
  free(a->ctx->scratch);
  a->ctx->scratch = calloc(1, bytes);
  if (!a->ctx->scratch) return -ENOMEM;
  a->ctx->scratch_size_bytes = (int)bytes;

  if (a->cfg.trace_ops && a->cfg.trace) { amdgpu_atom_debug = 1; rg_atom_logf = a->cfg.trace; }
  memset(&a->st, 0, sizeof a->st);
  g_active = a;
  double t0 = now_s();
  int r = amdgpu_atom_asic_init(a->ctx);
  uint32_t v;
  if (!a->failed && rg_rreg32(a->dev, a->cfg.reg_bar, a->cfg.flush_reg, &v)) fail(a, "final flush");
  a->st.seconds = now_s() - t0;
  g_active = NULL;
  amdgpu_atom_debug = 0;
  rg_atom_logf = NULL;
  if (a->cfg.trace) fflush(a->cfg.trace);
  if (st) *st = a->st;
  return a->failed ? -EIO : r;
}
