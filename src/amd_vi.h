/* Private to the amd-vi backend (src/amd_vi*.c). */
#ifndef RAWGPU_AMD_VI_H
#define RAWGPU_AMD_VI_H

#include "rg.h"
#include "amd_fw.h"
#include "amd_vi_regs.h"

#include <stdio.h>

#define VI_REG_BAR 5
#define VI_VRAM_BAR 0

/* Register access with a sticky error, like amdgpu's RREG32/WREG32 but able to
 * fail: after the first transport error every access is skipped and `err` says
 * why. Check it at the points where a failure matters. Offsets are amdgpu's
 * dword indices (mmFOO), not byte offsets. */
typedef struct {
  rg_dev *d;
  int err;
  FILE *trace; /* optional: one line per register access */
} vi_dev;

uint32_t vi_rreg(vi_dev *v, uint32_t idx);
void vi_wreg(vi_dev *v, uint32_t idx, uint32_t val);
uint32_t vi_rsmc(vi_dev *v, uint32_t addr); /* SMC indirect space (ixFOO), as vi_smc_rreg() */
void vi_wsmc(vi_dev *v, uint32_t addr, uint32_t val);
/* Poll until (reg & mask) == want. Returns 0, -ETIMEDOUT or the transport error. */
int vi_wait(vi_dev *v, uint32_t idx, uint32_t mask, uint32_t want, unsigned timeout_us);
void vi_udelay(unsigned us);
/* amdgpu_device_program_register_sequence(): {reg, and_mask, or_mask} triples */
void vi_program_sequence(vi_dev *v, const uint32_t *seq, size_t n);
/* vi_srbm_select(): which ME/pipe/queue/VMID the banked registers address */
void vi_srbm_select(vi_dev *v, uint32_t me, uint32_t pipe, uint32_t queue, uint32_t vmid);
int vi_open(vi_dev *v, rg_dev *d); /* maps the register BAR */
bool vi_posted(vi_dev *v, uint32_t *memsize_mb);
double vi_now(void);

#define REG_SET_FIELD(orig, reg, field, val) \
  (((orig) & ~reg##__##field##_MASK) | (reg##__##field##_MASK & ((uint32_t)(val) << reg##__##field##__SHIFT)))
#define REG_GET_FIELD(value, reg, field) (((value) & reg##__##field##_MASK) >> reg##__##field##__SHIFT)

/* ---- GMC v8: VRAM placement and the VMID0 GART (amd_vi_gmc.c) ---- */

/* VRAM layout rawgpu uses. Everything sits inside the 256 MB BAR so the CPU can
 * write it directly, above the first 16 MB that the VBIOS and VGA may use. */
#define VI_VRAM_GART_TABLE (16ull << 20) /* GART page table */
#define VI_VRAM_SCRATCH    (20ull << 20) /* system-aperture default page */
#define VI_VRAM_FREE       (32ull << 20) /* first byte free for users */

typedef struct {
  uint64_t vram_start, vram_size; /* MC address range of VRAM */
  uint64_t gart_start, gart_size; /* MC address range the GART translates */
  uint32_t next_page;             /* bump allocator, in 4 KB GART pages */
} vi_gart;

int vi_gmc_init(vi_dev *v, vi_gart *g, uint64_t dummy_page_bus);
/* Maps every 4 KB page of `b` into the GART; *gpu = MC address of byte 0. */
int vi_gart_map(vi_dev *v, vi_gart *g, const rg_dma *b, uint64_t *gpu);
void vi_hdp_flush(vi_dev *v);
void vi_hdp_invalidate(vi_dev *v);
void vi_gmc_report(vi_dev *v);
uint32_t vi_gmc_soft_reset(vi_dev *v);

/* ---- SDMA v3 (amd_vi_sdma.c) ---- */

/* Memory an engine uses for its ring or writeback: host DMA memory reached
 * through the GART (cpu != NULL), or VRAM inside BAR0 (cpu == NULL, bar = offset). */
typedef struct {
  void *cpu;
  uint64_t bar;
  uint64_t gpu; /* MC address */
} vi_mem;

typedef struct {
  vi_dev *v;
  int me;                /* engine 0 or 1 */
  uint32_t reg;          /* register offset of this engine (0 or 0x200) */
  vi_mem ring, wb;       /* wb: +0 read pointer (bytes), +16 fence, +32 ring-test slot */
  uint32_t *shadow;      /* CPU copy of a VRAM ring; == ring.cpu for a host ring */
  uint32_t ring_dw, wptr, committed; /* in dwords */
  uint32_t seq;
} vi_sdma;

int vi_sdma_load_ucode(vi_dev *v, int me, const amd_fw *fw);
int vi_sdma_soft_reset(vi_dev *v);
int vi_sdma_start(vi_sdma *s, vi_dev *v, int me, vi_mem ring, uint32_t ring_bytes, vi_mem wb);
void vi_sdma_stop(vi_dev *v, int me);
void vi_sdma_free(vi_sdma *s);
int vi_sdma_ring_test(vi_sdma *s, uint64_t target_gpu, volatile uint32_t *target_cpu, int64_t target_bar);
/* Queue a copy or fill (split into packets as needed) plus a fence; returns the
 * fence seq to wait for, or 0 on error. */
uint32_t vi_sdma_copy(vi_sdma *s, uint64_t dst, uint64_t src, uint64_t bytes);
uint32_t vi_sdma_fill(vi_sdma *s, uint64_t dst, uint32_t value, uint64_t bytes);
int vi_sdma_wait(vi_sdma *s, uint32_t seq, unsigned timeout_ms);
void vi_sdma_dump(vi_sdma *s);
void vi_sdma_report(vi_dev *v);

#endif
