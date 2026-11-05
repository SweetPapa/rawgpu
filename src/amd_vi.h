/* Private to the amd-vi backend (src/amd_vi*.c). */
#ifndef RAWGPU_AMD_VI_H
#define RAWGPU_AMD_VI_H

#include "rg.h"
#include "amd_fw.h"
#include "amd_kernel.h"
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
#define VI_VRAM_SMU_TOC    (24ull << 20) /* SMU_DRAMData_TOC: where each engine's microcode is */
#define VI_VRAM_SMU_BUF    (25ull << 20) /* the SMU's own 800 KB work buffer */
#define VI_VRAM_UCODE      (26ull << 20) /* engine microcode images, as the SMU loads them */
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
/* CPU writes to VRAM through BAR0 (offset < BAR size). Follow with vi_hdp_flush()
 * before an engine reads the data. */
int vi_vram_write(vi_dev *v, uint64_t off, const void *src, size_t len);
/* MC sequencer firmware, as gmc_v8_0_polaris_mc_load_microcode() loads it */
const char *vi_mc_fw_name(const rg_dev *d);
uint32_t vi_mc_ucode_version(vi_dev *v);
int vi_mc_load_ucode(vi_dev *v, const amd_fw *fw, double *train_s);
int vi_vram_zero(vi_dev *v, uint64_t off, size_t len);

/* ---- SMU7: SMC firmware and SMU-loaded engine microcode (amd_vi_smu.c) ---- */

/* UcodeLoadStatus bits (smu_ucode_xfer_vi.h UCODE_ID_*_MASK) */
#define VI_UCODE_SDMA0  0x002u
#define VI_UCODE_SDMA1  0x004u
#define VI_UCODE_CP_CE  0x008u
#define VI_UCODE_CP_PFP 0x010u
#define VI_UCODE_CP_ME  0x020u
#define VI_UCODE_CP_MEC 0x040u
#define VI_UCODE_RLC_G  0x400u

/* The engine images, placed in VRAM as amdgpu_ucode_init_bo() lays them out:
 * SDMA0, SDMA1, CE, PFP, ME, MEC (+ its jump table), RLC. */
#define VI_UCODE_IMAGES 7
typedef struct {
  amd_fw fw[VI_UCODE_IMAGES];
  uint64_t off[VI_UCODE_IMAGES]; /* VRAM offset of each image */
  uint64_t jt_off, end;           /* the MEC jump table's copy; first byte past the images */
} vi_ucode_set;
int vi_ucode_place(vi_dev *v, vi_ucode_set *u, const char *fw_dir);
void vi_ucode_free(vi_ucode_set *u);

bool vi_smc_running(vi_dev *v);                       /* smu7_is_smc_ram_running() */
int vi_smu_ucode_status(vi_dev *v, uint32_t *status); /* UcodeLoadStatus; -ENODEV if the SMC firmware is not up */
int vi_temp_c(vi_dev *v);                             /* GPU temperature in degrees C */
int vi_smu_clocks(vi_dev *v, uint32_t *sclk_mhz, uint32_t *mclk_mhz); /* as the SMC reports them */
void vi_smu_report(vi_dev *v);
const char *vi_smc_fw_name(const rg_dev *d, bool hard_key); /* the image amdgpu would load */
bool vi_smc_is_kicker(const rg_dev *d);
struct rg_atom;
struct rg_atom *vi_atom_open(rg_dev *d, FILE *trace, bool trace_ops); /* the VBIOS interpreter, as `post` uses it */

/* ---- SDMA v3 (amd_vi_sdma.c) ---- */

/* Memory an engine uses for its ring or writeback: host DMA memory reached
 * through the GART (cpu != NULL), or VRAM inside BAR0 (cpu == NULL, bar = offset). */
typedef struct {
  void *cpu;
  uint64_t bar;
  uint64_t gpu; /* MC address */
} vi_mem;
/* 32-bit read / bulk write of a vi_mem. A VRAM read invalidates HDP first, so
 * it sees what an engine wrote. Follow VRAM writes with vi_hdp_flush(). */
uint32_t vi_mem_read32(vi_dev *v, const vi_mem *m, uint32_t off);
void vi_mem_write(vi_dev *v, const vi_mem *m, uint64_t off, const void *src, size_t len);

typedef struct {
  vi_dev *v;
  int me;                /* engine 0 or 1 */
  uint32_t reg;          /* register offset of this engine (0 or 0x200) */
  vi_mem ring, wb;       /* wb: +0 read pointer (bytes), +16 fence, +32 ring-test slot */
  uint32_t *shadow;      /* CPU copy of a VRAM ring; == ring.cpu for a host ring */
  uint32_t ring_dw, wptr, committed; /* in dwords */
  uint32_t seq;
} vi_sdma;

/* Every SMU-loaded image ends in a 20-byte digest (smu_ucode_xfer_vi.h
 * SMU_DIGEST_SIZE_BYTES). The SMU checks it and does not store it in the
 * engine's RAM (docs/LOG.md, 2026-11-04). */
#define VI_SMU_DIGEST_DW 5u

/* Reads a halted engine's microcode RAM back and compares it with the image,
 * less its digest. Returns the number of mismatched dwords (first one in
 * *first_bad), or a negative error. Restores UCODE_ADDR. */
int vi_sdma_check_ucode(vi_dev *v, int me, const amd_fw *fw, uint32_t *first_bad);
int vi_sdma_soft_reset(vi_dev *v);
/* Golden registers, both engines halted, and engine `me`'s microcode checked (Polaris 10). */
int vi_sdma_init(vi_dev *v, int me, const char *fw_dir);
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

/* ---- GFX v8 compute: RLC, MEC and one compute queue (amd_vi_gfx.c) ---- */

/* Writeback block of a compute queue */
#define VI_CQ_WB_RPTR  0  /* the CP reports its read pointer here, in dwords */
#define VI_CQ_WB_WPTR  8  /* copy of the write pointer (CP_PQ_WPTR_POLL_CNTL is off; the doorbell carries it) */
#define VI_CQ_WB_FENCE 16 /* RELEASE_MEM fence */

typedef struct {
  vi_dev *v;
  uint32_t me, pipe, queue, doorbell;
  vi_mem ring, wb, mqd, eop;
  uint32_t *shadow;                  /* CPU copy of a VRAM ring; == ring.cpu for a host ring */
  uint32_t ring_dw, wptr, committed; /* in dwords */
  uint32_t seq;
} vi_cq;

/* gfx_v8_0_hw_init() for compute: golden registers, VMID0 shader memory
 * config, RLC resume, doorbells. Needs the SMU-loaded RLC and MEC microcode. */
int vi_gfx_compute_init(vi_dev *v);
/* gfx_v8_0_set_safe_mode() / unset_safe_mode(); no-op if the RLC is not running */
void vi_rlc_safe_mode(vi_dev *v, bool enter);
/* rawgpu's per-power-cycle state, in reserved VRAM: GFX brought up already?
 * post, mc-load and gfx-reset clear it. */
#define VI_VRAM_STATE (21ull << 20)
void vi_state_clear(vi_dev *v);
int vi_gfx_cu_count(vi_dev *v); /* active compute units */
/* Loads ME1 pipe 0 queue 0 from an MQD and activates it. All four areas
 * must be 256-byte aligned; eop is GFX8_MEC_HPD_SIZE (4 KB). */
int vi_cq_start(vi_cq *q, vi_dev *v, vi_mem ring, uint32_t ring_bytes, vi_mem wb, vi_mem mqd, vi_mem eop);
void vi_cq_stop(vi_cq *q); /* drains and deactivates the HQD */
int vi_cq_ring_test(vi_cq *q);
/* WRITE_DATA of one dword plus a fence; returns the fence seq, or 0 on error */
uint32_t vi_cq_write(vi_cq *q, uint64_t gpu, uint32_t value);
int vi_cq_wait(vi_cq *q, uint32_t seq, unsigned timeout_ms);
void vi_cq_dump(vi_cq *q);
#define VI_LDS_MAX (64u << 10) /* LDS per compute unit */
/* Can vi_cq_dispatch() run this kernel? (gfx803, no scratch, LDS within a CU,
 * only the private-segment and kernarg user SGPRs.) Prints why not. */
int vi_kernel_check(const amd_kernel *k);
/* Dispatches groups[0..2] work-groups of wg[0..2] work-items, code at MC
 * address code_gpu (256-byte aligned), then a fence; returns the fence seq or 0. */
uint32_t vi_cq_dispatch(vi_cq *q, const amd_kernel *k, uint64_t code_gpu, uint64_t kernarg_gpu, const uint32_t wg[3],
                        const uint32_t groups[3]);

#endif
