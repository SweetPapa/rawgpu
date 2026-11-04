/* SDMA v3 (VI system DMA engines) and `rawgpu sdma-test`.
 *
 * Bring-up follows Linux amdgpu's sdma_v3_0_gfx_resume().
 *
 * Microcode: vi_sdma_load_ucode() is amdgpu's old direct load
 * (sdma_v3_0_load_microcode(), Linux <= 4.14). Polaris ignores it: after a
 * direct load the microcode RAM still reads back its power-on contents
 * (docs/LOG.md, 2026-11-04). On Polaris the SMC has to load the image, which
 * rawgpu does not do yet, so sdma-test cannot pass there until it does.
 *
 * The ring and the writeback block (read pointer, fence, test slot) can live
 * in VRAM, written through BAR0, or in host DMA memory reached through the
 * GART. There are no interrupts: TinyGPU has no MSI path, so fences are polled.
 */
#include "amd_vi.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* tonga_sdma_pkt_open.h */
#define SDMA_OP_NOP 0
#define SDMA_OP_COPY 1
#define SDMA_OP_WRITE 2
#define SDMA_OP_FENCE 5
#define SDMA_OP_CONST_FILL 11
#define SDMA_SUBOP_COPY_LINEAR 0
#define SDMA_SUBOP_WRITE_LINEAR 0
#define SDMA_PKT_HEADER(op, sub) (((op) & 0xFFu) | ((sub) & 0xFFu) << 8)

#define SDMA1_OFFSET 0x200u          /* SDMA1_REGISTER_OFFSET */
#define SDMA_COPY_MAX 0x3fffe0u      /* sdma_v3_0_buffer_funcs.copy_max_bytes */
#define SDMA_CHUNK 0x200000u         /* rawgpu splits transfers into 2 MB packets */
#define SDMA_ALIGN_DW 16u            /* sdma_v3_0_ring_funcs.align_mask + 1 */
#define WB_RPTR 0
#define WB_FENCE 16
#define WB_TEST 32

/* sdma_v3_0.c golden_settings_polaris10_a11 */
static const uint32_t golden_polaris10[] = {
  mmSDMA0_CHICKEN_BITS, 0xfc910007, 0x00810007,
  mmSDMA0_CLK_CTRL, 0xff000fff, 0x00000000,
  mmSDMA0_GFX_IB_CNTL, 0x800f0111, 0x00000100,
  mmSDMA0_RLC0_IB_CNTL, 0x800f0111, 0x00000100,
  mmSDMA0_RLC1_IB_CNTL, 0x800f0111, 0x00000100,
  mmSDMA0_CHICKEN_BITS + SDMA1_OFFSET, 0xfc910007, 0x00810007,
  mmSDMA0_CLK_CTRL + SDMA1_OFFSET, 0xff000fff, 0x00000000,
  mmSDMA0_GFX_IB_CNTL + SDMA1_OFFSET, 0x800f0111, 0x00000100,
  mmSDMA0_RLC0_IB_CNTL + SDMA1_OFFSET, 0x800f0111, 0x00000100,
  mmSDMA0_RLC1_IB_CNTL + SDMA1_OFFSET, 0x800f0111, 0x00000100,
};
_Static_assert(mmSDMA1_UCODE_ADDR == mmSDMA0_UCODE_ADDR + SDMA1_OFFSET, "SDMA1 register offset");

static uint32_t eng(int me) { return me ? SDMA1_OFFSET : 0; }

/* ---- memory the engine shares with us ---- */

static uint32_t mem_read32(vi_dev *v, const vi_mem *m, uint32_t off) {
  if (m->cpu) return *(volatile uint32_t *)((uint8_t *)m->cpu + off);
  uint32_t x = 0xFFFFFFFFu;
  vi_hdp_invalidate(v); /* the engine's write may still sit behind HDP's read cache */
  if (!v->err) v->err = v->d->ops->mmio_read(v->d, VI_VRAM_BAR, m->bar + off, &x, 4);
  return x;
}

static void mem_write(vi_dev *v, const vi_mem *m, uint64_t off, const void *src, size_t len) {
  if (m->cpu) { memcpy((uint8_t *)m->cpu + off, src, len); return; }
  if (!v->err) v->err = v->d->ops->mmio_write(v->d, VI_VRAM_BAR, m->bar + off, src, len);
}

/* ---- engine control ---- */

/* sdma_v3_0_enable(false) + ctx_switch_enable(false): stop the rings, then halt the engine. */
void vi_sdma_stop(vi_dev *v, int me) {
  uint32_t o = eng(me);
  uint32_t x = vi_rreg(v, mmSDMA0_GFX_RB_CNTL + o);
  vi_wreg(v, mmSDMA0_GFX_RB_CNTL + o, REG_SET_FIELD(x, SDMA0_GFX_RB_CNTL, RB_ENABLE, 0));
  x = vi_rreg(v, mmSDMA0_GFX_IB_CNTL + o);
  vi_wreg(v, mmSDMA0_GFX_IB_CNTL + o, REG_SET_FIELD(x, SDMA0_GFX_IB_CNTL, IB_ENABLE, 0));
  x = vi_rreg(v, mmSDMA0_CNTL + o);
  x = REG_SET_FIELD(x, SDMA0_CNTL, AUTO_CTXSW_ENABLE, 0);
  vi_wreg(v, mmSDMA0_CNTL + o, REG_SET_FIELD(x, SDMA0_CNTL, ATC_L1_ENABLE, 1));
  x = vi_rreg(v, mmSDMA0_F32_CNTL + o);
  vi_wreg(v, mmSDMA0_F32_CNTL + o, REG_SET_FIELD(x, SDMA0_F32_CNTL, HALT, 1));
}

/* sdma_v3_0_soft_reset(): pulse the SRBM reset of both engines. */
int vi_sdma_soft_reset(vi_dev *v) {
  uint32_t bits = SRBM_SOFT_RESET__SOFT_RESET_SDMA_MASK | SRBM_SOFT_RESET__SOFT_RESET_SDMA1_MASK;
  uint32_t tmp = vi_rreg(v, mmSRBM_SOFT_RESET) | bits;
  vi_wreg(v, mmSRBM_SOFT_RESET, tmp);
  vi_rreg(v, mmSRBM_SOFT_RESET);
  vi_udelay(50);
  vi_wreg(v, mmSRBM_SOFT_RESET, tmp & ~bits);
  vi_rreg(v, mmSRBM_SOFT_RESET);
  vi_udelay(50);
  return v->err;
}

int vi_sdma_load_ucode(vi_dev *v, int me, const amd_fw *fw) {
  uint32_t o = eng(me);
  vi_wreg(v, mmSDMA0_UCODE_ADDR + o, 0);
  for (uint32_t i = 0; i < fw->ucode_dw; i++) vi_wreg(v, mmSDMA0_UCODE_DATA + o, fw->ucode[i]);
  vi_wreg(v, mmSDMA0_UCODE_ADDR + o, fw->ucode_version);
  return v->err;
}

int vi_sdma_start(vi_sdma *s, vi_dev *v, int me, vi_mem ring, uint32_t ring_bytes, vi_mem wb) {
  if (ring_bytes & (ring_bytes - 1) || ring_bytes < 4096 || ring.gpu & 0xFF) return -EINVAL;
  memset(s, 0, sizeof *s);
  s->v = v; s->me = me; s->reg = eng(me);
  s->ring = ring; s->wb = wb; s->ring_dw = ring_bytes / 4;
  s->shadow = ring.cpu ? ring.cpu : calloc(1, ring_bytes);
  if (!s->shadow) return -ENOMEM;
  static const uint8_t zero[64];
  memset(s->shadow, 0, ring_bytes);
  if (!ring.cpu) mem_write(v, &ring, 0, s->shadow, ring_bytes);
  mem_write(v, &wb, 0, zero, sizeof zero);
  vi_hdp_flush(v);
  uint32_t o = s->reg;

  for (uint32_t vmid = 0; vmid < 16; vmid++) {
    vi_srbm_select(v, 0, 0, 0, vmid);
    vi_wreg(v, mmSDMA0_GFX_VIRTUAL_ADDR + o, 0);
    vi_wreg(v, mmSDMA0_GFX_APE1_CNTL + o, 0);
  }
  vi_srbm_select(v, 0, 0, 0, 0);
  vi_wreg(v, mmSDMA0_TILING_CONFIG + o, vi_rreg(v, mmGB_ADDR_CONFIG) & 0x70);
  vi_wreg(v, mmSDMA0_SEM_WAIT_FAIL_TIMER_CNTL + o, 0);

  uint32_t rb_bufsz = 0;
  while ((1u << rb_bufsz) < s->ring_dw) rb_bufsz++;
  uint32_t rb_cntl = vi_rreg(v, mmSDMA0_GFX_RB_CNTL + o);
  rb_cntl = REG_SET_FIELD(rb_cntl, SDMA0_GFX_RB_CNTL, RB_SIZE, rb_bufsz);
  vi_wreg(v, mmSDMA0_GFX_RB_CNTL + o, rb_cntl);

  vi_wreg(v, mmSDMA0_GFX_RB_RPTR + o, 0);
  vi_wreg(v, mmSDMA0_GFX_RB_WPTR + o, 0);
  vi_wreg(v, mmSDMA0_GFX_IB_RPTR + o, 0);
  vi_wreg(v, mmSDMA0_GFX_IB_OFFSET + o, 0);
  uint64_t rptr_gpu = wb.gpu + WB_RPTR;
  vi_wreg(v, mmSDMA0_GFX_RB_RPTR_ADDR_HI + o, (uint32_t)(rptr_gpu >> 32));
  vi_wreg(v, mmSDMA0_GFX_RB_RPTR_ADDR_LO + o, (uint32_t)rptr_gpu & 0xFFFFFFFCu);
  rb_cntl = REG_SET_FIELD(rb_cntl, SDMA0_GFX_RB_CNTL, RPTR_WRITEBACK_ENABLE, 1);
  vi_wreg(v, mmSDMA0_GFX_RB_BASE + o, (uint32_t)(ring.gpu >> 8));
  vi_wreg(v, mmSDMA0_GFX_RB_BASE_HI + o, (uint32_t)(ring.gpu >> 40));

  uint32_t db = vi_rreg(v, mmSDMA0_GFX_DOORBELL + o);
  vi_wreg(v, mmSDMA0_GFX_DOORBELL + o, REG_SET_FIELD(db, SDMA0_GFX_DOORBELL, ENABLE, 0));
  vi_wreg(v, mmSDMA0_GFX_RB_WPTR_POLL_ADDR_LO + o, 0);
  vi_wreg(v, mmSDMA0_GFX_RB_WPTR_POLL_ADDR_HI + o, 0);
  uint32_t poll = vi_rreg(v, mmSDMA0_GFX_RB_WPTR_POLL_CNTL + o);
  vi_wreg(v, mmSDMA0_GFX_RB_WPTR_POLL_CNTL + o, REG_SET_FIELD(poll, SDMA0_GFX_RB_WPTR_POLL_CNTL, ENABLE, 0));

  rb_cntl = REG_SET_FIELD(rb_cntl, SDMA0_GFX_RB_CNTL, RB_ENABLE, 1);
  vi_wreg(v, mmSDMA0_GFX_RB_CNTL + o, rb_cntl);
  uint32_t ib = vi_rreg(v, mmSDMA0_GFX_IB_CNTL + o);
  vi_wreg(v, mmSDMA0_GFX_IB_CNTL + o, REG_SET_FIELD(ib, SDMA0_GFX_IB_CNTL, IB_ENABLE, 1));

  uint32_t f32 = vi_rreg(v, mmSDMA0_F32_CNTL + o); /* unhalt */
  vi_wreg(v, mmSDMA0_F32_CNTL + o, REG_SET_FIELD(f32, SDMA0_F32_CNTL, HALT, 0));
  return v->err;
}

void vi_sdma_free(vi_sdma *s) {
  if (s->shadow && s->shadow != s->ring.cpu) free(s->shadow);
  s->shadow = NULL;
}

/* ---- ring ---- */

static uint32_t ring_free(vi_sdma *s) {
  uint32_t rptr = (mem_read32(s->v, &s->wb, WB_RPTR) >> 2) & (s->ring_dw - 1);
  return (rptr - s->wptr - 1) & (s->ring_dw - 1);
}

/* Waits until `n` dwords fit. The engine reports progress through the rptr writeback. */
static int ring_reserve(vi_sdma *s, uint32_t n) {
  double end = vi_now() + 2.0;
  while (ring_free(s) < n + SDMA_ALIGN_DW) {
    if (s->v->err) return s->v->err;
    if (vi_now() > end) { fprintf(stderr, "rawgpu: SDMA%d ring stalled\n", s->me); return -ETIMEDOUT; }
  }
  return 0;
}

static void ring_put(vi_sdma *s, uint32_t dw) {
  s->shadow[s->wptr] = dw;
  s->wptr = (s->wptr + 1) & (s->ring_dw - 1);
}

/* amdgpu_ring_commit(): pad with NOPs to the alignment, copy a VRAM ring's new
 * dwords out through BAR0, then publish the write pointer. */
static void ring_commit(vi_sdma *s) {
  while (s->wptr & (SDMA_ALIGN_DW - 1)) ring_put(s, SDMA_PKT_HEADER(SDMA_OP_NOP, 0));
  if (!s->ring.cpu) {
    uint32_t a = s->committed, b = s->wptr;
    if (b < a) { mem_write(s->v, &s->ring, (uint64_t)a * 4, s->shadow + a, (size_t)(s->ring_dw - a) * 4); a = 0; }
    if (b > a) mem_write(s->v, &s->ring, (uint64_t)a * 4, s->shadow + a, (size_t)(b - a) * 4);
    vi_hdp_flush(s->v);
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST); /* ring contents before the write pointer */
  vi_wreg(s->v, mmSDMA0_GFX_RB_WPTR + s->reg, s->wptr << 2);
  s->committed = s->wptr;
}

/* sdma_v3_0_ring_emit_fence() minus the trap (no interrupts). */
static uint32_t emit_fence(vi_sdma *s) {
  uint32_t seq = ++s->seq;
  uint64_t a = s->wb.gpu + WB_FENCE;
  ring_put(s, SDMA_PKT_HEADER(SDMA_OP_NOP, 0)); /* "hang workaround" in amdgpu */
  ring_put(s, SDMA_PKT_HEADER(SDMA_OP_FENCE, 0));
  ring_put(s, (uint32_t)a);
  ring_put(s, (uint32_t)(a >> 32));
  ring_put(s, seq);
  return seq;
}

int vi_sdma_wait(vi_sdma *s, uint32_t seq, unsigned timeout_ms) {
  double end = vi_now() + timeout_ms / 1e3;
  uint32_t f;
  while ((int32_t)((f = mem_read32(s->v, &s->wb, WB_FENCE)) - seq) < 0) {
    if (s->v->err) return s->v->err;
    if (vi_now() > end) {
      fprintf(stderr, "rawgpu: SDMA%d fence %u timed out (fence reads %u, rptr %u, wptr %u)\n", s->me, seq, f,
              mem_read32(s->v, &s->wb, WB_RPTR) >> 2, s->wptr);
      return -ETIMEDOUT;
    }
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  return 0;
}

/* sdma_v3_0_ring_test_ring(): one WRITE_LINEAR packet to `target`, then poll it
 * (through target_cpu for host memory, or at target_bar in VRAM). */
int vi_sdma_ring_test(vi_sdma *s, uint64_t target_gpu, volatile uint32_t *target_cpu, int64_t target_bar) {
  vi_mem t = {.cpu = (void *)target_cpu, .bar = target_bar < 0 ? 0 : (uint64_t)target_bar, .gpu = target_gpu};
  uint32_t init = 0xCAFEDEADu;
  mem_write(s->v, &t, 0, &init, 4);
  vi_hdp_flush(s->v);
  int r = ring_reserve(s, 5);
  if (r) return r;
  ring_put(s, SDMA_PKT_HEADER(SDMA_OP_WRITE, SDMA_SUBOP_WRITE_LINEAR));
  ring_put(s, (uint32_t)target_gpu);
  ring_put(s, (uint32_t)(target_gpu >> 32));
  ring_put(s, 1); /* SDMA_PKT_WRITE_UNTILED_DW_3_COUNT(1) */
  ring_put(s, 0xDEADBEEFu);
  ring_commit(s);
  double end = vi_now() + 0.5;
  uint32_t x;
  while ((x = mem_read32(s->v, &t, 0)) != 0xDEADBEEFu) {
    if (s->v->err) return s->v->err;
    if (vi_now() > end) {
      fprintf(stderr, "rawgpu: SDMA%d write test: target still 0x%08x after 500 ms (rptr %u, wptr %u)\n", s->me, x,
              mem_read32(s->v, &s->wb, WB_RPTR) >> 2, s->wptr);
      return -ETIMEDOUT;
    }
  }
  return 0;
}

uint32_t vi_sdma_copy(vi_sdma *s, uint64_t dst, uint64_t src, uint64_t bytes) {
  while (bytes) {
    uint32_t n = bytes > SDMA_COPY_MAX ? SDMA_CHUNK : (uint32_t)bytes;
    if (ring_reserve(s, 7 + 5)) return 0;
    ring_put(s, SDMA_PKT_HEADER(SDMA_OP_COPY, SDMA_SUBOP_COPY_LINEAR)); /* sdma_v3_0_emit_copy_buffer() */
    ring_put(s, n);
    ring_put(s, 0); /* no endian swap */
    ring_put(s, (uint32_t)src);
    ring_put(s, (uint32_t)(src >> 32));
    ring_put(s, (uint32_t)dst);
    ring_put(s, (uint32_t)(dst >> 32));
    src += n; dst += n; bytes -= n;
  }
  uint32_t seq = emit_fence(s);
  ring_commit(s);
  return s->v->err ? 0 : seq;
}

uint32_t vi_sdma_fill(vi_sdma *s, uint64_t dst, uint32_t value, uint64_t bytes) {
  while (bytes) {
    uint32_t n = bytes > SDMA_COPY_MAX ? SDMA_CHUNK : (uint32_t)bytes;
    if (ring_reserve(s, 5 + 5)) return 0;
    ring_put(s, SDMA_PKT_HEADER(SDMA_OP_CONST_FILL, 0)); /* sdma_v3_0_emit_fill_buffer() */
    ring_put(s, (uint32_t)dst);
    ring_put(s, (uint32_t)(dst >> 32));
    ring_put(s, value);
    ring_put(s, n);
    dst += n; bytes -= n;
  }
  uint32_t seq = emit_fence(s);
  ring_commit(s);
  return s->v->err ? 0 : seq;
}

/* ---- diagnostics ---- */

void vi_sdma_report(vi_dev *v) {
  for (int me = 0; me < 2; me++) {
    uint32_t o = eng(me);
    uint32_t f32 = vi_rreg(v, mmSDMA0_F32_CNTL + o), st = vi_rreg(v, mmSDMA0_STATUS_REG + o);
    uint32_t rb = vi_rreg(v, mmSDMA0_GFX_RB_CNTL + o);
    printf("SDMA%d                  F32_CNTL 0x%08x (%s)  STATUS 0x%08x (%s)  RB_CNTL 0x%08x (ring %s)\n", me, f32,
           REG_GET_FIELD(f32, SDMA0_F32_CNTL, HALT) ? "halted" : "running", st,
           REG_GET_FIELD(st, SDMA0_STATUS_REG, IDLE) ? "idle" : "busy", rb,
           REG_GET_FIELD(rb, SDMA0_GFX_RB_CNTL, RB_ENABLE) ? "on" : "off");
    if (REG_GET_FIELD(rb, SDMA0_GFX_RB_CNTL, RB_ENABLE))
      printf("                       RB_BASE 0x%08x%08x  RPTR 0x%x  WPTR 0x%x\n", vi_rreg(v, mmSDMA0_GFX_RB_BASE_HI + o),
             vi_rreg(v, mmSDMA0_GFX_RB_BASE + o), vi_rreg(v, mmSDMA0_GFX_RB_RPTR + o), vi_rreg(v, mmSDMA0_GFX_RB_WPTR + o));
  }
}

/* What the engine, the VM and the PCIe link say right after a failure, before anything is halted. */
void vi_sdma_dump(vi_sdma *s) {
  vi_dev *v = s->v;
  uint32_t o = s->reg;
  int e = v->err;
  v->err = 0;
  printf("SDMA%d state:  F32_CNTL 0x%08x  STATUS 0x%08x  STATUS1 0x%08x  CONTEXT_STATUS 0x%08x\n", s->me,
         vi_rreg(v, mmSDMA0_F32_CNTL + o), vi_rreg(v, mmSDMA0_STATUS_REG + o), vi_rreg(v, mmSDMA0_STATUS1_REG + o),
         vi_rreg(v, mmSDMA0_GFX_CONTEXT_STATUS + o));
  printf("              RB_BASE 0x%08x_%08x  RB_RPTR 0x%x  RB_WPTR 0x%x  RPTR_ADDR 0x%08x_%08x\n",
         vi_rreg(v, mmSDMA0_GFX_RB_BASE_HI + o), vi_rreg(v, mmSDMA0_GFX_RB_BASE + o), vi_rreg(v, mmSDMA0_GFX_RB_RPTR + o),
         vi_rreg(v, mmSDMA0_GFX_RB_WPTR + o), vi_rreg(v, mmSDMA0_GFX_RB_RPTR_ADDR_HI + o),
         vi_rreg(v, mmSDMA0_GFX_RB_RPTR_ADDR_LO + o));
  printf("              SRBM_STATUS 0x%08x  SRBM_STATUS2 0x%08x  VM_L2_STATUS 0x%08x\n", vi_rreg(v, mmSRBM_STATUS),
         vi_rreg(v, mmSRBM_STATUS2), vi_rreg(v, mmVM_L2_STATUS));
  uint32_t fa = vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_ADDR);
  printf("              VM0 fault: status 0x%08x addr MC 0x%010llx client 0x%08x\n",
         vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_STATUS), (unsigned long long)fa << 12,
         vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_MCCLIENT));
  rg_pci_err pe;
  if (!rg_pci_errors(v->d, &pe)) rg_pci_print_errors(&pe);
  if (!v->err) v->err = e;
}

/* ---- rawgpu sdma-test ---- */

static uint32_t pat(uint32_t seed, size_t i) {
  uint32_t x = seed * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA77u;
  x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12;
  return x;
}

static double gbps(uint64_t bytes, double s) { return bytes / s / 1e9; }

/* Runs one transfer `reps` times and returns the best time. */
static int timed(vi_sdma *s, const char *what, bool fill, uint64_t dst, uint64_t src, uint64_t bytes, int reps,
                 double *best) {
  *best = 1e9;
  for (int i = 0; i < reps; i++) {
    double t0 = vi_now();
    uint32_t seq = fill ? vi_sdma_fill(s, dst, (uint32_t)src, bytes) : vi_sdma_copy(s, dst, src, bytes);
    int r = seq ? vi_sdma_wait(s, seq, 10000) : (s->v->err ? s->v->err : -EIO);
    if (r) { fprintf(stderr, "rawgpu: %s failed\n", what); return r; }
    double t = vi_now() - t0;
    if (t < *best) *best = t;
  }
  return 0;
}

static size_t count_bad(const uint32_t *got, uint32_t seed, size_t n) {
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += got[i] != pat(seed, i);
  return bad;
}

#define PHASE(fmt, ...) printf("[%d] " fmt "\n", ++phase, __VA_ARGS__)

int vi_cmd_sdma_test(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  uint64_t mb = 64;
  const char *fw_dir = NULL, *trace_path = NULL;
  bool host_ring = false, vram_only = false;
  int me = 0;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--fw") && i + 1 < argc) fw_dir = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!strcmp(argv[i], "--host-ring")) host_ring = true;
    else if (!strcmp(argv[i], "--vram-only")) vram_only = true;
    else if (!strcmp(argv[i], "--engine") && i + 1 < argc) me = atoi(argv[++i]) & 1;
    else if (argv[i][0] >= '1' && argv[i][0] <= '9') mb = strtoull(argv[i], NULL, 10);
    else return -EINVAL;
  }
  if (mb > 1024) { fprintf(stderr, "rawgpu: at most 1024 MB per buffer\n"); return -EINVAL; }
  if (!id->chip || strcmp(id->chip, "POLARIS10")) {
    fprintf(stderr, "rawgpu: sdma-test is only wired up for POLARIS10 so far (this is %s): the golden\n"
                    "        register tables and firmware names differ per chip.\n", id->chip ? id->chip : "unknown");
    return -ENOTSUP;
  }

  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize;
  if (!vi_posted(&v, &memsize)) {
    if (!v.err) fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post` first\n");
    return v.err ? v.err : -EIO;
  }
  amd_fw fw;
  if ((r = amd_fw_load(&fw, fw_dir, me ? "polaris10_sdma1" : "polaris10_sdma"))) return r;
  printf("firmware:     %s.bin v%u (feature %u), %u dwords\n", fw.name, fw.ucode_version, amd_fw_hdr32(&fw, 32),
         fw.ucode_dw);

  uint64_t bytes = mb << 20;
  rg_dma ctrl = {.fd = -1}, src = {.fd = -1}, dst = {.fd = -1};
  vi_sdma s = {0};
  int phase = 0;
  uint32_t *buf = NULL;
  if (trace_path && !(v.trace = fopen(trace_path, "w"))) { perror(trace_path); amd_fw_free(&fw); return -EIO; }

  rg_pci_err pe;
  if (!rg_pci_errors(d, &pe) && (pe.status & 0xF900 || pe.devsta & 0x2F || pe.uncor)) {
    printf("PCIe errors left from before (clearing them):\n");
    rg_pci_print_errors(&pe);
    rg_pci_clear_errors(d);
  }
  uint32_t busy = vi_rreg(&v, mmSRBM_STATUS2) & (SRBM_STATUS2__SDMA_BUSY_MASK | SRBM_STATUS2__SDMA1_BUSY_MASK);
  if (busy) {
    printf("SDMA busy from an earlier run (SRBM_STATUS2 0x%08x): soft-resetting both engines\n", busy);
    vi_sdma_stop(&v, 0);
    vi_sdma_stop(&v, 1);
    vi_sdma_soft_reset(&v);
    printf("  after reset: SRBM_STATUS 0x%08x SRBM_STATUS2 0x%08x\n", vi_rreg(&v, mmSRBM_STATUS), vi_rreg(&v, mmSRBM_STATUS2));
  }

  /* ctrl: ring (64 KB) | writeback (4 KB) | VM fault default page (4 KB) */
  if ((r = d->ops->dma_alloc(d, 1u << 20, &ctrl)) || (r = d->ops->dma_alloc(d, bytes, &src)) ||
      (r = d->ops->dma_alloc(d, bytes, &dst)))
    goto out;
  printf("host buffers: ctrl 1 MB @ bus 0x%llx, src %llu MB @ 0x%llx, dst @ 0x%llx (%d/%d/%d segments)\n",
         (unsigned long long)ctrl.seg[0].addr, (unsigned long long)mb, (unsigned long long)src.seg[0].addr,
         (unsigned long long)dst.seg[0].addr, ctrl.nseg, src.nseg, dst.nseg);

  vi_gart g;
  if ((r = vi_gmc_init(&v, &g, rg_dma_bus(&ctrl, 0x11000)))) goto out;
  uint64_t ctrl_gpu, src_gpu, dst_gpu;
  if ((r = vi_gart_map(&v, &g, &ctrl, &ctrl_gpu)) || (r = vi_gart_map(&v, &g, &src, &src_gpu)) ||
      (r = vi_gart_map(&v, &g, &dst, &dst_gpu)))
    goto out;
  uint64_t pte0 = 0;
  d->ops->mmio_read(d, VI_VRAM_BAR, VI_VRAM_GART_TABLE, &pte0, 8);
  printf("GART:         %llu MB at MC 0x%010llx, table at VRAM +%llu MB; PTE[0] reads back 0x%016llx\n",
         (unsigned long long)(g.gart_size >> 20), (unsigned long long)g.gart_start,
         (unsigned long long)(VI_VRAM_GART_TABLE >> 20), (unsigned long long)pte0);

  vi_program_sequence(&v, golden_polaris10, sizeof golden_polaris10 / sizeof golden_polaris10[0]);
  vi_sdma_stop(&v, 0);
  vi_sdma_stop(&v, 1);
  if ((r = vi_sdma_load_ucode(&v, me, &fw))) goto out;

  /* VRAM scratch for the VRAM-side tests, inside BAR0 */
  const uint64_t V_RING = VI_VRAM_FREE, V_WB = V_RING + 0x10000, V_A = 64ull << 20, V_B = 128ull << 20;
  vi_mem ring = host_ring ? (vi_mem){.cpu = ctrl.cpu, .gpu = ctrl_gpu}
                          : (vi_mem){.bar = V_RING, .gpu = g.vram_start + V_RING};
  vi_mem wb = host_ring ? (vi_mem){.cpu = (uint8_t *)ctrl.cpu + 0x10000, .gpu = ctrl_gpu + 0x10000}
                        : (vi_mem){.bar = V_WB, .gpu = g.vram_start + V_WB};
  if ((r = vi_sdma_start(&s, &v, me, ring, 64u << 10, wb))) goto out;
  printf("SDMA%d:        microcode loaded, ring of 64 KB in %s\n", me, host_ring ? "host memory (GART)" : "VRAM");

  /* 1: the engine runs packets and writes VRAM */
  if ((r = vi_sdma_ring_test(&s, g.vram_start + V_WB + WB_TEST, NULL, (int64_t)(V_WB + WB_TEST)))) {
    vi_sdma_dump(&s); goto out;
  }
  PHASE("engine:       %s", "WRITE packet landed in VRAM");

  /* 2: VRAM -> VRAM copy, checked through BAR0 */
  const size_t vn = 1u << 20; /* 4 MB pattern */
  if (!(buf = malloc(vn * 4))) { r = -ENOMEM; goto out; }
  for (size_t i = 0; i < vn; i++) buf[i] = pat(3, i);
  if ((r = d->ops->mmio_write(d, VI_VRAM_BAR, V_A, buf, vn * 4))) goto out;
  vi_hdp_flush(&v);
  uint32_t seq = vi_sdma_copy(&s, g.vram_start + V_B, g.vram_start + V_A, vn * 4);
  if (!seq || (r = vi_sdma_wait(&s, seq, 2000))) { vi_sdma_dump(&s); r = r ? r : -EIO; goto out; }
  memset(buf, 0, vn * 4);
  vi_hdp_invalidate(&v);
  if ((r = d->ops->mmio_read(d, VI_VRAM_BAR, V_B, buf, vn * 4))) goto out;
  size_t bad = count_bad(buf, 3, vn);
  PHASE("VRAM copy:    4 MB VRAM -> VRAM, read back through BAR0: %s", bad ? "MISMATCH" : "intact");
  if (bad) { r = -EIO; goto out; }
  if (vram_only) goto out;

  /* 3: posted writes to host memory through the GART */
  volatile uint32_t *slot = (volatile uint32_t *)((uint8_t *)ctrl.cpu + 0x10000 + WB_TEST);
  if ((r = vi_sdma_ring_test(&s, ctrl_gpu + 0x10000 + WB_TEST, slot, -1))) { vi_sdma_dump(&s); goto out; }
  PHASE("host write:   %s", "WRITE packet landed in host memory through the GART");

  /* 4: host -> VRAM, the GPU reading host memory */
  uint32_t *sp = src.cpu, *dp = dst.cpu;
  size_t n = bytes / 4;
  for (size_t i = 0; i < n; i++) sp[i] = pat(7, i);
  uint64_t vram_a = g.vram_start + (1ull << 30), vram_b = g.vram_start + (2ull << 30);
  seq = vi_sdma_copy(&s, vram_a, src_gpu, 4096);
  if (!seq || (r = vi_sdma_wait(&s, seq, 2000))) { vi_sdma_dump(&s); r = r ? r : -EIO; goto out; }
  PHASE("host read:    %s", "4 KB host -> VRAM copy completed");

  /* 5: full round trip and bandwidth */
  double t_up, t_vv, t_down, t_fill;
  int reps = 3;
  memset(dp, 0, bytes);
  if ((r = timed(&s, "host->VRAM copy", false, vram_a, src_gpu, bytes, reps, &t_up)) ||
      (r = timed(&s, "VRAM->VRAM copy", false, vram_b, vram_a, bytes, reps, &t_vv)) ||
      (r = timed(&s, "VRAM->host copy", false, dst_gpu, vram_b, bytes, reps, &t_down))) {
    vi_sdma_dump(&s); goto out;
  }
  bad = 0;
  for (size_t i = 0; i < n; i++) bad += dp[i] != sp[i];
  size_t spot_bad = 0; /* independent look at VRAM through MM_INDEX, not DMA */
  vi_hdp_invalidate(&v);
  for (size_t i = 0; i < 64; i++) {
    uint64_t pos = (1ull << 30) + i * 4 * (n / 64);
    vi_wreg(&v, mmMM_INDEX, (uint32_t)pos | 0x80000000u);
    vi_wreg(&v, mmMM_INDEX_HI, (uint32_t)(pos >> 31));
    spot_bad += vi_rreg(&v, mmMM_DATA) != sp[i * (n / 64)];
  }
  PHASE("round trip:   %llu MB host -> VRAM -> VRAM -> host: %s; VRAM spot check via MM_INDEX: %s",
        (unsigned long long)mb, bad ? "MISMATCH" : "intact", spot_bad ? "MISMATCH" : "64/64 OK");
  if ((r = timed(&s, "fill", true, dst_gpu, 0xA5A55A5Au, bytes, 1, &t_fill))) { vi_sdma_dump(&s); goto out; }
  size_t fill_bad = 0;
  for (size_t i = 0; i < n; i++) fill_bad += dp[i] != 0xA5A55A5Au;
  PHASE("fill:         %llu MB constant fill of host memory: %s", (unsigned long long)mb, fill_bad ? "MISMATCH" : "intact");

  printf("bandwidth (best of %d x %llu MB, SDMA%d, boot clocks):\n", reps, (unsigned long long)mb, me);
  printf("  host -> VRAM   %6.2f GB/s\n", gbps(bytes, t_up));
  printf("  VRAM -> host   %6.2f GB/s\n", gbps(bytes, t_down));
  printf("  VRAM -> VRAM   %6.2f GB/s  (each byte read and written)\n", gbps(bytes, t_vv));
  printf("  fill -> host   %6.2f GB/s\n", gbps(bytes, t_fill));
  r = bad || spot_bad || fill_bad ? -EIO : v.err;

out:
  /* Host buffers vanish when we disconnect, so the engine must not be left pointing at them. */
  { int e = v.err; v.err = 0; vi_sdma_stop(&v, me); if (!v.err) v.err = e; }
  vi_sdma_free(&s);
  free(buf);
  if (v.trace) fclose(v.trace);
  if (dst.cpu) d->ops->dma_free(d, &dst);
  if (src.cpu) d->ops->dma_free(d, &src);
  if (ctrl.cpu) d->ops->dma_free(d, &ctrl);
  amd_fw_free(&fw);
  if (r && r != -EIO) fprintf(stderr, "rawgpu: sdma-test failed (%d)\n", r);
  return r;
}
