/* GFX v8 compute: RLC and MEC bring-up, one compute queue, `rawgpu compute-test`.
 *
 * Follows Linux amdgpu at 05441d697f7f, gfx_v8_0.c, for what compute needs:
 *
 *   gfx_v8_0_hw_init()
 *     gfx_v8_0_init_golden_registers()   all of it, for Polaris 10
 *     gfx_v8_0_constants_init()          GRBM timeout, SH_STATIC_MEM_CONFIG, VMID0's SH_MEM_*
 *     gfx_v8_0_rlc_resume()              stop, soft reset, start
 *     gfx_v8_0_cp_resume()               doorbells, then one queue (below)
 *
 * The queue is loaded the way amdgpu loads its KIQ and amdkfd loads a user
 * queue (kgd_hqd_load()): build the MQD with gfx_v8_0_mqd_init(), then write
 * the HQD registers from it (gfx_v8_0_mqd_commit()). There is no KIQ and no
 * MAP_QUEUES. The queue is ME1 pipe 0 queue 0 in VMID0, on amdgpu's first
 * compute doorbell (AMDGPU_DOORBELL_MEC_RING0). Its write pointer goes to the
 * doorbell in BAR2, as gfx_v8_0_ring_set_wptr_compute() does.
 *
 * Left out, all for graphics or interrupts: the tiling table, render-backend
 * setup, PA_SC FIFO sizes, SPI arbitration, GDS, VMIDs 1-15, and the GUI-idle
 * interrupts (TinyGPU has no MSI path; rawgpu polls).
 */
#include "amd_vi.h"
#include "amd_kernel.h"
#include "upstream/vi_structs.h"

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* vid.h */
#define PACKET3(op, n) ((3u << 30) | (((op) & 0xFFu) << 8) | (((n) & 0x3FFFu) << 16))
#define PACKET3_NOP 0x10
#define PACKET3_WRITE_DATA 0x37
#define PACKET3_RELEASE_MEM 0x49
#define PACKET3_SET_UCONFIG_REG 0x79
#define PACKET3_SET_UCONFIG_REG_START 0xc000
#define PACKET3_SET_SH_REG 0x76
#define PACKET3_SET_SH_REG_START 0x2c00
#define PACKET3_DISPATCH_DIRECT 0x15
#define PACKET3_ACQUIRE_MEM 0x58
#define PACKET3_TC_WB_ACTION_ENA (1u << 18)
#define PACKET3_TCL1_ACTION_ENA (1u << 22)
#define PACKET3_TC_ACTION_ENA (1u << 23)
#define PACKET3_SH_KCACHE_ACTION_ENA (1u << 27)
#define PACKET3_SH_ICACHE_ACTION_ENA (1u << 29)
#define EF_AMDGPU_MACH_AMDGCN_GFX803 0x2a
#define WRITE_DATA_DST_SEL(x) ((uint32_t)(x) << 8) /* 5 = memory */
#define WR_CONFIRM (1u << 20)
#define WRITE_DATA_ENGINE_SEL(x) ((uint32_t)(x) << 30)
#define EVENT_TYPE(x) ((uint32_t)(x) << 0)
#define EVENT_INDEX(x) ((uint32_t)(x) << 8)
#define EOP_TC_WB_ACTION_EN (1u << 15)
#define EOP_TCL1_ACTION_EN (1u << 16)
#define EOP_TC_ACTION_EN (1u << 17)
#define DATA_SEL(x) ((uint32_t)(x) << 29)
#define INT_SEL(x) ((uint32_t)(x) << 24)
#define CACHE_FLUSH_AND_INV_TS_EVENT 0x14 /* gfx_8_0_enum.h */
#define RING_NOP PACKET3(PACKET3_NOP, 0x3FFF) /* gfx_v8_0_ring_funcs_compute.nop: a one-dword NOP */
#define RING_ALIGN_DW 256u                     /* .align_mask = 0xff */

/* gfx_8_0_enum.h, gfx_v8_0.c, amdgpu_doorbell.h */
#define MTYPE_NC 1
#define MTYPE_UC 3
#define SH_MEM_ALIGNMENT_MODE_UNALIGNED 3
#define GFX8_MEC_HPD_SIZE 4096
#define DOORBELL_KIQ 0x000
#define DOORBELL_MEC_RING0 0x010
#define DOORBELL_MEC_RING7 0x017
#define DEQUEUE_DRAIN_PIPE 1 /* enum hqd_dequeue_request_type */
#define SHARED_APERTURE_BASES 0x2000 /* gmc_v8: shared_aperture_start (0x2000000000000000) >> 48 */
#define POLARIS10_SE 4               /* shader engines; 1 SH each (VBIOS gfx_info) */
#define POLARIS10_CU_PER_SH 9        /* max_cu_per_sh (VBIOS gfx_info): 36 CUs over 4 SEs */
#define VI_DOORBELL_BAR 2

/* struct vi_mqd from cp_mqd_base_addr_lo mirrors the HQD registers from mmCP_MQD_BASE_ADDR */
#define MQD_REG_OFF(field, reg) (offsetof(struct vi_mqd, field) - offsetof(struct vi_mqd, cp_mqd_base_addr_lo) == \
                                 4 * (size_t)((reg) - mmCP_MQD_BASE_ADDR))
_Static_assert(MQD_REG_OFF(cp_hqd_active, mmCP_HQD_ACTIVE) && MQD_REG_OFF(cp_hqd_vmid, mmCP_HQD_VMID) &&
               MQD_REG_OFF(cp_hqd_eop_control, mmCP_HQD_EOP_CONTROL) &&
               MQD_REG_OFF(cp_hqd_eop_done_events, mmCP_HQD_EOP_EVENTS) &&
               MQD_REG_OFF(cp_hqd_error, mmCP_HQD_ERROR) && MQD_REG_OFF(cp_hqd_eop_wptr_mem, mmCP_HQD_EOP_WPTR_MEM),
               "vi_mqd layout vs HQD registers");

static uint32_t order_base_2(uint32_t n) { uint32_t o = 0; while ((1u << o) < n) o++; return o; }

/* ---- gfx_v8_0_hw_init(), the compute part ---- */

/* gfx_v8_0.c golden_settings_polaris10_a11 + polaris10_golden_common_all */
static const uint32_t golden_polaris10_a11[] = {
  mmATC_MISC_CG, 0x000c0fc0, 0x000c0200,
  mmCB_HW_CONTROL, 0x0001f3cf, 0x00007208,
  mmCB_HW_CONTROL_2, 0x0f000000, 0x0f000000,
  mmCB_HW_CONTROL_3, 0x000001ff, 0x00000040,
  mmDB_DEBUG2, 0xf00fffff, 0x00000400,
  mmPA_SC_ENHANCE, 0xffffffff, 0x20000001,
  mmPA_SC_LINE_STIPPLE_STATE, 0x0000ff0f, 0x00000000,
  mmPA_SC_RASTER_CONFIG, 0x3f3fffff, 0x16000012,
  mmPA_SC_RASTER_CONFIG_1, 0x0000003f, 0x0000002a,
  mmRLC_CGCG_CGLS_CTRL, 0x00000003, 0x0001003c,
  mmRLC_CGCG_CGLS_CTRL_3D, 0xffffffff, 0x0001003c,
  mmSQ_CONFIG, 0x07f80000, 0x07180000,
  mmTA_CNTL_AUX, 0x000f000f, 0x000b0000,
  mmTCC_CTRL, 0x00100000, 0xf31fff7f,
  mmTCP_ADDR_CONFIG, 0x000003ff, 0x000000f7,
  mmTCP_CHAN_STEER_HI, 0xffffffff, 0x00000000,
  mmVGT_RESET_DEBUG, 0x00000004, 0x00000004,
};
static const uint32_t golden_polaris10_common[] = {
  mmGRBM_GFX_INDEX, 0xffffffff, 0xe0000000,
  mmPA_SC_RASTER_CONFIG, 0xffffffff, 0x16000012,
  mmPA_SC_RASTER_CONFIG_1, 0xffffffff, 0x0000002A,
  mmGB_ADDR_CONFIG, 0xffffffff, 0x22011003,
  mmSPI_RESOURCE_RESERVE_CU_0, 0xffffffff, 0x00000800,
  mmSPI_RESOURCE_RESERVE_CU_1, 0xffffffff, 0x00000800,
  mmSPI_RESOURCE_RESERVE_EN_CU_0, 0xffffffff, 0x00FF7FBF,
  mmSPI_RESOURCE_RESERVE_EN_CU_1, 0xffffffff, 0x00FF7FAF,
};

/* gfx_v8_0_init_golden_registers(), CHIP_POLARIS10. Its I2C fixup is for
 * revision 0xc7 boards only. */
static void golden_registers(vi_dev *v) {
  vi_program_sequence(v, golden_polaris10_a11, sizeof golden_polaris10_a11 / sizeof golden_polaris10_a11[0]);
  vi_program_sequence(v, golden_polaris10_common, sizeof golden_polaris10_common / sizeof golden_polaris10_common[0]);
  uint32_t x = vi_rsmc(v, ixCG_ACLK_CNTL);
  vi_wsmc(v, ixCG_ACLK_CNTL, REG_SET_FIELD(x, CG_ACLK_CNTL, ACLK_DIVIDER, 0x18));
}

/* gfx_v8_0_constants_init(), the parts compute in VMID0 depends on */
static void constants(vi_dev *v) {
  vi_wreg(v, mmGRBM_CNTL, REG_SET_FIELD(vi_rreg(v, mmGRBM_CNTL), GRBM_CNTL, READ_TIMEOUT, 0xFF));
  uint32_t smc = REG_SET_FIELD(0, SH_STATIC_MEM_CONFIG, SWIZZLE_ENABLE, 1);
  smc = REG_SET_FIELD(smc, SH_STATIC_MEM_CONFIG, ELEMENT_SIZE, 1);
  smc = REG_SET_FIELD(smc, SH_STATIC_MEM_CONFIG, INDEX_STRIDE, 3);
  vi_wreg(v, mmSH_STATIC_MEM_CONFIG, smc);

  vi_srbm_select(v, 0, 0, 0, 0);
  uint32_t x = REG_SET_FIELD(0, SH_MEM_CONFIG, DEFAULT_MTYPE, MTYPE_UC);
  x = REG_SET_FIELD(x, SH_MEM_CONFIG, APE1_MTYPE, MTYPE_UC);
  x = REG_SET_FIELD(x, SH_MEM_CONFIG, ALIGNMENT_MODE, SH_MEM_ALIGNMENT_MODE_UNALIGNED);
  vi_wreg(v, mmSH_MEM_CONFIG, x);
  vi_wreg(v, mmSH_MEM_BASES, 0);
  vi_wreg(v, mmSH_MEM_APE1_BASE, 1);
  vi_wreg(v, mmSH_MEM_APE1_LIMIT, 0);
}

/* gfx_v8_0_select_se_sh(); ~0u = broadcast */
static void select_se_sh(vi_dev *v, uint32_t se, uint32_t sh) {
  uint32_t x = REG_SET_FIELD(0, GRBM_GFX_INDEX, INSTANCE_BROADCAST_WRITES, 1);
  x = se == ~0u ? REG_SET_FIELD(x, GRBM_GFX_INDEX, SE_BROADCAST_WRITES, 1) : REG_SET_FIELD(x, GRBM_GFX_INDEX, SE_INDEX, se);
  x = sh == ~0u ? REG_SET_FIELD(x, GRBM_GFX_INDEX, SH_BROADCAST_WRITES, 1) : REG_SET_FIELD(x, GRBM_GFX_INDEX, SH_INDEX, sh);
  vi_wreg(v, mmGRBM_GFX_INDEX, x);
}

/* gfx_v8_0_enable_gui_idle_interrupt(false) */
static void gui_idle_interrupt_off(vi_dev *v) {
  uint32_t x = vi_rreg(v, mmCP_INT_CNTL_RING0);
  x &= ~(CP_INT_CNTL_RING0__CNTX_BUSY_INT_ENABLE_MASK | CP_INT_CNTL_RING0__CNTX_EMPTY_INT_ENABLE_MASK |
         CP_INT_CNTL_RING0__CMP_BUSY_INT_ENABLE_MASK | CP_INT_CNTL_RING0__GFX_IDLE_INT_ENABLE_MASK);
  vi_wreg(v, mmCP_INT_CNTL_RING0, x);
}

/* gfx_v8_0_rlc_resume(): stop (and wait for the serdes), soft reset, start.
 * Polaris 10 has no RLC power-gating setup (gfx_v8_0_init_pg()). */
static int rlc_resume(vi_dev *v) {
  vi_wreg(v, mmRLC_CNTL, REG_SET_FIELD(vi_rreg(v, mmRLC_CNTL), RLC_CNTL, RLC_ENABLE_F32, 0));
  gui_idle_interrupt_off(v);
  for (uint32_t se = 0; se < POLARIS10_SE; se++) { /* gfx_v8_0_wait_for_rlc_serdes() */
    select_se_sh(v, se, 0);
    if (vi_wait(v, mmRLC_SERDES_CU_MASTER_BUSY, 0xFFFFFFFFu, 0, 100000) == -ETIMEDOUT)
      fprintf(stderr, "rawgpu: warning: RLC serdes busy on SE%u\n", se);
  }
  select_se_sh(v, ~0u, ~0u);
  uint32_t noncu = RLC_SERDES_NONCU_MASTER_BUSY__SE_MASTER_BUSY_MASK | RLC_SERDES_NONCU_MASTER_BUSY__GC_MASTER_BUSY_MASK |
                   RLC_SERDES_NONCU_MASTER_BUSY__TC0_MASTER_BUSY_MASK | RLC_SERDES_NONCU_MASTER_BUSY__TC1_MASTER_BUSY_MASK;
  if (vi_wait(v, mmRLC_SERDES_NONCU_MASTER_BUSY, noncu, 0, 100000) == -ETIMEDOUT)
    fprintf(stderr, "rawgpu: warning: RLC non-CU serdes busy\n");

  vi_wreg(v, mmGRBM_SOFT_RESET, REG_SET_FIELD(vi_rreg(v, mmGRBM_SOFT_RESET), GRBM_SOFT_RESET, SOFT_RESET_RLC, 1));
  vi_udelay(50);
  vi_wreg(v, mmGRBM_SOFT_RESET, REG_SET_FIELD(vi_rreg(v, mmGRBM_SOFT_RESET), GRBM_SOFT_RESET, SOFT_RESET_RLC, 0));
  vi_udelay(50);

  vi_wreg(v, mmRLC_CNTL, REG_SET_FIELD(vi_rreg(v, mmRLC_CNTL), RLC_CNTL, RLC_ENABLE_F32, 1));
  vi_udelay(50); /* the GUI-idle interrupts stay off */
  return v->err;
}

/* vi_enable_doorbell_aperture() and gfx_v8_0_set_mec_doorbell_range(), then
 * gfx_v8_0_cp_compute_enable(true). */
static void cp_doorbells_and_mec(vi_dev *v) {
  vi_wreg(v, mmBIF_DOORBELL_APER_EN,
          REG_SET_FIELD(vi_rreg(v, mmBIF_DOORBELL_APER_EN), BIF_DOORBELL_APER_EN, BIF_DOORBELL_APER_EN, 1));
  vi_wreg(v, mmCP_MEC_DOORBELL_RANGE_LOWER, DOORBELL_KIQ << 2);
  vi_wreg(v, mmCP_MEC_DOORBELL_RANGE_UPPER, DOORBELL_MEC_RING7 << 2);
  vi_wreg(v, mmCP_PQ_STATUS, REG_SET_FIELD(vi_rreg(v, mmCP_PQ_STATUS), CP_PQ_STATUS, DOORBELL_ENABLE, 1));
  vi_wreg(v, mmCP_MEC_CNTL, 0);
  vi_udelay(50);
}

/* gfx_v8_0_get_cu_active_bitmap(), summed over the shader engines */
int vi_gfx_cu_count(vi_dev *v) {
  int n = 0;
  for (uint32_t se = 0; se < POLARIS10_SE; se++) {
    select_se_sh(v, se, 0);
    uint32_t data = vi_rreg(v, mmCC_GC_SHADER_ARRAY_CONFIG) | vi_rreg(v, mmGC_USER_SHADER_ARRAY_CONFIG);
    n += __builtin_popcount(~REG_GET_FIELD(data, CC_GC_SHADER_ARRAY_CONFIG, INACTIVE_CUS) & ((1u << POLARIS10_CU_PER_SH) - 1));
  }
  select_se_sh(v, ~0u, ~0u);
  return v->err ? v->err : n;
}

/* gfx_v8_0_set_safe_mode() / gfx_v8_0_unset_safe_mode() */
void vi_rlc_safe_mode(vi_dev *v, bool enter) {
  if (!REG_GET_FIELD(vi_rreg(v, mmRLC_CNTL), RLC_CNTL, RLC_ENABLE_F32)) return; /* gfx_v8_0_is_rlc_enabled() */
  uint32_t data = vi_rreg(v, mmRLC_CNTL);
  data |= RLC_SAFE_MODE__CMD_MASK;
  data &= ~RLC_SAFE_MODE__MESSAGE_MASK;
  if (enter) data |= 1u << RLC_SAFE_MODE__MESSAGE__SHIFT;
  vi_wreg(v, mmRLC_SAFE_MODE, data);
  if (enter) {
    uint32_t want = RLC_GPM_STAT__GFX_CLOCK_STATUS_MASK | RLC_GPM_STAT__GFX_POWER_STATUS_MASK;
    if (vi_wait(v, mmRLC_GPM_STAT, want, want, 1000000) == -ETIMEDOUT)
      fprintf(stderr, "rawgpu: warning: RLC safe mode: GFX clock/power status not reported on\n");
  }
  if (vi_wait(v, mmRLC_SAFE_MODE, RLC_SAFE_MODE__CMD_MASK, 0, 1000000) == -ETIMEDOUT)
    fprintf(stderr, "rawgpu: warning: RLC did not acknowledge safe mode %s\n", enter ? "entry" : "exit");
}

/* The per-power-cycle state page in VRAM (inside BAR0). A 64-bit magic marks it valid. */
#define STATE_MAGIC 0x7261776770753031ull /* "rawgpu01" */
#define STATE_GFX_UP 1u
typedef struct { uint64_t magic; uint32_t flags, pad; } vi_state;

static uint32_t state_flags(vi_dev *v) {
  vi_state s = {0};
  vi_hdp_invalidate(v);
  if (!v->err) v->err = v->d->ops->mmio_read(v->d, VI_VRAM_BAR, VI_VRAM_STATE, &s, sizeof s);
  return s.magic == STATE_MAGIC ? s.flags : 0;
}

static void state_set(vi_dev *v, uint32_t flags) {
  vi_state s = {STATE_MAGIC, flags, 0};
  vi_vram_write(v, VI_VRAM_STATE, &s, sizeof s);
  vi_hdp_flush(v);
}

void vi_state_clear(vi_dev *v) {
  static const vi_state zero;
  vi_vram_write(v, VI_VRAM_STATE, &zero, sizeof zero);
  vi_hdp_flush(v);
}

/* gfx_v8_0_hw_init() runs once per power cycle in amdgpu: golden registers,
 * constants, RLC resume (which soft-resets the RLC) and CP resume. Later
 * rawgpu processes find the state page marked and only re-check the doorbells
 * and the MEC, so the RLC is not reset again while power management runs. */
int vi_gfx_compute_init(vi_dev *v) {
  if (state_flags(v) & STATE_GFX_UP) {
    cp_doorbells_and_mec(v);
    return v->err;
  }
  golden_registers(v);
  constants(v);
  int r = rlc_resume(v);
  if (r) return r;
  gui_idle_interrupt_off(v); /* gfx_v8_0_cp_resume() */
  cp_doorbells_and_mec(v);
  if (!v->err) state_set(v, STATE_GFX_UP);
  return v->err;
}

/* ---- the compute queue ---- */

/* kgd_hqd_destroy() without the IQ-timer workaround (nothing here arms IQ
 * timers): request a drain and wait for the HQD to go inactive. The queue
 * must already be selected with vi_srbm_select(). */
static int hqd_dequeue(vi_dev *v) {
  if (!REG_GET_FIELD(vi_rreg(v, mmCP_HQD_ACTIVE), CP_HQD_ACTIVE, ACTIVE)) return v->err;
  vi_wait(v, mmCP_HQD_DEQUEUE_REQUEST, CP_HQD_DEQUEUE_REQUEST__IQ_REQ_PEND_MASK, 0, 1000);
  vi_wreg(v, mmCP_HQD_DEQUEUE_REQUEST, DEQUEUE_DRAIN_PIPE);
  int r = vi_wait(v, mmCP_HQD_ACTIVE, CP_HQD_ACTIVE__ACTIVE_MASK, 0, 1000000);
  if (r == -ETIMEDOUT) fprintf(stderr, "rawgpu: compute queue did not go inactive after a drain request\n");
  return r;
}

/* gfx_v8_0_mqd_init(), for a queue the host activates itself (as the KIQ) */
static void mqd_init(vi_dev *v, vi_cq *q, struct vi_mqd *m) {
  m->header = 0xC0310800;
  m->compute_pipelinestat_enable = 0x00000001;
  m->compute_static_thread_mgmt_se0 = 0xffffffff;
  m->compute_static_thread_mgmt_se1 = 0xffffffff;
  m->compute_static_thread_mgmt_se2 = 0xffffffff;
  m->compute_static_thread_mgmt_se3 = 0xffffffff;
  m->compute_misc_reserved = 0x00000003;
  uint64_t cu_mask = q->mqd.gpu + offsetof(struct vi_mqd_allocation, dynamic_cu_mask);
  m->dynamic_cu_mask_addr_lo = (uint32_t)cu_mask;
  m->dynamic_cu_mask_addr_hi = (uint32_t)(cu_mask >> 32);
  uint64_t eop = q->eop.gpu >> 8;
  m->cp_hqd_eop_base_addr_lo = (uint32_t)eop;
  m->cp_hqd_eop_base_addr_hi = (uint32_t)(eop >> 32);
  m->cp_hqd_eop_control = REG_SET_FIELD(vi_rreg(v, mmCP_HQD_EOP_CONTROL), CP_HQD_EOP_CONTROL, EOP_SIZE,
                                        order_base_2(GFX8_MEC_HPD_SIZE / 4) - 1);

  m->cp_mqd_base_addr_lo = (uint32_t)q->mqd.gpu & 0xfffffffc;
  m->cp_mqd_base_addr_hi = (uint32_t)(q->mqd.gpu >> 32);
  m->cp_mqd_control = REG_SET_FIELD(vi_rreg(v, mmCP_MQD_CONTROL), CP_MQD_CONTROL, VMID, 0);

  uint64_t pq = q->ring.gpu >> 8;
  m->cp_hqd_pq_base_lo = (uint32_t)pq;
  m->cp_hqd_pq_base_hi = (uint32_t)(pq >> 32);
  uint32_t x = vi_rreg(v, mmCP_HQD_PQ_CONTROL);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, QUEUE_SIZE, order_base_2(q->ring_dw) - 1);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, RPTR_BLOCK_SIZE, order_base_2(4096 / 4) - 1);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, UNORD_DISPATCH, 0);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, ROQ_PQ_IB_FLIP, 0);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, PRIV_STATE, 1);
  x = REG_SET_FIELD(x, CP_HQD_PQ_CONTROL, KMD_QUEUE, 1);
  m->cp_hqd_pq_control = x;

  uint64_t rptr = q->wb.gpu + VI_CQ_WB_RPTR, wptr = q->wb.gpu + VI_CQ_WB_WPTR;
  m->cp_hqd_pq_rptr_report_addr_lo = (uint32_t)rptr & 0xfffffffc;
  m->cp_hqd_pq_rptr_report_addr_hi = (uint32_t)(rptr >> 32) & 0xffff;
  m->cp_hqd_pq_wptr_poll_addr_lo = (uint32_t)wptr & 0xfffffffc; /* only with CP_PQ_WPTR_POLL_CNTL.EN, which is off */
  m->cp_hqd_pq_wptr_poll_addr_hi = (uint32_t)(wptr >> 32) & 0xffff;

  x = vi_rreg(v, mmCP_HQD_PQ_DOORBELL_CONTROL);
  x = REG_SET_FIELD(x, CP_HQD_PQ_DOORBELL_CONTROL, DOORBELL_OFFSET, q->doorbell);
  x = REG_SET_FIELD(x, CP_HQD_PQ_DOORBELL_CONTROL, DOORBELL_EN, 1);
  x = REG_SET_FIELD(x, CP_HQD_PQ_DOORBELL_CONTROL, DOORBELL_SOURCE, 0);
  x = REG_SET_FIELD(x, CP_HQD_PQ_DOORBELL_CONTROL, DOORBELL_HIT, 0);
  m->cp_hqd_pq_doorbell_control = x;

  m->cp_hqd_pq_wptr = 0;
  m->cp_hqd_pq_rptr = vi_rreg(v, mmCP_HQD_PQ_RPTR);
  m->cp_hqd_vmid = 0;
  m->cp_hqd_persistent_state = REG_SET_FIELD(vi_rreg(v, mmCP_HQD_PERSISTENT_STATE), CP_HQD_PERSISTENT_STATE,
                                             PRELOAD_SIZE, 0x53);
  x = vi_rreg(v, mmCP_HQD_IB_CONTROL);
  x = REG_SET_FIELD(x, CP_HQD_IB_CONTROL, MIN_IB_AVAIL_SIZE, 3);
  m->cp_hqd_ib_control = REG_SET_FIELD(x, CP_HQD_IB_CONTROL, MTYPE, 3);
  m->cp_hqd_iq_timer = REG_SET_FIELD(vi_rreg(v, mmCP_HQD_IQ_TIMER), CP_HQD_IQ_TIMER, MTYPE, 3);
  m->cp_hqd_ctx_save_control = REG_SET_FIELD(vi_rreg(v, mmCP_HQD_CTX_SAVE_CONTROL), CP_HQD_CTX_SAVE_CONTROL, MTYPE, 3);

  m->cp_hqd_eop_rptr = vi_rreg(v, mmCP_HQD_EOP_RPTR);
  m->cp_hqd_eop_wptr = vi_rreg(v, mmCP_HQD_EOP_WPTR);
  m->cp_hqd_ctx_save_base_addr_lo = vi_rreg(v, mmCP_HQD_CTX_SAVE_BASE_ADDR_LO);
  m->cp_hqd_ctx_save_base_addr_hi = vi_rreg(v, mmCP_HQD_CTX_SAVE_BASE_ADDR_HI);
  m->cp_hqd_cntl_stack_offset = vi_rreg(v, mmCP_HQD_CNTL_STACK_OFFSET);
  m->cp_hqd_cntl_stack_size = vi_rreg(v, mmCP_HQD_CNTL_STACK_SIZE);
  m->cp_hqd_wg_state_offset = vi_rreg(v, mmCP_HQD_WG_STATE_OFFSET);
  m->cp_hqd_ctx_save_size = vi_rreg(v, mmCP_HQD_CTX_SAVE_SIZE);
  m->cp_hqd_eop_done_events = vi_rreg(v, mmCP_HQD_EOP_EVENTS);
  m->cp_hqd_error = vi_rreg(v, mmCP_HQD_ERROR);
  m->cp_hqd_eop_wptr_mem = vi_rreg(v, mmCP_HQD_EOP_WPTR_MEM);
  m->cp_hqd_eop_dones = vi_rreg(v, mmCP_HQD_EOP_DONES);

  x = vi_rreg(v, mmCP_HQD_QUANTUM);
  x = REG_SET_FIELD(x, CP_HQD_QUANTUM, QUANTUM_EN, 1);
  x = REG_SET_FIELD(x, CP_HQD_QUANTUM, QUANTUM_SCALE, 1);
  m->cp_hqd_quantum = REG_SET_FIELD(x, CP_HQD_QUANTUM, QUANTUM_DURATION, 10);
  m->cp_hqd_active = 1; /* amdgpu sets this for the KIQ: mqd_commit() then activates the queue */
}

/* gfx_v8_0_mqd_commit() */
static void mqd_commit(vi_dev *v, const struct vi_mqd *m) {
  const uint32_t *d = &m->cp_mqd_base_addr_lo;
  vi_wreg(v, mmCP_PQ_WPTR_POLL_CNTL, REG_SET_FIELD(vi_rreg(v, mmCP_PQ_WPTR_POLL_CNTL), CP_PQ_WPTR_POLL_CNTL, EN, 0));
  for (uint32_t r = mmCP_HQD_VMID; r <= mmCP_HQD_EOP_CONTROL; r++) vi_wreg(v, r, d[r - mmCP_MQD_BASE_ADDR]);
  vi_wreg(v, mmCP_HQD_EOP_RPTR, m->cp_hqd_eop_rptr); /* not Tonga: its errata leaves these alone */
  vi_wreg(v, mmCP_HQD_EOP_WPTR, m->cp_hqd_eop_wptr);
  vi_wreg(v, mmCP_HQD_EOP_WPTR_MEM, m->cp_hqd_eop_wptr_mem);
  for (uint32_t r = mmCP_HQD_EOP_EVENTS; r <= mmCP_HQD_ERROR; r++) vi_wreg(v, r, d[r - mmCP_MQD_BASE_ADDR]);
  for (uint32_t r = mmCP_MQD_BASE_ADDR; r <= mmCP_HQD_ACTIVE; r++) vi_wreg(v, r, d[r - mmCP_MQD_BASE_ADDR]);
}

int vi_cq_start(vi_cq *q, vi_dev *v, vi_mem ring, uint32_t ring_bytes, vi_mem wb, vi_mem mqd, vi_mem eop) {
  if (ring_bytes & (ring_bytes - 1) || ring_bytes < 4096 || ring.gpu & 0xFF || eop.gpu & 0xFF || mqd.gpu & 0xFF)
    return -EINVAL;
  if (vi_rreg(v, mmCP_MEC_CNTL) & (CP_MEC_CNTL__MEC_ME1_HALT_MASK)) {
    fprintf(stderr, "rawgpu: MEC1 is halted (CP_MEC_CNTL 0x%08x)\n", vi_rreg(v, mmCP_MEC_CNTL));
    return v->err ? v->err : -EIO;
  }
  memset(q, 0, sizeof *q);
  q->v = v; q->me = 1; q->pipe = 0; q->queue = 0; q->doorbell = DOORBELL_MEC_RING0;
  q->ring = ring; q->wb = wb; q->mqd = mqd; q->eop = eop; q->ring_dw = ring_bytes / 4;
  q->shadow = ring.cpu ? ring.cpu : calloc(1, ring_bytes);
  if (!q->shadow) return -ENOMEM;
  memset(q->shadow, 0, ring_bytes);
  static const uint8_t zero[GFX8_MEC_HPD_SIZE];
  if (!ring.cpu) vi_mem_write(v, &ring, 0, q->shadow, ring_bytes);
  vi_mem_write(v, &wb, 0, zero, 64);
  vi_mem_write(v, &eop, 0, zero, GFX8_MEC_HPD_SIZE);

  vi_srbm_select(v, q->me, q->pipe, q->queue, 0);
  int r = hqd_dequeue(v); /* left active by an earlier run */
  struct vi_mqd_allocation *a = calloc(1, sizeof *a);
  if (!a) r = -ENOMEM;
  if (!r) {
    a->dynamic_cu_mask = 0xFFFFFFFF; /* gfx_v8_0_kiq_init_queue() */
    a->dynamic_rb_mask = 0xFFFFFFFF;
    mqd_init(v, q, &a->mqd);
    vi_mem_write(v, &mqd, 0, a, sizeof *a);
    vi_hdp_flush(v);
    mqd_commit(v, &a->mqd);
  }
  uint32_t active = vi_rreg(v, mmCP_HQD_ACTIVE);
  vi_srbm_select(v, 0, 0, 0, 0);
  free(a);
  if (!r && !v->err && !REG_GET_FIELD(active, CP_HQD_ACTIVE, ACTIVE)) {
    fprintf(stderr, "rawgpu: compute queue did not activate (CP_HQD_ACTIVE 0x%08x)\n", active);
    r = -EIO;
  }
  return r ? r : v->err;
}

void vi_cq_stop(vi_cq *q) {
  if (!q->v) return;
  vi_dev *v = q->v;
  int e = v->err;
  v->err = 0;
  vi_srbm_select(v, q->me, q->pipe, q->queue, 0);
  hqd_dequeue(v);
  vi_srbm_select(v, 0, 0, 0, 0);
  if (!v->err) v->err = e;
  if (q->shadow && q->shadow != q->ring.cpu) free(q->shadow);
  q->shadow = NULL;
  q->v = NULL;
}

/* ---- the ring ---- */

static uint32_t cq_free(vi_cq *q) {
  uint32_t rptr = vi_mem_read32(q->v, &q->wb, VI_CQ_WB_RPTR) & (q->ring_dw - 1);
  return (rptr - q->wptr - 1) & (q->ring_dw - 1);
}

static int cq_reserve(vi_cq *q, uint32_t n) {
  double end = vi_now() + 2.0;
  while (cq_free(q) < n + RING_ALIGN_DW) {
    if (q->v->err) return q->v->err;
    if (vi_now() > end) { fprintf(stderr, "rawgpu: compute ring stalled\n"); return -ETIMEDOUT; }
  }
  return 0;
}

static void cq_put(vi_cq *q, uint32_t dw) {
  q->shadow[q->wptr] = dw;
  q->wptr = (q->wptr + 1) & (q->ring_dw - 1);
}

/* amdgpu_ring_commit() + gfx_v8_0_ring_set_wptr_compute(): pad to the
 * alignment, copy a VRAM ring's new dwords out, then ring the doorbell. */
static void cq_commit(vi_cq *q) {
  vi_dev *v = q->v;
  while (q->wptr & (RING_ALIGN_DW - 1)) cq_put(q, RING_NOP);
  if (!q->ring.cpu) {
    uint32_t a = q->committed, b = q->wptr;
    if (b < a) { vi_mem_write(v, &q->ring, (uint64_t)a * 4, q->shadow + a, (size_t)(q->ring_dw - a) * 4); a = 0; }
    if (b > a) vi_mem_write(v, &q->ring, (uint64_t)a * 4, q->shadow + a, (size_t)(b - a) * 4);
  }
  vi_mem_write(v, &q->wb, VI_CQ_WB_WPTR, &q->wptr, 4);
  vi_hdp_flush(v);
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  if (!v->err) v->err = rg_wreg32(v->d, VI_DOORBELL_BAR, q->doorbell * 4, q->wptr);
  q->committed = q->wptr;
}

/* gfx_v8_0_ring_emit_fence_compute() without the interrupt */
static uint32_t cq_emit_fence(vi_cq *q) {
  uint32_t seq = ++q->seq;
  uint64_t a = q->wb.gpu + VI_CQ_WB_FENCE;
  cq_put(q, PACKET3(PACKET3_RELEASE_MEM, 5));
  cq_put(q, EOP_TCL1_ACTION_EN | EOP_TC_ACTION_EN | EOP_TC_WB_ACTION_EN | EVENT_TYPE(CACHE_FLUSH_AND_INV_TS_EVENT) |
                EVENT_INDEX(5));
  cq_put(q, DATA_SEL(1) | INT_SEL(0));
  cq_put(q, (uint32_t)a & 0xfffffffc);
  cq_put(q, (uint32_t)(a >> 32));
  cq_put(q, seq);
  cq_put(q, 0);
  return seq;
}

int vi_cq_wait(vi_cq *q, uint32_t seq, unsigned timeout_ms) {
  double end = vi_now() + timeout_ms / 1e3;
  uint32_t f;
  while ((int32_t)((f = vi_mem_read32(q->v, &q->wb, VI_CQ_WB_FENCE)) - seq) < 0) {
    if (q->v->err) return q->v->err;
    if (vi_now() > end) {
      fprintf(stderr, "rawgpu: compute fence %u timed out (fence reads %u, rptr %u, wptr %u)\n", seq, f,
              vi_mem_read32(q->v, &q->wb, VI_CQ_WB_RPTR), q->wptr);
      return -ETIMEDOUT;
    }
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  return 0;
}

/* WRITE_DATA of one dword to memory (as gfx_v8_0_ring_emit_fence_kiq()), plus a fence */
uint32_t vi_cq_write(vi_cq *q, uint64_t gpu, uint32_t value) {
  if (cq_reserve(q, 5 + 7)) return 0;
  cq_put(q, PACKET3(PACKET3_WRITE_DATA, 3));
  cq_put(q, WRITE_DATA_ENGINE_SEL(0) | WRITE_DATA_DST_SEL(5) | WR_CONFIRM);
  cq_put(q, (uint32_t)gpu & 0xfffffffc);
  cq_put(q, (uint32_t)(gpu >> 32));
  cq_put(q, value);
  uint32_t seq = cq_emit_fence(q);
  cq_commit(q);
  return q->v->err ? 0 : seq;
}

/* gfx_v8_0_ring_test_ring(): SET_UCONFIG_REG to SCRATCH_REG0, then poll it */
int vi_cq_ring_test(vi_cq *q) {
  vi_dev *v = q->v;
  vi_wreg(v, mmSCRATCH_REG0, 0xCAFEDEAD);
  int r = cq_reserve(q, 3);
  if (r) return r;
  cq_put(q, PACKET3(PACKET3_SET_UCONFIG_REG, 1));
  cq_put(q, mmSCRATCH_REG0 - PACKET3_SET_UCONFIG_REG_START);
  cq_put(q, 0xDEADBEEF);
  cq_commit(q);
  r = vi_wait(v, mmSCRATCH_REG0, 0xFFFFFFFFu, 0xDEADBEEF, 500000);
  if (r == -ETIMEDOUT)
    fprintf(stderr, "rawgpu: compute ring test: SCRATCH_REG0 still 0x%08x after 500 ms (rptr %u, wptr %u)\n",
            vi_rreg(v, mmSCRATCH_REG0), vi_mem_read32(v, &q->wb, VI_CQ_WB_RPTR), q->wptr);
  return r;
}

/* ---- dispatch ---- */

int vi_kernel_check(const amd_kernel *k) {
  const char *why = NULL;
  uint16_t supported = AMD_KCP_PRIVATE_SEGMENT_BUFFER | AMD_KCP_KERNARG_SEGMENT_PTR;
  if (k->mach != EF_AMDGPU_MACH_AMDGCN_GFX803) why = "it is not built for gfx803 (Polaris): use -mcpu=gfx803";
  else if (k->private_segment_size || REG_GET_FIELD(k->rsrc2, COMPUTE_PGM_RSRC2, SCRATCH_EN))
    why = "it uses scratch memory, which rawgpu does not set up yet";
  else if (k->group_segment_size > VI_LDS_MAX) why = "it wants more than 64 KB of LDS";
  else if (k->code_properties & ~supported)
    why = "it wants user SGPRs beyond the private segment buffer and kernarg pointer (dispatch/queue pointers)";
  else if (k->entry & 0xFF) why = "its entry point is not 256-byte aligned";
  if (why) fprintf(stderr, "rawgpu: kernel %s: %s\n", k->name, why);
  return why ? -ENOTSUP : 0;
}

static void cq_set_sh(vi_cq *q, uint32_t reg, const uint32_t *vals, uint32_t n) {
  cq_put(q, PACKET3(PACKET3_SET_SH_REG, n));
  cq_put(q, reg - PACKET3_SET_SH_REG_START);
  for (uint32_t i = 0; i < n; i++) cq_put(q, vals[i]);
}

/* gfx_v8_0_emit_mem_sync_compute(): invalidate L1/L2 and the shader caches */
static void cq_mem_sync(vi_cq *q) {
  cq_put(q, PACKET3(PACKET3_ACQUIRE_MEM, 5));
  cq_put(q, PACKET3_TCL1_ACTION_ENA | PACKET3_TC_ACTION_ENA | PACKET3_SH_KCACHE_ACTION_ENA |
                PACKET3_SH_ICACHE_ACTION_ENA | PACKET3_TC_WB_ACTION_ENA); /* CP_COHER_CNTL */
  cq_put(q, 0xffffffff); /* CP_COHER_SIZE */
  cq_put(q, 0xff);       /* CP_COHER_SIZE_HI */
  cq_put(q, 0);          /* CP_COHER_BASE */
  cq_put(q, 0);          /* CP_COHER_BASE_HI */
  cq_put(q, 0x0000000A); /* poll interval */
}

/* The register state and DISPATCH_DIRECT amdgpu uses for its own compute
 * dispatch (gfx_v8_0_do_edc_gpr_workarounds()), with the program, resources
 * and user SGPRs from the kernel descriptor. */
uint32_t vi_cq_dispatch(vi_cq *q, const amd_kernel *k, uint64_t code_gpu, uint64_t kernarg_gpu, const uint32_t wg[3],
                        const uint32_t groups[3]) {
  uint32_t user[6], nu = 0;
  if (k->code_properties & AMD_KCP_PRIVATE_SEGMENT_BUFFER) { user[nu++] = 0; user[nu++] = 0; user[nu++] = 0; user[nu++] = 0; }
  if (k->code_properties & AMD_KCP_KERNARG_SEGMENT_PTR) { user[nu++] = (uint32_t)kernarg_gpu; user[nu++] = (uint32_t)(kernarg_gpu >> 32); }
  if (nu != REG_GET_FIELD(k->rsrc2, COMPUTE_PGM_RSRC2, USER_SGPR)) {
    fprintf(stderr, "rawgpu: kernel %s: descriptor enables %u user SGPRs, RSRC2 says %u\n", k->name, nu,
            REG_GET_FIELD(k->rsrc2, COMPUTE_PGM_RSRC2, USER_SGPR));
    return 0;
  }
  if (cq_reserve(q, 96)) return 0;
  uint64_t pgm = (code_gpu + k->entry) >> 8;
  const uint32_t start[3] = {0, 0, 0};
  /* LDS is allocated per work-group in 128-dword (512-byte) blocks on GFX7+ */
  uint32_t rsrc2 = REG_SET_FIELD(k->rsrc2, COMPUTE_PGM_RSRC2, LDS_SIZE, (k->group_segment_size + 511) / 512);
  const uint32_t pgm_addr[2] = {(uint32_t)pgm, (uint32_t)(pgm >> 32)}, rsrc[2] = {k->rsrc1, rsrc2};
  const uint32_t all_cus[2] = {0xffffffff, 0xffffffff}, zero = 0;
  cq_mem_sync(q);
  cq_set_sh(q, mmCOMPUTE_START_X, start, 3);
  cq_set_sh(q, mmCOMPUTE_NUM_THREAD_X, wg, 3);
  cq_set_sh(q, mmCOMPUTE_PGM_LO, pgm_addr, 2);
  cq_set_sh(q, mmCOMPUTE_PGM_RSRC1, rsrc, 2);
  cq_set_sh(q, mmCOMPUTE_RESOURCE_LIMITS, &zero, 1);
  cq_set_sh(q, mmCOMPUTE_STATIC_THREAD_MGMT_SE0, all_cus, 2);
  cq_set_sh(q, mmCOMPUTE_STATIC_THREAD_MGMT_SE2, all_cus, 2);
  cq_set_sh(q, mmCOMPUTE_TMPRING_SIZE, &zero, 1);
  if (nu) cq_set_sh(q, mmCOMPUTE_USER_DATA_0, user, nu);
  cq_put(q, PACKET3(PACKET3_DISPATCH_DIRECT, 3));
  cq_put(q, groups[0]);
  cq_put(q, groups[1]);
  cq_put(q, groups[2]);
  cq_put(q, REG_SET_FIELD(0, COMPUTE_DISPATCH_INITIATOR, COMPUTE_SHADER_EN, 1));
  uint32_t seq = cq_emit_fence(q); /* end-of-pipe: waits for the dispatch, flushes L2 */
  cq_commit(q);
  return q->v->err ? 0 : seq;
}

/* The queue's registers, the CP's status and the VM fault state, for a failure. */
void vi_cq_dump(vi_cq *q) {
  vi_dev *v = q->v;
  if (!v) return;
  int e = v->err;
  v->err = 0;
  vi_srbm_select(v, q->me, q->pipe, q->queue, 0);
  printf("HQD ME%u pipe %u queue %u: ACTIVE 0x%x  PQ_RPTR 0x%x  PQ_WPTR 0x%x  DOORBELL_CONTROL 0x%08x  ERROR 0x%08x\n",
         q->me, q->pipe, q->queue, vi_rreg(v, mmCP_HQD_ACTIVE), vi_rreg(v, mmCP_HQD_PQ_RPTR), vi_rreg(v, mmCP_HQD_PQ_WPTR),
         vi_rreg(v, mmCP_HQD_PQ_DOORBELL_CONTROL), vi_rreg(v, mmCP_HQD_ERROR));
  printf("                          PQ_BASE 0x%08x_%08x  PQ_CONTROL 0x%08x  DEQUEUE_REQUEST 0x%08x\n",
         vi_rreg(v, mmCP_HQD_PQ_BASE_HI), vi_rreg(v, mmCP_HQD_PQ_BASE), vi_rreg(v, mmCP_HQD_PQ_CONTROL),
         vi_rreg(v, mmCP_HQD_DEQUEUE_REQUEST));
  vi_srbm_select(v, 0, 0, 0, 0);
  printf("CP:  CP_MEC_CNTL 0x%08x  CP_CPC_STATUS 0x%08x  CP_CPC_BUSY_STAT 0x%08x  CP_CPC_STALLED_STAT1 0x%08x\n",
         vi_rreg(v, mmCP_MEC_CNTL), vi_rreg(v, mmCP_CPC_STATUS), vi_rreg(v, mmCP_CPC_BUSY_STAT),
         vi_rreg(v, mmCP_CPC_STALLED_STAT1));
  printf("     CP_PQ_STATUS 0x%08x  RLC_CNTL 0x%08x  RLC_STAT 0x%08x  GRBM_STATUS 0x%08x  SRBM_STATUS 0x%08x\n",
         vi_rreg(v, mmCP_PQ_STATUS), vi_rreg(v, mmRLC_CNTL), vi_rreg(v, mmRLC_STAT), vi_rreg(v, mmGRBM_STATUS),
         vi_rreg(v, mmSRBM_STATUS));
  uint32_t fa = vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_ADDR);
  printf("     VM0 fault: status 0x%08x addr MC 0x%010llx client 0x%08x\n",
         vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_STATUS), (unsigned long long)fa << 12,
         vi_rreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_MCCLIENT));
  rg_pci_err pe;
  if (!rg_pci_errors(v->d, &pe)) rg_pci_print_errors(&pe);
  if (!v->err) v->err = e;
}

/* ---- rawgpu gfx-reset ---- */

/* gfx_v8_0_deactivate_hqd(); the queue must be selected */
static int deactivate_hqd(vi_dev *v, uint32_t req) {
  int r = 0;
  if (vi_rreg(v, mmCP_HQD_ACTIVE) & CP_HQD_ACTIVE__ACTIVE_MASK) {
    vi_wreg(v, mmCP_HQD_DEQUEUE_REQUEST,
            REG_SET_FIELD(vi_rreg(v, mmCP_HQD_DEQUEUE_REQUEST), CP_HQD_DEQUEUE_REQUEST, DEQUEUE_REQ, req));
    r = vi_wait(v, mmCP_HQD_ACTIVE, CP_HQD_ACTIVE__ACTIVE_MASK, 0, 1000000);
  }
  vi_wreg(v, mmCP_HQD_DEQUEUE_REQUEST, 0);
  vi_wreg(v, mmCP_HQD_PQ_RPTR, 0);
  vi_wreg(v, mmCP_HQD_PQ_WPTR, 0);
  return r;
}

/* `rawgpu gfx-reset`: gfx_v8_0_soft_reset() for the one queue rawgpu uses.
 * Reset the HQD's waves (dequeue type 2), halt the MEC and RLC (the parts of
 * gfx_v8_0_hw_fini() that apply), stall and clear GFX's memory requests in
 * GMCON, pulse the GRBM and SRBM resets, unstall. The next compute command
 * brings the RLC and MEC back (vi_gfx_compute_init()). */
int vi_cmd_gfx_reset(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  printf("before: GRBM_STATUS 0x%08x, SRBM_STATUS 0x%08x\n", vi_rreg(&v, mmGRBM_STATUS), vi_rreg(&v, mmSRBM_STATUS));
  vi_srbm_select(&v, 1, 0, 0, 0);
  int dq = deactivate_hqd(&v, 2); /* RESET_WAVES */
  vi_srbm_select(&v, 0, 0, 0, 0);
  vi_udelay(50);
  printf("HQD deactivate: %s\n", dq ? "timed out (resetting anyway)" : "done");
  vi_wreg(&v, mmCP_MEC_CNTL, CP_MEC_CNTL__MEC_ME1_HALT_MASK | CP_MEC_CNTL__MEC_ME2_HALT_MASK);
  vi_wreg(&v, mmRLC_CNTL, REG_SET_FIELD(vi_rreg(&v, mmRLC_CNTL), RLC_CNTL, RLC_ENABLE_F32, 0));
  vi_udelay(50);

  uint32_t grbm = GRBM_SOFT_RESET__SOFT_RESET_RLC_MASK | GRBM_SOFT_RESET__SOFT_RESET_GFX_MASK |
                  GRBM_SOFT_RESET__SOFT_RESET_CP_MASK | GRBM_SOFT_RESET__SOFT_RESET_CPF_MASK |
                  GRBM_SOFT_RESET__SOFT_RESET_CPC_MASK | GRBM_SOFT_RESET__SOFT_RESET_CPG_MASK;
  uint32_t srbm = SRBM_SOFT_RESET__SOFT_RESET_GRBM_MASK | SRBM_SOFT_RESET__SOFT_RESET_SEM_MASK;
  uint32_t t = vi_rreg(&v, mmGMCON_DEBUG);
  vi_wreg(&v, mmGMCON_DEBUG, t | GMCON_DEBUG__GFX_STALL_MASK | GMCON_DEBUG__GFX_CLEAR_MASK);
  vi_udelay(100);
  t = vi_rreg(&v, mmGRBM_SOFT_RESET) | grbm;
  vi_wreg(&v, mmGRBM_SOFT_RESET, t);
  vi_rreg(&v, mmGRBM_SOFT_RESET);
  vi_udelay(100);
  vi_wreg(&v, mmGRBM_SOFT_RESET, t & ~grbm);
  vi_rreg(&v, mmGRBM_SOFT_RESET);
  vi_udelay(100);
  t = vi_rreg(&v, mmSRBM_SOFT_RESET) | srbm;
  vi_wreg(&v, mmSRBM_SOFT_RESET, t);
  vi_rreg(&v, mmSRBM_SOFT_RESET);
  vi_udelay(100);
  vi_wreg(&v, mmSRBM_SOFT_RESET, t & ~srbm);
  vi_rreg(&v, mmSRBM_SOFT_RESET);
  vi_udelay(100);
  t = vi_rreg(&v, mmGMCON_DEBUG);
  vi_wreg(&v, mmGMCON_DEBUG, t & ~(GMCON_DEBUG__GFX_STALL_MASK | GMCON_DEBUG__GFX_CLEAR_MASK));
  vi_udelay(100);
  vi_state_clear(&v); /* the next compute command brings GFX up again */
  printf("after:  GRBM_STATUS 0x%08x, SRBM_STATUS 0x%08x\n", vi_rreg(&v, mmGRBM_STATUS), vi_rreg(&v, mmSRBM_STATUS));
  rg_pci_err e;
  if (!rg_pci_errors(d, &e)) rg_pci_print_errors(&e);
  return v.err;
}

/* ---- rawgpu waves ---- */

/* gfx_v8_0.c wave_read_ind(): the SQ's per-wave registers through its index window */
static uint32_t wave_read_ind(vi_dev *v, uint32_t simd, uint32_t wave, uint32_t address) {
  vi_wreg(v, mmSQ_IND_INDEX, (wave << SQ_IND_INDEX__WAVE_ID__SHIFT) | (simd << SQ_IND_INDEX__SIMD_ID__SHIFT) |
                                 (address << SQ_IND_INDEX__INDEX__SHIFT) | SQ_IND_INDEX__FORCE_READ_MASK);
  return vi_rreg(v, mmSQ_IND_DATA);
}

/* `rawgpu waves`: every live wave, as amdgpu's debugfs wave reader walks them
 * (select SE/SH/CU with GRBM_GFX_INDEX, then SIMD and wave through SQ_IND_INDEX). */
int vi_cmd_waves(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  int n = 0;
  for (uint32_t se = 0; se < POLARIS10_SE; se++)
    for (uint32_t cu = 0; cu < POLARIS10_CU_PER_SH; cu++) {
      uint32_t x = REG_SET_FIELD(0, GRBM_GFX_INDEX, INSTANCE_INDEX, cu); /* gfx_v8_0_select_se_sh(se, 0, cu) */
      x = REG_SET_FIELD(x, GRBM_GFX_INDEX, SE_INDEX, se);
      x = REG_SET_FIELD(x, GRBM_GFX_INDEX, SH_INDEX, 0);
      vi_wreg(&v, mmGRBM_GFX_INDEX, x);
      for (uint32_t simd = 0; simd < 4; simd++)
        for (uint32_t w = 0; w < 10; w++) {
          uint32_t st = wave_read_ind(&v, simd, w, ixSQ_WAVE_STATUS);
          if (!(st & SQ_WAVE_STATUS__VALID_MASK)) continue;
          uint64_t pc = wave_read_ind(&v, simd, w, ixSQ_WAVE_PC_LO) |
                        (uint64_t)wave_read_ind(&v, simd, w, ixSQ_WAVE_PC_HI) << 32;
          printf("SE%u CU%u SIMD%u wave %u: PC 0x%010llx STATUS 0x%08x%s%s%s IB_STS 0x%08x TRAPSTS 0x%08x INST 0x%08x\n", se, cu,
                 simd, w, (unsigned long long)pc, st, st & SQ_WAVE_STATUS__HALT_MASK ? " HALT" : "",
                 st & SQ_WAVE_STATUS__TRAP_MASK ? " TRAP" : "", st & SQ_WAVE_STATUS__IN_BARRIER_MASK ? " BARRIER" : "",
                 wave_read_ind(&v, simd, w, ixSQ_WAVE_IB_STS), wave_read_ind(&v, simd, w, ixSQ_WAVE_TRAPSTS),
                 wave_read_ind(&v, simd, w, ixSQ_WAVE_INST_DW0));
          n++;
        }
    }
  select_se_sh(&v, ~0u, ~0u);
  printf("%d live wave%s; GRBM_STATUS 0x%08x\n", n, n == 1 ? "" : "s", vi_rreg(&v, mmGRBM_STATUS));
  return v.err;
}

/* ---- rawgpu compute-test ---- */

#define PHASE(fmt, ...) printf("[%d] " fmt "\n", ++phase, __VA_ARGS__)

int vi_cmd_compute_test(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  const char *trace_path = NULL, *kernel_path = "build/kernels/vadd.o";
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kernel_path = argv[++i];
    else return -EINVAL;
  }
  if (!id->chip || strcmp(id->chip, "POLARIS10")) {
    fprintf(stderr, "rawgpu: compute-test is only wired up for POLARIS10 so far (this is %s)\n",
            id->chip ? id->chip : "unknown");
    return -ENOTSUP;
  }
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize, status = 0;
  if (!vi_posted(&v, &memsize)) {
    if (!v.err) fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post`, then `rawgpu smc-load`\n");
    return v.err ? v.err : -EIO;
  }
  uint32_t need = VI_UCODE_CP_MEC | VI_UCODE_RLC_G;
  if (vi_smu_ucode_status(&v, &status) || (status & need) != need) {
    if (!v.err) fprintf(stderr, "rawgpu: the SMU has not loaded the MEC and RLC microcode (UcodeLoadStatus 0x%08x);\n"
                                "        run `rawgpu smc-load` first\n", status);
    return v.err ? v.err : -EIO;
  }
  uint64_t bar2 = 0;
  if ((r = rg_bar_map(d, VI_DOORBELL_BAR, &bar2))) { fprintf(stderr, "rawgpu: could not map doorbell BAR%d\n", VI_DOORBELL_BAR); return r; }
  if (trace_path && !(v.trace = fopen(trace_path, "w"))) { perror(trace_path); return -EIO; }

  rg_dma ctrl = {.fd = -1}, ha = {.fd = -1}, hb = {.fd = -1}, hc = {.fd = -1};
  vi_cq q = {0};
  amd_kernel k = {0};
  float *ref = NULL, *buf = NULL;
  int phase = 0;
  /* ctrl (host): writeback for the host-memory test at +0, VM fault default page at +0x11000 */
  if ((r = d->ops->dma_alloc(d, 1u << 20, &ctrl))) goto out;
  vi_gart g;
  uint64_t ctrl_gpu;
  if ((r = vi_gmc_init(&v, &g, rg_dma_bus(&ctrl, 0x11000))) || (r = vi_gart_map(&v, &g, &ctrl, &ctrl_gpu))) goto out;

  if ((r = vi_gfx_compute_init(&v))) goto out;
  printf("GFX:          up (RLC_CNTL 0x%08x), doorbells on (BAR2 %llu MB)\n", vi_rreg(&v, mmRLC_CNTL),
         (unsigned long long)(bar2 >> 20));

  /* VRAM for the queue, inside BAR0: ring 64 KB | MQD 4 KB | EOP 4 KB | writeback 4 KB */
  const uint64_t V_RING = VI_VRAM_FREE, V_MQD = V_RING + 0x10000, V_EOP = V_MQD + 0x1000, V_WB = V_EOP + 0x1000;
  const uint64_t V_DATA = V_WB + 0x1000;
#define VMEM(off) ((vi_mem){.bar = (off), .gpu = g.vram_start + (off)})
  if ((r = vi_cq_start(&q, &v, VMEM(V_RING), 64u << 10, VMEM(V_WB), VMEM(V_MQD), VMEM(V_EOP)))) { vi_cq_dump(&q); goto out; }
  printf("queue:        ME1 pipe 0 queue 0, VMID0, doorbell %u; ring 64 KB, MQD and EOP in VRAM; HQD active\n", q.doorbell);

  /* 1: the CP fetches the ring and executes a packet */
  if ((r = vi_cq_ring_test(&q))) { vi_cq_dump(&q); goto out; }
  PHASE("ring:         %s", "SET_UCONFIG_REG landed in SCRATCH_REG0");

  /* 2: WRITE_DATA to VRAM, then a RELEASE_MEM fence */
  vi_mem data = VMEM(V_DATA);
  uint32_t init = 0xCAFEDEADu;
  vi_mem_write(&v, &data, 0, &init, 4);
  vi_hdp_flush(&v);
  uint32_t seq = vi_cq_write(&q, data.gpu, 0x600DF00Du);
  if (!seq || (r = vi_cq_wait(&q, seq, 1000))) { vi_cq_dump(&q); r = r ? r : -EIO; goto out; }
  uint32_t got = vi_mem_read32(&v, &data, 0);
  PHASE("VRAM write:   WRITE_DATA + RELEASE_MEM fence %u; VRAM reads 0x%08x (%s)", seq, got,
        got == 0x600DF00Du ? "intact" : "MISMATCH");
  if (got != 0x600DF00Du) { r = -EIO; goto out; }

  /* 3: WRITE_DATA to host memory through the GART */
  volatile uint32_t *host = ctrl.cpu;
  host[0] = 0xCAFEDEADu;
  seq = vi_cq_write(&q, ctrl_gpu, 0x5EED5EEDu);
  if (!seq || (r = vi_cq_wait(&q, seq, 1000))) { vi_cq_dump(&q); r = r ? r : -EIO; goto out; }
  double t0 = vi_now();
  while (host[0] != 0x5EED5EEDu && vi_now() - t0 < 0.5) {}
  PHASE("host write:   WRITE_DATA through the GART; host memory reads 0x%08x (%s)", host[0],
        host[0] == 0x5EED5EEDu ? "intact" : "MISMATCH");
  if (host[0] != 0x5EED5EEDu) { r = -EIO; goto out; }

  /* 4: round-trip latency of a fenced packet, through the socket */
  double best = 1e9;
  for (int i = 0; i < 20; i++) {
    double t = vi_now();
    seq = vi_cq_write(&q, data.gpu, (uint32_t)i);
    if (!seq || (r = vi_cq_wait(&q, seq, 1000))) { vi_cq_dump(&q); r = r ? r : -EIO; goto out; }
    if (vi_now() - t < best) best = vi_now() - t;
  }
  PHASE("latency:      submit -> fence seen, best of 20: %.0f us", best * 1e6);

  /* 5-6: a compiled kernel, c[i] = a[i] + b[i] over 1M floats, first in host memory, then in VRAM */
  int kr = amd_kernel_load(&k, kernel_path, "vadd");
  if (kr == -ENOENT) { printf("dispatch:     skipped: %s is not built (make kernels)\n", kernel_path); goto out; }
  if (kr || (r = vi_kernel_check(&k))) { r = kr ? kr : r; goto out; }
  const uint32_t N = 1u << 20, WG = 64;
  const size_t bytes = (size_t)N * 4;
  const uint64_t V_CODE = V_DATA + 0x1000, CODE_REGION = ((k.code_size + 4095) & ~(size_t)4095) + 4096;
  const uint64_t V_KARG = V_CODE + CODE_REGION, V_A = 64ull << 20, V_B = 80ull << 20, V_C = 96ull << 20;
  if (!(ref = malloc(bytes)) || !(buf = malloc(bytes))) { r = -ENOMEM; goto out; }
  if ((r = vi_vram_zero(&v, V_CODE, CODE_REGION)) || (r = vi_vram_write(&v, V_CODE, k.code, k.code_size))) goto out;
  printf("kernel:       %s from %s: %zu bytes of gfx803 code, %u-byte kernargs, RSRC1 0x%08x RSRC2 0x%08x\n", k.name,
         kernel_path, k.code_size, k.kernarg_size, k.rsrc1, k.rsrc2);

  if ((r = d->ops->dma_alloc(d, bytes, &ha)) || (r = d->ops->dma_alloc(d, bytes, &hb)) || (r = d->ops->dma_alloc(d, bytes, &hc)))
    goto out;
  uint64_t a_gpu, b_gpu, c_gpu;
  if ((r = vi_gart_map(&v, &g, &ha, &a_gpu)) || (r = vi_gart_map(&v, &g, &hb, &b_gpu)) || (r = vi_gart_map(&v, &g, &hc, &c_gpu)))
    goto out;
  float *A = ha.cpu, *B = hb.cpu, *C = hc.cpu;
  for (uint32_t i = 0; i < N; i++) { /* exact in float: no rounding or denormals to differ on */
    A[i] = (float)(i % 1000) * 0.125f;
    B[i] = (float)((i * 7u) % 1013u) - 300.0f;
    ref[i] = A[i] + B[i];
  }
  struct { uint64_t a, b, c; uint32_t n, pad; } karg;
  for (int pass = 0; pass < 2; pass++) {
    bool vram = pass == 0; /* VRAM first: it does not depend on the GART path to host memory */
    if (vram) { /* inputs through BAR0; the output pre-filled so stale results cannot pass */
      memset(buf, 0xFF, bytes);
      if ((r = vi_vram_write(&v, V_A, A, bytes)) || (r = vi_vram_write(&v, V_B, B, bytes)) ||
          (r = vi_vram_write(&v, V_C, buf, bytes)))
        goto out;
      karg = (typeof(karg)){g.vram_start + V_A, g.vram_start + V_B, g.vram_start + V_C, N, 0};
    } else {
      memset(C, 0xFF, bytes);
      karg = (typeof(karg)){a_gpu, b_gpu, c_gpu, N, 0};
    }
    if ((r = vi_vram_write(&v, V_KARG, &karg, sizeof karg))) goto out;
    vi_hdp_flush(&v);
    double t = vi_now();
    const uint32_t wg[3] = {WG, 1, 1}, groups[3] = {N / WG, 1, 1};
    seq = vi_cq_dispatch(&q, &k, g.vram_start + V_CODE, g.vram_start + V_KARG, wg, groups);
    if (!seq || (r = vi_cq_wait(&q, seq, 5000))) { vi_cq_dump(&q); r = r ? r : -EIO; goto out; }
    t = vi_now() - t;
    const float *out = C;
    if (vram) {
      vi_hdp_invalidate(&v);
      if ((r = v.d->ops->mmio_read(v.d, VI_VRAM_BAR, V_C, buf, bytes))) goto out;
      out = buf;
    }
    size_t bad = 0, first = 0;
    for (uint32_t i = 0; i < N; i++)
      if (memcmp(&out[i], &ref[i], 4) && !bad++) first = i;
    PHASE("vadd %s %u floats in %u groups of %u: %s; %.2f ms submit-to-fence (%.1f GB/s of a, b, c)",
          vram ? "VRAM:" : "host:", N, N / WG, WG, bad ? "MISMATCH" : "matches the CPU", t * 1e3, 3.0 * bytes / t / 1e9);
    if (bad) {
      printf("              %zu wrong; c[%zu] = %g, expected %g\n", bad, first, (double)out[first], (double)ref[first]);
      r = -EIO;
      goto out;
    }
  }

out:
  vi_cq_stop(&q); /* the queue must not outlive the host memory it may point at */
  if (v.trace) { fclose(v.trace); printf("trace: %s\n", trace_path); }
  if (ctrl.cpu) d->ops->dma_free(d, &ctrl);
  if (ha.cpu) d->ops->dma_free(d, &ha);
  if (hb.cpu) d->ops->dma_free(d, &hb);
  if (hc.cpu) d->ops->dma_free(d, &hc);
  amd_kernel_free(&k);
  free(ref);
  free(buf);
  if (!r) r = v.err;
  if (r && r != -EIO) fprintf(stderr, "rawgpu: compute-test failed (%d)\n", r);
  return r;
}
