/* GMC v8 (VI memory controller): VRAM placement and the VMID0 GART.
 *
 * Follows Linux amdgpu's gmc_v8_0_mc_program() and gmc_v8_0_gart_enable(). The
 * GART lets the GPU reach host memory: a one-level page table in VRAM maps each
 * 4 KB GPU page in [gart_start, gart_end] to a bus address of a TinyGPU DMA
 * buffer (an IOVA behind Apple's DART). VMID0 is the only context set up;
 * everything rawgpu submits runs in it.
 *
 * The CPU writes the page table through BAR0. Those writes land via HDP, so
 * every update ends with an HDP flush and a TLB invalidate.
 */
#include "amd_vi.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* amdgpu_vm.h: VI PTE bits */
#define PTE_VALID      (1ull << 0)
#define PTE_SYSTEM     (1ull << 1)
#define PTE_SNOOPED    (1ull << 2)
#define PTE_EXECUTABLE (1ull << 4)
#define PTE_READABLE   (1ull << 5)
#define PTE_WRITEABLE  (1ull << 6)
#define PTE_ADDR_MASK  0x000000FFFFFFF000ull /* 39:12 */

#define GART_SIZE (256ull << 20)      /* amdgpu's default for Polaris (gmc_v8_0_mc_init) */
#define MC_MASK 0xFFFFFFFFFFull       /* 40-bit MC address space */
#define FOUR_GB (1ull << 32)
#define FRAGMENT_SIZE 9               /* amdgpu_vm_adjust_size(adev, 64, 9, 1, 40) */

/* gmc_v8_0.c golden_settings_polaris10_a11 */
static const uint32_t golden_polaris10[] = {
  mmMC_ARB_WTM_GRPWT_RD, 0x00000003, 0x00000000,
  mmVM_PRT_APERTURE0_LOW_ADDR, 0x0fffffff, 0x0fffffff,
  mmVM_PRT_APERTURE1_LOW_ADDR, 0x0fffffff, 0x0fffffff,
  mmVM_PRT_APERTURE2_LOW_ADDR, 0x0fffffff, 0x0fffffff,
  mmVM_PRT_APERTURE3_LOW_ADDR, 0x0fffffff, 0x0fffffff,
};

void vi_hdp_flush(vi_dev *v) { /* vi_flush_hdp() */
  vi_wreg(v, mmHDP_MEM_COHERENCY_FLUSH_CNTL, 1);
  vi_rreg(v, mmHDP_MEM_COHERENCY_FLUSH_CNTL);
}

void vi_hdp_invalidate(vi_dev *v) { /* vi_invalidate_hdp() */
  vi_wreg(v, mmHDP_DEBUG0, 1);
  vi_rreg(v, mmHDP_DEBUG0);
}

static void tlb_flush_vmid0(vi_dev *v) { /* gmc_v8_0_flush_gpu_tlb() */
  vi_wreg(v, mmVM_INVALIDATE_REQUEST, 1);
  vi_rreg(v, mmVM_INVALIDATE_RESPONSE);
}

static int mc_wait_idle(vi_dev *v) { /* gmc_v8_0_wait_for_idle() */
  uint32_t busy = SRBM_STATUS__MCB_BUSY_MASK | SRBM_STATUS__MCB_NON_DISPLAY_BUSY_MASK | SRBM_STATUS__MCC_BUSY_MASK |
                  SRBM_STATUS__MCD_BUSY_MASK | SRBM_STATUS__VMC_BUSY_MASK | SRBM_STATUS__VMC1_BUSY_MASK;
  int r = vi_wait(v, mmSRBM_STATUS, busy, 0, 100000);
  if (r == -ETIMEDOUT) fprintf(stderr, "rawgpu: warning: MC not idle (SRBM_STATUS 0x%08x)\n", vi_rreg(v, mmSRBM_STATUS));
  return v->err;
}

int vi_vram_write(vi_dev *v, uint64_t off, const void *src, size_t len) {
  if (v->err) return v->err;
  int r = v->d->ops->mmio_write(v->d, VI_VRAM_BAR, off, src, len);
  if (r) v->err = r;
  return r;
}

int vi_vram_zero(vi_dev *v, uint64_t off, size_t len) {
  void *z = calloc(1, len);
  if (!z) return -ENOMEM;
  int r = vi_vram_write(v, off, z, len);
  free(z);
  return r;
}

/* gmc_v8_0_mc_program(): apertures, AGP off, host FB access, HDP. VGA is left
 * alone: rawgpu drives no display, so whatever the VBIOS shows stays up. */
static void mc_program(vi_dev *v, const vi_gart *g) {
  for (uint32_t i = 0, j = 0; i < 32; i++, j += 6) /* HDP surface registers */
    for (uint32_t k = 0; k < 5; k++) vi_wreg(v, 0xb05 + j + k, 0);
  vi_wreg(v, mmHDP_REG_COHERENCY_FLUSH_CNTL, 0);
  mc_wait_idle(v);

  uint64_t vram_end = g->vram_start + g->vram_size - 1;
  vi_wreg(v, mmMC_VM_SYSTEM_APERTURE_LOW_ADDR, (uint32_t)(g->vram_start >> 12));
  vi_wreg(v, mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR, (uint32_t)(vram_end >> 12));
  vi_wreg(v, mmMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR, (uint32_t)((g->vram_start + VI_VRAM_SCRATCH) >> 12));
  /* amdgpu_gmc_set_agp_default(): agp_start = 0xffffffffffff, agp_end = 0, i.e. no AGP aperture */
  vi_wreg(v, mmMC_VM_AGP_BASE, 0);
  vi_wreg(v, mmMC_VM_AGP_TOP, 0);
  vi_wreg(v, mmMC_VM_AGP_BOT, (uint32_t)(0xffffffffffffull >> 22));
  mc_wait_idle(v);

  vi_wreg(v, mmBIF_FB_EN, BIF_FB_EN__FB_READ_EN_MASK | BIF_FB_EN__FB_WRITE_EN_MASK);
  uint32_t tmp = vi_rreg(v, mmHDP_MISC_CNTL);
  vi_wreg(v, mmHDP_MISC_CNTL, REG_SET_FIELD(tmp, HDP_MISC_CNTL, FLUSH_INVALIDATE_CACHE, 0));
  vi_wreg(v, mmHDP_HOST_PATH_CNTL, vi_rreg(v, mmHDP_HOST_PATH_CNTL));
}

/* gmc_v8_0_gart_enable(), VMID0 only. VMIDs 1-15 are left as the VBIOS set them. */
static void gart_enable(vi_dev *v, const vi_gart *g, uint64_t dummy_page_bus) {
  uint32_t tmp = vi_rreg(v, mmMC_VM_MX_L1_TLB_CNTL);
  tmp = REG_SET_FIELD(tmp, MC_VM_MX_L1_TLB_CNTL, ENABLE_L1_TLB, 1);
  tmp = REG_SET_FIELD(tmp, MC_VM_MX_L1_TLB_CNTL, ENABLE_L1_FRAGMENT_PROCESSING, 1);
  tmp = REG_SET_FIELD(tmp, MC_VM_MX_L1_TLB_CNTL, SYSTEM_ACCESS_MODE, 3);
  tmp = REG_SET_FIELD(tmp, MC_VM_MX_L1_TLB_CNTL, ENABLE_ADVANCED_DRIVER_MODEL, 1);
  tmp = REG_SET_FIELD(tmp, MC_VM_MX_L1_TLB_CNTL, SYSTEM_APERTURE_UNMAPPED_ACCESS, 0);
  vi_wreg(v, mmMC_VM_MX_L1_TLB_CNTL, tmp);

  tmp = vi_rreg(v, mmVM_L2_CNTL);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, ENABLE_L2_CACHE, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, ENABLE_L2_FRAGMENT_PROCESSING, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, ENABLE_L2_PTE_CACHE_LRU_UPDATE_BY_WRITE, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, ENABLE_L2_PDE0_CACHE_LRU_UPDATE_BY_WRITE, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, EFFECTIVE_L2_QUEUE_SIZE, 7);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, CONTEXT1_IDENTITY_ACCESS_MODE, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL, ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM_MEMORY, 1);
  vi_wreg(v, mmVM_L2_CNTL, tmp);
  tmp = vi_rreg(v, mmVM_L2_CNTL2);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL2, INVALIDATE_ALL_L1_TLBS, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL2, INVALIDATE_L2_CACHE, 1);
  vi_wreg(v, mmVM_L2_CNTL2, tmp);
  tmp = vi_rreg(v, mmVM_L2_CNTL3);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL3, L2_CACHE_BIGK_ASSOCIATIVITY, 1);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL3, BANK_SELECT, FRAGMENT_SIZE);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL3, L2_CACHE_BIGK_FRAGMENT_SIZE, FRAGMENT_SIZE);
  vi_wreg(v, mmVM_L2_CNTL3, tmp);
  tmp = vi_rreg(v, mmVM_L2_CNTL4); /* page tables live in VRAM: no physical/snooped PDE/PTE fetches */
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PDE_REQUEST_PHYSICAL, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PDE_REQUEST_SHARED, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PDE_REQUEST_SNOOP, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PTE_REQUEST_PHYSICAL, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PTE_REQUEST_SHARED, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT0_PTE_REQUEST_SNOOP, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PDE_REQUEST_PHYSICAL, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PDE_REQUEST_SHARED, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PDE_REQUEST_SNOOP, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PTE_REQUEST_PHYSICAL, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PTE_REQUEST_SHARED, 0);
  tmp = REG_SET_FIELD(tmp, VM_L2_CNTL4, VMC_TAP_CONTEXT1_PTE_REQUEST_SNOOP, 0);
  vi_wreg(v, mmVM_L2_CNTL4, tmp);

  uint64_t table_mc = g->vram_start + VI_VRAM_GART_TABLE;
  vi_wreg(v, mmVM_CONTEXT0_PAGE_TABLE_START_ADDR, (uint32_t)(g->gart_start >> 12));
  vi_wreg(v, mmVM_CONTEXT0_PAGE_TABLE_END_ADDR, (uint32_t)((g->gart_start + g->gart_size - 1) >> 12));
  vi_wreg(v, mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR, (uint32_t)(table_mc >> 12));
  vi_wreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR, (uint32_t)(dummy_page_bus >> 12));
  vi_wreg(v, mmVM_CONTEXT0_CNTL2, 0);
  tmp = vi_rreg(v, mmVM_CONTEXT0_CNTL);
  tmp = REG_SET_FIELD(tmp, VM_CONTEXT0_CNTL, ENABLE_CONTEXT, 1);
  tmp = REG_SET_FIELD(tmp, VM_CONTEXT0_CNTL, PAGE_TABLE_DEPTH, 0);
  tmp = REG_SET_FIELD(tmp, VM_CONTEXT0_CNTL, RANGE_PROTECTION_FAULT_ENABLE_DEFAULT, 1);
  vi_wreg(v, mmVM_CONTEXT0_CNTL, tmp);

  vi_wreg(v, mmVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR, 0);
  vi_wreg(v, mmVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR, 0);
  vi_wreg(v, mmVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET, 0);
  tlb_flush_vmid0(v);
}

int vi_gmc_init(vi_dev *v, vi_gart *g, uint64_t dummy_page_bus) {
  memset(g, 0, sizeof *g);
  uint32_t memsize;
  if (!vi_posted(v, &memsize)) return v->err ? v->err : -EIO;
  uint32_t loc = vi_rreg(v, mmMC_VM_FB_LOCATION);
  g->vram_start = (uint64_t)(loc & 0xFFFF) << 24; /* gmc_v8_0_vram_gtt_location() keeps the VBIOS's base */
  g->vram_size = (uint64_t)memsize << 20;

  /* amdgpu_gmc_gart_location(), BEST_FIT: below VRAM if that hole is the smaller fit, else at the top */
  uint64_t fb_end = g->vram_start + g->vram_size - 1;
  uint64_t size_bf = g->vram_start, size_af = MC_MASK + 1 - ((fb_end + FOUR_GB) & ~(FOUR_GB - 1));
  g->gart_size = GART_SIZE;
  rg_ranges_init(&g->pages, 0, g->gart_size >> 12);
  if ((size_bf >= g->gart_size && size_bf < size_af) || size_af < g->gart_size) g->gart_start = 0;
  else g->gart_start = MC_MASK - g->gart_size + 1;
  g->gart_start &= ~(FOUR_GB - 1);

  /* A GART this process's predecessor enabled (same table, same range) keeps its
   * memory-controller setup: amdgpu programs the MC once, at hw_init, and
   * reprogramming it while the SMC switches memory clocks is not something
   * amdgpu ever does. Only the table and the fault page are per process. */
  bool live = REG_GET_FIELD(vi_rreg(v, mmVM_CONTEXT0_CNTL), VM_CONTEXT0_CNTL, ENABLE_CONTEXT) &&
              vi_rreg(v, mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR) == (uint32_t)((g->vram_start + VI_VRAM_GART_TABLE) >> 12) &&
              vi_rreg(v, mmVM_CONTEXT0_PAGE_TABLE_START_ADDR) == (uint32_t)(g->gart_start >> 12);
  if (!live) {
    vi_program_sequence(v, golden_polaris10, sizeof golden_polaris10 / sizeof golden_polaris10[0]);
    mc_program(v, g);
  }
  if (v->err) return v->err;

  /* Stale VRAM can look like valid PTEs, so clear the whole table before enabling it. */
  int r = vi_vram_zero(v, VI_VRAM_GART_TABLE, (size_t)(g->gart_size >> 12) * 8);
  if (!r) r = vi_vram_zero(v, VI_VRAM_SCRATCH, 4096);
  if (r) return r;
  vi_hdp_flush(v);
  if (live) {
    vi_wreg(v, mmVM_CONTEXT0_PROTECTION_FAULT_DEFAULT_ADDR, (uint32_t)(dummy_page_bus >> 12));
    tlb_flush_vmid0(v);
  } else {
    gart_enable(v, g, dummy_page_bus);
  }
  return v->err;
}

int vi_gart_map(vi_dev *v, vi_gart *g, const rg_dma *b, uint64_t *gpu) {
  uint32_t pages = (uint32_t)((b->size + 4095) >> 12), total = (uint32_t)(g->gart_size >> 12);
  uint64_t first;
  if (rg_ranges_alloc(&g->pages, (pages + 15) & ~15u, 16, &first)) { /* keep mappings 64 KB aligned */
    fprintf(stderr, "rawgpu: GART full (%llu of %u pages free, largest run %llu; %u needed)\n",
            (unsigned long long)rg_ranges_free_bytes(&g->pages), total, (unsigned long long)rg_ranges_largest(&g->pages), pages);
    return -ENOSPC;
  }
  uint64_t *pte = malloc((size_t)pages * 8);
  if (!pte) return -ENOMEM;
  for (uint32_t i = 0; i < pages; i++) {
    uint64_t bus = rg_dma_bus(b, (uint64_t)i << 12);
    if (!bus || (bus & 0xFFF) || (bus & ~(PTE_ADDR_MASK | 0xFFF))) {
      fprintf(stderr, "rawgpu: bus address 0x%llx of page %u cannot go in a VI PTE\n", (unsigned long long)bus, i);
      free(pte);
      rg_ranges_free(&g->pages, first, (pages + 15) & ~15u);
      return -EINVAL;
    }
    /* amdgpu_ttm_tt_pte_flags() for cached system memory, plus gmc_v8's EXECUTABLE */
    pte[i] = bus | PTE_VALID | PTE_SYSTEM | PTE_SNOOPED | PTE_READABLE | PTE_WRITEABLE | PTE_EXECUTABLE;
  }
  int r = v->d->ops->mmio_write(v->d, VI_VRAM_BAR, VI_VRAM_GART_TABLE + first * 8, pte, (size_t)pages * 8);
  free(pte);
  if (r) return r;
  vi_hdp_flush(v);
  tlb_flush_vmid0(v);
  *gpu = g->gart_start + (first << 12);
  return v->err;
}

void vi_gart_unmap(vi_dev *v, vi_gart *g, uint64_t gpu, uint64_t size) {
  if (gpu < g->gart_start || !size) return;
  uint64_t first = (gpu - g->gart_start) >> 12, pages = (size + 4095) >> 12;
  uint64_t *zero = calloc(pages, 8);
  if (zero) { /* invalid PTEs: a stray access faults to the default page instead of reaching freed memory */
    v->d->ops->mmio_write(v->d, VI_VRAM_BAR, VI_VRAM_GART_TABLE + first * 8, zero, pages * 8);
    free(zero);
    vi_hdp_flush(v);
    tlb_flush_vmid0(v);
  }
  rg_ranges_free(&g->pages, first, (pages + 15) & ~15ull);
}

/* gmc_v8_0_{check,pre,post}_soft_reset() + gmc_v8_0_soft_reset(): if the MC or
 * VM report busy, black out the MC, pulse their SRBM resets and resume.
 * Returns the reset bits used (0 = nothing was busy). */
uint32_t vi_gmc_soft_reset(vi_dev *v) {
  uint32_t st = vi_rreg(v, mmSRBM_STATUS), bits = 0;
  if (st & SRBM_STATUS__VMC_BUSY_MASK) bits |= SRBM_SOFT_RESET__SOFT_RESET_VMC_MASK;
  if (st & (SRBM_STATUS__MCB_BUSY_MASK | SRBM_STATUS__MCB_NON_DISPLAY_BUSY_MASK | SRBM_STATUS__MCC_BUSY_MASK |
            SRBM_STATUS__MCD_BUSY_MASK))
    bits |= SRBM_SOFT_RESET__SOFT_RESET_MC_MASK;
  if (!bits || v->err) return 0;

  mc_wait_idle(v); /* gmc_v8_0_mc_stop() */
  uint32_t blackout = vi_rreg(v, mmMC_SHARED_BLACKOUT_CNTL);
  if (REG_GET_FIELD(blackout, MC_SHARED_BLACKOUT_CNTL, BLACKOUT_MODE) != 1) {
    vi_wreg(v, mmBIF_FB_EN, 0);
    vi_wreg(v, mmMC_SHARED_BLACKOUT_CNTL, REG_SET_FIELD(blackout, MC_SHARED_BLACKOUT_CNTL, BLACKOUT_MODE, 1));
  }
  vi_udelay(100);

  uint32_t tmp = vi_rreg(v, mmSRBM_SOFT_RESET) | bits;
  vi_wreg(v, mmSRBM_SOFT_RESET, tmp);
  vi_rreg(v, mmSRBM_SOFT_RESET);
  vi_udelay(50);
  vi_wreg(v, mmSRBM_SOFT_RESET, tmp & ~bits);
  vi_rreg(v, mmSRBM_SOFT_RESET);
  vi_udelay(50);

  tmp = vi_rreg(v, mmMC_SHARED_BLACKOUT_CNTL); /* gmc_v8_0_mc_resume() */
  vi_wreg(v, mmMC_SHARED_BLACKOUT_CNTL, REG_SET_FIELD(tmp, MC_SHARED_BLACKOUT_CNTL, BLACKOUT_MODE, 0));
  vi_wreg(v, mmBIF_FB_EN, BIF_FB_EN__FB_READ_EN_MASK | BIF_FB_EN__FB_WRITE_EN_MASK);
  return bits;
}

void vi_gmc_report(vi_dev *v) {
  struct { const char *name; uint32_t idx; bool page; } regs[] = {
    {"MC_VM_FB_LOCATION", mmMC_VM_FB_LOCATION, false},
    {"MC_VM_SYSTEM_APERTURE_LOW", mmMC_VM_SYSTEM_APERTURE_LOW_ADDR, true},
    {"MC_VM_SYSTEM_APERTURE_HIGH", mmMC_VM_SYSTEM_APERTURE_HIGH_ADDR, true},
    {"MC_VM_AGP_BOT/TOP", mmMC_VM_AGP_BOT, false},
    {"MC_VM_MX_L1_TLB_CNTL", mmMC_VM_MX_L1_TLB_CNTL, false}, {"VM_L2_CNTL", mmVM_L2_CNTL, false},
    {"VM_CONTEXT0_CNTL", mmVM_CONTEXT0_CNTL, false},
    {"VM_CONTEXT0_PT_START", mmVM_CONTEXT0_PAGE_TABLE_START_ADDR, true},
    {"VM_CONTEXT0_PT_END", mmVM_CONTEXT0_PAGE_TABLE_END_ADDR, true},
    {"VM_CONTEXT0_PT_BASE", mmVM_CONTEXT0_PAGE_TABLE_BASE_ADDR, true},
    {"VM_CONTEXT0_FAULT_STATUS", mmVM_CONTEXT0_PROTECTION_FAULT_STATUS, false},
    {"VM_CONTEXT0_FAULT_ADDR", mmVM_CONTEXT0_PROTECTION_FAULT_ADDR, true},
    {"VM_CONTEXT1_CNTL", mmVM_CONTEXT1_CNTL, false},
  };
  for (size_t i = 0; i < sizeof regs / sizeof regs[0]; i++) {
    uint32_t x = vi_rreg(v, regs[i].idx);
    printf("%-26s 0x%08x", regs[i].name, x);
    if (regs[i].idx == mmMC_VM_AGP_BOT) printf(" / 0x%08x%s", vi_rreg(v, mmMC_VM_AGP_TOP),
                                               x > vi_rreg(v, mmMC_VM_AGP_TOP) ? "   (AGP aperture off)" : "");
    if (regs[i].idx == mmVM_CONTEXT0_CNTL) printf("   GART %s", REG_GET_FIELD(x, VM_CONTEXT0_CNTL, ENABLE_CONTEXT) ? "on" : "off");
    if (regs[i].page) printf("   MC 0x%010llx", (unsigned long long)x << 12);
    printf("\n");
  }
}

/* ---- MC sequencer firmware ---- */

/* amdgpu's choice for Polaris 10 (gmc_v8_0_init_microcode()): the "k" image for P30 parts */
const char *vi_mc_fw_name(const rg_dev *d) {
  return d->device == 0x67DF && (d->revision == 0xE1 || d->revision == 0xF7) ? "polaris10_k_mc" : "polaris10_mc";
}

/* smu7_get_mc_microcode_version() */
uint32_t vi_mc_ucode_version(vi_dev *v) {
  vi_wreg(v, mmMC_SEQ_IO_DEBUG_INDEX, 0x9F);
  return vi_rreg(v, mmMC_SEQ_IO_DEBUG_DATA);
}

/* gmc_v8_0_polaris_mc_load_microcode(), which amdgpu runs on every Polaris in
 * gmc_v8_0_hw_init(), before power management starts. The VBIOS posts the card
 * with its own MC sequencer program; the driver's adds what memory clock
 * switching needs. Loading it retrains memory, so VRAM contents do not survive. */
int vi_mc_load_ucode(vi_dev *v, const amd_fw *fw, double *train_s) {
  /* mc_firmware_header_v1_0: common header, then io_debug_size_bytes, io_debug_array_offset_bytes */
  uint32_t io_bytes = amd_fw_hdr32(fw, 32), io_off = amd_fw_hdr32(fw, 36);
  if (amd_fw_hdr32(fw, 4) < 40 || io_bytes % 8 || (uint64_t)io_off + io_bytes > fw->size) {
    fprintf(stderr, "rawgpu: %s: IO register list (%u bytes at %u) is outside the file\n", fw->name, io_bytes, io_off);
    return -EINVAL;
  }
  const uint32_t *io = (const uint32_t *)(fw->data + io_off);
  uint32_t data = vi_rreg(v, mmMC_SEQ_MISC0);
  vi_wreg(v, mmMC_SEQ_MISC0, data & ~0x40u);
  for (uint32_t i = 0; i < io_bytes / 8; i++) { /* load mc io regs */
    vi_wreg(v, mmMC_SEQ_IO_DEBUG_INDEX, io[2 * i]);
    vi_wreg(v, mmMC_SEQ_IO_DEBUG_DATA, io[2 * i + 1]);
  }
  vi_wreg(v, mmMC_SEQ_SUP_CNTL, 0x00000008);
  vi_wreg(v, mmMC_SEQ_SUP_CNTL, 0x00000010);
  for (uint32_t i = 0; i < fw->ucode_dw; i++) vi_wreg(v, mmMC_SEQ_SUP_PGM, fw->ucode[i]);
  vi_wreg(v, mmMC_SEQ_SUP_CNTL, 0x00000008); /* put the engine back into the active state */
  vi_wreg(v, mmMC_SEQ_SUP_CNTL, 0x00000004);
  vi_wreg(v, mmMC_SEQ_SUP_CNTL, 0x00000001);
  double t0 = vi_now();
  int r = vi_wait(v, mmMC_SEQ_MISC0, 0x80, 0x80, 1000000); /* wait for training to complete */
  *train_s = vi_now() - t0;
  if (r == -ETIMEDOUT) fprintf(stderr, "rawgpu: MC training did not report done (MC_SEQ_MISC0 0x%08x)\n", vi_rreg(v, mmMC_SEQ_MISC0));
  return r;
}
