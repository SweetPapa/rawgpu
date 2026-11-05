/* The amd-vi host API (amd_vi_ctx.h). */
#include "amd_vi_ctx.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* VRAM the context keeps for itself, inside BAR0, above the SMU's area */
#define CTX_CQ_RING (VI_VRAM_FREE)              /* 64 KB compute ring */
#define CTX_CQ_MQD  (CTX_CQ_RING + 0x10000)     /* 4 KB each: MQD, EOP, writeback */
#define CTX_CQ_EOP  (CTX_CQ_MQD + 0x1000)
#define CTX_CQ_WB   (CTX_CQ_EOP + 0x1000)
#define CTX_KARG    (CTX_CQ_RING + 0x20000)     /* kernarg ring: dispatch packet + kernargs per launch */
#define CTX_KARG_SIZE ((33u << 20) - CTX_KARG)
#define CTX_CODE    (33ull << 20)               /* code images: must stay CPU-writable, so inside the BAR */
#define CTX_CODE_END (64ull << 20)
#define CTX_HEAP    (64ull << 20)               /* buffers and scratch: all the rest of VRAM */
#define STAGING_BYTES (16u << 20)
#define SCRATCH_MAX (256ull << 20)
#define TIMEOUT_MS 10000u

int vi_ctx_open(vi_ctx *c, rg_dev *d, const rg_ident *id) {
  memset(c, 0, sizeof *c);
  c->ctrl.fd = c->staging.fd = -1;
  if (!id->chip || strcmp(id->chip, "POLARIS10")) {
    fprintf(stderr, "rawgpu: the host API is only wired up for POLARIS10 so far (this is %s)\n",
            id->chip ? id->chip : "unknown");
    return -ENOTSUP;
  }
  vi_dev *v = &c->v;
  int r = vi_open(v, d);
  if (r) return r;
  uint32_t memsize, status = 0, need = VI_UCODE_SDMA0 | VI_UCODE_CP_MEC | VI_UCODE_RLC_G;
  if (!vi_posted(v, &memsize)) {
    if (!v->err) fprintf(stderr, "rawgpu: the card is not up; run `rawgpu post`, `rawgpu mc-load`, `rawgpu dpm-enable`\n");
    return v->err ? v->err : -EIO;
  }
  if (vi_smu_ucode_status(v, &status) || (status & need) != need) {
    if (!v->err)
      fprintf(stderr, "rawgpu: engine microcode is not loaded (UcodeLoadStatus 0x%08x); run `rawgpu dpm-enable` "
                      "(or `rawgpu smc-load` for boot clocks)\n", status);
    return v->err ? v->err : -EIO;
  }
  uint64_t bar2;
  if ((r = rg_bar_map(d, 2, &bar2))) return r;
  if ((r = d->ops->dma_alloc(d, 1u << 20, &c->ctrl)) || (r = d->ops->dma_alloc(d, STAGING_BYTES, &c->staging))) goto fail;
  if ((r = vi_gmc_init(v, &c->g, rg_dma_bus(&c->ctrl, 0x11000))) || (r = vi_gart_map(v, &c->g, &c->ctrl, &c->ctrl_gpu)) ||
      (r = vi_gart_map(v, &c->g, &c->staging, &c->staging_gpu)))
    goto fail;

  if ((r = vi_sdma_init(v, 0, NULL))) goto fail;
  vi_mem ring = {.cpu = c->ctrl.cpu, .gpu = c->ctrl_gpu};
  vi_mem wb = {.cpu = (uint8_t *)c->ctrl.cpu + 0x10000, .gpu = c->ctrl_gpu + 0x10000};
  if ((r = vi_sdma_start(&c->sdma, v, 0, ring, 64u << 10, wb))) goto fail;

  if ((r = vi_gfx_compute_init(v))) goto fail;
#define VMEM(off) ((vi_mem){.bar = (off), .gpu = c->g.vram_start + (off)})
  if ((r = vi_cq_start(&c->cq, v, VMEM(CTX_CQ_RING), 64u << 10, VMEM(CTX_CQ_WB), VMEM(CTX_CQ_MQD), VMEM(CTX_CQ_EOP)))) {
    vi_cq_dump(&c->cq);
    goto fail;
  }
  int cus = vi_gfx_cu_count(v);
  c->cus = cus > 0 ? (unsigned)cus : 36;
  rg_ranges_init(&c->code, CTX_CODE, CTX_CODE_END - CTX_CODE);
  rg_ranges_init(&c->heap, CTX_HEAP, c->g.vram_size - CTX_HEAP);
  return 0;
fail:
  vi_ctx_close(c);
  return r ? r : -EIO;
}

void vi_ctx_close(vi_ctx *c) {
  rg_dev *d = c->v.d;
  if (!d) return;
  if (c->last_seq) vi_ctx_finish(c);
  /* engines first: host memory they point at goes away with the DMA buffers */
  vi_cq_stop(&c->cq);
  { int e = c->v.err; c->v.err = 0; vi_sdma_stop(&c->v, 0); if (!c->v.err) c->v.err = e; }
  vi_sdma_free(&c->sdma);
  if (c->staging.cpu) d->ops->dma_free(d, &c->staging);
  if (c->ctrl.cpu) d->ops->dma_free(d, &c->ctrl);
  c->v.d = NULL;
}

/* ---- synchronization ---- */

bool vi_ctx_done(vi_ctx *c, uint32_t seq) {
  return (int32_t)(vi_mem_read32(&c->v, &c->cq.wb, VI_CQ_WB_FENCE) - seq) >= 0 || c->v.err;
}

int vi_ctx_wait(vi_ctx *c, uint32_t seq, unsigned timeout_ms) {
  if (!seq) return 0;
  int r = vi_cq_wait(&c->cq, seq, timeout_ms);
  if (r) vi_cq_dump(&c->cq);
  return r;
}

int vi_ctx_finish(vi_ctx *c) { return vi_ctx_wait(c, c->last_seq, TIMEOUT_MS); }

/* ---- memory ---- */

int vi_buf_alloc(vi_ctx *c, vi_buf *b, uint64_t size, bool host) {
  memset(b, 0, sizeof *b);
  b->dma.fd = -1;
  if (!size) return -EINVAL;
  size = (size + 4095) & ~4095ull;
  b->size = size;
  b->host = host;
  if (host) {
    int r = c->v.d->ops->dma_alloc(c->v.d, size, &b->dma);
    if (!r) r = vi_gart_map(&c->v, &c->g, &b->dma, &b->gpu);
    if (r) { if (b->dma.cpu) c->v.d->ops->dma_free(c->v.d, &b->dma); b->dma.cpu = NULL; return r; }
    b->cpu = b->dma.cpu;
    return 0;
  }
  uint64_t off;
  if (rg_ranges_alloc(&c->heap, size, 1u << 16, &off)) { /* 64 KB aligned */
    fprintf(stderr, "rawgpu: out of VRAM (%llu MB asked, %llu MB free, largest free block %llu MB)\n",
            (unsigned long long)(size >> 20), (unsigned long long)(rg_ranges_free_bytes(&c->heap) >> 20),
            (unsigned long long)(rg_ranges_largest(&c->heap) >> 20));
    return -ENOMEM;
  }
  b->gpu = c->g.vram_start + off;
  return 0;
}

void vi_buf_free(vi_ctx *c, vi_buf *b) {
  if (!b->size) return;
  if (b->host) {
    if (b->dma.cpu) {
      vi_ctx_finish(c); /* a kernel may still be using it */
      vi_gart_unmap(&c->v, &c->g, b->gpu, b->size);
      c->v.d->ops->dma_free(c->v.d, &b->dma);
    }
  } else {
    rg_ranges_free(&c->heap, b->gpu - c->g.vram_start, b->size);
  }
  memset(b, 0, sizeof *b);
}

static int sdma_sync(vi_ctx *c, uint32_t seq) {
  int r = seq ? vi_sdma_wait(&c->sdma, seq, TIMEOUT_MS) : (c->v.err ? c->v.err : -EIO);
  if (r) vi_sdma_dump(&c->sdma);
  return r;
}

int vi_buf_write(vi_ctx *c, vi_buf *b, uint64_t off, const void *src, uint64_t len) {
  if (off > b->size || len > b->size - off) return -EINVAL;
  int r = vi_ctx_finish(c);
  if (r) return r;
  if (b->host) { memcpy((uint8_t *)b->cpu + off, src, len); return 0; }
  for (uint64_t done = 0; done < len;) {
    uint64_t n = len - done < STAGING_BYTES ? len - done : STAGING_BYTES;
    memcpy(c->staging.cpu, (const uint8_t *)src + done, n);
    if ((r = sdma_sync(c, vi_sdma_copy(&c->sdma, b->gpu + off + done, c->staging_gpu, n)))) return r;
    done += n;
  }
  return 0;
}

int vi_buf_read(vi_ctx *c, const vi_buf *b, uint64_t off, void *dst, uint64_t len) {
  if (off > b->size || len > b->size - off) return -EINVAL;
  int r = vi_ctx_finish(c);
  if (r) return r;
  if (b->host) { memcpy(dst, (const uint8_t *)b->cpu + off, len); return 0; }
  for (uint64_t done = 0; done < len;) {
    uint64_t n = len - done < STAGING_BYTES ? len - done : STAGING_BYTES;
    if ((r = sdma_sync(c, vi_sdma_copy(&c->sdma, c->staging_gpu, b->gpu + off + done, n)))) return r;
    memcpy((uint8_t *)dst + done, c->staging.cpu, n);
    done += n;
  }
  return 0;
}

int vi_buf_copy(vi_ctx *c, vi_buf *dst, uint64_t doff, const vi_buf *src, uint64_t soff, uint64_t len) {
  if (doff > dst->size || len > dst->size - doff || soff > src->size || len > src->size - soff) return -EINVAL;
  if (!len) return 0;
  int r = vi_ctx_finish(c);
  return r ? r : sdma_sync(c, vi_sdma_copy(&c->sdma, dst->gpu + doff, src->gpu + soff, len));
}

int vi_buf_fill_range(vi_ctx *c, vi_buf *b, uint64_t off, uint64_t len, uint32_t value) {
  if ((off | len) & 3 || off > b->size || len > b->size - off) return -EINVAL;
  if (!len) return 0;
  int r = vi_ctx_finish(c);
  return r ? r : sdma_sync(c, vi_sdma_fill(&c->sdma, b->gpu + off, value, len));
}

int vi_buf_fill(vi_ctx *c, vi_buf *b, uint32_t value) { return vi_buf_fill_range(c, b, 0, b->size, value); }

/* ---- code ---- */

int vi_prog_load_mem(vi_ctx *c, vi_prog *pp, const void *data, size_t size, const char *label) {
  memset(pp, 0, sizeof *pp);
  int r = amd_program_load_mem(&pp->p, data, size, label);
  if (r) return r;
  if (pp->p.mach != EF_AMDGPU_MACH_AMDGCN_GFX803) {
    fprintf(stderr, "rawgpu: %s is not built for gfx803 (Polaris): compile with -mcpu=gfx803\n", label);
    amd_program_free(&pp->p);
    return -ENOTSUP;
  }
  /* zero padding after the image: the shader prefetches past its last instruction */
  pp->region = ((pp->p.image_size + 4095) & ~4095ull) + 4096;
  if (rg_ranges_alloc(&c->code, pp->region, 4096, &pp->off)) {
    fprintf(stderr, "rawgpu: no room for %s's %zu-byte image (code space is %llu MB)\n", label, pp->p.image_size,
            (unsigned long long)((CTX_CODE_END - CTX_CODE) >> 20));
    amd_program_free(&pp->p);
    return -ENOSPC;
  }
  pp->gpu = c->g.vram_start + pp->off;
  amd_program_relocate(&pp->p, pp->gpu);
  if ((r = vi_vram_zero(&c->v, pp->off, pp->region)) || (r = vi_vram_write(&c->v, pp->off, pp->p.image, pp->p.image_size))) {
    vi_prog_free(c, pp);
    return r;
  }
  vi_hdp_flush(&c->v);
  return c->v.err;
}

int vi_prog_load(vi_ctx *c, vi_prog *pp, const char *path) {
  memset(pp, 0, sizeof *pp);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "rawgpu: code object %s not found (make kernels builds rawgpu's own)\n", path); return -ENOENT; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 64 || sz > (64l << 20)) { fclose(f); fprintf(stderr, "rawgpu: %s: not a code object\n", path); return -EINVAL; }
  uint8_t *d = malloc((size_t)sz);
  if (!d) { fclose(f); return -ENOMEM; }
  size_t n = fread(d, 1, (size_t)sz, f);
  fclose(f);
  int r = n == (size_t)sz ? vi_prog_load_mem(c, pp, d, (size_t)sz, path) : -EIO;
  free(d);
  return r;
}

void vi_prog_free(vi_ctx *c, vi_prog *pp) {
  if (pp->region) {
    vi_ctx_finish(c); /* a queued launch may still run this code */
    rg_ranges_free(&c->code, pp->off, pp->region);
  }
  amd_program_free(&pp->p);
  memset(pp, 0, sizeof *pp);
}

int vi_kern_load(vi_ctx *c, vi_kern *kk, const char *path, const char *name) {
  memset(kk, 0, sizeof *kk);
  int r = vi_prog_load(c, &kk->prog, path);
  if (r) return r;
  if (!(kk->k = amd_program_kernel(&kk->prog.p, name))) r = -ENOENT;
  else r = vi_kernel_check(kk->k);
  if (r) vi_prog_free(c, &kk->prog);
  return r;
}

void vi_kern_free(vi_ctx *c, vi_kern *kk) {
  vi_prog_free(c, &kk->prog);
  kk->k = NULL;
}

/* ---- launches ---- */

/* hsa_kernel_dispatch_packet_t (HSA runtime spec 2.9.6): what the dispatch pointer points at */
typedef struct {
  uint16_t header, setup, workgroup_size_x, workgroup_size_y, workgroup_size_z, reserved0;
  uint32_t grid_size_x, grid_size_y, grid_size_z, private_segment_size, group_segment_size;
  uint64_t kernel_object, kernarg_address, reserved2, completion_signal;
} aql_dispatch;
_Static_assert(sizeof(aql_dispatch) == 64, "AQL packet layout");

static void put_hidden(uint8_t *karg, uint32_t cap, const amd_arg *a, uint64_t v) {
  if (a->offset > cap || a->size > 8 || a->size > cap - a->offset) return;
  for (uint32_t b = 0; b < a->size; b++) karg[a->offset + b] = (uint8_t)(v >> (8 * b));
}

/* Scratch for kernel k: grow the backing if it holds fewer than full-occupancy waves */
static int scratch_for(vi_ctx *c, const amd_kernel *k, vi_scratch_cfg *cfg) {
  memset(cfg, 0, sizeof *cfg);
  if (!k->private_segment_size) return 0;
  vi_scratch_cfg want;
  int r = vi_scratch_config(k->private_segment_size, c->cus, SCRATCH_MAX, &want);
  if (r) {
    fprintf(stderr, "rawgpu: kernel %s wants %u bytes of private memory per work-item; at most %llu are possible\n", k->name,
            k->private_segment_size, (unsigned long long)(SCRATCH_MAX / 4 / 64));
    return r;
  }
  if (want.bytes > c->scratch_bytes) {
    if ((r = vi_ctx_finish(c))) return r; /* nothing may be using the old backing */
    if (c->scratch_bytes) rg_ranges_free(&c->heap, c->scratch_off, c->scratch_bytes);
    c->scratch_bytes = 0;
    uint64_t off;
    if (rg_ranges_alloc(&c->heap, want.bytes, 1u << 16, &off)) {
      fprintf(stderr, "rawgpu: no VRAM for %llu MB of scratch\n", (unsigned long long)(want.bytes >> 20));
      return -ENOMEM;
    }
    c->scratch_off = off;
    c->scratch_bytes = want.bytes;
    vi_set_scratch_base(&c->v, c->g.vram_start + off);
  }
  return vi_scratch_config(k->private_segment_size, c->cus, c->scratch_bytes, cfg);
}

int vi_launch_async(vi_ctx *c, const vi_prog *pp, const amd_kernel *k, const void *args, uint32_t args_size,
                    uint32_t dyn_lds, const uint32_t grid[3], const uint32_t wg[3], uint32_t *seq) {
  *seq = 0;
  uint32_t groups[3];
  for (int i = 0; i < 3; i++) {
    if (!wg[i] || !grid[i] || grid[i] % wg[i]) {
      fprintf(stderr, "rawgpu: kernel %s: grid %u is not a positive multiple of the work-group size %u\n", k->name, grid[i], wg[i]);
      return -EINVAL;
    }
    groups[i] = grid[i] / wg[i];
  }
  uint64_t flat = (uint64_t)wg[0] * wg[1] * wg[2];
  if (flat > 1024 || (k->max_wg && flat > k->max_wg)) {
    fprintf(stderr, "rawgpu: kernel %s: work-group of %llu work-items; at most %u\n", k->name, (unsigned long long)flat,
            k->max_wg ? k->max_wg : 1024);
    return -EINVAL;
  }
  if (k->reqd_wg[0] && (wg[0] != k->reqd_wg[0] || wg[1] != k->reqd_wg[1] || wg[2] != k->reqd_wg[2])) {
    fprintf(stderr, "rawgpu: kernel %s requires work-groups of %ux%ux%u\n", k->name, k->reqd_wg[0], k->reqd_wg[1], k->reqd_wg[2]);
    return -EINVAL;
  }
  vi_scratch_cfg scfg;
  int r = scratch_for(c, k, &scfg);
  if (r) return r;

  /* the kernarg slot: dispatch packet, then the kernarg segment */
  uint32_t kbytes = (args_size > k->kernarg_size ? args_size : k->kernarg_size);
  uint32_t slot = (64 + ((kbytes + 15) & ~15u) + 255) & ~255u;
  if (slot > CTX_KARG_SIZE / 4) return -E2BIG;
  if (c->karg_head + slot > CTX_KARG_SIZE) { /* wrap: everything that used the ring must be done */
    if ((r = vi_ctx_finish(c))) return r;
    c->karg_head = 0;
  }
  uint8_t *buf = calloc(1, slot);
  if (!buf) return -ENOMEM;
  uint64_t slot_gpu = c->g.vram_start + CTX_KARG + c->karg_head;
  uint8_t *karg = buf + 64;
  if (args_size) memcpy(karg, args, args_size);
  unsigned dims = grid[2] > 1 ? 3 : grid[1] > 1 ? 2 : 1;
  for (uint32_t i = 0; i < k->nargs; i++) { /* code object v5 hidden arguments */
    const amd_arg *a = &k->args[i];
    switch (a->kind) {
    case AMD_ARG_HIDDEN_BLOCK_COUNT_X: case AMD_ARG_HIDDEN_BLOCK_COUNT_Y: case AMD_ARG_HIDDEN_BLOCK_COUNT_Z:
      put_hidden(karg, kbytes, a, groups[a->kind - AMD_ARG_HIDDEN_BLOCK_COUNT_X]); break;
    case AMD_ARG_HIDDEN_GROUP_SIZE_X: case AMD_ARG_HIDDEN_GROUP_SIZE_Y: case AMD_ARG_HIDDEN_GROUP_SIZE_Z:
      put_hidden(karg, kbytes, a, wg[a->kind - AMD_ARG_HIDDEN_GROUP_SIZE_X]); break;
    case AMD_ARG_HIDDEN_GRID_DIMS: put_hidden(karg, kbytes, a, dims); break;
    case AMD_ARG_HIDDEN_DYNAMIC_LDS_SIZE: put_hidden(karg, kbytes, a, dyn_lds); break;
    default: break; /* offsets, remainders (uniform grids), LDS size: zero */
    }
  }
  aql_dispatch pkt = {
    .header = 2 /* HSA_PACKET_TYPE_KERNEL_DISPATCH */, .setup = (uint16_t)dims,
    .workgroup_size_x = (uint16_t)wg[0], .workgroup_size_y = (uint16_t)wg[1], .workgroup_size_z = (uint16_t)wg[2],
    .grid_size_x = grid[0], .grid_size_y = grid[1], .grid_size_z = grid[2],
    .private_segment_size = k->private_segment_size, .group_segment_size = k->group_segment_size + dyn_lds,
    .kernel_object = pp->gpu + k->entry, .kernarg_address = slot_gpu + 64,
  };
  memcpy(buf, &pkt, sizeof pkt);
  r = vi_vram_write(&c->v, CTX_KARG + c->karg_head, buf, 64 + ((kbytes + 15) & ~15u));
  free(buf);
  if (r) return r;
  vi_hdp_flush(&c->v);
  c->karg_head += slot;

  vi_dispatch dd = {.code_gpu = pp->gpu + k->entry, .kernarg_gpu = slot_gpu + 64, .packet_gpu = slot_gpu,
                    .scratch_gpu = c->g.vram_start + c->scratch_off, .scratch = scfg, .dispatch_id = c->dispatch_id++,
                    .lds_bytes = k->group_segment_size + dyn_lds};
  uint32_t s = vi_cq_dispatch(&c->cq, k, &dd, wg, groups);
  if (!s) return c->v.err ? c->v.err : -EIO;
  c->last_seq = s;
  *seq = s;
  return 0;
}

int vi_launch(vi_ctx *c, const vi_kern *kk, const void *args, uint32_t args_size, const uint32_t grid[3],
              const uint32_t wg[3], double *seconds) {
  double t = vi_now();
  uint32_t seq;
  int r = vi_launch_async(c, &kk->prog, kk->k, args, args_size, 0, grid, wg, &seq);
  if (!r) r = vi_ctx_wait(c, seq, TIMEOUT_MS);
  if (seconds) *seconds = vi_now() - t;
  return r;
}
