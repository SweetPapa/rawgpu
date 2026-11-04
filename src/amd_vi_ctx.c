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
#define CTX_KARG    (CTX_CQ_WB + 0x1000)        /* kernarg block of the running launch */
#define CTX_KARG_MAX 4096u
#define CTX_CODE    (33ull << 20)               /* kernel code, up to 48 MB */
#define CTX_CODE_END (48ull << 20)
#define CTX_USER    (64ull << 20)               /* first VRAM byte for buffers */
#define STAGING_BYTES (16u << 20)
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
    if (!v->err) fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post`, then `rawgpu smc-load`\n");
    return v->err ? v->err : -EIO;
  }
  if (vi_smu_ucode_status(v, &status) || (status & need) != need) {
    if (!v->err) fprintf(stderr, "rawgpu: engine microcode is not loaded (UcodeLoadStatus 0x%08x); run `rawgpu smc-load`\n", status);
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
  c->code_next = CTX_CODE;
  c->vram_next = CTX_USER;
  return 0;
fail:
  vi_ctx_close(c);
  return r ? r : -EIO;
}

void vi_ctx_close(vi_ctx *c) {
  rg_dev *d = c->v.d;
  if (!d) return;
  /* engines first: host memory they point at goes away with the DMA buffers */
  vi_cq_stop(&c->cq);
  { int e = c->v.err; c->v.err = 0; vi_sdma_stop(&c->v, 0); if (!c->v.err) c->v.err = e; }
  vi_sdma_free(&c->sdma);
  if (c->staging.cpu) d->ops->dma_free(d, &c->staging);
  if (c->ctrl.cpu) d->ops->dma_free(d, &c->ctrl);
  c->v.d = NULL;
}

int vi_buf_alloc(vi_ctx *c, vi_buf *b, uint64_t size, bool host) {
  memset(b, 0, sizeof *b);
  b->dma.fd = -1;
  size = (size + 4095) & ~4095ull;
  b->size = size;
  b->host = host;
  if (host) {
    int r = c->v.d->ops->dma_alloc(c->v.d, size, &b->dma);
    if (!r) r = vi_gart_map(&c->v, &c->g, &b->dma, &b->gpu);
    if (r) { if (b->dma.cpu) c->v.d->ops->dma_free(c->v.d, &b->dma); return r; }
    b->cpu = b->dma.cpu;
    return 0;
  }
  uint64_t off = (c->vram_next + 0xFFFF) & ~0xFFFFull; /* 64 KB aligned */
  if (off + size > c->g.vram_size) {
    fprintf(stderr, "rawgpu: out of VRAM (%llu MB asked, %llu MB left)\n", (unsigned long long)(size >> 20),
            (unsigned long long)((c->g.vram_size - off) >> 20));
    return -ENOMEM;
  }
  c->vram_next = off + size;
  b->gpu = c->g.vram_start + off;
  return 0;
}

void vi_buf_free(vi_ctx *c, vi_buf *b) {
  if (b->host && b->dma.cpu) c->v.d->ops->dma_free(c->v.d, &b->dma);
  memset(b, 0, sizeof *b);
}

static int sdma_sync(vi_ctx *c, uint32_t seq) {
  int r = seq ? vi_sdma_wait(&c->sdma, seq, TIMEOUT_MS) : (c->v.err ? c->v.err : -EIO);
  if (r) vi_sdma_dump(&c->sdma);
  return r;
}

int vi_buf_write(vi_ctx *c, vi_buf *b, uint64_t off, const void *src, uint64_t len) {
  if (off > b->size || len > b->size - off) return -EINVAL;
  if (b->host) { memcpy((uint8_t *)b->cpu + off, src, len); return 0; }
  for (uint64_t done = 0; done < len;) {
    uint64_t n = len - done < STAGING_BYTES ? len - done : STAGING_BYTES;
    memcpy(c->staging.cpu, (const uint8_t *)src + done, n);
    int r = sdma_sync(c, vi_sdma_copy(&c->sdma, b->gpu + off + done, c->staging_gpu, n));
    if (r) return r;
    done += n;
  }
  return 0;
}

int vi_buf_read(vi_ctx *c, const vi_buf *b, uint64_t off, void *dst, uint64_t len) {
  if (off > b->size || len > b->size - off) return -EINVAL;
  if (b->host) { memcpy(dst, (const uint8_t *)b->cpu + off, len); return 0; }
  for (uint64_t done = 0; done < len;) {
    uint64_t n = len - done < STAGING_BYTES ? len - done : STAGING_BYTES;
    int r = sdma_sync(c, vi_sdma_copy(&c->sdma, c->staging_gpu, b->gpu + off + done, n));
    if (r) return r;
    memcpy((uint8_t *)dst + done, c->staging.cpu, n);
    done += n;
  }
  return 0;
}

int vi_buf_fill(vi_ctx *c, vi_buf *b, uint32_t value) {
  return sdma_sync(c, vi_sdma_fill(&c->sdma, b->gpu, value, b->size));
}

int vi_kern_load(vi_ctx *c, vi_kern *kk, const char *path, const char *name) {
  memset(kk, 0, sizeof *kk);
  int r = amd_kernel_load(&kk->k, path, name);
  if (!r) r = vi_kernel_check(&kk->k);
  if (r) { amd_kernel_free(&kk->k); return r; }
  /* zero padding after the code: the shader prefetches past its last instruction */
  uint64_t region = ((kk->k.code_size + 4095) & ~4095ull) + 4096;
  if (c->code_next + region > CTX_CODE_END) { amd_kernel_free(&kk->k); return -ENOSPC; }
  if ((r = vi_vram_zero(&c->v, c->code_next, region)) || (r = vi_vram_write(&c->v, c->code_next, kk->k.code, kk->k.code_size))) {
    amd_kernel_free(&kk->k);
    return r;
  }
  vi_hdp_flush(&c->v);
  kk->code_gpu = c->g.vram_start + c->code_next;
  c->code_next += region;
  return c->v.err;
}

void vi_kern_free(vi_kern *kk) { amd_kernel_free(&kk->k); }

int vi_launch(vi_ctx *c, const vi_kern *kk, const void *args, uint32_t args_size, const uint32_t grid[3],
              const uint32_t wg[3], double *seconds) {
  uint32_t groups[3];
  for (int i = 0; i < 3; i++) {
    if (!wg[i] || grid[i] % wg[i]) {
      fprintf(stderr, "rawgpu: grid %u is not a multiple of the work-group size %u\n", grid[i], wg[i]);
      return -EINVAL;
    }
    groups[i] = grid[i] / wg[i];
  }
  uint32_t karg_bytes = args_size > kk->k.kernarg_size ? args_size : kk->k.kernarg_size;
  if (karg_bytes > CTX_KARG_MAX) return -E2BIG;
  uint8_t karg[CTX_KARG_MAX] = {0}; /* hidden arguments, if the descriptor counts any, stay zero */
  memcpy(karg, args, args_size);
  int r = vi_vram_write(&c->v, CTX_KARG, karg, (karg_bytes + 15) & ~15u);
  if (r) return r;
  vi_hdp_flush(&c->v);
  double t = vi_now();
  uint32_t seq = vi_cq_dispatch(&c->cq, &kk->k, kk->code_gpu, c->g.vram_start + CTX_KARG, wg, groups);
  r = seq ? vi_cq_wait(&c->cq, seq, TIMEOUT_MS) : (c->v.err ? c->v.err : -EIO);
  if (seconds) *seconds = vi_now() - t;
  if (r) vi_cq_dump(&c->cq);
  return r;
}
