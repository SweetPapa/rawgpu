/* rawgpu's host API for the amd-vi backend: open the card, allocate buffers,
 * move data, run kernels. Built on the bring-up pieces in amd_vi*.c.
 *
 * A context owns the GART, SDMA0 (for copies and fills) and one compute queue.
 * Everything is synchronous: each call waits for the GPU before it returns.
 * The card must be posted and have run `smc-load` (see README).
 */
#ifndef RAWGPU_AMD_VI_CTX_H
#define RAWGPU_AMD_VI_CTX_H

#include "amd_vi.h"

typedef struct {
  vi_dev v;
  vi_gart g;
  rg_dma ctrl;              /* host: SDMA ring | SDMA writeback | VM fault page */
  rg_dma staging;           /* host bounce buffer for VRAM reads and writes */
  uint64_t ctrl_gpu, staging_gpu;
  vi_sdma sdma;
  vi_cq cq;
  uint64_t code_next;       /* VRAM bump allocators */
  uint64_t vram_next;
} vi_ctx;

typedef struct {
  uint64_t gpu;   /* MC address: what kernels and engines use */
  uint64_t size;
  bool host;      /* host memory (cpu is valid) or VRAM */
  void *cpu;
  rg_dma dma;     /* host buffers only */
} vi_buf;

typedef struct {
  amd_kernel k;
  uint64_t code_gpu;
} vi_kern;

int vi_ctx_open(vi_ctx *c, rg_dev *d, const rg_ident *id);
void vi_ctx_close(vi_ctx *c);

/* VRAM (host = false) comes from a bump allocator and is never reused within
 * a context. Host buffers are GART-mapped: together they must fit the 256 MB
 * GART, and their GART pages are not reclaimed either. */
int vi_buf_alloc(vi_ctx *c, vi_buf *b, uint64_t size, bool host);
void vi_buf_free(vi_ctx *c, vi_buf *b);
int vi_buf_write(vi_ctx *c, vi_buf *b, uint64_t off, const void *src, uint64_t len);
int vi_buf_read(vi_ctx *c, const vi_buf *b, uint64_t off, void *dst, uint64_t len);
int vi_buf_fill(vi_ctx *c, vi_buf *b, uint32_t value); /* whole buffer, by SDMA */

int vi_kern_load(vi_ctx *c, vi_kern *kk, const char *path, const char *name);
void vi_kern_free(vi_kern *kk);
/* grid[] counts work-items and must be a multiple of wg[]; `args` is the
 * kernarg block. *seconds (if non-NULL) gets submit-to-fence time. */
int vi_launch(vi_ctx *c, const vi_kern *kk, const void *args, uint32_t args_size, const uint32_t grid[3],
              const uint32_t wg[3], double *seconds);

#endif
