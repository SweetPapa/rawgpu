/* rawgpu's host API for the amd-vi backend: open the card, allocate memory,
 * move data, load code objects, run kernels. Built on the bring-up pieces in
 * amd_vi*.c; the public library (include/rawgpu.h) wraps it.
 *
 * A context owns the GART, SDMA0 (copies and fills) and one compute queue.
 * Launches are asynchronous and run in submission order; memory transfers
 * wait for every launch before they start and are complete when they return.
 * The card must be up: `rawgpu post`, `mc-load`, `dpm-enable` (see README).
 */
#ifndef RAWGPU_AMD_VI_CTX_H
#define RAWGPU_AMD_VI_CTX_H

#include "amd_kernel.h"
#include "amd_vi.h"

typedef struct {
  vi_dev v;
  vi_gart g;
  rg_dma ctrl;              /* host: SDMA ring | SDMA writeback | VM fault page */
  rg_dma staging;           /* host bounce buffer for VRAM reads and writes */
  uint64_t ctrl_gpu, staging_gpu;
  vi_sdma sdma;
  vi_cq cq;
  rg_ranges code;           /* free VRAM inside BAR0 for code images (byte offsets) */
  rg_ranges heap;           /* free VRAM for buffers and scratch (byte offsets) */
  uint32_t karg_head;       /* next free byte of the kernarg ring */
  uint32_t last_seq;        /* newest compute fence submitted */
  uint64_t dispatch_id;
  uint64_t scratch_off, scratch_bytes; /* VRAM backing for private memory; grows on demand */
  unsigned cus;
} vi_ctx;

typedef struct {
  uint64_t gpu;   /* MC address: what kernels and engines use */
  uint64_t size;
  bool host;      /* host memory (cpu is valid) or VRAM */
  void *cpu;
  rg_dma dma;     /* host buffers only */
} vi_buf;

typedef struct {
  amd_program p;
  uint64_t off, region; /* the VRAM the image occupies */
  uint64_t gpu;         /* MC address of the image */
} vi_prog;

/* one kernel loaded on its own (the CLI's benchmarks and tests) */
typedef struct {
  vi_prog prog;
  const amd_kernel *k;
} vi_kern;

int vi_ctx_open(vi_ctx *c, rg_dev *d, const rg_ident *id);
void vi_ctx_close(vi_ctx *c);

/* VRAM (host = false) or GART-mapped host memory, in 4 KB units. VRAM can be
 * all of the card's; host buffers share the 256 MB GART. */
int vi_buf_alloc(vi_ctx *c, vi_buf *b, uint64_t size, bool host);
void vi_buf_free(vi_ctx *c, vi_buf *b);
int vi_buf_write(vi_ctx *c, vi_buf *b, uint64_t off, const void *src, uint64_t len);
int vi_buf_read(vi_ctx *c, const vi_buf *b, uint64_t off, void *dst, uint64_t len);
/* SDMA copy between any two buffers; offsets and length in bytes */
int vi_buf_copy(vi_ctx *c, vi_buf *dst, uint64_t doff, const vi_buf *src, uint64_t soff, uint64_t len);
/* SDMA constant fill; off and len must be multiples of 4 */
int vi_buf_fill_range(vi_ctx *c, vi_buf *b, uint64_t off, uint64_t len, uint32_t value);
int vi_buf_fill(vi_ctx *c, vi_buf *b, uint32_t value); /* the whole buffer */

/* Load a code object: relocate it for its VRAM address and upload it */
int vi_prog_load(vi_ctx *c, vi_prog *pp, const char *path);
int vi_prog_load_mem(vi_ctx *c, vi_prog *pp, const void *data, size_t size, const char *label);
void vi_prog_free(vi_ctx *c, vi_prog *pp);

int vi_kern_load(vi_ctx *c, vi_kern *kk, const char *path, const char *name);
void vi_kern_free(vi_ctx *c, vi_kern *kk);

/* Queue a dispatch of kernel k from program pp. grid[] counts work-items and
 * must be a multiple of wg[]. `args` holds the explicit arguments laid out
 * as the kernel's metadata says (offset 0 = first argument); the hidden
 * arguments and the dispatch packet are filled in here. dyn_lds is the LDS
 * for __local pointer arguments, beyond the kernel's own (their offsets are
 * the caller's: from group_segment_size up). *seq gets the fence to wait for. */
int vi_launch_async(vi_ctx *c, const vi_prog *pp, const amd_kernel *k, const void *args, uint32_t args_size,
                    uint32_t dyn_lds, const uint32_t grid[3], const uint32_t wg[3], uint32_t *seq);
int vi_ctx_wait(vi_ctx *c, uint32_t seq, unsigned timeout_ms);
bool vi_ctx_done(vi_ctx *c, uint32_t seq);
int vi_ctx_finish(vi_ctx *c); /* every launch so far */
/* vi_launch_async() and wait; *seconds (if non-NULL) gets submit-to-fence time */
int vi_launch(vi_ctx *c, const vi_kern *kk, const void *args, uint32_t args_size, const uint32_t grid[3],
              const uint32_t wg[3], double *seconds);

#endif
