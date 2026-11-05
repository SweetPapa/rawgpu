/* rawgpu: run compute kernels on a PCIe GPU from userspace.
 *
 * The library behind the `rawgpu` CLI, for programs that want to use the GPU.
 * Today it drives an AMD Polaris card (RX 470/480/570/580/590) attached to an
 * Apple Silicon Mac through tinygrad's TinyGPU driver. See docs/LIBRARY.md.
 *
 *   rawgpu *g;
 *   rawgpu_open(&g, NULL, RAWGPU_OPEN_BRINGUP);
 *   rawgpu_buf *x = rawgpu_alloc(g, n * 4, RAWGPU_MEM_VRAM);
 *   rawgpu_write(g, x, 0, host, n * 4);
 *   rawgpu_program *p = rawgpu_program_build(g, source, NULL);
 *   rawgpu_kernel *k = rawgpu_kernel_get(p, "scale");
 *   rawgpu_set_arg_buf(k, 0, x);
 *   rawgpu_set_arg(k, 1, sizeof(float), &(float){2.0f});
 *   rawgpu_launch(g, k, 1, (size_t[]){n}, NULL);
 *   rawgpu_read(g, x, 0, host, n * 4);   // waits for the launch
 *
 * Kernels are OpenCL C (1.2 by default), compiled with clang for gfx803 and
 * linked with AMD's device libraries, or any self-contained AMDGPU code object
 * built for gfx803.
 *
 * Semantics
 * - Launches are asynchronous and run in the order they were made.
 *   Transfers (write, read, copy, fill) first wait for every launch, then
 *   complete before they return.
 * - Host buffers (RAWGPU_MEM_HOST) are shared with the CPU. Touch them from
 *   the CPU only when no launch that uses them is pending (rawgpu_finish()).
 * - Functions that can fail return 0 or a negative errno value (pointer
 *   returns: NULL; launches: event 0), and rawgpu_error() describes the last
 *   failure. Diagnostics from the driver also go to stderr.
 * - A context is not thread-safe, and one process at a time can use the GPU.
 */
#ifndef RAWGPU_H
#define RAWGPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RAWGPU_VERSION_MAJOR 0
#define RAWGPU_VERSION_MINOR 1

typedef struct rawgpu rawgpu;
typedef struct rawgpu_buf rawgpu_buf;
typedef struct rawgpu_program rawgpu_program;
typedef struct rawgpu_kernel rawgpu_kernel;
typedef uint64_t rawgpu_event; /* 0 = none / failure */

/* ---- device ---- */

/* If the card is not set up (after power-on or a Mac sleep), run the same
 * bring-up as `rawgpu post`, `rawgpu mc-load`, `rawgpu dpm-enable` first.
 * It prints progress to stdout and takes about 2 seconds. */
#define RAWGPU_OPEN_BRINGUP 1u

/* socket: the TinyGPU server's socket, NULL for the default. */
int rawgpu_open(rawgpu **out, const char *socket, unsigned flags);
void rawgpu_close(rawgpu *g);
/* the last failure's message, "" if none; valid until the next call (g may be NULL after a failed open) */
const char *rawgpu_error(const rawgpu *g);

typedef struct {
  char name[64];             /* e.g. "AMD POLARIS10, PCI 1002:67df rev e1, board 148c:2391" */
  char arch[16];             /* "gfx803" */
  unsigned compute_units;
  unsigned wavefront_size;   /* 64 */
  unsigned max_workgroup_size; /* 1024 (kernels default to 256 unless they ask for more) */
  unsigned local_mem_bytes;  /* LDS per work-group: 65536 */
  uint64_t vram_bytes, vram_free_bytes, vram_largest_free;
  int engine_mhz, memory_mhz; /* as the SMC reports them now; -1 if unknown */
  int temperature_c;
} rawgpu_info;
int rawgpu_get_info(rawgpu *g, rawgpu_info *info);

/* ---- memory ---- */

#define RAWGPU_MEM_VRAM 0u /* card memory: fastest for kernels (about 190 GB/s) */
#define RAWGPU_MEM_HOST 1u /* host memory the GPU reaches over PCIe (about 3 GB/s); rawgpu_buf_host() maps it */

rawgpu_buf *rawgpu_alloc(rawgpu *g, size_t bytes, unsigned flags);
void rawgpu_free(rawgpu *g, rawgpu_buf *b);
size_t rawgpu_buf_size(const rawgpu_buf *b);
uint64_t rawgpu_buf_gpu_addr(const rawgpu_buf *b); /* what a kernel's pointer argument holds */
void *rawgpu_buf_host(const rawgpu_buf *b);        /* the CPU mapping of a host buffer, NULL for VRAM */

int rawgpu_write(rawgpu *g, rawgpu_buf *dst, size_t offset, const void *src, size_t bytes);
int rawgpu_read(rawgpu *g, const rawgpu_buf *src, size_t offset, void *dst, size_t bytes);
int rawgpu_copy(rawgpu *g, rawgpu_buf *dst, size_t dst_offset, const rawgpu_buf *src, size_t src_offset, size_t bytes);
/* offset and bytes multiples of 4; the 32-bit pattern repeats */
int rawgpu_fill(rawgpu *g, rawgpu_buf *dst, size_t offset, size_t bytes, uint32_t pattern);

/* ---- code ---- */

/* OpenCL C source, compiled by clang (AMDGPU target) with AMD's device
 * libraries, and cached by content: the same source and options compile once.
 * options: extra clang flags separated by spaces (-D..., -cl-std=CL2.0,
 * -cl-fast-relaxed-math...), or NULL. Compiler errors are in rawgpu_error().
 * Environment: RAWGPU_CLANG, RAWGPU_DEVICE_LIBS, RAWGPU_CACHE override the
 * compiler, the bitcode directory and the cache directory. */
rawgpu_program *rawgpu_program_build(rawgpu *g, const char *source, const char *options);
/* an AMDGPU code object for gfx803 (relocatable or linked), from a file or memory */
rawgpu_program *rawgpu_program_load(rawgpu *g, const char *path);
rawgpu_program *rawgpu_program_load_mem(rawgpu *g, const void *data, size_t bytes);
/* waits for pending launches of its kernels */
void rawgpu_program_free(rawgpu *g, rawgpu_program *p);
unsigned rawgpu_program_num_kernels(const rawgpu_program *p);
const char *rawgpu_program_kernel_name(const rawgpu_program *p, unsigned index);

/* owned by the program; NULL if there is no such kernel or this GPU cannot run it */
rawgpu_kernel *rawgpu_kernel_get(rawgpu_program *p, const char *name);

enum { RAWGPU_ARG_VALUE = 0, RAWGPU_ARG_BUFFER = 1, RAWGPU_ARG_LOCAL = 2, RAWGPU_ARG_OTHER = 3 };
typedef struct {
  const char *name;      /* "" if the code object does not record names */
  const char *type_name; /* e.g. "float*", "uint" */
  unsigned size;         /* bytes in the kernarg segment */
  int kind;              /* RAWGPU_ARG_* */
  int is_const;
} rawgpu_arg_info;
typedef struct {
  unsigned num_args;
  unsigned max_workgroup_size;   /* the compiler's limit: 256 unless the kernel asks for more */
  unsigned reqd_workgroup_size[3]; /* 0 if any */
  unsigned local_mem_bytes;      /* static LDS */
  unsigned private_mem_bytes;    /* scratch per work-item */
  unsigned sgprs, vgprs;
} rawgpu_kernel_info;
int rawgpu_kernel_get_info(const rawgpu_kernel *k, rawgpu_kernel_info *info);
int rawgpu_kernel_arg_info(const rawgpu_kernel *k, unsigned index, rawgpu_arg_info *info);

/* Arguments, by index as in the kernel's signature. A value argument takes
 * exactly its size in bytes; a pointer to __global or __constant memory takes
 * a buffer (and an optional byte offset into it); a pointer to __local memory
 * takes the number of bytes to give it per work-group. */
int rawgpu_set_arg(rawgpu_kernel *k, unsigned index, size_t size, const void *value);
int rawgpu_set_arg_buf(rawgpu_kernel *k, unsigned index, const rawgpu_buf *b);
int rawgpu_set_arg_buf_offset(rawgpu_kernel *k, unsigned index, const rawgpu_buf *b, size_t offset);
int rawgpu_set_arg_local(rawgpu_kernel *k, unsigned index, size_t bytes);

/* ---- launches ---- */

/* dims 1-3. global[] counts work-items and must be a multiple of local[];
 * local NULL picks a work-group size (the kernel's required one, else the
 * largest power-of-two shape up to 256 that divides global). Every argument
 * must be set. Returns an event, 0 on error. */
rawgpu_event rawgpu_launch(rawgpu *g, rawgpu_kernel *k, unsigned dims, const size_t *global, const size_t *local);
int rawgpu_wait(rawgpu *g, rawgpu_event e); /* 10-second timeout: a kernel that runs longer is treated as hung */
int rawgpu_is_done(rawgpu *g, rawgpu_event e);  /* 1, 0, or a negative error */
int rawgpu_finish(rawgpu *g);

/* ---- built-in kernels ---- */

/* C = alpha * op(A) * op(B) + beta * C, row-major, op(X) = X or X^T.
 * M x N result, inner dimension K, leading dimensions in elements. A
 * register-blocked kernel (128x128 tiles, kernels/sgemm.cl); docs/LIBRARY.md
 * has measured speeds. beta == 0 never reads C. Asynchronous. */
rawgpu_event rawgpu_sgemm(rawgpu *g, int trans_a, int trans_b, size_t M, size_t N, size_t K, float alpha,
                          const rawgpu_buf *A, size_t lda, const rawgpu_buf *B, size_t ldb, float beta, rawgpu_buf *C,
                          size_t ldc);

#ifdef __cplusplus
}
#endif
#endif
