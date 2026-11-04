/* `rawgpu bench-compute`: memory bandwidth, FP32 throughput and an LDS-tiled
 * SGEMM, through the host API (amd_vi_ctx.h). Every result is checked
 * against the CPU before its time is printed. */
#include "amd_vi_ctx.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int best_of(vi_ctx *c, const vi_kern *k, const void *args, uint32_t n, const uint32_t grid[3],
                   const uint32_t wg[3], int reps, double *best) {
  *best = 1e9;
  for (int i = 0; i < reps; i++) {
    double t;
    int r = vi_launch(c, k, args, n, grid, wg, &t);
    if (r) return r;
    if (t < *best) *best = t;
  }
  return 0;
}

static uint32_t pat(uint32_t i) { uint32_t x = i * 0x9E3779B1u; x ^= x >> 15; return x * 0x2C1B3C6Du; }

/* 0: the engine clock under load, measured on the GPU: s_memtime ticks over
 * the host's wall time for a launch long enough that submit overhead is noise */
static int bench_clock(vi_ctx *c, const vi_kern *k) {
  vi_buf out;
  uint64_t got[3] = {0};
  int r = vi_buf_alloc(c, &out, 4096, true);
  struct { uint64_t out; uint32_t iters; } __attribute__((packed)) args = {out.gpu, 1u << 22};
  const uint32_t grid[3] = {64, 1, 1}, wg[3] = {64, 1, 1};
  double t = 0;
  if (!r) r = vi_launch(c, k, &args, sizeof args, grid, wg, &t);
  if (!r) r = vi_buf_read(c, &out, 0, got, sizeof got);
  if (!r)
    printf("clock:        one wave, %u loop iterations in %.1f ms: engine (s_memtime) %.1f MHz, reference "
           "(s_memrealtime) %.1f MHz\n", args.iters, t * 1e3, got[0] / t / 1e6, got[1] / t / 1e6);
  vi_buf_free(c, &out);
  return r;
}

/* 1: copy4 over 256 MB, VRAM to VRAM */
static int bench_copy(vi_ctx *c, const vi_kern *k) {
  const uint64_t bytes = 256ull << 20, head = 16ull << 20;
  vi_buf src, dst;
  uint32_t *buf = malloc(head);
  int r = buf ? 0 : -ENOMEM;
  if (!r) r = vi_buf_alloc(c, &src, bytes, false);
  if (!r) r = vi_buf_alloc(c, &dst, bytes, false);
  if (!r) r = vi_buf_fill(c, &src, 0x3F800000u); /* 1.0f, then a pattern in the first 16 MB */
  if (!r) r = vi_buf_fill(c, &dst, 0);
  for (uint64_t i = 0; !r && i < head / 4; i++) buf[i] = pat((uint32_t)i);
  if (!r) r = vi_buf_write(c, &src, 0, buf, head);
  struct { uint64_t src, dst; } args = {src.gpu, dst.gpu};
  const uint32_t grid[3] = {(uint32_t)(bytes / 16), 1, 1}, wg[3] = {256, 1, 1};
  double t = 0;
  if (!r) r = best_of(c, k, &args, sizeof args, grid, wg, 3, &t);
  size_t bad = 0;
  if (!r) r = vi_buf_read(c, &dst, 0, buf, head);
  for (uint64_t i = 0; !r && i < head / 4; i++) bad += buf[i] != pat((uint32_t)i);
  if (!r) r = vi_buf_read(c, &dst, bytes - (1u << 20), buf, 1u << 20);
  for (uint64_t i = 0; !r && i < (1u << 18); i++) bad += buf[i] != 0x3F800000u;
  if (!r)
    printf("copy:         256 MB VRAM -> VRAM: %.2f ms, %.1f GB/s (read + write); first 16 MB and last 1 MB %s\n",
           t * 1e3, 2.0 * bytes / t / 1e9, bad ? "MISMATCH" : "intact");
  free(buf);
  return r ? r : bad ? -EIO : 0;
}

/* 2: fma_peak / mad_peak, 1M work-items x `iters` rounds of 8 multiply-adds.
 * The CPU reference is fmaf() for the fused kernel, and a separately rounded
 * multiply and add for v_mad_f32 (separate statements, so C does not contract them). */
static float flops_ref(uint32_t i, uint32_t iters, float a, float b, bool fused) {
  float x[8];
  x[0] = (float)i * 1e-6f;
  for (int j = 1; j < 8; j++) x[j] = x[0] + 0.125f * (float)j;
  for (uint32_t k = 0; k < iters; k++)
    for (int j = 0; j < 8; j++) {
      if (fused) { x[j] = fmaf(x[j], a, b); continue; }
      float p = x[j] * a;
      x[j] = p + b;
    }
  return x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
}

static int bench_flops(vi_ctx *c, const vi_kern *k, bool fused) {
  const uint32_t n = 1u << 20, iters = 4096;
  const float a = 0.9990234375f, b = 0.0009765625f; /* exact in binary */
  vi_buf out;
  float *got = malloc((size_t)n * 4);
  int r = got ? vi_buf_alloc(c, &out, (uint64_t)n * 4, false) : -ENOMEM;
  struct { uint64_t out; uint32_t iters; float a, b; } __attribute__((packed)) args = {out.gpu, iters, a, b};
  const uint32_t grid[3] = {n, 1, 1}, wg[3] = {256, 1, 1};
  double t = 0;
  if (!r) r = best_of(c, k, &args, sizeof args, grid, wg, 3, &t);
  if (!r) r = vi_buf_read(c, &out, 0, got, (size_t)n * 4);
  size_t bad = 0, checked = 0;
  for (uint32_t i = 0; !r && i < n; i += 256, checked++) {
    float want = flops_ref(i, iters, a, b, fused);
    bad += memcmp(&got[i], &want, 4) != 0;
  }
  if (!r)
    printf("fp32 %s:     %u work-items x %u x 8 %s: %.2f ms, %.0f GFLOPS; %zu samples %s the CPU\n", fused ? "fma" : "mad",
           n, iters, fused ? "FMAs (fused: quarter rate on Polaris)" : "MADs (v_mad_f32: full rate)", t * 1e3,
           (double)n * iters * 16 / t / 1e9, checked, bad ? "DIFFER from" : "match");
  free(got);
  return r ? r : bad ? -EIO : 0;
}

/* 3: sgemm n x n. Entries are multiples of 1/16 in [-0.5, 0.5], so every
 * partial sum is exact in float and any summation order gives the same bits. */
static int bench_sgemm(vi_ctx *c, const vi_kern *k, uint32_t n) {
  size_t elems = (size_t)n * n, bytes = elems * 4;
  float *A = malloc(bytes), *B = malloc(bytes), *C = malloc(bytes);
  vi_buf a, b, cc;
  int r = A && B && C ? 0 : -ENOMEM;
  for (size_t i = 0; !r && i < elems; i++) {
    A[i] = (float)((int)((i * 7 + 3) % 17) - 8) / 16.0f;
    B[i] = (float)((int)((i * 5 + 1) % 13) - 6) / 16.0f;
  }
  if (!r) r = vi_buf_alloc(c, &a, bytes, false);
  if (!r) r = vi_buf_alloc(c, &b, bytes, false);
  if (!r) r = vi_buf_alloc(c, &cc, bytes, false);
  if (!r) r = vi_buf_write(c, &a, 0, A, bytes);
  if (!r) r = vi_buf_write(c, &b, 0, B, bytes);
  if (!r) r = vi_buf_fill(c, &cc, 0xFFFFFFFFu);
  struct { uint64_t a, b, c; uint32_t n; } __attribute__((packed)) args = {a.gpu, b.gpu, cc.gpu, n};
  const uint32_t grid[3] = {n, n, 1}, wg[3] = {16, 16, 1};
  double t = 0;
  if (!r) r = best_of(c, k, &args, sizeof args, grid, wg, 3, &t);
  if (!r) r = vi_buf_read(c, &cc, 0, C, bytes);
  size_t bad = 0, rows = 0;
  for (uint32_t row = 0; !r && row < n; row += n / 64 ? n / 64 : 1, rows++)
    for (uint32_t col = 0; col < n; col++) {
      float acc = 0.0f;
      for (uint32_t kk = 0; kk < n; kk++) acc = fmaf(A[(size_t)row * n + kk], B[(size_t)kk * n + col], acc);
      bad += memcmp(&acc, &C[(size_t)row * n + col], 4) != 0;
    }
  if (!r)
    printf("sgemm:        %ux%u, 16x16 LDS tiles: %.2f ms, %.0f GFLOPS; %zu rows %s the CPU\n", n, n, t * 1e3,
           2.0 * n * n * n / t / 1e9, rows, bad ? "DIFFER from" : "match");
  free(A);
  free(B);
  free(C);
  return r ? r : bad ? -EIO : 0;
}

int vi_cmd_bench_compute(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  const char *dir = "build/kernels";
  uint32_t n = 1024;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--kernels") && i + 1 < argc) dir = argv[++i];
    else if (!strcmp(argv[i], "--sgemm") && i + 1 < argc) n = (uint32_t)strtoul(argv[++i], NULL, 10);
    else return -EINVAL;
  }
  if (n < 16 || n > 4096 || n % 16) { fprintf(stderr, "rawgpu: --sgemm N must be a multiple of 16 in 16..4096\n"); return -EINVAL; }
  vi_ctx c;
  int r = vi_ctx_open(&c, d, id);
  if (r) return r;
  vi_kern kc = {0}, kf = {0}, km = {0}, ks = {0}, kk = {0};
  char path[512];
  snprintf(path, sizeof path, "%s/copy.o", dir);
  if (!r) r = vi_kern_load(&c, &kc, path, "copy4");
  snprintf(path, sizeof path, "%s/fma.o", dir);
  if (!r) r = vi_kern_load(&c, &kf, path, "fma_peak");
  if (!r) r = vi_kern_load(&c, &km, path, "mad_peak");
  snprintf(path, sizeof path, "%s/sgemm.o", dir);
  if (!r) r = vi_kern_load(&c, &ks, path, "sgemm");
  snprintf(path, sizeof path, "%s/clock.o", dir);
  if (!r) r = vi_kern_load(&c, &kk, path, "clock_probe");
  if (!r) {
    printf("card:         %d compute units active; temperature %d C\n", vi_gfx_cu_count(&c.v), vi_temp_c(&c.v));
    r = bench_clock(&c, &kk);
  }
  if (!r) {
    r = bench_copy(&c, &kc);
  }
  if (!r) r = bench_flops(&c, &km, false);
  if (!r) r = bench_flops(&c, &kf, true);
  if (!r) r = bench_sgemm(&c, &ks, n);
  printf("temperature:  %d C\n", vi_temp_c(&c.v));
  vi_kern_free(&kc);
  vi_kern_free(&kf);
  vi_kern_free(&km);
  vi_kern_free(&ks);
  vi_kern_free(&kk);
  vi_ctx_close(&c);
  if (r && r != -EIO) fprintf(stderr, "rawgpu: bench-compute failed (%d)\n", r);
  return r;
}
