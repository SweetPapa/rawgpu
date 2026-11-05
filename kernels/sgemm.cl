/* SGEMM for Polaris (gfx803): C = alpha * op(A) * op(B) + beta * C, all
 * row-major, M x N result, inner dimension K, leading dimensions in floats.
 *
 * sgemm_nn/nt/tn/tt: register-blocked. A 16x16 work-group computes a 128x128
 * tile of C; each work-item owns 8x8 of it, in two 4-row by two 4-column
 * blocks 64 apart, so its LDS reads are float4 broadcasts with no bank
 * conflicts. K advances 8 at a time through double-buffered LDS slices: the
 * next slice is loaded from memory while the current one is multiplied, and
 * one barrier per slice keeps the two apart. Any M, N, K: edge tiles load
 * zeros and store only inside C.
 *
 * The multiply-adds are a * b + c, which becomes v_mac_f32: full rate. Fused
 * fma is quarter rate on Polaris, and v_mac flushes denormals like the rest
 * of this GPU's float code (results can differ from a fused CPU GEMM in the
 * last bit).
 *
 * Pad rows whose length is a multiple of a large power of two. A tile's 128
 * rows then evidently collide in the caches or memory channels: an nt
 * product with K = 2048 runs at 0.8 TFLOPS with lda = ldb = 2048, and at 4.6
 * with 2080 (a pad of 4 floats is enough).
 *
 * sgemm16: the first, simple kernel (one output per work-item, 16x16 tiles),
 * kept for comparison. n x n, n a multiple of 16. */

#define TM 128           /* tile rows (M) and columns (N) */
#define TK 8             /* K per LDS slice */
#define LS (TM + 4)      /* LDS row stride in floats: 528 bytes, 16-aligned, two k rows on different banks */
#ifndef WAVES_PER_EU
#define WAVES_PER_EU 2   /* at most 128 VGPRs, so two waves per SIMD can hide each other's latency */
#endif
#ifndef KUNROLL
#define KUNROLL 2        /* unrolling all 8 k steps lets the scheduler hoist every LDS read and spill */
#endif
#define STR_(x) #x
#define PRAGMA_UNROLL_(n) STR_(unroll n)
#define KUNROLL_PRAGMA PRAGMA_UNROLL_(KUNROLL)

/* 4 consecutive floats from p[off], the first `valid` of them real, the rest 0 */
static inline float4 load4(__global const float *p, size_t off, int valid) {
  if (valid >= 4) return vload4(0, p + off);
  float4 r = (float4)(0.0f);
  if (valid > 0) r.x = p[off];
  if (valid > 1) r.y = p[off + 1];
  if (valid > 2) r.z = p[off + 2];
  return r;
}

static inline int clamp4(long n) { return n <= 0 ? 0 : n >= 4 ? 4 : (int)n; }

/* One TK x TM slice of an operand, held in a register by each of the 256
 * work-items until it goes into LDS. Rows of the slice are k; columns are
 * the M (or N) index. `kcontig`: the operand's k index is the contiguous one
 * in memory (A not transposed, B transposed): each work-item reads 4 k values
 * of one row and stores them down an LDS column. Otherwise it reads 4
 * consecutive columns of one k row and stores them as a float4. */
static inline float4 slice_load(__global const float *X, uint ld, int kcontig, uint base, uint R, uint k0, uint K, uint tid) {
  if (kcontig) {
    uint r = tid >> 1, kk = (tid & 1) * 4, row = base + r;
    return load4(X, (size_t)row * ld + k0 + kk, row < R ? clamp4((long)K - (k0 + kk)) : 0);
  }
  uint kr = tid >> 5, c = (tid & 31) * 4, k = k0 + kr;
  return load4(X, (size_t)k * ld + base + c, k < K ? clamp4((long)R - (base + c)) : 0);
}

static inline void slice_store(__local float *S, int kcontig, uint tid, float4 v) {
  if (kcontig) {
    uint r = tid >> 1, kk = (tid & 1) * 4;
    S[(kk + 0) * LS + r] = v.x;
    S[(kk + 1) * LS + r] = v.y;
    S[(kk + 2) * LS + r] = v.z;
    S[(kk + 3) * LS + r] = v.w;
  } else {
    uint kr = tid >> 5, c = (tid & 31) * 4;
    *(__local float4 *)&S[kr * LS + c] = v;
  }
}

static inline void store_row(__global float *C, uint ldc, uint row, uint col, uint M, uint N, float4 v, float alpha,
                             float beta) {
  if (row >= M || col >= N) return;
  __global float *p = C + (size_t)row * ldc + col;
  int n = clamp4((long)N - col);
  if (n == 4) {
    float4 o = alpha * v;
    if (beta != 0.0f) o = beta * vload4(0, p) + o;
    vstore4(o, 0, p);
    return;
  }
  float e[4] = {v.x, v.y, v.z, v.w};
  for (int j = 0; j < n; j++) p[j] = beta != 0.0f ? beta * p[j] + alpha * e[j] : alpha * e[j];
}

/* The SGEMM itself, for A, B, C, M, N, K, lda, ldb, ldc, alpha, beta in scope. The tile's work-group
 * returns at once if SKIP (an expression of m0 and n0, the tile's first row and column) is true, and
 * multiplies over k in [KBEG, KEND) only (KBEG a multiple of TK). The plain kernels pass 0, 0, K. */
#define SGEMM_BODY(A_KCONTIG, B_KCONTIG, SKIP, KBEG, KEND) \
    __local float As[2][TK * LS] __attribute__((aligned(16)));                                                     \
    __local float Bs[2][TK * LS] __attribute__((aligned(16)));                                                     \
    uint tx = get_local_id(0), ty = get_local_id(1), tid = ty * 16 + tx;                                           \
    uint n0 = get_group_id(0) * TM, m0 = get_group_id(1) * TM;                                                     \
    if (SKIP) return;                                                                                              \
    uint kbeg = (KBEG), kend = (KEND);                                                                             \
    float acc[8][8];                                                                                               \
    for (int i = 0; i < 8; i++)                                                                                    \
      for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;                                                                \
    uint nk = kend > kbeg ? (kend - kbeg + TK - 1) / TK : 0;                                                       \
    float4 ra = slice_load(A, lda, A_KCONTIG, m0, M, kbeg, kend, tid);                                             \
    float4 rb = slice_load(B, ldb, B_KCONTIG, n0, N, kbeg, kend, tid);                                             \
    slice_store(As[0], A_KCONTIG, tid, ra);                                                                        \
    slice_store(Bs[0], B_KCONTIG, tid, rb);                                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \
    for (uint t = 0; t < nk; t++) {                                                                                \
      uint cur = t & 1;                                                                                            \
      bool more = t + 1 < nk;                                                                                      \
      if (more) {                                                                                                  \
        ra = slice_load(A, lda, A_KCONTIG, m0, M, kbeg + (t + 1) * TK, kend, tid);                                 \
        rb = slice_load(B, ldb, B_KCONTIG, n0, N, kbeg + (t + 1) * TK, kend, tid);                                 \
      }                                                                                                            \
      __local const float *as = As[cur], *bs = Bs[cur];                                                            \
      _Pragma(KUNROLL_PRAGMA) for (int k = 0; k < TK; k++) {                                                        \
        float4 a0 = *(__local const float4 *)&as[k * LS + ty * 4];                                                 \
        float4 a1 = *(__local const float4 *)&as[k * LS + 64 + ty * 4];                                            \
        float4 b0 = *(__local const float4 *)&bs[k * LS + tx * 4];                                                 \
        float4 b1 = *(__local const float4 *)&bs[k * LS + 64 + tx * 4];                                            \
        float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};                                             \
        float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};                                             \
        _Pragma("unroll") for (int i = 0; i < 8; i++)                                                              \
          _Pragma("unroll") for (int j = 0; j < 8; j++) acc[i][j] = a[i] * b[j] + acc[i][j];                       \
      }                                                                                                            \
      if (more) {                                                                                                  \
        slice_store(As[cur ^ 1], A_KCONTIG, tid, ra);                                                              \
        slice_store(Bs[cur ^ 1], B_KCONTIG, tid, rb);                                                              \
      }                                                                                                            \
      barrier(CLK_LOCAL_MEM_FENCE);                                                                                \
    }                                                                                                              \
    for (int i = 0; i < 8; i++) {                                                                                  \
      uint row = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));                                                \
      store_row(C, ldc, row, n0 + tx * 4, M, N, (float4)(acc[i][0], acc[i][1], acc[i][2], acc[i][3]), alpha, beta); \
      store_row(C, ldc, row, n0 + 64 + tx * 4, M, N, (float4)(acc[i][4], acc[i][5], acc[i][6], acc[i][7]), alpha,   \
                beta);                                                                                             \
    }

#define SGEMM(NAME, A_KCONTIG, B_KCONTIG)                                                                          \
  __kernel __attribute__((reqd_work_group_size(16, 16, 1), amdgpu_waves_per_eu(WAVES_PER_EU))) void NAME(       \
      __global const float *A, __global const float *B, __global float *C, uint M, uint N, uint K, uint lda,       \
      uint ldb, uint ldc, float alpha, float beta) {                                                                \
    SGEMM_BODY(A_KCONTIG, B_KCONTIG, 0, 0, K)                                                                      \
  }

/* Sliding-window attention: query q sees key k when |q - k| < window. band_keys() is the range of keys
 * that any query of the 128-row tile starting at m0 can see, its start rounded down to a multiple of TK:
 * the keys the P V product of that tile multiplies over (the softmax must leave zeros at the others). */
static inline uint2 band_keys(uint m0, uint window, uint K) {
  if (!window) return (uint2)(0, K);
  return (uint2)(m0 + 1 > window ? (m0 + 1 - window) & ~(uint)(TK - 1) : 0, min(K, m0 + TM + window - 1));
}

/* Strided batched SGEMM: one product per work-group layer z = b * heads + h (the grid's third dimension).
 * Each operand starts at base + b * s?b + h * s?h (elements); B's head index is h / kv_group, so
 * grouped-query attention can share one key or value head between kv_group query heads.
 *
 * `window` (0: none) is for sliding-window attention. sgemm_batched_nt, scores = Q K^T, skips the
 * 128 x 128 tiles where no |row - column| is below window (their scores are left unwritten);
 * sgemm_batched_nn, P V, multiplies each tile of rows over band_keys() only. */
#define SGEMM_BATCHED(NAME, A_KCONTIG, B_KCONTIG, SKIP, KBEG, KEND)                                                \
  __kernel __attribute__((reqd_work_group_size(16, 16, 1), amdgpu_waves_per_eu(WAVES_PER_EU))) void NAME(       \
      __global const float *A0, __global const float *B0, __global float *C0, uint M, uint N, uint K, uint lda,    \
      uint ldb, uint ldc, float alpha, float beta, uint heads, uint kv_group, ulong sAb, ulong sAh, ulong sBb,      \
      ulong sBh, ulong sCb, ulong sCh, uint window) {                                                               \
    uint z = get_group_id(2), zb = z / heads, zh = z % heads;                                                      \
    __global const float *A = A0 + zb * sAb + zh * sAh;                                                            \
    __global const float *B = B0 + zb * sBb + (zh / kv_group) * sBh;                                               \
    __global float *C = C0 + zb * sCb + zh * sCh;                                                                  \
    SGEMM_BODY(A_KCONTIG, B_KCONTIG, SKIP, KBEG, KEND)                                                             \
  }

/* op(A) = A: k contiguous. op(B) = B: n contiguous; op(B) = B^T: k contiguous. */
SGEMM(sgemm_nn, 1, 0)
SGEMM(sgemm_nt, 1, 1)
SGEMM(sgemm_tn, 0, 0)
SGEMM(sgemm_tt, 0, 1)
#define OUT_OF_BAND (window && (n0 > m0 + TM - 2 + window || m0 > n0 + TM - 2 + window))
SGEMM_BATCHED(sgemm_batched_nn, 1, 0, 0, band_keys(m0, window, K).x, band_keys(m0, window, K).y)
SGEMM_BATCHED(sgemm_batched_nt, 1, 1, OUT_OF_BAND, 0, K)

/* The original 16x16-tile kernel. */
#define TS 16
__kernel __attribute__((reqd_work_group_size(TS, TS, 1)))
void sgemm16(__global const float *A, __global const float *B, __global float *C, uint n) {
  __local float As[TS][TS], Bs[TS][TS];
  uint tx = get_local_id(0), ty = get_local_id(1);
  uint col = get_group_id(0) * TS + tx, row = get_group_id(1) * TS + ty;
  float acc = 0.0f;
  for (uint t = 0; t < n; t += TS) {
    As[ty][tx] = A[row * n + t + tx];
    Bs[ty][tx] = B[(t + ty) * n + col];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < TS; k++) acc = As[ty][k] * Bs[k][tx] + acc;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  C[row * n + col] = acc;
}
