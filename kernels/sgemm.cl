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

/* The SGEMM itself, for A, B, C, M, N, K, lda, ldb, ldc, alpha, beta in scope */
#define SGEMM_BODY(A_KCONTIG, B_KCONTIG) \
    __local float As[2][TK * LS] __attribute__((aligned(16)));                                                     \
    __local float Bs[2][TK * LS] __attribute__((aligned(16)));                                                     \
    uint tx = get_local_id(0), ty = get_local_id(1), tid = ty * 16 + tx;                                           \
    uint n0 = get_group_id(0) * TM, m0 = get_group_id(1) * TM;                                                     \
    float acc[8][8];                                                                                               \
    for (int i = 0; i < 8; i++)                                                                                    \
      for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;                                                                \
    uint nk = (K + TK - 1) / TK;                                                                                   \
    float4 ra = slice_load(A, lda, A_KCONTIG, m0, M, 0, K, tid);                                                   \
    float4 rb = slice_load(B, ldb, B_KCONTIG, n0, N, 0, K, tid);                                                   \
    slice_store(As[0], A_KCONTIG, tid, ra);                                                                        \
    slice_store(Bs[0], B_KCONTIG, tid, rb);                                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \
    for (uint t = 0; t < nk; t++) {                                                                                \
      uint cur = t & 1;                                                                                            \
      bool more = t + 1 < nk;                                                                                      \
      if (more) {                                                                                                  \
        ra = slice_load(A, lda, A_KCONTIG, m0, M, (t + 1) * TK, K, tid);                                           \
        rb = slice_load(B, ldb, B_KCONTIG, n0, N, (t + 1) * TK, K, tid);                                           \
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
    SGEMM_BODY(A_KCONTIG, B_KCONTIG)                                                                               \
  }

/* Strided batched SGEMM: one product per work-group layer z = b * heads + h (the grid's third dimension).
 * Each operand starts at base + b * s?b + h * s?h (elements); B's head index is h / kv_group, so
 * grouped-query attention can share one key or value head between kv_group query heads. */
#define SGEMM_BATCHED(NAME, A_KCONTIG, B_KCONTIG)                                                                  \
  __kernel __attribute__((reqd_work_group_size(16, 16, 1), amdgpu_waves_per_eu(WAVES_PER_EU))) void NAME(       \
      __global const float *A0, __global const float *B0, __global float *C0, uint M, uint N, uint K, uint lda,    \
      uint ldb, uint ldc, float alpha, float beta, uint heads, uint kv_group, ulong sAb, ulong sAh, ulong sBb,      \
      ulong sBh, ulong sCb, ulong sCh) {                                                                            \
    uint z = get_group_id(2), zb = z / heads, zh = z % heads;                                                      \
    __global const float *A = A0 + zb * sAb + zh * sAh;                                                            \
    __global const float *B = B0 + zb * sBb + (zh / kv_group) * sBh;                                               \
    __global float *C = C0 + zb * sCb + zh * sCh;                                                                  \
    SGEMM_BODY(A_KCONTIG, B_KCONTIG)                                                                               \
  }

/* op(A) = A: k contiguous. op(B) = B: n contiguous; op(B) = B^T: k contiguous. */
SGEMM(sgemm_nn, 1, 0)
SGEMM(sgemm_nt, 1, 1)
SGEMM(sgemm_tn, 0, 0)
SGEMM(sgemm_tt, 0, 1)
SGEMM_BATCHED(sgemm_batched_nn, 1, 0)
SGEMM_BATCHED(sgemm_batched_nt, 1, 1)

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
