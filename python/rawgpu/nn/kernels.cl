/* rawgpu.nn: the kernels behind training, forward and backward. Compiled after kernels/sgemm.cl, whose
 * SGEMM_BODY the batched products below reuse.
 *
 * Every tensor is 2-D and float32 unless noted: `rows` x `cols`, rows `ld` floats apart (ld >= cols;
 * wider rows keep the matrix products fast at power-of-two widths). Element-wise kernels take one
 * work-item per element; row kernels one work-group of WG per row (or per group of `dim` columns).
 *
 * Backward kernels take `beta`: 0 writes the gradient, 1 adds it to what is there (the autograd engine
 * passes 1 when a tensor already has a gradient from another use). With beta 0 the destination is never
 * read, so it can start as uninitialized memory. */

#define WG 64

static inline float gelu_tanh(float x) { /* gelu_pytorch_tanh */
  return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static inline float wg_sum(float v, __local float *red) {
  uint l = get_local_id(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (l < s) red[l] += red[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}

static inline float wg_max(float v, __local float *red) {
  uint l = get_local_id(0);
  red[l] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint s = WG / 2; s > 0; s >>= 1) {
    if (l < s) red[l] = fmax(red[l], red[l + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}

/* Polaris has no float atomics in global memory: compare-and-swap on the bits. */
static inline void atomic_add_f(volatile __global float *p, float v) {
  volatile __global uint *u = (volatile __global uint *)p;
  uint old = *u, assumed;
  do {
    assumed = old;
    old = atomic_cmpxchg(u, assumed, as_uint(as_float(assumed) + v));
  } while (old != assumed);
}

static inline float acc(float beta, __global const float *p) { return beta != 0.0f ? *p : 0.0f; }

#define ELEM_INDEX                                                                                                 \
  size_t i = get_global_id(0);                                                                                     \
  if (i >= (size_t)rows * cols) return;                                                                            \
  size_t r = i / cols, c = i - r * cols;

/* ---- filling and copying ---- */

__kernel void fill(__global float *x, uint ld, uint rows, uint cols, float value) {
  ELEM_INDEX
  x[r * ld + c] = value;
}

/* y = alpha * x + beta * y (beta 0: y is not read). 4-byte words: also copies int32 tensors (alpha 1, beta 0
 * with the `words` flag, which skips the float arithmetic). */
__kernel void axpby(__global const float *x, uint ldx, __global float *y, uint ldy, uint rows, uint cols, float alpha,
                    float beta, uint words) {
  ELEM_INDEX
  if (words) {
    ((__global uint *)y)[r * ldy + c] = ((__global const uint *)x)[r * ldx + c];
    return;
  }
  float v = alpha * x[r * ldx + c];
  if (beta != 0.0f) v += beta * y[r * ldy + c];
  y[r * ldy + c] = v;
}

/* y[r, c] = beta * y + alpha * src, where src is broadcast by mode: 0 full [rows, cols], 1 one row [1, cols],
 * 2 one column [rows, 1] (lds apart), 3 one value */
static inline size_t bcast_off(uint mode, size_t r, size_t c, uint ld) {
  return mode == 0 ? r * ld + c : mode == 1 ? c : mode == 2 ? r * ld : 0;
}

__kernel void broadcast(__global const float *src, uint lds, uint mode, __global float *y, uint ldy, uint rows,
                        uint cols, float alpha, float beta) {
  ELEM_INDEX
  float v = alpha * src[bcast_off(mode, r, c, lds)];
  if (beta != 0.0f) v += beta * y[r * ldy + c];
  y[r * ldy + c] = v;
}

/* ---- element-wise ---- */

enum { U_AFFINE, U_RELU, U_GELU, U_SILU, U_TANH, U_SIGMOID, U_EXP, U_LOG, U_SQUARE, U_SQRT, U_SOFTPLUS, U_ABS };

static inline float act_f(uint op, float x, float a, float b) {
  switch (op) {
    case U_AFFINE: return a * x + b;
    case U_RELU: return fmax(x, 0.0f);
    case U_GELU: return gelu_tanh(x);
    case U_SILU: return x / (1.0f + exp(-x));
    case U_TANH: return tanh(x);
    case U_SIGMOID: return 1.0f / (1.0f + exp(-x));
    case U_EXP: return exp(x);
    case U_LOG: return log(x);
    case U_SQUARE: return x * x;
    case U_SQRT: return sqrt(x);
    case U_SOFTPLUS: return x > 20.0f ? x : log1p(exp(x));
    case U_ABS: return fabs(x);
  }
  return x;
}

/* d f(x) / dx, given x and y = f(x) */
static inline float act_d(uint op, float x, float y, float a) {
  switch (op) {
    case U_AFFINE: return a;
    case U_RELU: return x > 0.0f ? 1.0f : 0.0f;
    case U_GELU: {
      const float k = 0.7978845608028654f;
      float x2 = x * x, t = tanh(k * (x + 0.044715f * x2 * x));
      return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x2);
    }
    case U_SILU: {
      float s = 1.0f / (1.0f + exp(-x));
      return s * (1.0f + x * (1.0f - s));
    }
    case U_TANH: return 1.0f - y * y;
    case U_SIGMOID: return y * (1.0f - y);
    case U_EXP: return y;
    case U_LOG: return 1.0f / x;
    case U_SQUARE: return 2.0f * x;
    case U_SQRT: return y > 0.0f ? 0.5f / y : 0.0f;
    case U_SOFTPLUS: return 1.0f / (1.0f + exp(-x));
    case U_ABS: return x > 0.0f ? 1.0f : x < 0.0f ? -1.0f : 0.0f;
  }
  return 1.0f;
}

__kernel void unary(__global const float *x, uint ldx, __global float *y, uint ldy, uint rows, uint cols, uint op,
                    float a, float b) {
  ELEM_INDEX
  y[r * ldy + c] = act_f(op, x[r * ldx + c], a, b);
}

/* dx = beta * dx + dy * f'(x) */
__kernel void unary_bwd(__global const float *x, uint ldx, __global const float *y, uint ldy, __global const float *dy,
                        uint lddy, __global float *dx, uint lddx, uint rows, uint cols, uint op, float a, float beta) {
  ELEM_INDEX
  float g = dy[r * lddy + c] * act_d(op, x[r * ldx + c], y[r * ldy + c], a);
  dx[r * lddx + c] = acc(beta, dx + r * lddx + c) + g;
}

/* ---- dropout ---- */

/* lowbias32 (Chris Wellons' integer hash): full avalanche in two multiplies */
static inline uint hash32(uint x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

/* y = beta * y + x * keep * scale, keep = hash(seed, r * cols + c) >= cut: each element dropped with
 * probability cut / 2^32. The mask is a function of the seed and the element's logical index (not of ld),
 * so the backward pass is this kernel again on the gradient with the same seed, and nothing is stored.
 * y may be x (in place, beta 0). */
__kernel void dropout(__global const float *x, uint ldx, __global float *y, uint ldy, uint rows, uint cols, uint cut,
                      float scale, uint seed, float beta) {
  ELEM_INDEX
  uint k = hash32(seed ^ 0x9e3779b9U);
  uint h = hash32(hash32((uint)i + k) ^ seed);
  float v = h >= cut ? x[r * ldx + c] * scale : 0.0f;
  y[r * ldy + c] = acc(beta, y + r * ldy + c) + v;
}

/* z = beta_z * z + (op 0: alpha * x + beta * y; 1: alpha * x * y; 2: alpha * x / y), y broadcast by ymode
 * (as in broadcast()) */
__kernel void binary(__global const float *x, uint ldx, __global const float *y, uint ldy, __global float *z, uint ldz,
                     uint rows, uint cols, uint op, uint ymode, float alpha, float beta, float beta_z) {
  ELEM_INDEX
  float a = x[r * ldx + c], b = y[bcast_off(ymode, r, c, ldy)];
  float v = op == 0 ? alpha * a + beta * b : op == 1 ? alpha * a * b : alpha * a / b;
  z[r * ldz + c] = acc(beta_z, z + r * ldz + c) + v;
}

/* ---- reductions ---- */

/* out[ch, c] = beta * out[ch, c] + alpha * (sum of x[r, c] over the chunk's rows [ch * per, (ch + 1) * per)).
 * Grid (cols, chunks). Two launches sum any number of rows: chunks of partial sums, then their sum. */
__kernel void colsum(__global const float *x, uint ldx, uint rows, uint cols, uint per, __global float *out, uint ldo,
                     float alpha, float beta) {
  uint c = get_global_id(0), ch = get_global_id(1);
  if (c >= cols) return;
  uint r0 = ch * per, r1 = min(rows, r0 + per);
  float s = 0.0f;
  for (uint r = r0; r < r1; r++) s += x[(size_t)r * ldx + c];
  __global float *o = out + (size_t)ch * ldo + c;
  *o = acc(beta, o) + alpha * s;
}

/* out[r * ldo] = beta * out + alpha * sum of row r (op 0) or of its squares (op 1). One work-group per row. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rowsum(__global const float *x, uint ldx, uint cols, __global float *out, uint ldo, float alpha, float beta, uint op) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx;
  float s = 0.0f;
  for (uint i = l; i < cols; i += WG) s += op ? xr[i] * xr[i] : xr[i];
  s = wg_sum(s, red);
  if (l == 0) out[r * ldo] = acc(beta, out + r * ldo) + alpha * s;
}

/* ---- normalization ---- */

/* RMSNorm over groups of `dim` columns: row-group g covers x[r, j * dim + i] for r = g / groups,
 * j = g % groups. y = x * rstd * (w ? w : 1) (plus 1 on w with plus_one, the Gemma 1-3 convention).
 * rstd[g] is kept for the backward pass. One work-group per row-group. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rmsnorm_fwd(__global const float *x, uint ldx, __global const float *w, __global float *y, uint ldy,
                 __global float *rstd, uint groups, uint dim, float eps, uint plus_one) {
  __local float red[WG];
  size_t g = get_group_id(0), r = g / groups, j = g % groups;
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx + j * dim;
  __global float *yr = y + r * ldy + j * dim;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += xr[i] * xr[i];
  float inv = rsqrt(wg_sum(s, red) / (float)dim + eps);
  if (l == 0) rstd[g] = inv;
  for (uint i = l; i < dim; i += WG) yr[i] = xr[i] * inv * (w ? w[i] + (plus_one ? 1.0f : 0.0f) : 1.0f);
}

/* dx = rstd * (dy*w - xhat * mean(dy*w * xhat)); dwp[g, i] = dy * xhat (summed over row-groups by colsum for
 * the weight's gradient), if dwp is given */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rmsnorm_bwd(__global const float *x, uint ldx, __global const float *w, __global const float *rstd,
                 __global const float *dy, uint lddy, __global float *dx, uint lddx, __global float *dwp, uint groups,
                 uint dim, uint plus_one, float beta) {
  __local float red[WG];
  size_t g = get_group_id(0), r = g / groups, j = g % groups;
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx + j * dim;
  __global const float *dyr = dy + r * lddy + j * dim;
  __global float *dxr = dx + r * lddx + j * dim;
  float inv = rstd[g], s = 0.0f;
  for (uint i = l; i < dim; i += WG) {
    float wi = w ? w[i] + (plus_one ? 1.0f : 0.0f) : 1.0f;
    s += dyr[i] * wi * xr[i] * inv;
  }
  float m = wg_sum(s, red) / (float)dim;
  for (uint i = l; i < dim; i += WG) {
    float wi = w ? w[i] + (plus_one ? 1.0f : 0.0f) : 1.0f, xh = xr[i] * inv;
    dxr[i] = acc(beta, dxr + i) + inv * (dyr[i] * wi - xh * m);
    if (dwp) dwp[g * dim + i] = dyr[i] * xh;
  }
}

/* LayerNorm over each row: y = (x - mean) * rstd * w + b (w, b may be NULL) */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void layernorm_fwd(__global const float *x, uint ldx, __global const float *w, __global const float *b,
                   __global float *y, uint ldy, __global float *stats, uint dim, float eps) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx;
  __global float *yr = y + r * ldy;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += xr[i];
  float mean = wg_sum(s, red) / (float)dim, v = 0.0f;
  for (uint i = l; i < dim; i += WG) {
    float d = xr[i] - mean;
    v += d * d;
  }
  float inv = rsqrt(wg_sum(v, red) / (float)dim + eps);
  if (l == 0) {
    stats[2 * r] = mean;
    stats[2 * r + 1] = inv;
  }
  for (uint i = l; i < dim; i += WG) yr[i] = (xr[i] - mean) * inv * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void layernorm_bwd(__global const float *x, uint ldx, __global const float *w, __global const float *stats,
                   __global const float *dy, uint lddy, __global float *dx, uint lddx, __global float *dwp, uint dim,
                   float beta) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx, *dyr = dy + r * lddy;
  __global float *dxr = dx + r * lddx;
  float mean = stats[2 * r], inv = stats[2 * r + 1], s1 = 0.0f, s2 = 0.0f;
  for (uint i = l; i < dim; i += WG) {
    float g = dyr[i] * (w ? w[i] : 1.0f), xh = (xr[i] - mean) * inv;
    s1 += g;
    s2 += g * xh;
  }
  float m1 = wg_sum(s1, red) / (float)dim, m2 = wg_sum(s2, red) / (float)dim;
  for (uint i = l; i < dim; i += WG) {
    float g = dyr[i] * (w ? w[i] : 1.0f), xh = (xr[i] - mean) * inv;
    dxr[i] = acc(beta, dxr + i) + inv * (g - m1 - xh * m2);
    if (dwp) dwp[r * dim + i] = dyr[i] * xh;
  }
}

/* y = x / max(|x|, eps) per row; nrm[r] keeps the divisor */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void l2norm_fwd(__global const float *x, uint ldx, __global float *y, uint ldy, __global float *nrm, uint dim, float eps) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += xr[i] * xr[i];
  float n = fmax(sqrt(wg_sum(s, red)), eps);
  if (l == 0) nrm[r] = n;
  for (uint i = l; i < dim; i += WG) y[r * ldy + i] = xr[i] / n;
}

/* dx = (dy - y * (y . dy)) / n */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void l2norm_bwd(__global const float *y, uint ldy, __global const float *nrm, __global const float *dy, uint lddy,
                __global float *dx, uint lddx, uint dim, float beta) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *yr = y + r * ldy, *dyr = dy + r * lddy;
  __global float *dxr = dx + r * lddx;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += yr[i] * dyr[i];
  float d = wg_sum(s, red), n = nrm[r];
  for (uint i = l; i < dim; i += WG) dxr[i] = acc(beta, dxr + i) + (dyr[i] - yr[i] * d) / n;
}

/* ---- softmax and losses ---- */

/* Softmax of each row (in place or not). To mask columns, add a large negative number to them first. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void softmax_fwd(__global const float *x, uint ldx, __global float *y, uint ldy, uint cols) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx;
  float m = -INFINITY;
  for (uint i = l; i < cols; i += WG) m = fmax(m, xr[i]);
  m = wg_max(m, red);
  float s = 0.0f;
  for (uint i = l; i < cols; i += WG) s += exp(xr[i] - m);
  float inv = 1.0f / wg_sum(s, red);
  for (uint i = l; i < cols; i += WG) y[r * ldy + i] = exp(xr[i] - m) * inv;
}

/* dx = y * (dy - sum(dy * y)) * scale */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void softmax_bwd(__global const float *y, uint ldy, __global const float *dy, uint lddy, __global float *dx, uint lddx,
                 uint cols, float scale, float beta) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *yr = y + r * ldy, *dyr = dy + r * lddy;
  __global float *dxr = dx + r * lddx;
  float s = 0.0f;
  for (uint i = l; i < cols; i += WG) s += yr[i] * dyr[i];
  s = wg_sum(s, red);
  for (uint i = l; i < cols; i += WG) dxr[i] = acc(beta, dxr + i) + yr[i] * (dyr[i] - s) * scale;
}

/* Cross entropy per row against an integer label (label < 0: the row is ignored, loss 0), with label
 * smoothing eps: loss = lse - (1 - eps) * x[label] - eps * mean(x). lse[r] is kept. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void ce_fwd(__global const float *x, uint ldx, __global const int *labels, uint cols, __global float *loss,
            __global float *lse, float smoothing) {
  __local float red[WG];
  size_t r = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *xr = x + r * ldx;
  float m = -INFINITY, sum = 0.0f;
  for (uint i = l; i < cols; i += WG) m = fmax(m, xr[i]);
  m = wg_max(m, red);
  float s = 0.0f;
  for (uint i = l; i < cols; i += WG) {
    s += exp(xr[i] - m);
    sum += xr[i];
  }
  s = wg_sum(s, red);
  sum = wg_sum(sum, red);
  if (l == 0) {
    float z = m + log(s);
    int y = labels[r];
    lse[r] = z;
    loss[r] = y < 0 ? 0.0f : z - (1.0f - smoothing) * xr[y] - smoothing * sum / (float)cols;
  }
}

/* dx = beta * dx + dl[0] * scale * (softmax(x) - target), target the smoothed one-hot */
__kernel void ce_bwd(__global const float *x, uint ldx, __global const int *labels, __global const float *lse,
                     __global const float *dl, float scale, __global float *dx, uint lddx, uint rows, uint cols,
                     float smoothing, float beta) {
  ELEM_INDEX
  int y = labels[r];
  float g = 0.0f;
  if (y >= 0) {
    float p = exp(x[r * ldx + c] - lse[r]);
    float t = (c == (uint)y ? 1.0f - smoothing : 0.0f) + smoothing / (float)cols;
    g = dl[0] * scale * (p - t);
  }
  dx[r * lddx + c] = acc(beta, dx + r * lddx + c) + g;
}

/* Binary cross entropy with logits, per element, against targets t in [0, 1]:
 * loss = softplus(x) - t * x; d/dx = sigmoid(x) - t. A target below 0 masks the element out. */
__kernel void bce_fwd(__global const float *x, uint ldx, __global const float *t, uint ldt, __global float *loss,
                      uint ldl, uint rows, uint cols) {
  ELEM_INDEX
  float v = x[r * ldx + c], tv = t[r * ldt + c];
  float sp = v > 0.0f ? v + log1p(exp(-v)) : log1p(exp(v));
  loss[r * ldl + c] = tv < 0.0f ? 0.0f : sp - tv * v;
}

__kernel void bce_bwd(__global const float *x, uint ldx, __global const float *t, uint ldt, __global const float *dl,
                      float scale, __global float *dx, uint lddx, uint rows, uint cols, float beta) {
  ELEM_INDEX
  float v = x[r * ldx + c], tv = t[r * ldt + c];
  float g = tv < 0.0f ? 0.0f : dl[0] * scale * (1.0f / (1.0f + exp(-v)) - tv);
  dx[r * lddx + c] = acc(beta, dx + r * lddx + c) + g;
}

/* ---- attention ---- */

/* Scores S [batch * heads * T (queries), T (keys)], rows ld apart: softmax over the keys of scale * S, in
 * place, with keys masked out at or past lens[b] (lens NULL: none), after the query (causal), or `window`
 * or more positions away (window 0: none). A row with no visible key becomes zeros. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attn_softmax_fwd(__global float *S, uint ld, __global const uint *lens, uint heads, uint T, float scale,
                      uint causal, uint window) {
  __local float red[WG];
  size_t row = get_group_id(0);
  uint l = get_local_id(0), q = (uint)(row % T), b = (uint)(row / T / heads);
  uint hi = lens ? min(lens[b], T) : T, lo = 0;
  if (causal) hi = min(hi, q + 1);
  if (window) {
    lo = q >= window ? q - window + 1 : 0;
    hi = min(hi, q + window);
  }
  __global float *r = S + row * ld;
  float m = -INFINITY;
  for (uint j = lo + l; j < hi; j += WG) m = fmax(m, r[j] * scale);
  m = wg_max(m, red);
  float s = 0.0f;
  for (uint j = lo + l; j < hi; j += WG) s += exp(r[j] * scale - m);
  s = wg_sum(s, red);
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  for (uint j = l; j < T; j += WG) r[j] = (j >= lo && j < hi) ? exp(r[j] * scale - m) * inv : 0.0f;
}

/* dS = scale * P * (dP - sum(dP * P)), in place in dP */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void attn_softmax_bwd(__global const float *P, uint ldp, __global float *dP, uint ldd, uint T, float scale) {
  __local float red[WG];
  size_t row = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *p = P + row * ldp;
  __global float *d = dP + row * ldd;
  float s = 0.0f;
  for (uint j = l; j < T; j += WG) s += p[j] * d[j];
  s = wg_sum(s, red);
  for (uint j = l; j < T; j += WG) d[j] = scale * p[j] * (d[j] - s);
}

/* Rotary position embedding, rotate-half form, on `groups` heads of hd columns in each row:
 * (a, b) -> (a cos - b sin, b cos + a sin) at angle pos * inv_freq[j], pos = row % T. sign -1 rotates back
 * (the backward pass). */
__kernel void rope(__global const float *x, uint ldx, __global float *y, uint ldy, __global const float *inv_freq,
                   uint rows, uint groups, uint hd, uint T, float sign, float beta) {
  uint half_hd = hd / 2;
  size_t i = get_global_id(0);
  if (i >= (size_t)rows * groups * half_hd) return;
  size_t r = i / (groups * half_hd);
  uint rem = (uint)(i - r * groups * half_hd), g = rem / half_hd, j = rem % half_hd;
  float pos = (float)(r % T), cs, sn = sincos(pos * inv_freq[j], &cs);
  sn *= sign;
  __global const float *xr = x + r * ldx + g * hd;
  __global float *yr = y + r * ldy + g * hd;
  float a = xr[j], b = xr[j + half_hd];
  float ya = a * cs - b * sn, yb = b * cs + a * sn;
  yr[j] = acc(beta, yr + j) + ya;
  yr[j + half_hd] = acc(beta, yr + j + half_hd) + yb;
}

/* ---- gathers and pooling ---- */

/* out[n, :] = table[ids[n], :] (ids < 0: zeros) */
__kernel void gather_rows(__global const int *ids, __global const float *table, uint ldt, __global float *out, uint ldo,
                          uint rows, uint cols) {
  ELEM_INDEX
  int id = ids[r];
  out[r * ldo + c] = id < 0 ? 0.0f : table[(size_t)id * ldt + c];
}

/* dtable[ids[n], :] += dout[n, :], atomically (ids repeat) */
__kernel void scatter_add_rows(__global const int *ids, __global const float *dout, uint lddo, __global float *dtable,
                               uint ldt, uint rows, uint cols) {
  ELEM_INDEX
  int id = ids[r];
  if (id >= 0) atomic_add_f(dtable + (size_t)id * ldt + c, dout[r * lddo + c]);
}

/* y[b, c] = mean of x[b * T + t, c] over t < lens[b]. Grid (cols, batch). */
__kernel void pool_mean_fwd(__global const float *x, uint ldx, __global const uint *lens, uint T, uint cols,
                            __global float *y, uint ldy) {
  uint c = get_global_id(0), b = get_global_id(1);
  if (c >= cols) return;
  uint len = min(lens[b], T);
  __global const float *p = x + (size_t)b * T * ldx + c;
  float s = 0.0f;
  for (uint t = 0; t < len; t++) s += p[(size_t)t * ldx];
  y[(size_t)b * ldy + c] = s / (float)max(len, 1u);
}

__kernel void pool_mean_bwd(__global const float *dy, uint lddy, __global const uint *lens, uint T,
                            __global float *dx, uint lddx, uint rows, uint cols, float beta) {
  ELEM_INDEX
  uint b = (uint)(r / T), t = (uint)(r % T), len = min(lens[b], T);
  float g = t < len ? dy[(size_t)b * lddy + c] / (float)max(len, 1u) : 0.0f;
  dx[r * lddx + c] = acc(beta, dx + r * lddx + c) + g;
}

/* ---- optimizers ---- */

/* Partial sums of squares of a gradient: work-group w of `parts` writes out[w]. WG 256. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sumsq_parts(__global const float *x, uint ld, uint rows, uint cols, __global float *out) {
  __local float red[256];
  uint l = get_local_id(0), w = get_group_id(0), parts = get_num_groups(0);
  size_t n = (size_t)rows * cols;
  float s = 0.0f;
  for (size_t i = (size_t)w * 256 + l; i < n; i += (size_t)parts * 256) {
    size_t r = i / cols, c = i - r * cols;
    float v = x[r * ld + c];
    s += v * v;
  }
  red[l] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint k = 128; k > 0; k >>= 1) {
    if (l < k) red[l] += red[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) out[w] = red[0];
}

/* out[0] = the clipping factor min(1, max_norm / norm), out[1] = the norm, from n partial sums. One WG. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void clip_factor(__global const float *parts, uint n, float max_norm, __global float *out) {
  __local float red[256];
  uint l = get_local_id(0);
  float s = 0.0f;
  for (uint i = l; i < n; i += 256) s += parts[i];
  red[l] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint k = 128; k > 0; k >>= 1) {
    if (l < k) red[l] += red[l + k];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) {
    float norm = sqrt(red[0]);
    out[0] = (max_norm > 0.0f && norm > max_norm) ? max_norm / (norm + 1e-6f) : 1.0f;
    out[1] = norm;
  }
}

/* AdamW, decoupled weight decay. gscale (may be NULL) multiplies the gradient: the clipping factor, read on
 * the GPU so the step needs no round trip to the host. A non-finite gradient norm skips the step. */
__kernel void adamw(__global float *p, __global const float *g, __global float *m, __global float *v, uint ld,
                    uint ldg, uint rows, uint cols, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2,
                    __global const float *gscale) {
  ELEM_INDEX
  if (gscale && !isfinite(gscale[1])) return;
  size_t o = r * ld + c;
  float gr = g[r * ldg + c] * (gscale ? gscale[0] : 1.0f);
  float mo = b1 * m[o] + (1.0f - b1) * gr, vo = b2 * v[o] + (1.0f - b2) * gr * gr;
  m[o] = mo;
  v[o] = vo;
  p[o] -= lr * ((mo / bc1) / (sqrt(vo / bc2) + eps) + wd * p[o]);
}

/* SGD with momentum (buffer m), decoupled weight decay */
__kernel void sgd(__global float *p, __global const float *g, __global float *m, uint ld, uint ldg, uint rows, uint cols,
                  float lr, float momentum, float wd, __global const float *gscale) {
  ELEM_INDEX
  if (gscale && !isfinite(gscale[1])) return;
  size_t o = r * ld + c;
  float gr = g[r * ldg + c] * (gscale ? gscale[0] : 1.0f);
  float mo = m ? momentum * m[o] + gr : gr;
  if (m) m[o] = mo;
  p[o] -= lr * (mo + wd * p[o]);
}

/* t = decay * t + (1 - decay) * s: an exponential moving average of weights (a JEPA target encoder) */
__kernel void ema(__global float *t, uint ldt, __global const float *s, uint lds, uint rows, uint cols, float decay) {
  ELEM_INDEX
  t[r * ldt + c] = decay * t[r * ldt + c] + (1.0f - decay) * s[r * lds + c];
}

/* ---- batched products ----
 * One product per layer z = b * heads + h of the grid's third dimension, each operand at base + b * s?b +
 * h * s?h elements, with B's head index h / kv_group (grouped-query attention). All four transposes. */
#define BGEMM(NAME, A_KCONTIG, B_KCONTIG)                                                                          \
  __kernel __attribute__((reqd_work_group_size(16, 16, 1), amdgpu_waves_per_eu(WAVES_PER_EU))) void NAME(       \
      __global const float *A0, __global const float *B0, __global float *C0, uint M, uint N, uint K, uint lda,    \
      uint ldb, uint ldc, float alpha, float beta, uint heads, uint kv_group, ulong sAb, ulong sAh, ulong sBb,      \
      ulong sBh, ulong sCb, ulong sCh) {                                                                            \
    uint z = get_group_id(2), zb = z / heads, zh = z % heads;                                                      \
    __global const float *A = A0 + zb * sAb + zh * sAh;                                                            \
    __global const float *B = B0 + zb * sBb + (zh / kv_group) * sBh;                                               \
    __global float *C = C0 + zb * sCb + zh * sCh;                                                                  \
    SGEMM_BODY(A_KCONTIG, B_KCONTIG, 0, 0, K)                                                                      \
  }

BGEMM(bgemm_nn, 1, 0)
BGEMM(bgemm_nt, 1, 1)
BGEMM(bgemm_tn, 0, 0)
BGEMM(bgemm_tt, 0, 1)
