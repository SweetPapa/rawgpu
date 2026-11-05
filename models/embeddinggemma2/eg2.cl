/* EmbeddingGemma 2's text encoder on rawgpu: the row-wise and element-wise
 * steps. The matrix products are kernels/sgemm.cl's, which the engine
 * compiles into the same program. Everything is float32 and row-major.
 * Activations are [tokens, features], with tokens = batch x seq_len, each
 * sequence right-padded to seq_len.
 *
 * The math follows transformers' modeling_embedding_gemma2.py:
 *   - RMSNorm is x * (mean(x^2) + eps)^-0.5 * weight (no 1 + weight);
 *   - attention is bidirectional, with scale 1.0 and no soft-capping;
 *   - q, k and v are RMS-normed per head (v without a weight);
 *   - RoPE rotates the two halves of each head. */

#define WG 64

static inline float gelu_tanh(float x) { /* gelu_pytorch_tanh */
  return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* sum over the work-group of 64 (one wave) */
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

/* out[row] = in[row] * rsqrt(mean(in[row]^2) + eps) * w (w NULL: no weight). One work-group per row. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void rmsnorm(__global const float *in, __global float *out, __global const float *w, uint dim, uint in_stride,
             uint out_stride, float eps) {
  __local float red[WG];
  size_t row = get_group_id(0);
  uint l = get_local_id(0);
  __global const float *x = in + row * in_stride;
  __global float *y = out + row * out_stride;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += x[i] * x[i];
  float inv = rsqrt(wg_sum(s, red) / (float)dim + eps);
  for (uint i = l; i < dim; i += WG) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

/* x[row] = (x[row] + rmsnorm(o[row]) * w) * scale: a residual add after a post-norm */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void add_rmsnorm(__global float *x, __global const float *o, __global const float *w, uint dim, float eps, float scale) {
  __local float red[WG];
  size_t row = get_group_id(0);
  uint l = get_local_id(0);
  __global float *xr = x + row * dim;
  __global const float *orow = o + row * dim;
  float s = 0.0f;
  for (uint i = l; i < dim; i += WG) s += orow[i] * orow[i];
  float inv = rsqrt(wg_sum(s, red) / (float)dim + eps);
  for (uint i = l; i < dim; i += WG) xr[i] = (xr[i] + orow[i] * inv * w[i]) * scale;
}

/* The fused q/k/v projection's output, in place: [tokens, (heads + 2 kv_heads) * hd], q heads first,
 * then k, then v. One work-group per (token, head slot): RMS-norm the head (q and k with their weights,
 * v without), then rotate q and k by the token's position (RoPE, the rotate-half form, with the
 * frequencies from the host so they match torch's float32 ones). */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void qkv_post(__global float *qkv, __global const float *q_w, __global const float *k_w, __global const float *inv_freq,
              uint heads, uint kv_heads, uint hd, uint seq_len, float eps) {
  __local float red[WG];
  uint slots = heads + 2 * kv_heads;
  size_t g = get_group_id(0), tok = g / slots;
  uint slot = (uint)(g % slots), l = get_local_id(0);
  __global float *x = qkv + tok * (size_t)slots * hd + (size_t)slot * hd;
  bool is_q = slot < heads, is_k = !is_q && slot < heads + kv_heads;
  __global const float *w = is_q ? q_w : is_k ? k_w : 0;
  float s = 0.0f;
  for (uint i = l; i < hd; i += WG) s += x[i] * x[i];
  float inv = rsqrt(wg_sum(s, red) / (float)hd + eps);
  uint half_hd = hd / 2;
  float pos = (float)(tok % seq_len);
  for (uint j = l; j < half_hd; j += WG) {
    float a = x[j] * inv * (w ? w[j] : 1.0f), b = x[j + half_hd] * inv * (w ? w[j + half_hd] : 1.0f);
    if (w) { /* q and k: rotate */
      float cs, sn = sincos(pos * inv_freq[j], &cs); /* OpenCL's sincos returns the sine */
      x[j] = a * cs - b * sn;
      x[j + half_hd] = b * cs + a * sn;
    } else {
      x[j] = a;
      x[j + half_hd] = b;
    }
  }
}

/* Softmax of each row of the attention scores [batch * heads, seq_len (query), seq_len (key)], in
 * place, after masking: keys at or past the sequence's length, and for sliding layers keys `window`
 * or more positions from the query. A row with no visible key becomes all zeros. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void softmax_mask(__global float *S, __global const uint *lens, uint heads, uint seq_len, uint window) {
  __local float red[WG];
  size_t row = get_group_id(0);
  uint l = get_local_id(0);
  uint q = (uint)(row % seq_len), b = (uint)(row / seq_len / heads), len = lens[b];
  __global float *r = S + row * seq_len;
  uint lo = 0, hi = len;
  if (window) {
    lo = q >= window ? q - window + 1 : 0;
    hi = min(hi, q + window);
  }
  float m = -INFINITY;
  for (uint j = lo + l; j < hi; j += WG) m = fmax(m, r[j]);
  m = wg_max(m, red);
  float s = 0.0f;
  for (uint j = l; j < seq_len; j += WG) {
    float e = (j >= lo && j < hi) ? exp(r[j] - m) : 0.0f;
    r[j] = e;
    s += e;
  }
  s = wg_sum(s, red);
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  for (uint j = l; j < seq_len; j += WG) r[j] *= inv;
}

/* out[t, i] = gelu(gu[t, i]) * gu[t, inter + i]: the MLP's gate and up halves from one fused product */
__kernel void gelu_mul(__global const float *gu, __global float *out, uint inter, uint total) {
  size_t i = get_global_id(0);
  if (i >= total) return;
  size_t t = i / inter, c = i % inter;
  out[i] = gelu_tanh(gu[t * 2 * inter + c]) * gu[t * 2 * inter + inter + c];
}

/* g[t, c] = gelu(g[t, c]) * ple[t, layer, c]: the per-layer-embedding gate */
__kernel void ple_gate(__global float *g, __global const float *ple, uint dim, uint layers, uint layer, uint total) {
  size_t i = get_global_id(0);
  if (i >= total) return;
  size_t t = i / dim, c = i % dim;
  g[i] = gelu_tanh(g[i]) * ple[(t * layers + layer) * dim + c];
}

/* Mean over each sequence's real tokens, then L2-normalize: [batch, dim]. One work-group per sequence. */
__kernel __attribute__((reqd_work_group_size(WG, 1, 1)))
void pool_normalize(__global const float *e, __global float *out, __global const uint *lens, uint seq_len, uint dim) {
  __local float red[WG];
  uint b = get_group_id(0), l = get_local_id(0), len = lens[b];
  __global const float *base = e + (size_t)b * seq_len * dim;
  float sq = 0.0f;
  for (uint c = l; c < dim; c += WG) {
    float acc = 0.0f;
    for (uint t = 0; t < len; t++) acc += base[(size_t)t * dim + c];
    acc /= (float)max(len, 1u);
    out[(size_t)b * dim + c] = acc;
    sq += acc * acc;
  }
  float inv = rsqrt(fmax(wg_sum(sq, red), 1e-24f));
  for (uint c = l; c < dim; c += WG) out[(size_t)b * dim + c] *= inv;
}
