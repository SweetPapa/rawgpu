/* Kernels for `rawgpu kernel-test`: the parts of the kernel ABI rawgpu
 * provides, each checked against the CPU. Ordinary OpenCL C 1.2: the
 * built-ins come from AMD's device libraries, linked in at compile time. */

/* 1. Work-item functions in a 3-D launch: 16 values per work-item. */
__kernel void workitem(__global uint *out) {
  size_t gx = get_global_id(0), gy = get_global_id(1), gz = get_global_id(2);
  size_t lin = (gz * get_global_size(1) + gy) * get_global_size(0) + gx;
  __global uint *o = out + lin * 16;
  o[0] = gx; o[1] = gy; o[2] = gz;
  o[3] = get_local_id(0); o[4] = get_local_id(1); o[5] = get_local_id(2);
  o[6] = get_group_id(0); o[7] = get_group_id(1); o[8] = get_group_id(2);
  o[9] = get_local_size(0); o[10] = get_local_size(1); o[11] = get_local_size(2);
  o[12] = get_num_groups(0); o[13] = get_num_groups(1); o[14] = get_num_groups(2);
  o[15] = get_work_dim();
}

/* 2. The dispatch packet directly, as code object v4 and hand-written
 * kernels read it: grid and work-group sizes, dimensions, LDS size. */
typedef struct {
  ushort header, setup, wg_x, wg_y, wg_z, reserved0;
  uint grid_x, grid_y, grid_z, private_size, group_size;
} aql_packet;
__kernel void packet(__global uint *out) {
  __constant aql_packet *p = (__constant aql_packet *)__builtin_amdgcn_dispatch_ptr();
  size_t i = get_global_id(0);
  out[4 * i + 0] = p->grid_x;
  out[4 * i + 1] = p->wg_x;
  out[4 * i + 2] = p->setup & 3;
  out[4 * i + 3] = p->group_size;
}

/* 3. Private memory with dynamic indexing: scratch. `words` per work-item. */
#define SCRATCH_SMALL 256  /* 1 KB per work-item: every wave fits at once */
#define SCRATCH_BIG 2048   /* 8 KB per work-item: the backing holds fewer waves than the GPU can run */
__kernel void scratch_small(__global uint *out, __global const uint *in, uint n) {
  size_t i = get_global_id(0);
  uint a[SCRATCH_SMALL];
  for (uint k = 0; k < SCRATCH_SMALL; k++) a[k] = in[(i + k) % n] ^ k;
  uint j = in[i] % SCRATCH_SMALL, m = (in[i] >> 9) % SCRATCH_SMALL;
  a[j] += 1;
  out[i] = a[j] + a[m] * 3u;
}
__kernel void scratch_big(__global uint *out, __global const uint *in, uint n) {
  size_t i = get_global_id(0);
  uint a[SCRATCH_BIG];
  for (uint k = 0; k < SCRATCH_BIG; k++) a[k] = in[(i + k) % n] ^ k;
  uint j = in[i] % SCRATCH_BIG, m = (in[i] >> 11) % SCRATCH_BIG;
  a[j] += 1;
  out[i] = a[j] + a[m] * 3u;
}

/* 4. Program-scope constants: a relocated reference into .rodata. */
__constant uint primes[8] = {2, 3, 5, 7, 11, 13, 17, 19};
typedef struct { float scale; uint add; } params_t;
__constant params_t params = {2.5f, 7};
__kernel void constants(__global uint *out) {
  size_t i = get_global_id(0);
  out[i] = primes[i & 7] * (uint)(params.scale * 4.0f) + params.add;
}

/* 5. Atomics on VRAM and LDS, with barriers. */
__kernel void atomics(__global int *g, __global int *per_group) {
  __local int l;
  if (get_local_id(0) == 0) l = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(&l, 1);
  atomic_add(&g[0], 1);
  atomic_max(&g[1], (int)get_global_id(0));
  atomic_xor(&g[2], (int)get_global_id(0));
  atomic_min(&g[3], -(int)get_global_id(0));
  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) per_group[get_group_id(0)] = l;
}

/* 6. A tree reduction in a dynamic __local argument (clSetKernelArg(NULL)). */
__kernel void reduce(__global const float *in, __global float *out, __local float *tmp) {
  size_t l = get_local_id(0), n = get_local_size(0);
  tmp[l] = in[get_global_id(0)];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (size_t s = n / 2; s > 0; s >>= 1) {
    if (l < s) tmp[l] += tmp[l + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (l == 0) out[get_group_id(0)] = tmp[0];
}

/* 7. Vector loads, saturating conversions, stores. */
__kernel void vectors(__global const float *in, __global uchar *out) {
  size_t i = get_global_id(0);
  float4 v = vload4(i, in);
  vstore4(convert_uchar4_sat_rte(v * 300.0f), i, out);
}

/* 8. Single-precision math from ocml: out = f(x, y) for f number `fn`. */
__kernel void math(__global float *out, __global const float *x, __global const float *y, uint fn) {
  size_t i = get_global_id(0);
  float a = x[i], b = y[i], r;
  switch (fn) {
  case 0: r = sin(a); break;
  case 1: r = cos(a); break;
  case 2: r = tan(a); break;
  case 3: r = asin(a); break;
  case 4: r = acos(a); break;
  case 5: r = atan(a); break;
  case 6: r = atan2(a, b); break;
  case 7: r = sinh(a); break;
  case 8: r = cosh(a); break;
  case 9: r = tanh(a); break;
  case 10: r = exp(a); break;
  case 11: r = exp2(a); break;
  case 12: r = exp10(a); break;
  case 13: r = expm1(a); break;
  case 14: r = log(a); break;
  case 15: r = log2(a); break;
  case 16: r = log10(a); break;
  case 17: r = log1p(a); break;
  case 18: r = sqrt(a); break;
  case 19: r = rsqrt(a); break;
  case 20: r = cbrt(a); break;
  case 21: r = pow(a, b); break;
  case 22: r = erf(a); break;
  case 23: r = erfc(a); break;
  case 24: r = tgamma(a); break;
  case 25: r = lgamma(a); break;
  case 26: r = fmod(a, b); break;
  case 27: r = hypot(a, b); break;
  case 28: r = a / b; break;
  default: r = 0.0f;
  }
  out[i] = r;
}

/* 9. Double precision (1/16 rate on Polaris, but there). */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void dmath(__global double *out, __global const double *x) {
  size_t i = get_global_id(0);
  double v = x[i];
  out[2 * i] = exp(v) * sqrt(v);
  out[2 * i + 1] = sin(v) / v;
}
