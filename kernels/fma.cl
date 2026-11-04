/* FP32 throughput, fused: 8 independent FMA chains per work-item, `iters` rounds
 * (16 FLOPs per round). The sum is stored so nothing is optimized away; the
 * host recomputes it with fmaf() and expects the same bits. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void fma_peak(__global float *out, unsigned iters, float a, float b) {
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  float x0 = (float)i * 1e-6f, x1 = x0 + 0.125f, x2 = x0 + 0.25f, x3 = x0 + 0.375f;
  float x4 = x0 + 0.5f, x5 = x0 + 0.625f, x6 = x0 + 0.75f, x7 = x0 + 0.875f;
  for (unsigned k = 0; k < iters; k++) {
    x0 = __builtin_fmaf(x0, a, b); x1 = __builtin_fmaf(x1, a, b);
    x2 = __builtin_fmaf(x2, a, b); x3 = __builtin_fmaf(x3, a, b);
    x4 = __builtin_fmaf(x4, a, b); x5 = __builtin_fmaf(x5, a, b);
    x6 = __builtin_fmaf(x6, a, b); x7 = __builtin_fmaf(x7, a, b);
  }
  out[i] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

/* The same with v_mad_f32 (multiply, round, add, round; denormals flushed).
 * Polaris issues it at full rate, where fused fma is quarter rate. OpenCL
 * contracts x * a + b into fmuladd, which LLVM lowers to v_mad/v_mac when
 * f32 denormals are flushed (the default for gfx803). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mad_peak(__global float *out, unsigned iters, float a, float b) {
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  float x0 = (float)i * 1e-6f, x1 = x0 + 0.125f, x2 = x0 + 0.25f, x3 = x0 + 0.375f;
  float x4 = x0 + 0.5f, x5 = x0 + 0.625f, x6 = x0 + 0.75f, x7 = x0 + 0.875f;
  for (unsigned k = 0; k < iters; k++) {
    x0 = x0 * a + b; x1 = x1 * a + b;
    x2 = x2 * a + b; x3 = x3 * a + b;
    x4 = x4 * a + b; x5 = x5 * a + b;
    x6 = x6 * a + b; x7 = x7 * a + b;
  }
  out[i] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
