/* rawgpu's first kernel: c[i] = a[i] + b[i]. Built for gfx803 (Polaris) with
 * Homebrew LLVM; no ROCm device library, so the work-item index comes from
 * the amdgcn builtins. One work-group is 64 work-items (one wavefront). */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void vadd(__global const float *a, __global const float *b, __global float *c, unsigned n) {
  unsigned i = __builtin_amdgcn_workgroup_id_x() * 64 + __builtin_amdgcn_workitem_id_x();
  if (i < n) c[i] = a[i] + b[i];
}
