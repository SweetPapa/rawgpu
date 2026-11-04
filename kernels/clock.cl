/* Clock probe: one wavefront spins on a dependent FMA chain between two
 * reads of the shader clock counter (s_memtime) and the constant real-time
 * counter (s_memrealtime). With the host's wall time for the whole launch,
 * that gives the engine clock under load and the real-time counter's rate. */
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void clock_probe(__global ulong *out, unsigned iters) {
  ulong t0 = __builtin_amdgcn_s_memtime(), r0 = __builtin_amdgcn_s_memrealtime();
  float x = (float)__builtin_amdgcn_workitem_id_x();
  for (unsigned k = 0; k < iters; k++) x = __builtin_fmaf(x, 0.999f, 0.001f);
  ulong t1 = __builtin_amdgcn_s_memtime(), r1 = __builtin_amdgcn_s_memrealtime();
  if (__builtin_amdgcn_workitem_id_x() == 0) {
    out[0] = t1 - t0;
    out[1] = r1 - r0;
    out[2] = as_uint(x);
  }
}
