/* C = A * B for n x n row-major matrices, n a multiple of 16. Each 16x16
 * work-group stages 16x16 tiles of A and B in LDS. The multiply-add is
 * a * b + c, which becomes v_mac_f32 (full rate; fused fma is quarter rate on
 * Polaris). Without device libraries, the barrier is s_barrier between
 * workgroup fences. */
#define TS 16
#define BARRIER()                                              \
  do {                                                         \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");     \
    __builtin_amdgcn_s_barrier();                              \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");     \
  } while (0)

__kernel __attribute__((reqd_work_group_size(TS, TS, 1)))
void sgemm(__global const float *A, __global const float *B, __global float *C, unsigned n) {
  __local float As[TS][TS], Bs[TS][TS];
  unsigned tx = __builtin_amdgcn_workitem_id_x(), ty = __builtin_amdgcn_workitem_id_y();
  unsigned col = __builtin_amdgcn_workgroup_id_x() * TS + tx, row = __builtin_amdgcn_workgroup_id_y() * TS + ty;
  float acc = 0.0f;
  for (unsigned t = 0; t < n; t += TS) {
    As[ty][tx] = A[row * n + t + tx];
    Bs[ty][tx] = B[(t + ty) * n + col];
    BARRIER();
    for (unsigned k = 0; k < TS; k++) acc = As[ty][k] * Bs[k][tx] + acc;
    BARRIER();
  }
  C[row * n + col] = acc;
}
