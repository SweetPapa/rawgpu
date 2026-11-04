/* Streaming copy, 16 bytes per work-item: measures memory bandwidth.
 * Launch one work-item per float4, in work-groups of 256. */
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void copy4(__global const float4 *src, __global float4 *dst) {
  size_t i = (size_t)__builtin_amdgcn_workgroup_id_x() * 256 + __builtin_amdgcn_workitem_id_x();
  dst[i] = src[i];
}
