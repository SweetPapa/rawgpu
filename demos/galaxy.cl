/* Galaxy collision: direct-sum N-body gravity, then additive splatting.
 *
 * nbody_step: every star feels every other star (N^2 interactions, about 20
 * FLOPs each), with softening. Work-groups stage 256 positions at a time in
 * LDS, so each position is read from memory once per group. A kick-drift
 * leapfrog step, double-buffered.
 *
 * galaxy_splat: project each star through the camera and add its light into
 * an integer accumulation buffer with atomics, bilinearly over four pixels.
 * galaxy_resolve: tone-map the buffer into the frame and clear it. */

#define TILE 256

__kernel __attribute__((reqd_work_group_size(TILE, 1, 1)))
void nbody_step(__global const float4 *pos_in, __global const float4 *vel_in, __global float4 *pos_out,
                __global float4 *vel_out, uint n, float dt, float eps2) {
  __local float4 tile[TILE];
  uint i = get_global_id(0), l = get_local_id(0);
  float4 p = pos_in[i];
  float3 a = (float3)(0.0f);
  for (uint base = 0; base < n; base += TILE) {
    tile[l] = pos_in[base + l];
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll 8
    for (uint j = 0; j < TILE; j++) {
      float4 q = tile[j];
      float3 d = q.xyz - p.xyz;
      float inv = native_rsqrt(dot(d, d) + eps2);
      a += d * (q.w * inv * inv * inv);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float4 v = vel_in[i];
  v.xyz += a * dt;
  p.xyz += v.xyz * dt;
  pos_out[i] = p;
  vel_out[i] = v;
}

__kernel void galaxy_splat(__global const float4 *pos, __global const float4 *vel, uint n, __global uint *accum,
                           uint w, uint h, float4 cam_pos, float4 cam_target, float exposure) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float3 ro = cam_pos.xyz;
  float3 fw = normalize(cam_target.xyz - ro), rt = normalize(cross(fw, (float3)(0.0f, 1.0f, 0.0f))), up = cross(rt, fw);
  float3 d = pos[i].xyz - ro;
  float z = dot(d, fw);
  if (z < 0.5f) return;
  float sx = dot(d, rt) / z * 1.7f, sy = dot(d, up) / z * 1.7f;
  float px = (sx * (float)h + (float)w) * 0.5f, py = ((float)h - sy * (float)h) * 0.5f;
  if (px < 0.0f || py < 0.0f || px >= (float)(w - 1) || py >= (float)(h - 1)) return;
  float4 v = vel[i];
  float speed = length(v.xyz);
  float3 c = v.w < 0.5f ? (float3)(0.45f, 0.65f, 1.0f) : (float3)(1.0f, 0.62f, 0.32f); /* the two galaxies */
  c = mix(c, (float3)(1.0f, 1.0f, 1.0f), clamp(speed * 0.025f, 0.0f, 0.6f));       /* fast stars run hot */
  float bright = exposure * (pos[i].w > 1.0f ? 30.0f : 1.0f) / (1.0f + 0.00015f * z * z);
  int x0 = (int)px, y0 = (int)py;
  float fx = px - (float)x0, fy = py - (float)y0;
  float wts[4] = {(1.0f - fx) * (1.0f - fy), fx * (1.0f - fy), (1.0f - fx) * fy, fx * fy};
  int idx[4] = {y0 * (int)w + x0, y0 * (int)w + x0 + 1, (y0 + 1) * (int)w + x0, (y0 + 1) * (int)w + x0 + 1};
  for (int k = 0; k < 4; k++) {
    float3 e = c * (bright * wts[k] * 256.0f);
    atomic_add(&accum[3 * idx[k] + 0], (uint)e.x);
    atomic_add(&accum[3 * idx[k] + 1], (uint)e.y);
    atomic_add(&accum[3 * idx[k] + 2], (uint)e.z);
  }
}

__kernel void galaxy_resolve(__global uint *accum, __global uint *out, uint w, uint h) {
  uint x = get_global_id(0), y = get_global_id(1);
  if (x >= w || y >= h) return;
  uint i = y * w + x;
  float3 e = (float3)((float)accum[3 * i], (float)accum[3 * i + 1], (float)accum[3 * i + 2]) * (1.0f / 256.0f);
  /* a little glow from the neighbours, then 1 - exp(-light) */
  if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
    uint nb[4] = {i - 1, i + 1, i - w, i + w};
    for (int k = 0; k < 4; k++)
      e += 0.12f * (float3)((float)accum[3 * nb[k]], (float)accum[3 * nb[k] + 1], (float)accum[3 * nb[k] + 2]) *
           (1.0f / 256.0f);
  }
  float3 c = 1.0f - native_exp(-e * 0.35f);
  c = native_powr(clamp(c, 0.0f, 1.0f), (float3)(1.0f / 2.2f));
  out[i] = (uint)(c.x * 255.0f + 0.5f) | (uint)(c.y * 255.0f + 0.5f) << 8 | (uint)(c.z * 255.0f + 0.5f) << 16 | 0xff000000u;
}

__kernel void galaxy_clear(__global uint *accum, uint n) {
  uint i = get_global_id(0);
  if (i < n) accum[i] = 0;
}
