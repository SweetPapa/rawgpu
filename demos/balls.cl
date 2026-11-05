/* Bouncing balls: rigid spheres in a box, simulated and ray-traced on the GPU.
 *
 * balls_step: one physics substep. Every ball checks every other (N^2, fine
 * for a few thousand), pushes out of overlaps and exchanges impulses with
 * restitution, then bounces off the floor and the box walls. Positions and
 * velocities are double-buffered, so each step reads a consistent state.
 *
 * balls_render: one ray per pixel against every sphere and the floor, plus a
 * shadow ray toward the sun, one reflection bounce, and analytic ambient
 * occlusion from the nearby spheres. Spheres are read through __constant
 * with a uniform index, so the whole wave shares each load. */

#define FLOOR_Y 0.0f
#define GRAVITY 9.81f

typedef struct { float3 ro, rd; } ray_t;

static inline uint pack_rgba(float3 c) {
  c = clamp(c, 0.0f, 1.0f);
  uint r = (uint)(c.x * 255.0f + 0.5f), g = (uint)(c.y * 255.0f + 0.5f), b = (uint)(c.z * 255.0f + 0.5f);
  return r | g << 8 | b << 16 | 0xff000000u;
}

static inline uint wang(uint x) {
  x = (x ^ 61u) ^ (x >> 16); x *= 9u; x ^= x >> 4; x *= 0x27d4eb2du; x ^= x >> 15;
  return x;
}
static inline float rnd(uint *s) { *s = wang(*s + 0x9e3779b9u); return (float)(*s >> 8) * (1.0f / 16777216.0f); }

/* pos: xyz, radius. vel: xyz, unused. Mass goes as radius^3. */
__kernel void balls_step(__global const float4 *pos_in, __global const float4 *vel_in, __global float4 *pos_out,
                         __global float4 *vel_out, uint n, float dt, float box, float restitution) {
  uint i = get_global_id(0);
  if (i >= n) return;
  float4 pi = pos_in[i];
  float3 p = pi.xyz, v = vel_in[i].xyz, dv = 0.0f, dp = 0.0f;
  float r = pi.w, mi = r * r * r;
  for (uint j = 0; j < n; j++) {
    if (j == i) continue;
    float4 pj = pos_in[j];
    float3 d = p - pj.xyz;
    float rr = r + pj.w, d2 = dot(d, d);
    if (d2 >= rr * rr || d2 < 1e-12f) continue;
    float dist = native_sqrt(d2);
    float3 nrm = d / dist;
    float mj = pj.w * pj.w * pj.w, share = mj / (mi + mj);
    dp += nrm * (rr - dist) * share;                     /* out of the overlap, by mass */
    float vn = dot(v - vel_in[j].xyz, nrm);
    if (vn < 0.0f) dv -= (1.0f + restitution) * vn * share * nrm;
  }
  v += dv;
  v.y -= GRAVITY * dt;
  v *= 1.0f - 0.02f * dt;                                /* a little air drag */
  p += dp + v * dt;
  if (p.y - r < FLOOR_Y) {                               /* floor: bounce, and rolling friction */
    p.y = FLOOR_Y + r;
    if (v.y < 0.0f) v.y = -v.y * restitution;
    v.xz *= 1.0f - 1.5f * dt;
  }
  float lim = box - r;
  if (p.x < -lim) { p.x = -lim; v.x = fabs(v.x) * restitution; }
  if (p.x > lim) { p.x = lim; v.x = -fabs(v.x) * restitution; }
  if (p.z < -lim) { p.z = -lim; v.z = fabs(v.z) * restitution; }
  if (p.z > lim) { p.z = lim; v.z = -fabs(v.z) * restitution; }
  pos_out[i] = (float4)(p, r);
  vel_out[i] = (float4)(v, 0.0f);
}

/* Throw every ball upward from where it is, with a little scatter. */
__kernel void balls_kick(__global float4 *vel, uint n, uint seed, float strength) {
  uint i = get_global_id(0);
  if (i >= n) return;
  uint s = wang(i * 747796405u + seed);
  float3 v = vel[i].xyz;
  v += (float3)((rnd(&s) - 0.5f) * strength * 0.6f, strength * (0.6f + 0.6f * rnd(&s)), (rnd(&s) - 0.5f) * strength * 0.6f);
  vel[i] = (float4)(v, 0.0f);
}

/* ---- rendering ---- */

static inline float3 sky(float3 rd, float3 sun) {
  float y = clamp(rd.y, -1.0f, 1.0f);
  float3 c = mix((float3)(0.80f, 0.86f, 0.95f), (float3)(0.22f, 0.42f, 0.80f), sqrt(clamp(y, 0.0f, 1.0f)));
  if (y < 0.0f) c = mix((float3)(0.80f, 0.86f, 0.95f), (float3)(0.35f, 0.33f, 0.30f), clamp(-y * 3.0f, 0.0f, 1.0f));
  float s = max(dot(rd, sun), 0.0f);
  c += (float3)(1.0f, 0.75f, 0.45f) * (0.3f * native_powr(s, 12.0f) + 4.0f * native_powr(s, 900.0f));
  return c;
}

/* nearest hit along a ray: t, and what: sphere index, -1 floor, -2 nothing */
static inline float trace(float3 ro, float3 rd, __constant const float4 *sph, uint n, int *what) {
  float tbest = 1e30f;
  *what = -2;
  for (uint j = 0; j < n; j++) {
    float4 s = sph[j];
    float3 oc = ro - s.xyz;
    float b = dot(oc, rd), c = dot(oc, oc) - s.w * s.w, h = b * b - c;
    if (h > 0.0f) {
      float t = -b - native_sqrt(h);
      if (t > 1e-3f && t < tbest) { tbest = t; *what = (int)j; }
    }
  }
  if (rd.y < 0.0f) {
    float t = (FLOOR_Y - ro.y) / rd.y;
    if (t > 1e-3f && t < tbest) { tbest = t; *what = -1; }
  }
  return tbest;
}

/* any sphere between p and the sun? soft-edged by the closest miss */
static inline float shadow(float3 ro, float3 rd, __constant const float4 *sph, uint n) {
  float res = 1.0f;
  for (uint j = 0; j < n; j++) {
    float4 s = sph[j];
    float3 oc = ro - s.xyz;
    float b = dot(oc, rd);
    if (b > 0.0f) continue; /* sphere behind the point */
    float h = dot(oc, oc) - b * b;  /* squared distance from the ray to the centre */
    float d = native_sqrt(max(h, 0.0f)) - s.w;
    res = min(res, clamp(8.0f * d / -b, 0.0f, 1.0f));
    if (res <= 0.0f) break;
  }
  return res * res * (3.0f - 2.0f * res);
}

/* ambient occlusion at p (normal nrm) from every sphere, analytically (a sphere's solid angle, cosine-weighted) */
static inline float occlusion(float3 p, float3 nrm, __constant const float4 *sph, uint n, int self) {
  float ao = 1.0f;
  for (uint j = 0; j < n; j++) {
    if ((int)j == self) continue;
    float4 s = sph[j];
    float3 d = s.xyz - p;
    float l2 = dot(d, d);
    float cosv = dot(nrm, d) * native_rsqrt(l2);
    ao *= 1.0f - clamp(cosv * s.w * s.w / l2, 0.0f, 0.9f); /* bounded: touching balls overlap a little */
  }
  return clamp(ao, 0.0f, 1.0f);
}

static inline float3 floor_color(float3 p, float box) {
  bool inside = fabs(p.x) < box && fabs(p.z) < box;
  int cx = (int)floor(p.x * 1.0f), cz = (int)floor(p.z * 1.0f);
  float chk = ((cx + cz) & 1) ? 1.0f : 0.0f;
  float3 a = inside ? mix((float3)(0.82f, 0.82f, 0.80f), (float3)(0.55f, 0.56f, 0.58f), chk) : (float3)(0.20f, 0.21f, 0.23f);
  float edge = min(fabs(fabs(p.x) - box), fabs(fabs(p.z) - box));
  if (inside && edge < 0.06f) a = (float3)(0.95f, 0.55f, 0.15f); /* the box's rim */
  return a;
}

/* direct light + sky at a hit; no further bounces */
static inline float3 shade(float3 p, float3 rd, int what, float3 sun, __constant const float4 *sph,
                           __constant const float4 *mat, uint n, float box, bool with_ao, float3 *nrm_out, float *refl_out) {
  float3 nrm, alb;
  float refl, spec;
  if (what >= 0) {
    nrm = normalize(p - sph[what].xyz);
    float4 m = mat[what];
    alb = m.xyz;
    refl = m.w;
    spec = 0.4f + m.w;
  } else {
    nrm = (float3)(0.0f, 1.0f, 0.0f);
    alb = floor_color(p, box);
    refl = 0.12f;
    spec = 0.1f;
  }
  float dif = max(dot(nrm, sun), 0.0f);
  if (dif > 0.0f) dif *= shadow(p + nrm * 1e-3f, sun, sph, n);
  float ao = with_ao ? occlusion(p, nrm, sph, n, what) : 0.8f; /* in reflections AO barely shows */
  float3 h = normalize(sun - rd);
  float sp = native_powr(max(dot(nrm, h), 0.0f), 60.0f) * dif * spec;
  float3 col = alb * ((float3)(1.0f, 0.92f, 0.80f) * 2.2f * dif + (float3)(0.35f, 0.45f, 0.65f) * 0.9f * ao *
                      (0.6f + 0.4f * nrm.y)) + (float3)(1.0f, 0.95f, 0.85f) * sp * 2.0f;
  *nrm_out = nrm;
  *refl_out = refl;
  return col;
}

static inline float3 aces(float3 x) { return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f); }

/* what a reflected ray sees, lit without AO */
static inline float3 reflected(float3 p, float3 r, float3 sun, __constant const float4 *sph, __constant const float4 *mat,
                               uint n, float box) {
  int w2;
  float t2 = trace(p, r, sph, n, &w2);
  if (w2 == -2) return sky(r, sun);
  float3 n2;
  float rf2;
  return shade(p + t2 * r, r, w2, sun, sph, mat, n, box, false, &n2, &rf2);
}

static inline float3 ray_dir(uint x, uint y, uint w, uint h, float2 off, float3 rt, float3 up, float3 fw) {
  float2 uv = ((float2)((float)x, (float)(h - 1 - y)) + off) * 2.0f - (float2)((float)w, (float)h);
  uv /= (float)h;
  return normalize(uv.x * rt + uv.y * up + 1.8f * fw);
}

/* One ray per pixel: hit, shade, one reflection bounce, haze. A mirror ball
 * reflects hundreds of other balls into a few pixels, which one reflected ray
 * per pixel turns into noise, so a shiny ball's pixel takes four more
 * reflected rays from points across the pixel (the ball's own lighting is
 * computed once). */
__kernel void balls_render(__global uint *out, uint w, uint h, __constant const float4 *sph, __constant const float4 *mat,
                           uint n, float4 cam_pos, float4 cam_target, float box) {
  uint x = get_global_id(0), y = get_global_id(1);
  if (x >= w || y >= h) return;
  float3 ro = cam_pos.xyz, ta = cam_target.xyz;
  float3 fw = normalize(ta - ro), rt = normalize(cross(fw, (float3)(0.0f, 1.0f, 0.0f))), up = cross(rt, fw);
  float3 sun = normalize((float3)(0.55f, 0.75f, 0.35f));
  float3 rd = ray_dir(x, y, w, h, (float2)(0.0f), rt, up, fw);
  int what;
  float t = trace(ro, rd, sph, n, &what);
  float3 col;
  if (what == -2) col = sky(rd, sun);
  else {
    float3 p = ro + t * rd, nrm;
    float refl;
    col = shade(p, rd, what, sun, sph, mat, n, box, true, &nrm, &refl);
    if (refl > 0.0f) {
      float fres = clamp(refl + (1.0f - refl) * native_powr(1.0f - max(dot(-rd, nrm), 0.0f), 5.0f) * 0.5f, 0.0f, 1.0f);
      float3 r1 = rd - 2.0f * dot(rd, nrm) * nrm;
      /* matte balls reflect only the sky: a traced ray at their low strength adds noise, not detail */
      float3 rc = what >= 0 && refl <= 0.3f ? sky(r1, sun) : reflected(p + nrm * 1e-3f, r1, sun, sph, mat, n, box);
      if (what >= 0 && refl > 0.3f) {
        const float2 offs[4] = {(float2)(-0.125f, -0.375f), (float2)(0.375f, -0.125f), (float2)(0.125f, 0.375f),
                                (float2)(-0.375f, 0.125f)};
        float4 sp = sph[what];
        for (int k = 0; k < 4; k++) { /* the same ball, hit a little elsewhere in the pixel */
          float3 rs = ray_dir(x, y, w, h, offs[k], rt, up, fw), oc = ro - sp.xyz;
          float b = dot(oc, rs), hh = b * b - dot(oc, oc) + sp.w * sp.w;
          if (hh <= 0.0f) { rc += rc / (float)(k + 1); continue; } /* past its edge: reuse what we have */
          float3 ps = ro + (-b - native_sqrt(hh)) * rs, ns = normalize(ps - sp.xyz);
          rc += reflected(ps + ns * 1e-3f, rs - 2.0f * dot(rs, ns) * ns, sun, sph, mat, n, box);
        }
        rc *= 0.2f;
      }
      col = mix(col, rc, fres);
    }
    col = mix(col, sky(rd, sun), 1.0f - native_exp(-0.0006f * t * t)); /* distance haze */
  }
  col = aces(max(col, 0.0f) * 0.9f); /* never a negative into powr(): that is a NaN pixel */
  out[y * w + x] = pack_rgba(native_powr(col, (float3)(1.0f / 2.2f)));
}
