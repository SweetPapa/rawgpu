/* Valley flyover: a procedural landscape, ray-marched per pixel.
 *
 * The terrain is fractal value noise whose octaves are damped by the slope
 * accumulated so far, which gives smooth valleys and sharp ridges. Each pixel
 * marches its ray over the height field, then shades the hit:
 *   - materials by height and slope: rock, grass, sand, snow;
 *   - a soft shadow marched toward the sun;
 *   - water with waves, fresnel reflection and depth tint;
 *   - clouds, distance fog and sun glare.
 * valley_heights evaluates the same terrain at points the host asks for, so
 * the camera can fly above it. */

#ifndef MARCH_OCTAVES
#define MARCH_OCTAVES 7     /* octaves while marching; normals get more */
#endif
#ifndef DETAIL_OCTAVES
#define DETAIL_OCTAVES 12
#endif
#ifndef MAX_STEPS
#define MAX_STEPS 260
#endif

#define TERRAIN_FREQ (1.0f / 420.0f)
#define TERRAIN_HEIGHT 260.0f
#define SEA_LEVEL 0.0f
#define FAR 9000.0f

static inline float hash2(int2 p) {
  uint h = (uint)p.x * 0x8da6b343u + (uint)p.y * 0xd8163841u + 0x9e3779b9u;
  h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
  return (float)(h >> 8) * (1.0f / 16777216.0f);
}

/* value noise in [0,1] and its gradient: (value, d/dx, d/dy) */
static inline float3 noised(float2 x) {
  float2 p = floor(x), w = x - p;
  float2 u = w * w * w * (w * (w * 6.0f - 15.0f) + 10.0f);
  float2 du = 30.0f * w * w * (w * (w - 2.0f) + 1.0f);
  int2 i = convert_int2(p);
  float a = hash2(i), b = hash2(i + (int2)(1, 0)), c = hash2(i + (int2)(0, 1)), d = hash2(i + (int2)(1, 1));
  float k1 = b - a, k2 = c - a, k4 = a - b - c + d;
  return (float3)(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y, du.x * (k1 + k4 * u.y), du.y * (k2 + k4 * u.x));
}

static inline float noise(float2 x) { return noised(x).x; }

static inline float terrain(float2 xz, int octaves) {
  float2 p = xz * TERRAIN_FREQ, d = (float2)(0.0f);
  float a = 0.0f, b = 0.5f;
  for (int i = 0; i < octaves; i++) {
    float3 n = noised(p);
    d += n.yz;
    a += b * n.x / (1.0f + dot(d, d));
    b *= 0.5f;
    p = (float2)(1.6f * p.x - 1.2f * p.y, 1.2f * p.x + 1.6f * p.y); /* rotate and double */
  }
  /* broad valleys: a slow wave lowers a band the camera can follow */
  float valley = 0.35f * smoothstep(0.0f, 1.0f, fabs(sin(xz.x * 0.0011f + 0.6f * sin(xz.y * 0.0005f))));
  return TERRAIN_HEIGHT * (a * 1.9f - 0.32f - 0.35f + valley);
}

static inline float3 terrain_normal(float3 p, float t) {
  float e = 0.15f + 0.0012f * t;
  float hx = terrain(p.xz + (float2)(e, 0.0f), DETAIL_OCTAVES) - terrain(p.xz - (float2)(e, 0.0f), DETAIL_OCTAVES);
  float hz = terrain(p.xz + (float2)(0.0f, e), DETAIL_OCTAVES) - terrain(p.xz - (float2)(0.0f, e), DETAIL_OCTAVES);
  return normalize((float3)(-hx, 2.0f * e, -hz));
}

static inline float march(float3 ro, float3 rd, float tmax) {
  float t = 1.0f;
  for (int i = 0; i < MAX_STEPS; i++) {
    float3 p = ro + t * rd;
    float h = p.y - terrain(p.xz, MARCH_OCTAVES);
    if (h < 0.0012f * t || t > tmax) break;
    if (p.y > TERRAIN_HEIGHT * 1.6f && rd.y >= 0.0f) return tmax + 1.0f; /* above everything, climbing */
    t += 0.42f * h;
  }
  return t;
}

static inline float soft_shadow(float3 ro, float3 rd) {
  float res = 1.0f, t = 3.0f;
  for (int i = 0; i < 56; i++) {
    float3 p = ro + t * rd;
    float h = p.y - terrain(p.xz, 6);
    res = min(res, 10.0f * h / t);
    if (res < 0.002f || p.y > TERRAIN_HEIGHT * 1.6f) break;
    t += clamp(h, 2.0f + 0.01f * t, 80.0f);
  }
  return clamp(res, 0.0f, 1.0f);
}

static inline float3 sky(float3 ro, float3 rd, float3 sun, float time) {
  float y = max(rd.y, 0.0f);
  float3 col = mix((float3)(0.72f, 0.80f, 0.92f), (float3)(0.20f, 0.38f, 0.72f), native_powr(y, 0.45f));
  float s = max(dot(rd, sun), 0.0f);
  col += (float3)(1.0f, 0.62f, 0.32f) * (0.28f * native_powr(s, 6.0f) + 0.5f * native_powr(s, 64.0f));
  col += (float3)(1.0f, 0.90f, 0.70f) * 12.0f * native_powr(s, 3000.0f); /* the sun's disc */
  if (rd.y > 0.0f) { /* a cloud deck */
    float tc = (2400.0f - ro.y) / rd.y;
    float2 uv = (ro.xz + tc * rd.xz) * 0.00022f + (float2)(time * 0.004f, time * 0.002f);
    float c = 0.0f, a = 0.5f;
    for (int i = 0; i < 6; i++) { c += a * noise(uv); uv = uv * 2.03f + (float2)(1.7f, 9.2f); a *= 0.5f; }
    float cover = smoothstep(0.48f, 0.78f, c) * smoothstep(0.0f, 0.12f, rd.y);
    float3 cloud = mix((float3)(0.95f, 0.92f, 0.88f), (float3)(1.0f, 0.75f, 0.55f), 0.4f * native_powr(s, 4.0f));
    col = mix(col, cloud, cover * 0.85f);
  }
  return col;
}

static inline float3 fog_color(float3 rd, float3 sun) {
  float s = max(dot(rd, sun), 0.0f);
  return mix((float3)(0.66f, 0.74f, 0.86f), (float3)(1.0f, 0.78f, 0.52f), native_powr(s, 8.0f));
}

static inline float3 aces(float3 x) { return clamp((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f); }

static inline uint pack_rgba(float3 c) {
  c = clamp(c, 0.0f, 1.0f);
  return (uint)(c.x * 255.0f + 0.5f) | (uint)(c.y * 255.0f + 0.5f) << 8 | (uint)(c.z * 255.0f + 0.5f) << 16 | 0xff000000u;
}

static inline float3 land(float3 p, float3 n, float3 rd, float t, float3 sun) {
  float h = p.y;
  float var = noise(p.xz * 0.05f), var2 = noise(p.xz * 0.004f);
  float3 rock = mix((float3)(0.30f, 0.26f, 0.22f), (float3)(0.42f, 0.38f, 0.33f), var);
  float3 grass = mix((float3)(0.16f, 0.27f, 0.08f), (float3)(0.30f, 0.33f, 0.12f), var2);
  float3 alb = rock;
  float flat = smoothstep(0.72f, 0.92f, n.y);
  alb = mix(alb, grass, smoothstep(0.62f, 0.85f, n.y) * smoothstep(175.0f, 120.0f, h + 40.0f * var2));
  alb = mix(alb, (float3)(0.62f, 0.56f, 0.42f), smoothstep(6.0f, 1.5f, h) * flat); /* beaches */
  float snow = smoothstep(150.0f, 175.0f, h + 45.0f * var2) * smoothstep(0.45f, 0.75f, n.y);
  alb = mix(alb, (float3)(0.92f, 0.94f, 0.98f), snow);

  float dif = max(dot(n, sun), 0.0f);
  if (dif > 0.0f) dif *= soft_shadow(p + n * 0.5f, sun);
  float amb = 0.5f + 0.5f * n.y;
  float bounce = clamp(-n.y + 0.3f, 0.0f, 1.0f) * 0.4f;
  float3 light = (float3)(1.0f, 0.80f, 0.55f) * 3.8f * dif + (float3)(0.36f, 0.50f, 0.80f) * 0.6f * amb +
                 (float3)(0.40f, 0.30f, 0.20f) * bounce;
  float3 col = alb * light;
  float3 hv = normalize(sun - rd);
  col += snow * (float3)(1.0f, 0.9f, 0.8f) * native_powr(max(dot(n, hv), 0.0f), 40.0f) * dif * 0.6f;
  return col;
}

__kernel void valley(__global uint *out, uint w, uint h, float4 cam_pos, float4 cam_target, float time) {
  uint x = get_global_id(0), y = get_global_id(1);
  if (x >= w || y >= h) return;
  float3 ro = cam_pos.xyz, ta = cam_target.xyz;
  float roll = cam_pos.w;
  float3 fw = normalize(ta - ro);
  float3 upw = (float3)(sin(roll), cos(roll), 0.0f);
  float3 rt = normalize(cross(fw, upw)), up = cross(rt, fw);
  float2 uv = ((float2)((float)x, (float)(h - 1 - y)) * 2.0f - (float2)((float)w, (float)h)) / (float)h;
  float3 rd = normalize(uv.x * rt + uv.y * up + 1.6f * fw);
  float3 sun = normalize((float3)(-0.80f, 0.30f, 0.38f)); /* low, from the side: long shadows, strong relief */

  float tw = rd.y < 0.0f ? (SEA_LEVEL - ro.y) / rd.y : 1e9f;
  float t = march(ro, rd, min(FAR, tw));
  float3 col;
  float dist;
  if (tw < t && tw < FAR) { /* water first */
    float3 p = ro + tw * rd;
    float2 q = p.xz * 0.06f;
    float3 n1 = noised(q + (float2)(time * 0.35f, time * 0.20f)), n2 = noised(q * 2.7f - (float2)(time * 0.25f, -time * 0.4f));
    float3 nrm = normalize((float3)(-(n1.y + 0.5f * n2.y) * 0.08f, 1.0f, -(n1.z + 0.5f * n2.z) * 0.08f));
    float3 r = rd - 2.0f * dot(rd, nrm) * nrm;
    float fres = 0.02f + 0.98f * native_powr(1.0f - max(dot(-rd, nrm), 0.0f), 5.0f);
    float depth = SEA_LEVEL - terrain(p.xz, 5);
    float3 water = mix((float3)(0.10f, 0.22f, 0.20f), (float3)(0.02f, 0.06f, 0.12f), smoothstep(0.0f, 25.0f, depth));
    float tr = march(p + (float3)(0.0f, 0.5f, 0.0f), r, 2500.0f); /* the terrain in the reflection, coarsely */
    float3 rc = tr < 2500.0f ? fog_color(r, sun) * 0.55f : sky(p, r, sun, time);
    col = mix(water * (0.4f + 0.6f * max(sun.y, 0.0f)), rc, fres);
    col += (float3)(1.0f, 0.85f, 0.6f) * 6.0f * native_powr(max(dot(r, sun), 0.0f), 400.0f);
    col = mix(col, (float3)(0.55f, 0.50f, 0.40f), smoothstep(1.5f, 0.0f, depth) * 0.5f); /* foam at the shore */
    dist = tw;
  } else if (t < FAR) {
    float3 p = ro + t * rd;
    col = land(p, terrain_normal(p, t), rd, t, sun);
    dist = t;
  } else {
    col = sky(ro, rd, sun, time);
    dist = -1.0f;
  }
  if (dist > 0.0f) { /* fog: denser low down, glowing toward the sun */
    float fog = 1.0f - native_exp(-dist * 0.00018f * native_exp(-max(ro.y + rd.y * dist * 0.5f, 0.0f) * 0.0025f));
    col = mix(col, fog_color(rd, sun), clamp(fog, 0.0f, 1.0f));
  }
  col = aces(max(col, 0.0f) * 0.75f);
  float2 v = (float2)((float)x / (float)w, (float)y / (float)h) - 0.5f;
  col *= 1.0f - 0.35f * dot(v, v); /* vignette */
  out[y * w + x] = pack_rgba(native_powr(col, (float3)(1.0f / 2.2f)));
}

/* terrain height under each (x, z), for the camera path */
__kernel void valley_heights(__global float *out, __global const float2 *xz, uint n) {
  uint i = get_global_id(0);
  if (i < n) out[i] = max(terrain(xz[i], 9), SEA_LEVEL);
}
