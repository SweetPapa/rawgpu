/* `rawgpu kernel-test`: the kernel ABI rawgpu provides, checked against the
 * CPU (kernels/abi_test.cl). Work-item functions, the dispatch packet,
 * scratch, relocated constants, atomics, dynamic LDS, vector conversions,
 * and the accuracy of every ocml function it calls. */
#include "amd_vi_ctx.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  vi_ctx *c;
  vi_prog *p;
  int failed;
} kt;

/* the kernarg block, filled by argument index from the metadata */
typedef struct { uint8_t b[512]; uint32_t size; } kargs;

static void karg(kargs *a, const amd_kernel *k, uint32_t idx, const void *v, uint32_t n) {
  uint32_t seen = 0;
  for (uint32_t i = 0; i < k->nargs; i++) {
    const amd_arg *g = &k->args[i];
    if (g->kind >= AMD_ARG_HIDDEN_GLOBAL_OFFSET_X) continue;
    if (seen++ != idx) continue;
    if (g->offset + n <= sizeof a->b) {
      memcpy(a->b + g->offset, v, n);
      if (g->offset + n > a->size) a->size = g->offset + n;
    }
    return;
  }
}

static const amd_kernel *kern(kt *t, const char *name) { return amd_program_kernel(&t->p->p, name); }

static int run(kt *t, const amd_kernel *k, const kargs *a, uint32_t dyn_lds, uint32_t g0, uint32_t g1, uint32_t g2,
               uint32_t w0, uint32_t w1, uint32_t w2, double *secs) {
  if (vi_kernel_check(k)) return -ENOTSUP;
  const uint32_t grid[3] = {g0, g1, g2}, wg[3] = {w0, w1, w2};
  uint32_t seq;
  double t0 = vi_now();
  int r = vi_launch_async(t->c, t->p, k, a->b, a->size, dyn_lds, grid, wg, &seq);
  if (!r) r = vi_ctx_wait(t->c, seq, 20000);
  if (secs) *secs = vi_now() - t0;
  return r;
}

static void result(kt *t, const char *what, bool ok, const char *detail) {
  printf("%-14s %s%s%s\n", what, ok ? "ok" : "FAILED", detail[0] ? ": " : "", detail);
  if (!ok) t->failed++;
}

static uint32_t xs(uint32_t *s) { *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s; }
static float urand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * (float)(xs(s) >> 8) * (1.0f / 16777216.0f); }
static float lrand(uint32_t *s, float lo, float hi) { return expf(urand(s, logf(lo), logf(hi))); } /* log-uniform */

static int t_workitem(kt *t) {
  const amd_kernel *k = kern(t, "workitem");
  const uint32_t G[3] = {32, 6, 4}, W[3] = {8, 3, 2}, n = 32 * 6 * 4;
  vi_buf o;
  int r = vi_buf_alloc(t->c, &o, n * 64, false);
  if (r) return r;
  uint32_t *h = malloc(n * 64);
  kargs a = {0};
  karg(&a, k, 0, &o.gpu, 8);
  if (!(r = run(t, k, &a, 0, G[0], G[1], G[2], W[0], W[1], W[2], NULL)) && !(r = vi_buf_read(t->c, &o, 0, h, n * 64))) {
    int bad = 0;
    char d[128] = "";
    for (uint32_t z = 0; z < G[2]; z++)
      for (uint32_t y = 0; y < G[1]; y++)
        for (uint32_t x = 0; x < G[0]; x++) {
          const uint32_t *v = h + ((z * G[1] + y) * G[0] + x) * 16;
          uint32_t want[16] = {x, y, z, x % W[0], y % W[1], z % W[2], x / W[0], y / W[1], z / W[2], W[0], W[1], W[2],
                               G[0] / W[0], G[1] / W[1], G[2] / W[2], 3};
          for (int j = 0; j < 16; j++)
            if (v[j] != want[j] && !bad++)
              snprintf(d, sizeof d, "work-item (%u,%u,%u) value %d is %u, want %u", x, y, z, j, v[j], want[j]);
        }
    if (!bad) snprintf(d, sizeof d, "%u work-items in a 32x6x4 grid of 8x3x2 groups: every id and size right", n);
    result(t, "work-items", !bad, d);
  }
  free(h);
  vi_buf_free(t->c, &o);
  return r;
}

static int t_packet(kt *t) {
  const amd_kernel *k = kern(t, "packet");
  const uint32_t n = 4096, wg = 128;
  vi_buf o;
  int r = vi_buf_alloc(t->c, &o, n * 16, false);
  if (r) return r;
  uint32_t *h = malloc(n * 16);
  kargs a = {0};
  karg(&a, k, 0, &o.gpu, 8);
  if (!(r = run(t, k, &a, 0, n, 1, 1, wg, 1, 1, NULL)) && !(r = vi_buf_read(t->c, &o, 0, h, n * 16))) {
    int bad = 0;
    for (uint32_t i = 0; i < n; i++)
      bad += h[4 * i] != n || h[4 * i + 1] != wg || h[4 * i + 2] != 1 || h[4 * i + 3] != k->group_segment_size;
    char d[128];
    snprintf(d, sizeof d, "grid %u, work-group %u, 1 dimension read through the dispatch pointer%s", h[0], h[1],
             bad ? " (some work-items saw other values)" : "");
    result(t, "AQL packet", !bad, d);
  }
  free(h);
  vi_buf_free(t->c, &o);
  return r;
}

static int t_scratch(kt *t, const char *name, uint32_t words, uint32_t n) {
  const amd_kernel *k = kern(t, name);
  vi_buf in, out;
  int r = vi_buf_alloc(t->c, &in, n * 4, false);
  if (r) return r;
  if ((r = vi_buf_alloc(t->c, &out, n * 4, false))) { vi_buf_free(t->c, &in); return r; }
  uint32_t *hi = malloc(n * 4), *ho = malloc(n * 4), s = 12345;
  for (uint32_t i = 0; i < n; i++) hi[i] = xs(&s);
  kargs a = {0};
  karg(&a, k, 0, &out.gpu, 8);
  karg(&a, k, 1, &in.gpu, 8);
  karg(&a, k, 2, &n, 4);
  double secs = 0;
  if (!(r = vi_buf_write(t->c, &in, 0, hi, n * 4)) && !(r = vi_buf_fill(t->c, &out, 0xDEADBEEF)) &&
      !(r = run(t, k, &a, 0, n, 1, 1, 256, 1, 1, &secs)) && !(r = vi_buf_read(t->c, &out, 0, ho, n * 4))) {
    int bad = 0;
    uint32_t first = 0;
    for (uint32_t i = 0; i < n; i++) {
      uint32_t j = hi[i] % words, m = (hi[i] >> (words == 256 ? 9 : 11)) % words;
      uint32_t aj = (hi[(i + j) % n] ^ j) + 1, am = m == j ? aj : (hi[(i + m) % n] ^ m);
      if (ho[i] != aj + am * 3u && !bad++) first = i;
    }
    vi_scratch_cfg cfg;
    vi_scratch_config(k->private_segment_size, t->c->cus, t->c->scratch_bytes, &cfg);
    char d[200];
    snprintf(d, sizeof d, "%u work-items x %u B private: %u waves at a time in %llu MB of scratch; %.1f ms%s", n,
             k->private_segment_size, cfg.tmpring & 0xfff, (unsigned long long)(t->c->scratch_bytes >> 20), secs * 1e3,
             bad ? " (wrong results)" : "");
    if (bad) printf("               first wrong: work-item %u reads 0x%08x\n", first, ho[first]);
    result(t, words == 256 ? "scratch 1 KB" : "scratch 8 KB", !bad, d);
  }
  free(hi);
  free(ho);
  vi_buf_free(t->c, &in);
  vi_buf_free(t->c, &out);
  return r;
}

static int t_constants(kt *t) {
  const amd_kernel *k = kern(t, "constants");
  const uint32_t n = 1024, primes[8] = {2, 3, 5, 7, 11, 13, 17, 19};
  vi_buf o;
  int r = vi_buf_alloc(t->c, &o, n * 4, false);
  if (r) return r;
  uint32_t h[1024];
  kargs a = {0};
  karg(&a, k, 0, &o.gpu, 8);
  if (!(r = run(t, k, &a, 0, n, 1, 1, 64, 1, 1, NULL)) && !(r = vi_buf_read(t->c, &o, 0, h, sizeof h))) {
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += h[i] != primes[i & 7] * 10 + 7;
    char d[128];
    snprintf(d, sizeof d, "__constant table and struct read through %u relocations", t->p->p.nrelocs);
    result(t, "constants", !bad, d);
  }
  vi_buf_free(t->c, &o);
  return r;
}

static int t_atomics(kt *t) {
  const amd_kernel *k = kern(t, "atomics");
  const uint32_t n = 1u << 18, wg = 256, groups = n / wg;
  vi_buf g, pg;
  int r = vi_buf_alloc(t->c, &g, 4096, false);
  if (r) return r;
  if ((r = vi_buf_alloc(t->c, &pg, groups * 4, false))) { vi_buf_free(t->c, &g); return r; }
  int32_t init[4] = {0, -1, 0, 0}, hg[4], *hp = malloc(groups * 4);
  kargs a = {0};
  karg(&a, k, 0, &g.gpu, 8);
  karg(&a, k, 1, &pg.gpu, 8);
  if (!(r = vi_buf_write(t->c, &g, 0, init, sizeof init)) && !(r = run(t, k, &a, 0, n, 1, 1, wg, 1, 1, NULL)) &&
      !(r = vi_buf_read(t->c, &g, 0, hg, sizeof hg)) && !(r = vi_buf_read(t->c, &pg, 0, hp, groups * 4))) {
    int32_t x = 0;
    for (uint32_t i = 0; i < n; i++) x ^= (int32_t)i;
    int bad = hg[0] != (int32_t)n || hg[1] != (int32_t)n - 1 || hg[2] != x || hg[3] != -(int32_t)(n - 1);
    for (uint32_t i = 0; i < groups; i++) bad += hp[i] != (int32_t)wg;
    char d[160];
    snprintf(d, sizeof d, "%u work-items: global add/max/xor/min %d/%d/%s/%d, LDS add per group %d", n, hg[0], hg[1],
             hg[2] == x ? "right" : "WRONG", hg[3], hp[0]);
    result(t, "atomics", !bad, d);
  }
  free(hp);
  vi_buf_free(t->c, &g);
  vi_buf_free(t->c, &pg);
  return r;
}

static int t_reduce(kt *t) {
  const amd_kernel *k = kern(t, "reduce");
  const uint32_t n = 1u << 20, wg = 256, groups = n / wg;
  vi_buf in, out;
  int r = vi_buf_alloc(t->c, &in, n * 4, false);
  if (r) return r;
  if ((r = vi_buf_alloc(t->c, &out, groups * 4, false))) { vi_buf_free(t->c, &in); return r; }
  float *hi = malloc(n * 4), *ho = malloc(groups * 4);
  for (uint32_t i = 0; i < n; i++) hi[i] = (float)(i % 97); /* small integers: every partial sum is exact */
  uint32_t lds_off = (k->group_segment_size + 15) & ~15u, lds = wg * 4;
  kargs a = {0};
  karg(&a, k, 0, &in.gpu, 8);
  karg(&a, k, 1, &out.gpu, 8);
  karg(&a, k, 2, &lds_off, 4);
  if (!(r = vi_buf_write(t->c, &in, 0, hi, n * 4)) && !(r = run(t, k, &a, lds_off - k->group_segment_size + lds, n, 1, 1, wg, 1, 1, NULL)) &&
      !(r = vi_buf_read(t->c, &out, 0, ho, groups * 4))) {
    int bad = 0;
    for (uint32_t g = 0; g < groups; g++) {
      float s = 0;
      for (uint32_t i = 0; i < wg; i++) s += hi[g * wg + i];
      bad += ho[g] != s;
    }
    char d[128];
    snprintf(d, sizeof d, "%u groups summed %u floats each in a %u-byte __local argument", groups, wg, lds);
    result(t, "dynamic LDS", !bad, d);
  }
  free(hi);
  free(ho);
  vi_buf_free(t->c, &in);
  vi_buf_free(t->c, &out);
  return r;
}

static int t_vectors(kt *t) {
  const amd_kernel *k = kern(t, "vectors");
  const uint32_t n = 4096; /* float4s */
  vi_buf in, out;
  int r = vi_buf_alloc(t->c, &in, n * 16, false);
  if (r) return r;
  if ((r = vi_buf_alloc(t->c, &out, n * 4, false))) { vi_buf_free(t->c, &in); return r; }
  float *hi = malloc(n * 16);
  uint8_t *ho = malloc(n * 4);
  uint32_t s = 99;
  for (uint32_t i = 0; i < n * 4; i++) hi[i] = urand(&s, -0.5f, 1.5f);
  kargs a = {0};
  karg(&a, k, 0, &in.gpu, 8);
  karg(&a, k, 1, &out.gpu, 8);
  if (!(r = vi_buf_write(t->c, &in, 0, hi, n * 16)) && !(r = run(t, k, &a, 0, n, 1, 1, 64, 1, 1, NULL)) &&
      !(r = vi_buf_read(t->c, &out, 0, ho, n * 4))) {
    int bad = 0;
    for (uint32_t i = 0; i < n * 4; i++) {
      float v = rintf(hi[i] * 300.0f);
      uint8_t want = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
      bad += ho[i] != want;
    }
    result(t, "vectors", !bad, "vload4, convert_uchar4_sat_rte, vstore4");
  }
  free(hi);
  free(ho);
  vi_buf_free(t->c, &in);
  vi_buf_free(t->c, &out);
  return r;
}

/* ---- math accuracy ---- */

typedef struct {
  const char *name;
  double (*f1)(double);
  double (*f2)(double, double);
  float lo, hi, lo2, hi2; /* x range, y range */
  bool logx;              /* log-uniform x */
  float max_ulp;          /* OpenCL 1.2 full profile limit (section 7.4) */
} mfn;

static double d_rsqrt(double x) { return 1.0 / sqrt(x); }
static double d_exp10(double x) { return pow(10.0, x); }
static double d_div(double a, double b) { return a / b; }

static const mfn fns[] = {
  {"sin", sin, NULL, -100, 100, 0, 0, false, 4},      {"cos", cos, NULL, -100, 100, 0, 0, false, 4},
  {"tan", tan, NULL, -10, 10, 0, 0, false, 5},        {"asin", asin, NULL, -1, 1, 0, 0, false, 4},
  {"acos", acos, NULL, -1, 1, 0, 0, false, 4},        {"atan", atan, NULL, -100, 100, 0, 0, false, 5},
  {"atan2", NULL, atan2, -10, 10, -10, 10, false, 6}, {"sinh", sinh, NULL, -80, 80, 0, 0, false, 4},
  {"cosh", cosh, NULL, -80, 80, 0, 0, false, 4},      {"tanh", tanh, NULL, -10, 10, 0, 0, false, 5},
  {"exp", exp, NULL, -80, 80, 0, 0, false, 3},        {"exp2", exp2, NULL, -120, 120, 0, 0, false, 3},
  {"exp10", d_exp10, NULL, -35, 35, 0, 0, false, 3},  {"expm1", expm1, NULL, -10, 10, 0, 0, false, 3},
  {"log", log, NULL, 1e-30f, 1e30f, 0, 0, true, 3},   {"log2", log2, NULL, 1e-30f, 1e30f, 0, 0, true, 3},
  {"log10", log10, NULL, 1e-30f, 1e30f, 0, 0, true, 3}, {"log1p", log1p, NULL, -0.9f, 100, 0, 0, false, 2},
  {"sqrt", sqrt, NULL, 1e-30f, 1e30f, 0, 0, true, 3}, {"rsqrt", d_rsqrt, NULL, 1e-30f, 1e30f, 0, 0, true, 2},
  {"cbrt", cbrt, NULL, 1e-30f, 1e30f, 0, 0, true, 2}, {"pow", NULL, pow, 1e-3f, 100, -10, 10, true, 16},
  {"erf", erf, NULL, -6, 6, 0, 0, false, 16},         {"erfc", erfc, NULL, -6, 9, 0, 0, false, 16},
  {"tgamma", tgamma, NULL, 0.1f, 30, 0, 0, false, 16}, {"lgamma", lgamma, NULL, 0.5f, 1000, 0, 0, false, 64},
  {"fmod", NULL, fmod, -1000, 1000, 0.1f, 10, false, 0}, {"hypot", NULL, hypot, -1e10f, 1e10f, -1e10f, 1e10f, false, 4},
  {"divide", NULL, d_div, -1e10f, 1e10f, -1e3f, 1e3f, false, 2.5f},
};

/* error of a float result against the double-precision reference, in ULPs of the reference */
static double ulps(float got, double ref) {
  if (isnan(ref)) return isnan(got) ? 0 : 1e9;
  if (isinf(ref)) return isinf(got) && signbit(got) == signbit(ref) ? 0 : 1e9;
  if (fabs(ref) < FLT_MIN && fabsf(got) < FLT_MIN) return 0; /* denormals flush to zero on this GPU (daz) */
  if (fabs(ref) > FLT_MAX) return isinf(got) ? 0 : 1e9;
  int e;
  frexp(ref, &e);
  double ulp = ldexp(1.0, (e - 1 > FLT_MIN_EXP - 1 ? e - 1 : FLT_MIN_EXP - 1) - 23);
  return fabs((double)got - ref) / ulp;
}

static int t_math(kt *t) {
  const amd_kernel *k = kern(t, "math");
  const uint32_t n = 1u << 16;
  vi_buf bx, by, bo;
  int r;
  if ((r = vi_buf_alloc(t->c, &bx, n * 4, false))) return r;
  if ((r = vi_buf_alloc(t->c, &by, n * 4, false))) { vi_buf_free(t->c, &bx); return r; }
  if ((r = vi_buf_alloc(t->c, &bo, n * 4, false))) { vi_buf_free(t->c, &bx); vi_buf_free(t->c, &by); return r; }
  float *x = malloc(n * 4), *y = malloc(n * 4), *o = malloc(n * 4);
  int bad = 0, total = (int)(sizeof fns / sizeof fns[0]);
  char line[1024] = "";
  for (uint32_t fn = 0; fn < sizeof fns / sizeof fns[0] && !r; fn++) {
    const mfn *m = &fns[fn];
    uint32_t s = 777 + fn;
    for (uint32_t i = 0; i < n; i++) {
      x[i] = m->logx ? lrand(&s, m->lo, m->hi) : urand(&s, m->lo, m->hi);
      y[i] = m->f2 ? urand(&s, m->lo2, m->hi2) : 0;
    }
    kargs a = {0};
    karg(&a, k, 0, &bo.gpu, 8);
    karg(&a, k, 1, &bx.gpu, 8);
    karg(&a, k, 2, &by.gpu, 8);
    karg(&a, k, 3, &fn, 4);
    if ((r = vi_buf_write(t->c, &bx, 0, x, n * 4)) || (r = vi_buf_write(t->c, &by, 0, y, n * 4)) ||
        (r = run(t, k, &a, 0, n, 1, 1, 256, 1, 1, NULL)) || (r = vi_buf_read(t->c, &bo, 0, o, n * 4)))
      break;
    double worst = 0;
    uint32_t wi = 0;
    for (uint32_t i = 0; i < n; i++) {
      double ref = m->f1 ? m->f1(x[i]) : m->f2(x[i], y[i]);
      double u = ulps(o[i], ref);
      if (u > worst) { worst = u; wi = i; }
    }
    bool ok = worst <= m->max_ulp;
    bad += !ok;
    char item[64];
    snprintf(item, sizeof item, "%s %.2f%s", m->name, worst, ok ? "" : "!");
    if (strlen(line) + strlen(item) + 3 < sizeof line) { if (line[0]) strcat(line, ", "); strcat(line, item); }
    if (!ok) printf("               %s(%.9g%s%.9g) = %.9g, want %.9g (%.1f ulp, limit %.1f)\n", m->name, (double)x[wi],
                    m->f2 ? ", " : "", m->f2 ? (double)y[wi] : 0.0, (double)o[wi],
                    m->f1 ? m->f1(x[wi]) : m->f2(x[wi], y[wi]), worst, (double)m->max_ulp);
  }
  if (!r) {
    char d[64];
    snprintf(d, sizeof d, "%d of %d functions within the OpenCL 1.2 limits; max ulp:", total - bad, total);
    result(t, "math (ocml)", !bad, d);
    /* wrap the list */
    for (char *p = line; *p;) {
      size_t take = strlen(p);
      if (take > 100) { take = 100; while (take && p[take] != ',') take--; take = take ? take + 1 : 100; }
      printf("               %.*s\n", (int)take, p);
      p += take;
      while (*p == ' ') p++;
    }
  }
  free(x);
  free(y);
  free(o);
  vi_buf_free(t->c, &bx);
  vi_buf_free(t->c, &by);
  vi_buf_free(t->c, &bo);
  return r;
}

static int t_dmath(kt *t) {
  const amd_kernel *k = kern(t, "dmath");
  const uint32_t n = 1u << 14;
  vi_buf bx, bo;
  int r = vi_buf_alloc(t->c, &bx, n * 8, false);
  if (r) return r;
  if ((r = vi_buf_alloc(t->c, &bo, n * 16, false))) { vi_buf_free(t->c, &bx); return r; }
  double *x = malloc(n * 8), *o = malloc(n * 16);
  uint32_t s = 4242;
  for (uint32_t i = 0; i < n; i++) x[i] = 0.001 + 50.0 * (xs(&s) / 4294967296.0);
  kargs a = {0};
  karg(&a, k, 0, &bo.gpu, 8);
  karg(&a, k, 1, &bx.gpu, 8);
  if (!(r = vi_buf_write(t->c, &bx, 0, x, n * 8)) && !(r = run(t, k, &a, 0, n, 1, 1, 256, 1, 1, NULL)) &&
      !(r = vi_buf_read(t->c, &bo, 0, o, n * 16))) {
    double worst = 0;
    for (uint32_t i = 0; i < n; i++) {
      double e1 = fabs(o[2 * i] - exp(x[i]) * sqrt(x[i])) / fabs(exp(x[i]) * sqrt(x[i]));
      double e2 = fabs(o[2 * i + 1] - sin(x[i]) / x[i]) / fmax(fabs(sin(x[i]) / x[i]), 1e-300);
      worst = fmax(worst, fmax(e1, e2));
    }
    char d[128];
    snprintf(d, sizeof d, "exp(x)*sqrt(x) and sin(x)/x in double: worst relative error %.1e", worst);
    result(t, "double", worst < 1e-14, d);
  }
  free(x);
  free(o);
  vi_buf_free(t->c, &bx);
  vi_buf_free(t->c, &bo);
  return r;
}

static bool skipped(int argc, char **argv, const char *name) {
  for (int i = 0; i + 1 < argc; i++)
    if (!strcmp(argv[i], "--skip") && strstr(argv[i + 1], name)) return true;
  return false;
}

int vi_cmd_kernel_test(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  const char *path = "build/kernels/abi_test.o";
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--skip")) i++;
    else path = argv[i];
  }
  vi_ctx c;
  int r = vi_ctx_open(&c, d, id);
  if (r) return r;
  vi_prog p;
  if ((r = vi_prog_load(&c, &p, path))) { vi_ctx_close(&c); return r; }
  printf("program:       %s: %u kernels, %zu-byte image at VRAM +%llu MB, %u relocations\n", path, p.p.nkernels,
         p.p.image_size, (unsigned long long)(p.off >> 20), p.p.nrelocs);
  kt t = {&c, &p, 0};
  /* in order of how much they can hang the GPU if something is wrong */
  static const struct { const char *name; int (*fn)(kt *); } tests[] = {
    {"workitem", t_workitem}, {"packet", t_packet}, {"constants", t_constants}, {"reduce", t_reduce},
    {"vectors", t_vectors}, {"double", t_dmath}, {"math", t_math},
  };
  for (size_t i = 0; i < sizeof tests / sizeof tests[0] && !r; i++)
    if (skipped(argc, argv, tests[i].name)) printf("%-14s skipped\n", tests[i].name);
    else r = tests[i].fn(&t);
  if (!r && !skipped(argc, argv, "scratch")) r = t_scratch(&t, "scratch_small", 256, 1u << 16);
  if (!r && !skipped(argc, argv, "scratch")) r = t_scratch(&t, "scratch_big", 2048, 1u << 16);
  if (!r && !skipped(argc, argv, "atomics")) r = t_atomics(&t);
  if (!r) printf("%s\n", t.failed ? "SOME CHECKS FAILED" : "all kernel ABI checks passed");
  vi_prog_free(&c, &p);
  vi_ctx_close(&c);
  return r ? r : t.failed ? -EIO : 0;
}
