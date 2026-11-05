/* The rawgpu library (include/rawgpu.h), over the amd-vi host API (amd_vi_ctx.h). */
#include "rawgpu.h"

#include "amd_kernel.h"
#include "amd_vi_ctx.h"
#include "rg.h"

#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

/* set by the Makefile: where this checkout keeps its compiler and bitcode */
#ifndef RAWGPU_DEFAULT_CLANG
#define RAWGPU_DEFAULT_CLANG "/opt/homebrew/opt/llvm/bin/clang"
#endif
#ifndef RAWGPU_DEFAULT_LOCAL
#define RAWGPU_DEFAULT_LOCAL "local"
#endif

/* built-in kernels, embedded at build time (build/gen/embedded.c) */
extern const unsigned char rawgpu_embedded_sgemm[];
extern const size_t rawgpu_embedded_sgemm_size;

#define ARG_MAX_BYTES 4096u

struct rawgpu_kernel {
  rawgpu_program *prog;
  const amd_kernel *k;
  bool ok;                /* vi_kernel_check() passed */
  uint32_t nexplicit;
  uint32_t *meta;         /* explicit index -> k->args index */
  uint32_t *local_bytes;  /* per explicit argument: __local size, 0 otherwise */
  uint8_t *set;           /* per explicit argument: set yet? */
  uint8_t karg[ARG_MAX_BYTES];
  uint32_t karg_size;
};

struct rawgpu_program {
  rawgpu *g;
  vi_prog vp;
  rawgpu_kernel *kernels; /* one per kernel in the code object */
  rawgpu_program *prev, *next;
};

struct rawgpu_buf {
  vi_buf b;
  rawgpu_buf *prev, *next;
};

struct rawgpu {
  rg_dev d;
  rg_ident id;
  vi_ctx c;
  bool dev_open, ctx_open;
  uint32_t last_seq;
  uint64_t epoch;         /* high bits of 64-bit events */
  rawgpu_program *blas;   /* the embedded SGEMM kernels, loaded on first use */
  rawgpu_buf *bufs;       /* everything still allocated, freed by rawgpu_close() */
  rawgpu_program *progs;
  char err[2048];
};

static char open_err[2048]; /* for a failed rawgpu_open(), which has no context to keep it in */

static int fail(rawgpu *g, int r, const char *fmt, ...) {
  char *dst = g ? g->err : open_err;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(dst, sizeof g->err, fmt, ap);
  va_end(ap);
  return r ? r : -EIO;
}

const char *rawgpu_error(const rawgpu *g) { return g ? g->err : open_err; }

/* ---- device ---- */

static const rg_cmd *find_cmd(const char *name) {
  for (const rg_cmd *c = rg_backend_amd_vi.cmds; c && c->name; c++)
    if (!strcmp(c->name, name)) return c;
  return NULL;
}

/* Is the card up enough for a context? (posted and the engines' microcode loaded) */
static bool card_up(rg_dev *d) {
  vi_dev v;
  if (vi_open(&v, d)) return false;
  uint32_t memsize, status = 0;
  return vi_posted(&v, &memsize) && !vi_smu_ucode_status(&v, &status) &&
         (status & (VI_UCODE_SDMA0 | VI_UCODE_CP_MEC | VI_UCODE_RLC_G)) == (VI_UCODE_SDMA0 | VI_UCODE_CP_MEC | VI_UCODE_RLC_G);
}

static int bringup(rawgpu *g) {
  if (!getenv("RAWGPU_FIRMWARE")) {
    struct stat st;
    if (stat("local/firmware", &st)) setenv("RAWGPU_FIRMWARE", RAWGPU_DEFAULT_LOCAL "/firmware", 0);
  }
  const char *steps[] = {"post", "mc-load", "dpm-enable"};
  for (int i = 0; i < 3; i++) {
    const rg_cmd *c = find_cmd(steps[i]);
    if (!c) return fail(g, -ENOSYS, "this build has no `%s` step", steps[i]);
    printf("rawgpu: bring-up: %s\n", steps[i]);
    fflush(stdout);
    char *argv[1] = {NULL};
    int r = c->run(&g->d, &g->id, 0, argv);
    if (r) return fail(g, r, "bring-up step `rawgpu %s` failed (%d); see the messages above", steps[i], r);
  }
  return 0;
}

int rawgpu_open(rawgpu **out, const char *socket, unsigned flags) {
  *out = NULL;
  open_err[0] = 0;
  rawgpu *g = calloc(1, sizeof *g);
  if (!g) return fail(NULL, -ENOMEM, "out of memory");
  int r = rg_open_tinygpu(&g->d, socket);
  if (r) { free(g); return fail(NULL, r, "cannot reach the TinyGPU server (is TinyGPU running, and is the GPU attached?)"); }
  g->dev_open = true;
  if ((r = rg_pci_identify(&g->d))) { rawgpu_close(g); return fail(NULL, r, "no device answered PCI config reads"); }
  rg_identify_family(&g->d, &g->id);
  if (g->id.backend != &rg_backend_amd_vi || !g->id.chip || strcmp(g->id.chip, "POLARIS10")) {
    fail(NULL, -ENOTSUP, "this GPU (%04x:%04x, %s) is not supported yet: rawgpu drives Polaris 10 cards", g->d.vendor,
         g->d.device, g->id.chip ? g->id.chip : g->id.gen_name);
    rawgpu_close(g);
    return -ENOTSUP;
  }
  if (!card_up(&g->d)) {
    if (!(flags & RAWGPU_OPEN_BRINGUP)) {
      fail(NULL, -EIO, "the GPU is not set up (it was powered on or the Mac slept): open with RAWGPU_OPEN_BRINGUP, "
                       "or run `rawgpu post`, `rawgpu mc-load`, `rawgpu dpm-enable`");
      rawgpu_close(g);
      return -EIO;
    }
    if ((r = bringup(g))) { memcpy(open_err, g->err, sizeof open_err); rawgpu_close(g); return r; }
  }
  if ((r = vi_ctx_open(&g->c, &g->d, &g->id))) { rawgpu_close(g); return fail(NULL, r, "GPU context setup failed (%d)", r); }
  g->ctx_open = true;
  *out = g;
  return 0;
}

void rawgpu_close(rawgpu *g) {
  if (!g) return;
  if (g->ctx_open) vi_ctx_finish(&g->c);
  while (g->progs) rawgpu_program_free(g, g->progs);
  while (g->bufs) rawgpu_free(g, g->bufs);
  if (g->ctx_open) vi_ctx_close(&g->c);
  if (g->dev_open) rg_close(&g->d);
  free(g);
}

int rawgpu_get_info(rawgpu *g, rawgpu_info *info) {
  memset(info, 0, sizeof *info);
  snprintf(info->name, sizeof info->name, "%s (%s), %04x:%04x rev %02x", g->id.gen_name, g->id.chip, g->d.vendor,
           g->d.device, g->d.revision);
  snprintf(info->arch, sizeof info->arch, "gfx803");
  info->compute_units = g->c.cus;
  info->wavefront_size = 64;
  info->max_workgroup_size = 1024;
  info->local_mem_bytes = VI_LDS_MAX;
  info->vram_bytes = g->c.g.vram_size;
  info->vram_free_bytes = rg_ranges_free_bytes(&g->c.heap);
  info->vram_largest_free = rg_ranges_largest(&g->c.heap);
  uint32_t sclk = 0, mclk = 0;
  if (!vi_smu_clocks(&g->c.v, &sclk, &mclk)) { info->engine_mhz = (int)sclk; info->memory_mhz = (int)mclk; }
  else info->engine_mhz = info->memory_mhz = -1;
  info->temperature_c = vi_temp_c(&g->c.v);
  return g->c.v.err ? fail(g, g->c.v.err, "register access failed") : 0;
}

/* ---- memory ---- */

rawgpu_buf *rawgpu_alloc(rawgpu *g, size_t bytes, unsigned flags) {
  if (!bytes) { fail(g, -EINVAL, "a buffer needs at least 1 byte"); return NULL; }
  rawgpu_buf *b = calloc(1, sizeof *b);
  if (!b) { fail(g, -ENOMEM, "out of memory"); return NULL; }
  int r = vi_buf_alloc(&g->c, &b->b, bytes, (flags & RAWGPU_MEM_HOST) != 0);
  if (r) {
    free(b);
    fail(g, r, "cannot allocate %zu bytes of %s (%d)", bytes, flags & RAWGPU_MEM_HOST ? "host memory" : "VRAM", r);
    return NULL;
  }
  b->next = g->bufs;
  if (g->bufs) g->bufs->prev = b;
  g->bufs = b;
  return b;
}

void rawgpu_free(rawgpu *g, rawgpu_buf *b) {
  if (!b) return;
  vi_buf_free(&g->c, &b->b);
  if (b->prev) b->prev->next = b->next;
  else g->bufs = b->next;
  if (b->next) b->next->prev = b->prev;
  free(b);
}

size_t rawgpu_buf_size(const rawgpu_buf *b) { return b->b.size; }
uint64_t rawgpu_buf_gpu_addr(const rawgpu_buf *b) { return b->b.gpu; }
void *rawgpu_buf_host(const rawgpu_buf *b) { return b->b.host ? b->b.cpu : NULL; }

int rawgpu_write(rawgpu *g, rawgpu_buf *dst, size_t off, const void *src, size_t n) {
  int r = vi_buf_write(&g->c, &dst->b, off, src, n);
  return r ? fail(g, r, "write of %zu bytes at offset %zu failed (%d)", n, off, r) : 0;
}

int rawgpu_read(rawgpu *g, const rawgpu_buf *src, size_t off, void *dst, size_t n) {
  int r = vi_buf_read(&g->c, &src->b, off, dst, n);
  return r ? fail(g, r, "read of %zu bytes at offset %zu failed (%d)", n, off, r) : 0;
}

int rawgpu_copy(rawgpu *g, rawgpu_buf *dst, size_t doff, const rawgpu_buf *src, size_t soff, size_t n) {
  int r = vi_buf_copy(&g->c, &dst->b, doff, &src->b, soff, n);
  return r ? fail(g, r, "copy of %zu bytes failed (%d)", n, r) : 0;
}

int rawgpu_fill(rawgpu *g, rawgpu_buf *dst, size_t off, size_t n, uint32_t pattern) {
  int r = vi_buf_fill_range(&g->c, &dst->b, off, n, pattern);
  return r ? fail(g, r, "fill of %zu bytes at offset %zu failed (%d; both must be multiples of 4)", n, off, r) : 0;
}

/* ---- code ---- */

static void kernel_init(rawgpu_program *p, rawgpu_kernel *rk, const amd_kernel *k) {
  rk->prog = p;
  rk->k = k;
  for (uint32_t i = 0; i < k->nargs; i++)
    if (k->args[i].kind < AMD_ARG_HIDDEN_GLOBAL_OFFSET_X) rk->nexplicit++;
  uint32_t n = rk->nexplicit ? rk->nexplicit : 1;
  rk->meta = calloc(n, sizeof *rk->meta);
  rk->local_bytes = calloc(n, sizeof *rk->local_bytes);
  rk->set = calloc(n, 1);
  for (uint32_t i = 0, e = 0; i < k->nargs && rk->meta; i++)
    if (k->args[i].kind < AMD_ARG_HIDDEN_GLOBAL_OFFSET_X) rk->meta[e++] = i;
}

static rawgpu_program *program_wrap(rawgpu *g, rawgpu_program *p) {
  p->g = g;
  p->next = g->progs;
  if (g->progs) g->progs->prev = p;
  g->progs = p;
  p->kernels = calloc(p->vp.p.nkernels, sizeof *p->kernels);
  if (!p->kernels) { rawgpu_program_free(g, p); fail(g, -ENOMEM, "out of memory"); return NULL; }
  for (uint32_t i = 0; i < p->vp.p.nkernels; i++) kernel_init(p, &p->kernels[i], &p->vp.p.kernels[i]);
  for (uint32_t i = 0; i < p->vp.p.nkernels; i++)
    if (!p->kernels[i].meta || !p->kernels[i].local_bytes || !p->kernels[i].set) {
      rawgpu_program_free(g, p);
      fail(g, -ENOMEM, "out of memory");
      return NULL;
    }
  return p;
}

rawgpu_program *rawgpu_program_load_mem(rawgpu *g, const void *data, size_t bytes) {
  rawgpu_program *p = calloc(1, sizeof *p);
  if (!p) { fail(g, -ENOMEM, "out of memory"); return NULL; }
  int r = vi_prog_load_mem(&g->c, &p->vp, data, bytes, "code object");
  if (r) { free(p); fail(g, r, "cannot load the code object (%d); see the message above", r); return NULL; }
  return program_wrap(g, p);
}

rawgpu_program *rawgpu_program_load(rawgpu *g, const char *path) {
  rawgpu_program *p = calloc(1, sizeof *p);
  if (!p) { fail(g, -ENOMEM, "out of memory"); return NULL; }
  int r = vi_prog_load(&g->c, &p->vp, path);
  if (r) { free(p); fail(g, r, "cannot load %s (%d); see the message above", path, r); return NULL; }
  return program_wrap(g, p);
}

void rawgpu_program_free(rawgpu *g, rawgpu_program *p) {
  if (!p) return;
  if (p == g->blas) g->blas = NULL;
  if (p->prev) p->prev->next = p->next;
  else if (g->progs == p) g->progs = p->next;
  if (p->next) p->next->prev = p->prev;
  for (uint32_t i = 0; p->kernels && i < p->vp.p.nkernels; i++) {
    free(p->kernels[i].meta);
    free(p->kernels[i].local_bytes);
    free(p->kernels[i].set);
  }
  free(p->kernels);
  vi_prog_free(&g->c, &p->vp);
  free(p);
}

unsigned rawgpu_program_num_kernels(const rawgpu_program *p) { return p->vp.p.nkernels; }
const char *rawgpu_program_kernel_name(const rawgpu_program *p, unsigned i) {
  return i < p->vp.p.nkernels ? p->vp.p.kernels[i].name : NULL;
}

/* FNV-1a, for cache keys */
static uint64_t fnv(uint64_t h, const void *data, size_t n) {
  const uint8_t *p = data;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

static int mkdirs(const char *path) {
  char tmp[1024];
  snprintf(tmp, sizeof tmp, "%s", path);
  for (char *p = tmp + 1; *p; p++)
    if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
  return mkdir(tmp, 0755) && errno != EEXIST ? -errno : 0;
}

static void cache_dir(char *out, size_t cap) {
  const char *e = getenv("RAWGPU_CACHE"), *home = getenv("HOME");
  if (e) snprintf(out, cap, "%s", e);
#ifdef __APPLE__
  else if (home) snprintf(out, cap, "%s/Library/Caches/rawgpu", home);
#else
  else if (getenv("XDG_CACHE_HOME")) snprintf(out, cap, "%s/rawgpu", getenv("XDG_CACHE_HOME"));
  else if (home) snprintf(out, cap, "%s/.cache/rawgpu", home);
#endif
  else snprintf(out, cap, "/tmp/rawgpu-cache");
}

static int write_file(const char *path, const void *data, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) return -errno;
  size_t w = fwrite(data, 1, n, f);
  return fclose(f) || w != n ? -EIO : 0;
}

rawgpu_program *rawgpu_program_build(rawgpu *g, const char *source, const char *options) {
  const char *clang = getenv("RAWGPU_CLANG"), *libs = getenv("RAWGPU_DEVICE_LIBS");
  if (!clang) clang = RAWGPU_DEFAULT_CLANG;
  char libdir[1024];
  if (libs) snprintf(libdir, sizeof libdir, "%s", libs);
  else snprintf(libdir, sizeof libdir, "%s/device-libs", RAWGPU_DEFAULT_LOCAL);
  if (!options) options = "";
  char dir[1024];
  cache_dir(dir, sizeof dir);
  int r = mkdirs(dir);
  if (r) { fail(g, r, "cannot create the kernel cache %s", dir); return NULL; }

  uint64_t h = 0xcbf29ce484222325ull;
  const char *salt = "rawgpu-kernel-cache-1 gfx803";
  h = fnv(h, salt, strlen(salt) + 1);
  h = fnv(h, clang, strlen(clang) + 1);
  h = fnv(h, libdir, strlen(libdir) + 1);
  h = fnv(h, options, strlen(options) + 1);
  h = fnv(h, source, strlen(source));
  char obj[1200], src[1200], log[1200], tmp[1200];
  snprintf(obj, sizeof obj, "%s/%016llx.o", dir, (unsigned long long)h);
  struct stat st;
  if (!stat(obj, &st)) {
    rawgpu_program *p = rawgpu_program_load(g, obj);
    if (p) return p;
    unlink(obj); /* stale or damaged: rebuild */
  }
  snprintf(src, sizeof src, "%s/%016llx.cl", dir, (unsigned long long)h);
  snprintf(log, sizeof log, "%s/%016llx.log", dir, (unsigned long long)h);
  snprintf(tmp, sizeof tmp, "%s/%016llx.%d.tmp", dir, (unsigned long long)h, (int)getpid());
  if ((r = write_file(src, source, strlen(source)))) { fail(g, r, "cannot write %s", src); return NULL; }

  /* argv: clang, our flags, the caller's options (split on spaces), -c -o tmp src */
  char *opts = strdup(options), *argv[160];
  int n = 0;
  char devlib[1100];
  snprintf(devlib, sizeof devlib, "--rocm-device-lib-path=%s", libdir);
  argv[n++] = (char *)clang;
  argv[n++] = "-x";
  argv[n++] = "cl";
  if (!strstr(options, "-cl-std=")) argv[n++] = "-cl-std=CL1.2";
  argv[n++] = "-target";
  argv[n++] = "amdgcn-amd-amdhsa";
  argv[n++] = "-mcpu=gfx803";
  argv[n++] = "-O2";
  argv[n++] = devlib;
  for (char *s = opts ? strtok(opts, " \t\n") : NULL; s && n < 150; s = strtok(NULL, " \t\n")) argv[n++] = s;
  argv[n++] = "-c";
  argv[n++] = "-o";
  argv[n++] = tmp;
  argv[n++] = src;
  argv[n] = NULL;

  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addopen(&fa, 2, log, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  pid_t pid;
  int sr = posix_spawn(&pid, clang, &fa, NULL, argv, environ), status = 0;
  posix_spawn_file_actions_destroy(&fa);
  free(opts);
  if (sr) {
    fail(g, -sr, "cannot run the kernel compiler %s (%s); install Homebrew's llvm or set RAWGPU_CLANG", clang,
         strerror(sr));
    return NULL;
  }
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  if (!WIFEXITED(status) || WEXITSTATUS(status)) {
    char msg[1700] = "";
    FILE *f = fopen(log, "r");
    if (f) { size_t m = fread(msg, 1, sizeof msg - 1, f); msg[m] = 0; fclose(f); }
    unlink(tmp);
    fail(g, -EINVAL, "kernel compile failed:\n%s", msg[0] ? msg : "(no compiler output)");
    return NULL;
  }
  if (rename(tmp, obj)) { fail(g, -errno, "cannot store %s", obj); return NULL; }
  return rawgpu_program_load(g, obj);
}

rawgpu_kernel *rawgpu_kernel_get(rawgpu_program *p, const char *name) {
  for (uint32_t i = 0; i < p->vp.p.nkernels; i++) {
    rawgpu_kernel *k = &p->kernels[i];
    if (strcmp(k->k->name, name)) continue;
    if (!k->ok) {
      if (vi_kernel_check(k->k)) { fail(p->g, -ENOTSUP, "kernel %s cannot run on this GPU; see the message above", name); return NULL; }
      if (!k->k->has_metadata) { fail(p->g, -ENOTSUP, "kernel %s has no metadata, so its arguments are unknown", name); return NULL; }
      k->ok = true;
    }
    return k;
  }
  char list[512] = "";
  for (uint32_t i = 0; i < p->vp.p.nkernels; i++) {
    if (strlen(list) + strlen(p->vp.p.kernels[i].name) + 3 >= sizeof list) break;
    if (i) strcat(list, ", ");
    strcat(list, p->vp.p.kernels[i].name);
  }
  fail(p->g, -ENOENT, "no kernel named %s (the program has: %s)", name, list);
  return NULL;
}

int rawgpu_kernel_get_info(const rawgpu_kernel *rk, rawgpu_kernel_info *info) {
  const amd_kernel *k = rk->k;
  memset(info, 0, sizeof *info);
  info->num_args = rk->nexplicit;
  info->max_workgroup_size = k->max_wg ? k->max_wg : 256;
  for (int i = 0; i < 3; i++) info->reqd_workgroup_size[i] = k->reqd_wg[i];
  info->local_mem_bytes = k->group_segment_size;
  info->private_mem_bytes = k->private_segment_size;
  info->sgprs = k->sgprs;
  info->vgprs = k->vgprs;
  return 0;
}

static int arg_kind(const amd_arg *a) {
  switch (a->kind) {
  case AMD_ARG_BY_VALUE: return RAWGPU_ARG_VALUE;
  case AMD_ARG_GLOBAL_BUFFER: return RAWGPU_ARG_BUFFER;
  case AMD_ARG_DYNAMIC_SHARED_POINTER: return RAWGPU_ARG_LOCAL;
  default: return RAWGPU_ARG_OTHER;
  }
}

int rawgpu_kernel_arg_info(const rawgpu_kernel *rk, unsigned i, rawgpu_arg_info *info) {
  if (i >= rk->nexplicit) return -EINVAL;
  const amd_arg *a = &rk->k->args[rk->meta[i]];
  info->name = a->name;
  info->type_name = a->type_name;
  info->size = a->size;
  info->kind = arg_kind(a);
  info->is_const = a->is_const;
  return 0;
}

static const amd_arg *arg_at(rawgpu_kernel *rk, unsigned i, int want, const char *what) {
  rawgpu *g = rk->prog->g;
  if (i >= rk->nexplicit) {
    fail(g, -EINVAL, "kernel %s has %u arguments; there is no argument %u", rk->k->name, rk->nexplicit, i);
    return NULL;
  }
  const amd_arg *a = &rk->k->args[rk->meta[i]];
  if (arg_kind(a) != want) {
    fail(g, -EINVAL, "kernel %s argument %u (%s %s) is not %s", rk->k->name, i, a->type_name, a->name, what);
    return NULL;
  }
  if (a->offset + a->size > ARG_MAX_BYTES) { fail(g, -E2BIG, "kernel %s: argument %u is past 4 KB of kernargs", rk->k->name, i); return NULL; }
  return a;
}

static void put_arg(rawgpu_kernel *rk, const amd_arg *a, const void *v, size_t n, unsigned i) {
  memset(rk->karg + a->offset, 0, a->size);
  memcpy(rk->karg + a->offset, v, n < a->size ? n : a->size);
  if (a->offset + a->size > rk->karg_size) rk->karg_size = a->offset + a->size;
  rk->set[i] = 1;
}

int rawgpu_set_arg(rawgpu_kernel *rk, unsigned i, size_t size, const void *value) {
  const amd_arg *a = arg_at(rk, i, RAWGPU_ARG_VALUE, "a value");
  if (!a) return -EINVAL;
  if (size != a->size)
    return fail(rk->prog->g, -EINVAL, "kernel %s argument %u (%s %s) takes %u bytes, not %zu", rk->k->name, i,
                a->type_name, a->name, a->size, size);
  put_arg(rk, a, value, size, i);
  return 0;
}

int rawgpu_set_arg_buf_offset(rawgpu_kernel *rk, unsigned i, const rawgpu_buf *b, size_t offset) {
  const amd_arg *a = arg_at(rk, i, RAWGPU_ARG_BUFFER, "a buffer pointer");
  if (!a) return -EINVAL;
  if (b && offset > b->b.size) return fail(rk->prog->g, -EINVAL, "offset %zu is past the buffer's %llu bytes", offset,
                                           (unsigned long long)b->b.size);
  uint64_t addr = b ? b->b.gpu + offset : 0; /* NULL buffer: a null pointer */
  put_arg(rk, a, &addr, 8, i);
  return 0;
}

int rawgpu_set_arg_buf(rawgpu_kernel *rk, unsigned i, const rawgpu_buf *b) { return rawgpu_set_arg_buf_offset(rk, i, b, 0); }

int rawgpu_set_arg_local(rawgpu_kernel *rk, unsigned i, size_t bytes) {
  const amd_arg *a = arg_at(rk, i, RAWGPU_ARG_LOCAL, "a __local pointer");
  if (!a) return -EINVAL;
  if (!bytes || bytes > VI_LDS_MAX) return fail(rk->prog->g, -EINVAL, "__local argument of %zu bytes; 1 to 65536", bytes);
  rk->local_bytes[i] = (uint32_t)bytes;
  rk->set[i] = 1;
  return 0;
}

/* ---- launches ---- */

static rawgpu_event to_event(rawgpu *g, uint32_t seq) {
  if (g->last_seq && seq < g->last_seq) g->epoch += 1ull << 32; /* the 32-bit fence wrapped */
  g->last_seq = seq;
  return g->epoch | seq;
}

/* the largest power of two <= cap dividing n */
static uint32_t pow2_divisor(size_t n, uint32_t cap) {
  uint32_t d = 1;
  while (d * 2 <= cap && n % (d * 2) == 0) d *= 2;
  return d;
}

rawgpu_event rawgpu_launch(rawgpu *g, rawgpu_kernel *rk, unsigned dims, const size_t *global, const size_t *local) {
  const amd_kernel *k = rk->k;
  if (dims < 1 || dims > 3) { fail(g, -EINVAL, "dims must be 1, 2 or 3"); return 0; }
  uint32_t grid[3] = {1, 1, 1}, wg[3] = {1, 1, 1};
  for (unsigned d = 0; d < dims; d++) {
    if (!global[d] || global[d] > 0xffffffffu) { fail(g, -EINVAL, "global size %zu in dimension %u", global[d], d); return 0; }
    grid[d] = (uint32_t)global[d];
  }
  uint32_t cap = k->max_wg ? k->max_wg : 256;
  if (local) {
    for (unsigned d = 0; d < dims; d++) wg[d] = (uint32_t)local[d];
  } else if (k->reqd_wg[0]) {
    memcpy(wg, k->reqd_wg, sizeof wg);
  } else { /* x first, then what is left for y and z */
    wg[0] = pow2_divisor(grid[0], dims == 1 ? cap : cap < 16 ? cap : 16);
    if (dims > 1) wg[1] = pow2_divisor(grid[1], cap / wg[0]);
    if (dims > 2) wg[2] = pow2_divisor(grid[2], cap / (wg[0] * wg[1]));
  }
  /* arguments: every one set; __local ones placed after the static LDS */
  uint32_t lds = (k->group_segment_size + 15) & ~15u;
  for (uint32_t i = 0; i < rk->nexplicit; i++) {
    if (!rk->set[i]) {
      const amd_arg *a = &k->args[rk->meta[i]];
      fail(g, -EINVAL, "kernel %s: argument %u (%s %s) is not set", k->name, i, a->type_name, a->name);
      return 0;
    }
    if (rk->local_bytes[i]) {
      const amd_arg *a = &k->args[rk->meta[i]];
      uint32_t off = lds;
      put_arg(rk, a, &off, 4, i);
      lds = (lds + rk->local_bytes[i] + 15) & ~15u;
    }
  }
  if (lds > VI_LDS_MAX) { fail(g, -E2BIG, "kernel %s: %u bytes of LDS per work-group; at most 65536", k->name, lds); return 0; }
  uint32_t dyn = lds > k->group_segment_size ? lds - k->group_segment_size : 0, seq;
  int r = vi_launch_async(&g->c, &rk->prog->vp, k, rk->karg, rk->karg_size, dyn, grid, wg, &seq);
  if (r) {
    fail(g, r, "launch of %s (global %ux%ux%u, local %ux%ux%u) failed (%d); see the message above", k->name, grid[0],
         grid[1], grid[2], wg[0], wg[1], wg[2], r);
    return 0;
  }
  return to_event(g, seq);
}

int rawgpu_wait(rawgpu *g, rawgpu_event e) {
  if (!e) return 0;
  int r = vi_ctx_wait(&g->c, (uint32_t)e, 10000);
  return r ? fail(g, r, "waiting for the GPU failed (%d); a kernel may be hung (rawgpu waves shows it)", r) : 0;
}

int rawgpu_is_done(rawgpu *g, rawgpu_event e) {
  if (!e) return 1;
  bool done = vi_ctx_done(&g->c, (uint32_t)e);
  return g->c.v.err ? fail(g, g->c.v.err, "register access failed") : done;
}

int rawgpu_finish(rawgpu *g) {
  int r = vi_ctx_finish(&g->c);
  return r ? fail(g, r, "waiting for the GPU failed (%d)", r) : 0;
}

/* ---- built-in kernels ---- */

rawgpu_event rawgpu_sgemm(rawgpu *g, int ta, int tb, size_t M, size_t N, size_t K, float alpha, const rawgpu_buf *A,
                          size_t lda, const rawgpu_buf *B, size_t ldb, float beta, rawgpu_buf *C, size_t ldc) {
  if (!M || !N || !K) { fail(g, -EINVAL, "sgemm of an empty matrix"); return 0; }
  if (M > 0xffffff80u || N > 0xffffff80u || K > 0xffffffffu || lda > 0xffffffffu || ldb > 0xffffffffu || ldc > 0xffffffffu) {
    fail(g, -E2BIG, "sgemm dimensions must fit in 32 bits");
    return 0;
  }
  /* op(A) is M x K: A is M x K (lda >= K) or K x M (lda >= M); likewise B */
  size_t arows = ta ? K : M, acols = ta ? M : K, brows = tb ? N : K, bcols = tb ? K : N;
  if (lda < acols || ldb < bcols || ldc < N) { fail(g, -EINVAL, "sgemm: a leading dimension is smaller than its row"); return 0; }
  if ((arows - 1) * lda + acols > A->b.size / 4 || (brows - 1) * ldb + bcols > B->b.size / 4 || (M - 1) * ldc + N > C->b.size / 4) {
    fail(g, -EINVAL, "sgemm: a matrix does not fit in its buffer");
    return 0;
  }
  if (!g->blas && !(g->blas = rawgpu_program_load_mem(g, rawgpu_embedded_sgemm, rawgpu_embedded_sgemm_size))) return 0;
  static const char *names[2][2] = {{"sgemm_nn", "sgemm_nt"}, {"sgemm_tn", "sgemm_tt"}};
  rawgpu_kernel *k = rawgpu_kernel_get(g->blas, names[!!ta][!!tb]);
  if (!k) return 0;
  struct __attribute__((packed)) {
    uint64_t a, b, c;
    uint32_t m, n, k, lda, ldb, ldc;
    float alpha, beta;
  } args = {A->b.gpu, B->b.gpu, C->b.gpu, (uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)lda, (uint32_t)ldb,
            (uint32_t)ldc, alpha, beta};
  /* the layout kernels/sgemm.cl declares: check it against the metadata once */
  if (k->nexplicit != 11 || k->k->args[k->meta[10]].offset != 52) {
    fail(g, -EINVAL, "the embedded %s does not have the argument layout this library expects", k->k->name);
    return 0;
  }
  memcpy(k->karg, &args, sizeof args);
  k->karg_size = sizeof args;
  memset(k->set, 1, k->nexplicit);
  const size_t global[2] = {(N + 127) / 128 * 16, (M + 127) / 128 * 16}, local[2] = {16, 16};
  return rawgpu_launch(g, k, 2, global, local);
}
