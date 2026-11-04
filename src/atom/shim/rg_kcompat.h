/* Minimal Linux-kernel API surface needed to compile amdgpu's atom.c, unmodified,
 * in userspace. Every shim header in this directory just includes this file.
 *
 * Delays and the jump-loop watchdog call back into rawgpu (src/atom/rg_atom.c):
 * a delay must first flush register writes still queued in the TinyGPU socket,
 * or a "write, wait 10 us, write" sequence would reach the hardware without the wait.
 */
#ifndef RG_KCOMPAT_H
#define RG_KCOMPAT_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "rawgpu's ATOM shim assumes a little-endian host"
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;
typedef uint16_t __le16;
typedef uint32_t __le32;

/* single-threaded CLI: locks are no-ops */
struct mutex { int unused; };
#define mutex_init(m) ((void)(m))
#define mutex_lock(m) ((void)(m))
#define mutex_unlock(m) ((void)(m))

#define GFP_KERNEL 0
#define kzalloc(sz, gfp) calloc(1, (sz))
#define kcalloc(n, sz, gfp) calloc((n), (sz))
#define kmalloc(sz, gfp) malloc(sz)
#define kfree(p) free((void *)(p))
#define kzalloc_obj(T, ...) calloc(1, sizeof(T))

/* logging: opcode-level debug output goes to rg_atom_logf when amdgpu_atom_debug is set */
extern FILE *rg_atom_logf;
#define KERN_DEBUG ""
#define KERN_INFO ""
#define KERN_ERR ""
#define printk(...) fprintf(rg_atom_logf ? rg_atom_logf : stderr, __VA_ARGS__)
#define pr_info(...) fprintf(stderr, "atom: " __VA_ARGS__)
#define pr_err(...) fprintf(stderr, "atom: " __VA_ARGS__)
#define pr_warn(...) fprintf(stderr, "atom: " __VA_ARGS__)
#define DRM_ERROR(...) fprintf(stderr, "atom: error: " __VA_ARGS__)
#define DRM_INFO(...) fprintf(stderr, "atom: " __VA_ARGS__)
#define DRM_DEBUG(...) ((void)0)
#define drm_info(dev, ...) ((void)(dev), fprintf(stderr, "atom: " __VA_ARGS__))

/* time */
void rg_atom_udelay(unsigned long us);
unsigned long rg_atom_jiffies(void); /* milliseconds */
#define HZ 1000
#define jiffies rg_atom_jiffies()
#define time_after(a, b) ((long)((b) - (a)) < 0)
#define jiffies_to_msecs(j) ((unsigned int)(j))
#define msecs_to_jiffies(m) ((unsigned long)(m))
#define udelay(us) rg_atom_udelay((unsigned long)(us))
#define mdelay(ms) rg_atom_udelay((unsigned long)(ms) * 1000ul)
#define msleep(ms) rg_atom_udelay((unsigned long)(ms) * 1000ul)
#define drm_can_sleep() 1

/* arithmetic: do_div divides a u64 in place and yields the remainder */
#define do_div(n, base) __extension__({ uint32_t __b = (base); uint32_t __r = (uint32_t)((n) % __b); (n) /= __b; __r; })
#define lower_32_bits(n) ((uint32_t)((n) & 0xffffffffu))
#define upper_32_bits(n) ((uint32_t)(((uint64_t)(n)) >> 32))

/* misc */
#include <errno.h>
#define str_yes_no(b) ((b) ? "yes" : "no")
#define OPTIMIZER_HIDE_VAR(v) __asm__ volatile("" : "+r"(v))
#define EXPORT_SYMBOL(x)
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define cpu_to_le16(x) ((uint16_t)(x))
#define cpu_to_le32(x) ((uint32_t)(x))
#define le16_to_cpu(x) ((uint16_t)(x))
#define le32_to_cpu(x) ((uint32_t)(x))
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

static inline uint32_t get_unaligned_le32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint16_t get_unaligned_le16(const void *p) { uint16_t v; memcpy(&v, p, 2); return v; }

static inline long strscpy(char *dst, const char *src, size_t n) {
  if (!n) return -7; /* -E2BIG */
  size_t l = strnlen(src, n);
  if (l == n) { memcpy(dst, src, n - 1); dst[n - 1] = 0; return -7; }
  memcpy(dst, src, l + 1);
  return (long)l;
}

static inline int scnprintf(char *buf, size_t size, const char *fmt, ...) {
  if (!size) return 0;
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, size, fmt, ap);
  va_end(ap);
  if (n < 0) return 0;
  return (size_t)n >= size ? (int)size - 1 : n;
}

#endif
