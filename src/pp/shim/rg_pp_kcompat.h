/* Kernel API for amdgpu's powerplay code, compiled in userspace (src/pp/upstream).
 * Every shim header under src/pp/shim/ includes this. Only what the vendored
 * files use is here; see docs/DPM.md. Delays and logging call into rg_pp.c. */
#ifndef RG_PP_KCOMPAT_H
#define RG_PP_KCOMPAT_H

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "rawgpu's powerplay shim assumes a little-endian host"
#endif

/* ---- types ---- */
typedef uint8_t u8, __u8;
typedef uint16_t u16, __u16;
typedef uint32_t u32, __u32;
typedef uint64_t u64, __u64;
typedef int8_t s8, __s8;
typedef int16_t s16, __s16;
typedef int32_t s32, __s32;
typedef int64_t s64, __s64;
typedef uint16_t __le16, __be16;
typedef uint32_t __le32, __be32;
typedef uint64_t __le64, __be64;
typedef uint64_t dma_addr_t, phys_addr_t, resource_size_t;
typedef unsigned int uint;
typedef unsigned long ulong;

/* ---- attributes and compiler helpers ---- */
#define __iomem
#define __user
#define __force
#define __packed __attribute__((packed))
#define __aligned(x) __attribute__((aligned(x)))
#define __maybe_unused __attribute__((unused))
#define __always_unused __attribute__((unused))
#define __must_check
#define __always_inline inline __attribute__((always_inline))
#define noinline __attribute__((noinline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define fallthrough __attribute__((fallthrough))
#define BUILD_BUG_ON(c) _Static_assert(!(c), #c)
#define static_assert(c, ...) _Static_assert(c, #c)
#define IS_ENABLED(x) 0
#define EXPORT_SYMBOL(x)
#define EXPORT_SYMBOL_GPL(x)
#define MODULE_FIRMWARE(x)
#define READ_ONCE(x) (*(volatile __typeof__(x) *)&(x))
#define WRITE_ONCE(x, v) (*(volatile __typeof__(x) *)&(x) = (v))

/* ---- memory ---- */
#define GFP_KERNEL 0
#define GFP_ATOMIC 0
#define kzalloc(sz, gfp) calloc(1, (sz))
#define kcalloc(n, sz, gfp) calloc((n), (sz))
#define kmalloc(sz, gfp) malloc(sz)
#define kmalloc_array(n, sz, gfp) calloc((n), (sz))
#define kvzalloc(sz, gfp) calloc(1, (sz))
#define kfree(p) free((void *)(p))
#define kvfree(p) free((void *)(p))
#define krealloc(p, sz, gfp) realloc((p), (sz))
/* newer kernels: kzalloc_obj(T) and kzalloc_objs(T, n) */
#define kzalloc_obj(T, ...) ((__typeof__(T) *)calloc(1, sizeof(T)))
#define kzalloc_objs(T, n, ...) ((__typeof__(T) *)calloc((n), sizeof(T)))
#define kzalloc_flex(T, member, n, ...) \
  ((__typeof__(T) *)calloc(1, sizeof(T) + (size_t)(n) * sizeof(((__typeof__(T) *)0)->member[0])))
static inline void *kmemdup(const void *src, size_t len, int gfp) {
  void *p = malloc(len);
  if (p) memcpy(p, src, len);
  return p;
}
#define memcpy_toio(dst, src, n) memcpy((dst), (src), (n))
#define memcpy_fromio(dst, src, n) memcpy((dst), (src), (n))
#define memset_io(dst, c, n) memset((dst), (c), (n))
#define struct_size(p, member, n) (sizeof(*(p)) + (size_t)(n) * sizeof((p)->member[0]))
#define flex_array_size(p, member, n) ((size_t)(n) * sizeof((p)->member[0]))

/* ---- logging: rg_pp.c sends it to stderr with a prefix ---- */
void rg_pp_log(const char *level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
#define KERN_ERR "error"
#define KERN_WARNING "warning"
#define KERN_INFO "info"
#define KERN_DEBUG "debug"
#define printk(...) rg_pp_log("info", __VA_ARGS__)
#define pr_err(...) rg_pp_log("error", __VA_ARGS__)
#define pr_warn(...) rg_pp_log("warning", __VA_ARGS__)
#define pr_warn_once(...) rg_pp_log("warning", __VA_ARGS__)
#define pr_info(...) rg_pp_log("info", __VA_ARGS__)
#define pr_notice(...) rg_pp_log("info", __VA_ARGS__)
#define pr_debug(...) rg_pp_log("debug", __VA_ARGS__)
#define pr_err_once(...) rg_pp_log("error", __VA_ARGS__)
#define pr_warn_ratelimited(...) rg_pp_log("warning", __VA_ARGS__)
#define pr_err_ratelimited(...) rg_pp_log("error", __VA_ARGS__)
#define dev_emerg(d, ...) ((void)(d), rg_pp_log("EMERGENCY", __VA_ARGS__))
#define dev_crit(d, ...) ((void)(d), rg_pp_log("critical", __VA_ARGS__))
#define pr_info_once(...) rg_pp_log("info", __VA_ARGS__)
#define dev_err(d, ...) ((void)(d), rg_pp_log("error", __VA_ARGS__))
#define dev_warn(d, ...) ((void)(d), rg_pp_log("warning", __VA_ARGS__))
#define dev_info(d, ...) ((void)(d), rg_pp_log("info", __VA_ARGS__))
#define dev_dbg(d, ...) ((void)(d), rg_pp_log("debug", __VA_ARGS__))
#define dev_err_once dev_err
#define dev_warn_once dev_warn
#define drm_err(d, ...) ((void)(d), rg_pp_log("error", __VA_ARGS__))
#define drm_warn(d, ...) ((void)(d), rg_pp_log("warning", __VA_ARGS__))
#define drm_info(d, ...) ((void)(d), rg_pp_log("info", __VA_ARGS__))
#define drm_dbg(d, ...) ((void)(d), rg_pp_log("debug", __VA_ARGS__))
#define drm_dbg_driver drm_dbg
#define DRM_ERROR(...) rg_pp_log("error", __VA_ARGS__)
#define DRM_WARN(...) rg_pp_log("warning", __VA_ARGS__)
#define DRM_INFO(...) rg_pp_log("info", __VA_ARGS__)
#define DRM_DEBUG(...) rg_pp_log("debug", __VA_ARGS__)
#define DRM_DEBUG_DRIVER(...) rg_pp_log("debug", __VA_ARGS__)
#define WARN_ON(c) ({ int __c = !!(c); if (__c) rg_pp_log("warning", "WARN_ON(%s) at %s:%d\n", #c, __FILE__, __LINE__); __c; })
#define WARN_ON_ONCE(c) WARN_ON(c)
#define WARN(c, ...) ({ int __c = !!(c); if (__c) rg_pp_log("warning", __VA_ARGS__); __c; })
#define WARN_ONCE WARN
#define BUG_ON(c) do { if (c) { rg_pp_log("error", "BUG_ON(%s) at %s:%d\n", #c, __FILE__, __LINE__); abort(); } } while (0)
#define BUG() BUG_ON(1)

/* ---- time: rg_pp.c drains queued register writes before sleeping ---- */
void rg_pp_udelay(unsigned long us);
unsigned long rg_pp_jiffies(void);
#define HZ 1000
#define jiffies rg_pp_jiffies()
#define time_after(a, b) ((long)((b) - (a)) < 0)
#define time_before(a, b) time_after(b, a)
#define msecs_to_jiffies(m) ((unsigned long)(m))
#define jiffies_to_msecs(j) ((unsigned int)(j))
#define udelay(us) rg_pp_udelay((unsigned long)(us))
#define ndelay(ns) rg_pp_udelay(((unsigned long)(ns) + 999) / 1000)
#define mdelay(ms) rg_pp_udelay((unsigned long)(ms) * 1000ul)
#define msleep(ms) rg_pp_udelay((unsigned long)(ms) * 1000ul)
#define usleep_range(a, b) rg_pp_udelay((unsigned long)(a))
#define might_sleep() ((void)0)

/* ---- arithmetic ---- */
#define do_div(n, base) __extension__({ uint32_t __b = (base); uint32_t __r = (uint32_t)((n) % __b); (n) /= __b; __r; })
static inline uint64_t div_u64(uint64_t a, uint32_t b) { return a / b; }
static inline int64_t div_s64(int64_t a, int32_t b) { return a / b; }
static inline uint64_t div64_u64(uint64_t a, uint64_t b) { return a / b; }
static inline int64_t div64_s64(int64_t a, int64_t b) { return a / b; }
static inline uint64_t div_u64_rem(uint64_t a, uint32_t b, uint32_t *r) { *r = (uint32_t)(a % b); return a / b; }
#define lower_32_bits(n) ((uint32_t)((n) & 0xffffffffu))
#define upper_32_bits(n) ((uint32_t)(((uint64_t)(n)) >> 32))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define BIT(n) (1ul << (n))
#define BIT_ULL(n) (1ull << (n))
#define GENMASK(h, l) (((~0ul) << (l)) & (~0ul >> (sizeof(long) * 8 - 1 - (h))))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define DIV_ROUND_UP_ULL(n, d) DIV_ROUND_UP((unsigned long long)(n), (d))
#define DIV_ROUND_CLOSEST(x, d) (((x) + ((d) / 2)) / (d))
#define DIV_ROUND_CLOSEST_ULL(x, d) ({ unsigned long long __x = (x), __d = (d); (__x + __d / 2) / __d; })
#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define rounddown(x, y) ((x) - ((x) % (y)))
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((__typeof__(x))(a) - 1))
#define min(a, b) ({ __typeof__(a) __a = (a); __typeof__(b) __b = (b); __a < __b ? __a : __b; })
#define max(a, b) ({ __typeof__(a) __a = (a); __typeof__(b) __b = (b); __a > __b ? __a : __b; })
#define min3(a, b, c) min(min(a, b), c)
#define max3(a, b, c) max(max(a, b), c)
#define min_t(t, a, b) ({ t __a = (a); t __b = (b); __a < __b ? __a : __b; })
#define max_t(t, a, b) ({ t __a = (a); t __b = (b); __a > __b ? __a : __b; })
#define clamp(v, lo, hi) min(max(v, lo), hi)
#define clamp_t(t, v, lo, hi) min_t(t, max_t(t, v, lo), hi)
#define clamp_val(v, lo, hi) clamp(v, lo, hi)
#define swap(a, b) do { __typeof__(a) __t = (a); (a) = (b); (b) = __t; } while (0)
#define abs(x) ({ __typeof__(x) __x = (x); __x < 0 ? -__x : __x; })
#include <strings.h> /* ffs(), fls(): same semantics as the kernel's */
static inline unsigned long __ffs(unsigned long x) { return (unsigned long)__builtin_ctzl(x); }
static inline unsigned int hweight32(uint32_t x) { return (unsigned int)__builtin_popcount(x); }
#define ilog2(n) ((int)(63 - __builtin_clzll((unsigned long long)(n))))
#define is_power_of_2(n) ((n) != 0 && (((n) & ((n) - 1)) == 0))

/* ---- byte order (little-endian host) ---- */
#define cpu_to_le16(x) ((uint16_t)(x))
#define cpu_to_le32(x) ((uint32_t)(x))
#define cpu_to_le64(x) ((uint64_t)(x))
#define le16_to_cpu(x) ((uint16_t)(x))
#define le32_to_cpu(x) ((uint32_t)(x))
#define le64_to_cpu(x) ((uint64_t)(x))
#define cpu_to_be16(x) __builtin_bswap16((uint16_t)(x))
#define cpu_to_be32(x) __builtin_bswap32((uint32_t)(x))
#define cpu_to_be64(x) __builtin_bswap64((uint64_t)(x))
#define be16_to_cpu(x) __builtin_bswap16((uint16_t)(x))
#define be32_to_cpu(x) __builtin_bswap32((uint32_t)(x))
#define be64_to_cpu(x) __builtin_bswap64((uint64_t)(x))
#define le16_to_cpup(p) (*(const uint16_t *)(p))
#define le32_to_cpup(p) (*(const uint32_t *)(p))
static inline uint32_t get_unaligned_le32(const void *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint16_t get_unaligned_le16(const void *p) { uint16_t v; memcpy(&v, p, 2); return v; }

/* ---- locking: rawgpu is single-threaded ---- */
struct mutex { int unused; };
typedef struct { int unused; } spinlock_t;
#define mutex_init(m) ((void)(m))
#define mutex_destroy(m) ((void)(m))
#define mutex_lock(m) ((void)(m))
#define mutex_unlock(m) ((void)(m))
#define spin_lock_init(l) ((void)(l))
#define spin_lock_irqsave(l, f) ((void)(l), (void)(f))
#define spin_unlock_irqrestore(l, f) ((void)(l), (void)(f))
#define local_irq_save(f) ((void)(f))
#define local_irq_restore(f) ((void)(f))
#define preempt_disable() ((void)0)
#define preempt_enable() ((void)0)

/* ---- strings ---- */
static inline int scnprintf(char *buf, size_t size, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, size, fmt, ap);
  va_end(ap);
  if (n < 0) return 0;
  return (size_t)n >= size ? (size ? (int)size - 1 : 0) : n;
}
#define PAGE_SIZE 4096
#define PAGE_SHIFT 12
#define offset_in_page(p) ((unsigned long)(p) & (PAGE_SIZE - 1))
#define sysfs_emit(buf, ...) scnprintf((buf), PAGE_SIZE, __VA_ARGS__)
#define sysfs_emit_at(buf, at, ...) scnprintf((buf) + (at), PAGE_SIZE - (at), __VA_ARGS__)
static inline long strscpy(char *dst, const char *src, size_t n) {
  if (!n) return -E2BIG;
  size_t l = strnlen(src, n - 1);
  memcpy(dst, src, l);
  dst[l] = 0;
  return (long)l;
}
#define str_yes_no(b) ((b) ? "yes" : "no")

/* ---- kernel-only errno values ---- */
#ifndef ENOTSUPP
#define ENOTSUPP 524
#endif
#ifndef ETIME
#define ETIME 62
#endif
#ifndef ERESTARTSYS
#define ERESTARTSYS 512
#endif

/* ---- devices ---- */
struct device { int unused; };
struct pci_dev {
  uint16_t vendor, device, subsystem_vendor, subsystem_device;
  uint8_t revision;
  struct device dev;
};
struct firmware { size_t size; const uint8_t *data; };
/* workqueues: hwmgr embeds one for its thermal work; rawgpu never schedules it */
struct work_struct { void (*func)(struct work_struct *); };
struct delayed_work { struct work_struct work; };
#define INIT_WORK(w, f) ((w)->func = (f))
#define INIT_DELAYED_WORK(w, f) ((w)->work.func = (f))
#define schedule_work(w) ((void)(w), true)
#define schedule_delayed_work(w, d) ((void)(w), (void)(d), true)
#define cancel_delayed_work_sync(w) ((void)(w), false)
#define cancel_work_sync(w) ((void)(w), false)
#define container_of(ptr, type, member) ((type *)((char *)(ptr) - offsetof(type, member)))
struct seq_file { int unused; };
#define seq_printf(m, ...) ((void)(m))
void rg_pp_poweroff(const char *why); /* orderly_poweroff() on a thermal emergency: stop everything */
#define orderly_poweroff(force) (rg_pp_poweroff("orderly_poweroff"), 0)

#endif
