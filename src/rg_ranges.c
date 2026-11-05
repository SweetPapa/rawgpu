/* rg_ranges.h */
#include "rg_ranges.h"

#include <errno.h>
#include <stdbool.h>
#include <string.h>

void rg_ranges_init(rg_ranges *r, uint64_t off, uint64_t size) {
  r->n = size ? 1 : 0;
  r->e[0] = (rg_extent){off, size};
}

static void erase(rg_ranges *r, uint32_t i) {
  memmove(&r->e[i], &r->e[i + 1], (r->n - i - 1) * sizeof r->e[0]);
  r->n--;
}

static int insert(rg_ranges *r, uint32_t i, rg_extent x) {
  if (r->n == RG_RANGES_MAX) return -ENOSPC;
  memmove(&r->e[i + 1], &r->e[i], (r->n - i) * sizeof r->e[0]);
  r->e[i] = x;
  r->n++;
  return 0;
}

int rg_ranges_alloc(rg_ranges *r, uint64_t size, uint64_t align, uint64_t *off) {
  if (!size || (align & (align - 1))) return -EINVAL;
  if (!align) align = 1;
  for (uint32_t i = 0; i < r->n; i++) {
    rg_extent x = r->e[i];
    uint64_t a = (x.off + align - 1) & ~(align - 1), end = x.off + x.size;
    if (a < x.off || a > end || end - a < size) continue;
    /* split x into [x.off, a) used-before, [a, a+size) taken, [a+size, end) after */
    uint64_t before = a - x.off, after = end - (a + size);
    if (before && after) {
      if (r->n == RG_RANGES_MAX) continue; /* would need one more extent */
      r->e[i].size = before;
      insert(r, i + 1, (rg_extent){a + size, after});
    } else if (before) r->e[i].size = before;
    else if (after) r->e[i] = (rg_extent){a + size, after};
    else erase(r, i);
    *off = a;
    return 0;
  }
  return -ENOMEM;
}

int rg_ranges_free(rg_ranges *r, uint64_t off, uint64_t size) {
  if (!size) return 0;
  uint32_t i = 0;
  while (i < r->n && r->e[i].off < off) i++;
  bool left = i > 0 && r->e[i - 1].off + r->e[i - 1].size == off;
  bool right = i < r->n && off + size == r->e[i].off;
  if (left && right) {
    r->e[i - 1].size += size + r->e[i].size;
    erase(r, i);
  } else if (left) r->e[i - 1].size += size;
  else if (right) r->e[i] = (rg_extent){off, size + r->e[i].size};
  else return insert(r, i, (rg_extent){off, size});
  return 0;
}

uint64_t rg_ranges_free_bytes(const rg_ranges *r) {
  uint64_t t = 0;
  for (uint32_t i = 0; i < r->n; i++) t += r->e[i].size;
  return t;
}

uint64_t rg_ranges_largest(const rg_ranges *r) {
  uint64_t t = 0;
  for (uint32_t i = 0; i < r->n; i++)
    if (r->e[i].size > t) t = r->e[i].size;
  return t;
}
