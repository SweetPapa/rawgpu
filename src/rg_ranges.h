/* A first-fit allocator over one address range, for VRAM and GART pages.
 * The free list is a fixed array kept sorted and coalesced, so the struct can
 * live anywhere without setup beyond rg_ranges_init(). */
#ifndef RAWGPU_RG_RANGES_H
#define RAWGPU_RG_RANGES_H

#include <stdint.h>

#define RG_RANGES_MAX 256

typedef struct { uint64_t off, size; } rg_extent;
typedef struct {
  uint32_t n;
  rg_extent e[RG_RANGES_MAX]; /* free extents, sorted by offset, never adjacent */
} rg_ranges;

void rg_ranges_init(rg_ranges *r, uint64_t off, uint64_t size);
/* align is a power of two. -ENOMEM if no free extent fits. */
int rg_ranges_alloc(rg_ranges *r, uint64_t size, uint64_t align, uint64_t *off);
/* -ENOSPC if the free list is full (the range is then lost, not corrupted) */
int rg_ranges_free(rg_ranges *r, uint64_t off, uint64_t size);
uint64_t rg_ranges_free_bytes(const rg_ranges *r);
uint64_t rg_ranges_largest(const rg_ranges *r);

#endif
