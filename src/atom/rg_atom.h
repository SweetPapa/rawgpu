/* rawgpu bridge to amdgpu's ATOM BIOS interpreter (src/atom/upstream/atom.c).
 * Kernel types stay behind this header; the rest of rawgpu only sees these. */
#ifndef RAWGPU_RG_ATOM_H
#define RAWGPU_RG_ATOM_H

#include <stdio.h>

#include "../rg.h"

typedef struct rg_atom rg_atom;

typedef struct {
  int reg_bar;          /* BAR holding the register aperture */
  uint32_t mm_index;    /* byte offsets of the indirect window used for registers past the BAR */
  uint32_t mm_data;
  uint32_t flush_reg;   /* side-effect-free register; reading it drains queued writes */
  FILE *trace;          /* register-level trace (R/W/D lines), or NULL */
  bool trace_ops;       /* also send amdgpu's opcode-level ATOM debug output to `trace` */
} rg_atom_cfg;

typedef struct {
  uint64_t reads, writes, indirect, delays, delay_us;
  double seconds;
} rg_atom_stats;

rg_atom *rg_atom_open(rg_dev *d, const uint8_t *bios, size_t len, const rg_atom_cfg *cfg);
void rg_atom_close(rg_atom *a);

void rg_atom_print_info(rg_atom *a);
int rg_atom_default_clocks(rg_atom *a, uint32_t *sclk_10khz, uint32_t *mclk_10khz);
/* first VRAM_UsageByFirmware entry: where the VBIOS keeps its own scratch in VRAM */
int rg_atom_fw_vram_reserve(rg_atom *a, uint32_t *start, uint32_t *kb);

/* Runs ASIC_Init with the BIOS default clocks, as amdgpu_atom_asic_init() does.
 * 0 = success; 1 = the BIOS lacks clocks or the table; <0 = interpreter or transport failure. */
int rg_atom_asic_init(rg_atom *a, rg_atom_stats *st);

#endif
