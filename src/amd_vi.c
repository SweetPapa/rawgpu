/* Backend: AMD GCN3/GCN4 "Volcanic Islands" dGPUs (Tonga, Fiji, Polaris 10/11/12, VegaM).
 *
 * Register offsets come from Linux amdgpu (drivers/gpu/drm/amd/include/asic_reg/,
 * MIT-licensed); the Linux headers give dword indices, so byte offset = index * 4.
 *
 * Everything here is read-only except the ROM read, which writes the SMC
 * index register exactly as amdgpu's vi_read_bios_from_rom() does before the
 * card is posted.
 */
#include "rg.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VI_REG_BAR 5

/* bif_5_0_d.h */
#define mmCONFIG_MEMSIZE   (0x150aU * 4)
#define mmBIOS_SCRATCH_0   (0x5c9U * 4)
/* smu_7_1_1_d.h / smu_7_1_3_d.h (same values on Polaris) */
#define mmSMC_IND_INDEX_11 (0x1ACU * 4)
#define mmSMC_IND_DATA_11  (0x1ADU * 4)
#define ixROM_INDEX        0xc0600010U
#define ixROM_DATA         0xc0600014U

#define VI_ROM_MAX (256u << 10)

static int map_regs(rg_dev *d) {
  uint64_t sz = 0;
  int r = rg_bar_map(d, VI_REG_BAR, &sz);
  if (r) { fprintf(stderr, "rawgpu: could not map register BAR%d\n", VI_REG_BAR); return r; }
  if (sz < mmCONFIG_MEMSIZE + 4) { fprintf(stderr, "rawgpu: register BAR too small (%llu)\n", (unsigned long long)sz); return -EINVAL; }
  return 0;
}

/* Streams `n` dwords of the VBIOS starting at byte 0 through the auto-incrementing ROM_DATA port. */
static int rom_read(rg_dev *d, uint32_t *dst, size_t n) {
  int r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_INDEX_11, ixROM_INDEX))) return r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_DATA_11, 0))) return r;
  if ((r = rg_wreg32(d, VI_REG_BAR, mmSMC_IND_INDEX_11, ixROM_DATA))) return r;
  for (size_t i = 0; i < n; i++)
    if ((r = rg_rreg32(d, VI_REG_BAR, mmSMC_IND_DATA_11, &dst[i]))) return r;
  return 0;
}

static int vi_vbios_read(rg_dev *d, uint8_t **out, size_t *len) {
  int r = map_regs(d);
  if (r) return r;
  uint32_t hdr[128];
  if ((r = rom_read(d, hdr, 128))) return r;
  const uint8_t *h = (const uint8_t *)hdr;
  if (h[0] != 0x55 || h[1] != 0xAA) {
    fprintf(stderr, "rawgpu: no option-ROM signature (got %02x %02x)\n", h[0], h[1]);
    return -ENOENT;
  }
  size_t size = (size_t)h[2] * 512;
  if (size == 0 || size > VI_ROM_MAX) size = 64u << 10;
  uint8_t *buf = malloc(size);
  if (!buf) return -ENOMEM;
  if ((r = rom_read(d, (uint32_t *)buf, size / 4))) { free(buf); return r; }
  *out = buf;
  *len = size;
  return 0;
}

static int vi_probe(rg_dev *d, const rg_ident *id) {
  (void)id;
  int r = map_regs(d);
  if (r) return r;

  uint32_t memsize = 0;
  if ((r = rg_rreg32(d, VI_REG_BAR, mmCONFIG_MEMSIZE, &memsize))) return r;
  bool posted = memsize != 0 && memsize != 0xFFFFFFFFu; /* amdgpu_device_need_post() */
  printf("  posted:          %s (CONFIG_MEMSIZE = 0x%08x%s)\n", posted ? "yes" : "NO",
         memsize, posted ? "" : "; nothing has run the VBIOS init, so VRAM is not usable yet");
  if (posted) printf("  VRAM:            %u MB\n", memsize);

  printf("  BIOS scratch:   ");
  for (int i = 0; i < 8; i++) {
    uint32_t v = 0;
    if ((r = rg_rreg32(d, VI_REG_BAR, mmBIOS_SCRATCH_0 + 4u * (uint32_t)i, &v))) return r;
    printf(" %08x", v);
  }
  printf("\n");

  uint8_t *bios = NULL;
  size_t len = 0;
  if ((r = vi_vbios_read(d, &bios, &len))) {
    printf("  VBIOS:           unreadable (%d)\n", r);
    return 0;
  }
  rg_atom_report(bios, len);
  free(bios);
  return 0;
}

const rg_backend rg_backend_amd_vi = {
  .name = "amd-vi",
  .reg_bar = VI_REG_BAR,
  .safe_reg = mmBIOS_SCRATCH_0,
  .probe = vi_probe,
  .vbios_read = vi_vbios_read,
};
