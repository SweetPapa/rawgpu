/* PCI ID -> chip family -> backend.
 *
 * This is the "swap the card" layer. Identification is data-driven (the AMD
 * table is generated from Linux amdgpu); each family either has a rawgpu
 * backend or a one-line note saying what can drive it today.
 */
#include "rg.h"

#include <stddef.h>
#include <string.h>

#include "amd_ids.inc"

typedef struct { const char *chip; rg_gen gen; } chip_gen;

static const chip_gen amd_chip_gens[] = {
  {"TAHITI", RG_GEN_AMD_SI}, {"PITCAIRN", RG_GEN_AMD_SI}, {"VERDE", RG_GEN_AMD_SI},
  {"OLAND", RG_GEN_AMD_SI}, {"HAINAN", RG_GEN_AMD_SI},
  {"BONAIRE", RG_GEN_AMD_CIK}, {"HAWAII", RG_GEN_AMD_CIK}, {"KAVERI", RG_GEN_AMD_CIK},
  {"KABINI", RG_GEN_AMD_CIK}, {"MULLINS", RG_GEN_AMD_CIK},
  {"TOPAZ", RG_GEN_AMD_VI}, {"TONGA", RG_GEN_AMD_VI}, {"FIJI", RG_GEN_AMD_VI},
  {"CARRIZO", RG_GEN_AMD_VI}, {"STONEY", RG_GEN_AMD_VI}, {"POLARIS10", RG_GEN_AMD_VI},
  {"POLARIS11", RG_GEN_AMD_VI}, {"POLARIS12", RG_GEN_AMD_VI}, {"VEGAM", RG_GEN_AMD_VI},
  {"VEGA10", RG_GEN_AMD_GFX9}, {"VEGA12", RG_GEN_AMD_GFX9}, {"VEGA20", RG_GEN_AMD_GFX9},
  {"RAVEN", RG_GEN_AMD_GFX9}, {"RENOIR", RG_GEN_AMD_GFX9}, {"ARCTURUS", RG_GEN_AMD_GFX9},
  {"ALDEBARAN", RG_GEN_AMD_GFX9},
  {"NAVI10", RG_GEN_AMD_NAVI1X}, {"NAVI12", RG_GEN_AMD_NAVI1X}, {"NAVI14", RG_GEN_AMD_NAVI1X},
  {"CYAN_SKILLFISH", RG_GEN_AMD_NAVI1X},
  {"SIENNA_CICHLID", RG_GEN_AMD_NAVI2X}, {"NAVY_FLOUNDER", RG_GEN_AMD_NAVI2X},
  {"DIMGREY_CAVEFISH", RG_GEN_AMD_NAVI2X}, {"BEIGE_GOBY", RG_GEN_AMD_NAVI2X},
  {"YELLOW_CARP", RG_GEN_AMD_NAVI2X},
};

static const char *gen_name(rg_gen g) {
  switch (g) {
  case RG_GEN_AMD_SI: return "AMD GCN1 (Southern Islands)";
  case RG_GEN_AMD_CIK: return "AMD GCN2 (Sea Islands / CIK)";
  case RG_GEN_AMD_VI: return "AMD GCN3/GCN4 (Volcanic Islands: Tonga/Fiji/Polaris)";
  case RG_GEN_AMD_GFX9: return "AMD GCN5 (Vega / early CDNA)";
  case RG_GEN_AMD_NAVI1X: return "AMD RDNA1";
  case RG_GEN_AMD_NAVI2X: return "AMD RDNA2";
  case RG_GEN_AMD_IPDISC: return "AMD, newer than the static ID table (RDNA3/RDNA4/CDNA3+)";
  case RG_GEN_NVIDIA: return "NVIDIA";
  default: return "unknown";
  }
}

static const char *support_note(rg_gen g, uint32_t flags) {
  if (flags & RG_AMD_APU) return "integrated GPU (APU), not a PCIe card: out of scope";
  switch (g) {
  case RG_GEN_AMD_VI:
    return "rawgpu amd-vi backend: probe + VBIOS today; bring-up targets Polaris first";
  case RG_GEN_AMD_SI:
  case RG_GEN_AMD_CIK:
    return "no backend yet; closest relative of amd-vi (GCN, ATOM BIOS, SMC firmware)";
  case RG_GEN_AMD_GFX9:
  case RG_GEN_AMD_NAVI1X:
  case RG_GEN_AMD_NAVI2X:
    return "no backend yet; tinygrad does not drive this generation either";
  case RG_GEN_AMD_IPDISC:
    return "tinygrad's AM driver drives RDNA3+ over this same TinyGPU transport";
  case RG_GEN_NVIDIA:
    return "tinygrad drives Ampere+ over TinyGPU; rawgpu has no NVIDIA backend";
  default:
    return "TinyGPU only binds AMD (0x1002) and NVIDIA (0x10de); other vendors need their own driver extension";
  }
}

static const rg_backend *backend_for(rg_gen g, uint32_t flags) {
  if (flags & RG_AMD_APU) return NULL;
  if (g == RG_GEN_AMD_VI) return &rg_backend_amd_vi;
  return NULL;
}

void rg_identify_family(const rg_dev *d, rg_ident *out) {
  memset(out, 0, sizeof *out);
  if (d->vendor == 0x10DE) {
    out->gen = RG_GEN_NVIDIA;
  } else if (d->vendor == 0x1002) {
    out->gen = RG_GEN_AMD_IPDISC; /* amdgpu's fallback for display-class AMD devices */
    for (size_t i = 0; i < sizeof rg_amd_ids / sizeof rg_amd_ids[0]; i++) {
      if (rg_amd_ids[i].device != d->device) continue;
      out->chip = rg_amd_ids[i].chip;
      out->flags = rg_amd_ids[i].flags;
      out->gen = RG_GEN_UNKNOWN;
      for (size_t j = 0; j < sizeof amd_chip_gens / sizeof amd_chip_gens[0]; j++)
        if (!strcmp(amd_chip_gens[j].chip, out->chip)) out->gen = amd_chip_gens[j].gen;
      break;
    }
  }
  out->gen_name = gen_name(out->gen);
  out->backend = backend_for(out->gen, out->flags);
  out->support = support_note(out->gen, out->flags);
}
