/* Generic PCI config-space decoding. Nothing vendor-specific lives here. */
#include "rg.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

int rg_pci_identify(rg_dev *d) {
  uint32_t id, cls, ss;
  int r;
  if ((r = rg_cfg_read(d, 0x00, 4, &id))) return r;
  if ((r = rg_cfg_read(d, 0x08, 4, &cls))) return r;
  if ((r = rg_cfg_read(d, 0x2C, 4, &ss))) return r;
  if (id == 0xFFFFFFFFu || id == 0) return -ENODEV;
  d->vendor = id & 0xFFFF;
  d->device = id >> 16;
  d->revision = cls & 0xFF;
  d->class_code = cls >> 8;
  d->subsys_vendor = ss & 0xFFFF;
  d->subsys_device = ss >> 16;
  return 0;
}

int rg_pci_find_cap(rg_dev *d, uint8_t want) {
  uint32_t status, ptr;
  if (rg_cfg_read(d, 0x06, 2, &status) || !(status & 0x10)) return 0; /* no capability list */
  if (rg_cfg_read(d, 0x34, 1, &ptr)) return 0;
  for (int guard = 0; ptr >= 0x40 && ptr < 0x100 && guard < 48; guard++) {
    uint32_t hdr;
    if (rg_cfg_read(d, ptr & ~3u, 2, &hdr)) return 0;
    if ((hdr & 0xFF) == want) return (int)(ptr & ~3u);
    ptr = (hdr >> 8) & 0xFF;
  }
  return 0;
}

int rg_pci_link(rg_dev *d, int *cur_gen, int *cur_width, int *max_gen, int *max_width) {
  int cap = rg_pci_find_cap(d, RG_PCI_CAP_EXP);
  if (!cap) return -ENOENT;
  uint32_t lcap, lsta;
  int r;
  if ((r = rg_cfg_read(d, (uint32_t)cap + 0x0C, 4, &lcap))) return r;
  if ((r = rg_cfg_read(d, (uint32_t)cap + 0x12, 2, &lsta))) return r;
  *max_gen = (int)(lcap & 0xF);
  *max_width = (int)((lcap >> 4) & 0x3F);
  *cur_gen = (int)(lsta & 0xF);
  *cur_width = (int)((lsta >> 4) & 0x3F);
  return 0;
}

/* Decodes the raw BAR registers only; sizing is the transport's job (rg_bar_map).
 * The upper dword of a 64-bit BAR is marked upper_half and has no meaning alone. */
int rg_pci_bars(rg_dev *d, rg_bar_desc bars[RG_MAX_BARS]) {
  uint32_t raw[RG_MAX_BARS];
  for (int i = 0; i < RG_MAX_BARS; i++)
    if (rg_cfg_read(d, 0x10 + 4u * (uint32_t)i, 4, &raw[i])) return -EIO;
  for (int i = 0; i < RG_MAX_BARS; i++) {
    rg_bar_desc *b = &bars[i];
    *b = (rg_bar_desc){0};
    if (i > 0 && bars[i - 1].is_64 && !bars[i - 1].upper_half) { b->upper_half = true; continue; }
    uint32_t lo = raw[i];
    b->is_io = lo & 1;
    b->is_64 = !b->is_io && ((lo >> 1) & 3) == 2 && i + 1 < RG_MAX_BARS;
    b->prefetch = !b->is_io && (lo & 8);
    uint64_t hi = b->is_64 ? raw[i + 1] : 0;
    b->raw = (hi << 32) | (b->is_io ? (lo & ~3u) : (lo & ~0xFu));
  }
  return 0;
}

const char *rg_pci_class_name(uint32_t c) {
  switch (c >> 8) {
  case 0x0300: return "VGA-compatible display controller";
  case 0x0302: return "3D controller";
  case 0x0380: return "display controller";
  case 0x0403: return "HD audio";
  case 0x1200: return "processing accelerator";
  default: return "other";
  }
}

const char *rg_pci_vendor_name(uint16_t v) {
  switch (v) {
  case 0x1002: return "AMD/ATI";
  case 0x10DE: return "NVIDIA";
  case 0x8086: return "Intel";
  case 0x148C: return "PowerColor (TUL)";
  case 0x1DA2: return "Sapphire";
  case 0x1043: return "ASUS";
  case 0x1462: return "MSI";
  case 0x1458: return "Gigabyte";
  case 0x1682: return "XFX";
  case 0x174B: return "Sapphire (PC Partner)";
  case 0x1849: return "ASRock";
  case 0x19DA: return "Zotac";
  case 0x3842: return "EVGA";
  case 0x106B: return "Apple";
  default: return "unknown vendor";
  }
}

int rg_pci_find_ext_cap(rg_dev *d, uint16_t want) {
  uint32_t off = 0x100;
  for (int guard = 0; off >= 0x100 && off < 0x1000 && guard < 64; guard++) {
    uint32_t hdr;
    if (rg_cfg_read(d, off, 4, &hdr) || hdr == 0 || hdr == 0xFFFFFFFFu) return 0;
    if ((hdr & 0xFFFF) == want) return (int)off;
    off = hdr >> 20;
  }
  return 0;
}

/* PCI status, PCIe device status and AER: what went wrong on the link, if anything. */
int rg_pci_errors(rg_dev *d, rg_pci_err *e) {
  memset(e, 0, sizeof *e);
  int r = rg_cfg_read(d, 0x06, 2, &e->status);
  int cap = rg_pci_find_cap(d, RG_PCI_CAP_EXP);
  if (!r && cap) r = rg_cfg_read(d, (uint32_t)cap + 0x0A, 2, &e->devsta);
  int aer = rg_pci_find_ext_cap(d, 0x0001);
  e->has_aer = aer != 0;
  if (!r && aer) {
    r = rg_cfg_read(d, (uint32_t)aer + 0x04, 4, &e->uncor);
    if (!r) r = rg_cfg_read(d, (uint32_t)aer + 0x10, 4, &e->cor);
    for (int i = 0; !r && i < 4; i++) r = rg_cfg_read(d, (uint32_t)aer + 0x1C + 4u * (uint32_t)i, 4, &e->hdr_log[i]);
  }
  return r;
}

/* Write-1-to-clear everything rg_pci_errors() reports. */
int rg_pci_clear_errors(rg_dev *d) {
  rg_pci_err e;
  int r = rg_pci_errors(d, &e);
  if (r) return r;
  r = rg_cfg_write(d, 0x06, 2, e.status & 0xF900);
  int cap = rg_pci_find_cap(d, RG_PCI_CAP_EXP);
  if (!r && cap) r = rg_cfg_write(d, (uint32_t)cap + 0x0A, 2, e.devsta & 0x000F);
  int aer = rg_pci_find_ext_cap(d, 0x0001);
  if (!r && aer) r = rg_cfg_write(d, (uint32_t)aer + 0x04, 4, e.uncor);
  if (!r && aer) r = rg_cfg_write(d, (uint32_t)aer + 0x10, 4, e.cor);
  return r;
}

void rg_pci_print_errors(const rg_pci_err *e) {
  static const struct { uint32_t bit; const char *name; } st[] = {
    {1u << 15, "detected parity error"}, {1u << 14, "signaled system error"}, {1u << 13, "received master abort"},
    {1u << 12, "received target abort"}, {1u << 11, "signaled target abort"}, {1u << 8, "master data parity error"}};
  static const struct { uint32_t bit; const char *name; } un[] = {
    {1u << 4, "data link protocol"}, {1u << 5, "surprise down"}, {1u << 12, "poisoned TLP"},
    {1u << 13, "flow control protocol"}, {1u << 14, "completion timeout"}, {1u << 15, "completer abort"},
    {1u << 16, "unexpected completion"}, {1u << 17, "receiver overflow"}, {1u << 18, "malformed TLP"},
    {1u << 19, "ECRC"}, {1u << 20, "unsupported request"}, {1u << 21, "ACS violation"}};
  printf("PCI status 0x%04x:", e->status);
  int any = 0;
  for (size_t i = 0; i < sizeof st / sizeof st[0]; i++) if (e->status & st[i].bit) { printf(" [%s]", st[i].name); any = 1; }
  printf("%s\n", any ? "" : " no errors");
  printf("PCIe device status 0x%04x:%s%s%s%s%s\n", e->devsta, e->devsta & 1 ? " [correctable]" : "",
         e->devsta & 2 ? " [non-fatal]" : "", e->devsta & 4 ? " [fatal]" : "", e->devsta & 8 ? " [unsupported request]" : "",
         e->devsta & 0xF ? "" : " no errors");
  if (!e->has_aer) { printf("AER: not present\n"); return; }
  printf("AER uncorrectable 0x%08x:", e->uncor);
  for (size_t i = 0; i < sizeof un / sizeof un[0]; i++) if (e->uncor & un[i].bit) printf(" [%s]", un[i].name);
  printf("%s\nAER correctable 0x%08x\n", e->uncor ? "" : " none", e->cor);
  if (e->uncor)
    printf("AER header log: %08x %08x %08x %08x\n", e->hdr_log[0], e->hdr_log[1], e->hdr_log[2], e->hdr_log[3]);
}
