/* rawgpu CLI. Every command is read-only unless noted. */
#include "rg.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *usage =
  "usage: rawgpu [--sock PATH] <command>\n"
  "\n"
  "  probe            identify the card, link and BARs, then run its family's probe (default)\n"
  "  cfg              hex-dump PCI config space\n"
  "  vbios -o FILE    save the VBIOS image (writes the ROM index register, as amdgpu does)\n"
  "  bench [N]        round-trip latency of config and register reads\n"
  "  dma-test [KB]    allocate a DMA buffer and print its bus addresses (no GPU activity)\n"
  "  atom-info        list the VBIOS command tables and ASIC_Init parameters (no writes)\n"
  "  post [--trace FILE] [--trace-ops] [--force]\n"
  "                   WRITES: run the VBIOS ASIC_Init so the card's memory works\n"
  "  vram-test [--enable-fb]\n"
  "                   WRITES VRAM: pattern test through BAR0 (needs a posted card)\n";

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmd_probe(rg_dev *d, const rg_ident *id) {
  printf("device:            %04x:%04x rev %02x  (%s)\n", d->vendor, d->device, d->revision,
         rg_pci_vendor_name(d->vendor));
  printf("  class:           %06x %s\n", d->class_code, rg_pci_class_name(d->class_code));
  printf("  board:           %04x:%04x  (%s)\n", d->subsys_vendor, d->subsys_device,
         rg_pci_vendor_name(d->subsys_vendor));
  printf("  family:          %s%s%s\n", id->gen_name, id->chip ? " / " : "", id->chip ? id->chip : "");
  printf("  support:         %s\n", id->support);

  int cg, cw, mg, mw;
  if (!rg_pci_link(d, &cg, &cw, &mg, &mw))
    printf("  PCIe link:       Gen%d x%d (card supports Gen%d x%d)\n", cg, cw, mg, mw);

  rg_bar_desc bars[RG_MAX_BARS];
  if (!rg_pci_bars(d, bars)) {
    for (int i = 0; i < RG_MAX_BARS; i++) {
      if (bars[i].upper_half || (!bars[i].raw && !bars[i].is_io)) continue;
      uint64_t sz = 0;
      if (!bars[i].is_io) rg_bar_map(d, i, &sz);
      printf("  BAR%d:            %s%s%s", i, bars[i].is_io ? "I/O" : "mem",
             bars[i].is_64 ? " 64-bit" : "", bars[i].prefetch ? " prefetchable" : "");
      if (sz) printf(", %llu %s", sz >= (1u << 20) ? (unsigned long long)(sz >> 20) : (unsigned long long)(sz >> 10),
                     sz >= (1u << 20) ? "MB" : "KB");
      else if (bars[i].is_io) printf(" (not usable: Apple Silicon has no I/O port space)");
      printf("\n");
    }
  }

  if (!id->backend) { printf("  backend:         none\n"); return 0; }
  printf("  backend:         %s\n", id->backend->name);
  return id->backend->probe(d, id);
}

static int cmd_cfg(rg_dev *d) {
  for (uint32_t off = 0; off < 0x100; off += 16) {
    printf("%03x:", off);
    for (uint32_t i = 0; i < 16; i += 4) {
      uint32_t v = 0;
      if (rg_cfg_read(d, off + i, 4, &v)) return -EIO;
      printf(" %02x %02x %02x %02x", v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, v >> 24);
    }
    printf("\n");
  }
  return 0;
}

static int cmd_vbios(rg_dev *d, const rg_ident *id, const char *path) {
  if (!id->backend || !id->backend->vbios_read) { fprintf(stderr, "rawgpu: no VBIOS reader for this family\n"); return -ENOTSUP; }
  uint8_t *buf = NULL;
  size_t len = 0;
  int r = id->backend->vbios_read(d, &buf, &len);
  if (r) return r;
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(buf, 1, len, f) != len) { perror(path); free(buf); if (f) fclose(f); return -EIO; }
  fclose(f);
  printf("wrote %zu bytes to %s\n", len, path);
  rg_atom_report(buf, len);
  free(buf);
  return 0;
}

static int cmd_bench(rg_dev *d, const rg_ident *id, int n) {
  uint32_t v;
  double t0 = now_us();
  for (int i = 0; i < n; i++) if (rg_cfg_read(d, 0, 4, &v)) return -EIO;
  double cfg = (now_us() - t0) / n;
  printf("config read:   %7.1f us/op over %d ops\n", cfg, n);
  if (id->backend) {
    if (rg_bar_map(d, id->backend->reg_bar, NULL)) return -EIO;
    t0 = now_us();
    for (int i = 0; i < n; i++) if (rg_rreg32(d, id->backend->reg_bar, id->backend->safe_reg, &v)) return -EIO;
    printf("register read: %7.1f us/op over %d ops (BAR%d +0x%x)\n", (now_us() - t0) / n, n,
           id->backend->reg_bar, id->backend->safe_reg);
  }
  printf("(each op is a socket round trip to the TinyGPU server plus one PCIe read)\n");
  return 0;
}

static int cmd_dma_test(rg_dev *d, size_t kb) {
  rg_dma b;
  int r = d->ops->dma_alloc(d, kb << 10, &b);
  if (r) return r;
  printf("DMA buffer: %zu KB mapped at %p, %d segment(s)\n", b.size >> 10, b.cpu, b.nseg);
  for (int i = 0; i < b.nseg; i++)
    printf("  bus 0x%010llx  len 0x%llx\n", (unsigned long long)b.seg[i].addr, (unsigned long long)b.seg[i].len);
  ((volatile uint32_t *)b.cpu)[0] = 0xC0FFEE;
  printf("CPU write/read of the buffer: %s\n", ((volatile uint32_t *)b.cpu)[0] == 0xC0FFEE ? "OK" : "FAILED");
  d->ops->dma_free(d, &b);
  return 0;
}

int main(int argc, char **argv) {
  const char *sock = NULL;
  int i = 1;
  if (i + 1 < argc && !strcmp(argv[i], "--sock")) { sock = argv[i + 1]; i += 2; }
  const char *cmd = i < argc ? argv[i++] : "probe";
  if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help") || !strcmp(cmd, "help")) { fputs(usage, stdout); return 0; }

  rg_dev d;
  int r = rg_open_tinygpu(&d, sock);
  if (r) return 1;
  if ((r = rg_pci_identify(&d))) { fprintf(stderr, "rawgpu: no device answered config reads (%d)\n", r); rg_close(&d); return 1; }
  rg_ident id;
  rg_identify_family(&d, &id);

  if (!strcmp(cmd, "probe")) r = cmd_probe(&d, &id);
  else if (!strcmp(cmd, "cfg")) r = cmd_cfg(&d);
  else if (!strcmp(cmd, "vbios")) {
    if (i + 1 < argc && !strcmp(argv[i], "-o")) r = cmd_vbios(&d, &id, argv[i + 1]);
    else { fputs(usage, stderr); r = -EINVAL; }
  } else if (!strcmp(cmd, "bench")) r = cmd_bench(&d, &id, i < argc ? atoi(argv[i]) : 2000);
  else if (!strcmp(cmd, "dma-test")) r = cmd_dma_test(&d, i < argc ? (size_t)atoi(argv[i]) : 1024);
  else if (!strcmp(cmd, "atom-info") || !strcmp(cmd, "post") || !strcmp(cmd, "vram-test")) {
    const rg_backend *b = id.backend;
    if (!b || !b->atom_info) { fprintf(stderr, "rawgpu: no ATOM support for this family\n"); r = -ENOTSUP; }
    else if (!strcmp(cmd, "atom-info")) r = b->atom_info(&d);
    else if (!strcmp(cmd, "post")) {
      rg_post_opts o = {0};
      for (; i < argc; i++) {
        if (!strcmp(argv[i], "--trace") && i + 1 < argc) o.trace_path = argv[++i];
        else if (!strcmp(argv[i], "--trace-ops")) o.trace_ops = true;
        else if (!strcmp(argv[i], "--force")) o.force = true;
        else { fputs(usage, stderr); r = -EINVAL; break; }
      }
      if (!r) r = b->post(&d, &o);
    } else {
      bool en = i < argc && !strcmp(argv[i], "--enable-fb");
      r = b->vram_test(&d, en);
    }
  } else { fputs(usage, stderr); r = -EINVAL; }

  rg_close(&d);
  return r ? 1 : 0;
}
