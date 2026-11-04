/* Minimal ELF64 reader for AMDGPU code objects: finds a kernel's code and its
 * kernel descriptor. Field layouts are from the ELF spec and LLVM's
 * AMDGPUUsage ("Kernel Descriptor"). Everything read is bounds-checked: a
 * code object is input data. */
#include "amd_kernel.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EM_AMDGPU 224
#define ELFOSABI_AMDGPU_HSA 64
#define SHT_SYMTAB 2
#define SHT_RELA 4
#define SHT_REL 9
#define STT_OBJECT 1
#define STT_FUNC 2
#define KD_SIZE 64

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t le64(const uint8_t *p) { return le32(p) | (uint64_t)le32(p + 4) << 32; }

typedef struct { uint32_t name, type, link, info; uint64_t off, size; } shdr;

static int bad(const char *path, const char *why) {
  fprintf(stderr, "rawgpu: %s: %s\n", path, why);
  return -EINVAL;
}

int amd_kernel_load(amd_kernel *k, const char *path, const char *name) {
  memset(k, 0, sizeof *k);
  snprintf(k->name, sizeof k->name, "%s", name);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "rawgpu: kernel %s not found (make kernels builds it)\n", path); return -ENOENT; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 64 || sz > (64l << 20)) { fclose(f); return bad(path, "not a code object (size)"); }
  uint8_t *d = malloc((size_t)sz);
  if (!d) { fclose(f); return -ENOMEM; }
  size_t n = fread(d, 1, (size_t)sz, f);
  fclose(f);
  int r = -EINVAL;
  shdr *sh = NULL;
  if (n != (size_t)sz) { r = -EIO; goto out; }

  if (memcmp(d, "\x7f" "ELF", 4) || d[4] != 2 || d[5] != 1) { r = bad(path, "not a little-endian ELF64 file"); goto out; }
  if (d[7] != ELFOSABI_AMDGPU_HSA || le16(d + 18) != EM_AMDGPU) { r = bad(path, "not an AMDGPU HSA code object"); goto out; }
  k->mach = le32(d + 48) & 0xff;
  uint64_t shoff = le64(d + 40);
  uint16_t shentsize = le16(d + 58), shnum = le16(d + 60);
  if (shentsize != 64 || !shnum || shoff > (uint64_t)sz || (uint64_t)shnum * 64 > (uint64_t)sz - shoff) {
    r = bad(path, "bad section table"); goto out;
  }
  if (!(sh = calloc(shnum, sizeof *sh))) { r = -ENOMEM; goto out; }
  for (uint16_t i = 0; i < shnum; i++) {
    const uint8_t *p = d + shoff + 64ull * i;
    sh[i] = (shdr){le32(p), le32(p + 4), le32(p + 40), le32(p + 44), le64(p + 24), le64(p + 32)};
    if (sh[i].type != 8 /* NOBITS */ && (sh[i].off > (uint64_t)sz || sh[i].size > (uint64_t)sz - sh[i].off)) {
      r = bad(path, "section outside the file"); goto out;
    }
  }

  /* the kernel's symbol (FUNC) and its descriptor (<name>.kd, OBJECT) */
  char kd_name[80];
  snprintf(kd_name, sizeof kd_name, "%s.kd", name);
  int64_t code_sec = -1, kd_sec = -1;
  uint64_t code_off = 0, kd_off = 0;
  for (uint16_t i = 0; i < shnum; i++) {
    if (sh[i].type != SHT_SYMTAB || sh[i].link >= shnum) continue;
    const shdr *st = &sh[sh[i].link];
    for (uint64_t o = 0; o + 24 <= sh[i].size; o += 24) {
      const uint8_t *s = d + sh[i].off + o;
      uint32_t nm = le32(s);
      uint8_t type = s[4] & 0xf;
      uint16_t shndx = le16(s + 6);
      if (nm >= st->size || shndx == 0 || shndx >= shnum) continue;
      const char *sn = (const char *)d + st->off + nm;
      size_t maxlen = st->size - nm;
      if (type == STT_FUNC && strnlen(sn, maxlen) == strlen(name) && !strncmp(sn, name, maxlen)) {
        code_sec = shndx; code_off = le64(s + 8);
      } else if (type == STT_OBJECT && strnlen(sn, maxlen) == strlen(kd_name) && !strncmp(sn, kd_name, maxlen)) {
        kd_sec = shndx; kd_off = le64(s + 8);
      }
    }
  }
  if (code_sec < 0 || kd_sec < 0) {
    fprintf(stderr, "rawgpu: %s: no kernel named %s (need both %s and %s.kd)\n", path, name, name, name);
    goto out;
  }
  for (uint16_t i = 0; i < shnum; i++)
    if ((sh[i].type == SHT_RELA || sh[i].type == SHT_REL) && sh[i].info == (uint32_t)code_sec) {
      r = bad(path, "the code has relocations; link it (ld.lld) first or avoid globals"); goto out;
    }
  const shdr *cs = &sh[code_sec], *ks = &sh[kd_sec];
  if (code_off >= cs->size || kd_off > ks->size || ks->size - kd_off < KD_SIZE) { r = bad(path, "symbol outside its section"); goto out; }

  const uint8_t *kd = d + ks->off + kd_off;
  k->group_segment_size = le32(kd + 0);
  k->private_segment_size = le32(kd + 4);
  k->kernarg_size = le32(kd + 8);
  k->rsrc3 = le32(kd + 44);
  k->rsrc1 = le32(kd + 48);
  k->rsrc2 = le32(kd + 52);
  k->code_properties = le16(kd + 56);
  k->entry = code_off;
  k->code_size = cs->size;
  if (!(k->code = malloc(cs->size))) { r = -ENOMEM; goto out; }
  memcpy(k->code, d + cs->off, cs->size);
  r = 0;
out:
  free(sh);
  free(d);
  return r;
}

void amd_kernel_free(amd_kernel *k) {
  free(k->code);
  k->code = NULL;
}
