#include "amd_fw.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* set by the Makefile: this checkout's local/ folder; empty in release builds */
#ifndef RAWGPU_DEFAULT_LOCAL
#define RAWGPU_DEFAULT_LOCAL ""
#endif

static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

uint32_t amd_fw_hdr32(const amd_fw *fw, size_t off) { return off + 4 <= fw->size ? le32(fw->data + off) : 0; }

const char *rg_data_dir(void) {
  static char dir[1024];
  if (dir[0]) return dir;
  const char *env = getenv("RAWGPU_HOME"), *home = getenv("HOME");
  struct stat st;
  if (env && *env) snprintf(dir, sizeof dir, "%s", env);
  else if (RAWGPU_DEFAULT_LOCAL[0] && !stat(RAWGPU_DEFAULT_LOCAL, &st) && S_ISDIR(st.st_mode))
    snprintf(dir, sizeof dir, "%s", RAWGPU_DEFAULT_LOCAL);
  else snprintf(dir, sizeof dir, "%s/Library/Application Support/rawgpu", home ? home : "");
  return dir;
}

static FILE *open_fw(const char *dir, const char *name, char *path, size_t n) {
  const char *fmt[] = {"%s/amdgpu/%s.bin", "%s/%s.bin"};
  for (int i = 0; i < 2; i++) {
    snprintf(path, n, fmt[i], dir, name);
    FILE *f = fopen(path, "rb");
    if (f) return f;
  }
  return NULL;
}

int amd_fw_load(amd_fw *fw, const char *dir, const char *name) {
  memset(fw, 0, sizeof *fw);
  snprintf(fw->name, sizeof fw->name, "%s", name);
  char def[1100];
  if (!dir) dir = getenv("RAWGPU_FIRMWARE");
  if (!dir) {
    snprintf(def, sizeof def, "%s/firmware", rg_data_dir());
    dir = def;
  }

  char path[1024];
  FILE *f = open_fw(dir, name, path, sizeof path);
  if (!f) {
    fprintf(stderr, "rawgpu: firmware %s.bin not found under %s. Run fetch-firmware.sh (tools/ in a\n"
                    "        checkout, share/rawgpu/ in a release), or point RAWGPU_FIRMWARE at a linux-firmware\n"
                    "        checkout. `rawgpu doctor` checks the setup.\n", name, dir);
    return -ENOENT;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 32 || sz > (16l << 20)) { fclose(f); fprintf(stderr, "rawgpu: %s: bad size %ld\n", path, sz); return -EINVAL; }
  fw->data = malloc((size_t)sz);
  if (!fw->data) { fclose(f); return -ENOMEM; }
  size_t got = fread(fw->data, 1, (size_t)sz, f);
  fclose(f);
  fw->size = (size_t)sz;
  if (got != fw->size) { amd_fw_free(fw); return -EIO; }

  /* amdgpu_ucode_validate(): the header's size must match the file */
  const uint8_t *h = fw->data;
  uint32_t size_bytes = le32(h + 0), hdr_bytes = le32(h + 4);
  fw->hdr_major = le16(h + 8); fw->hdr_minor = le16(h + 10);
  fw->ip_major = le16(h + 12); fw->ip_minor = le16(h + 14);
  fw->ucode_version = le32(h + 16);
  uint32_t ucode_bytes = le32(h + 20), ucode_off = le32(h + 24);
  if (size_bytes != fw->size || hdr_bytes < 32 || hdr_bytes > fw->size || ucode_bytes % 4 ||
      ucode_off % 4 || (uint64_t)ucode_off + ucode_bytes > fw->size) {
    fprintf(stderr, "rawgpu: %s: header does not match the file (size %u vs %zu, ucode %u@%u)\n", path,
            size_bytes, fw->size, ucode_bytes, ucode_off);
    amd_fw_free(fw);
    return -EINVAL;
  }
  fw->ucode = (const uint32_t *)(fw->data + ucode_off); /* host is little-endian, as rg_kcompat.h checks */
  fw->ucode_dw = ucode_bytes / 4;
  return 0;
}

void amd_fw_free(amd_fw *fw) {
  free(fw->data);
  fw->data = NULL;
  fw->ucode = NULL;
}
