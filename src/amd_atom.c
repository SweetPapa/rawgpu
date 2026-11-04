/* AMD ATOM BIOS header parsing. Offsets and string-finding rules follow Linux
 * amdgpu (atom.c, atom.h, atombios.h, atomfirmware.h; MIT-licensed). This is the
 * read-only half; the interpreter that executes ATOM command tables (needed to
 * post the card) is the next milestone. */
#include "rg.h"

#include <stdio.h>
#include <string.h>

#define ATOM_ROM_TABLE_PTR 0x48 /* u16: offset of ATOM_ROM_HEADER */
#define OFF_NUM_STRINGS 0x2f
#define OFF_STRING_START 0x6e
#define OFF_PART_NUMBER 0x80
#define OFF_DATE 0x50

static uint16_t rd16(const uint8_t *b, size_t len, size_t off) {
  return off + 1 < len ? (uint16_t)(b[off] | b[off + 1] << 8) : 0;
}

static void printable(char *dst, size_t cap, const uint8_t *src, size_t avail) {
  size_t n = 0;
  while (n + 1 < cap && n < avail && src[n] >= ' ' && src[n] <= '~') { dst[n] = (char)src[n]; n++; }
  while (n && dst[n - 1] == ' ') n--;
  dst[n] = 0;
}

static const uint8_t *find(const uint8_t *b, size_t len, const char *s, size_t from, size_t to) {
  size_t sl = strlen(s);
  if (to > len) to = len;
  for (size_t i = from; i + sl <= to; i++)
    if (!memcmp(b + i, s, sl)) return b + i;
  return NULL;
}

int rg_atom_report(const uint8_t *b, size_t len) {
  if (len < 0x100 || b[0] != 0x55 || b[1] != 0xAA) { printf("  VBIOS:           no option-ROM signature\n"); return -1; }
  size_t img = (size_t)b[2] * 512;
  uint8_t sum = 0;
  for (size_t i = 0; i < img && i < len; i++) sum += b[i];

  uint16_t hdr = rd16(b, len, ATOM_ROM_TABLE_PTR);
  bool atom = hdr && hdr + 0x24 <= len && !memcmp(b + hdr + 4, "ATOM", 4);
  printf("  VBIOS:           %zu KB image, checksum %s, %s\n", img / 1024, sum == 0 ? "OK" : "BAD",
         atom ? "ATOM BIOS" : "not an ATOM BIOS");
  if (!atom) return -1;

  char s[80];
  /* Part number: first of the ATOM strings, else the fixed offset. */
  const uint8_t *pn = b[OFF_NUM_STRINGS] ? b + rd16(b, len, OFF_STRING_START) : b + OFF_PART_NUMBER;
  if (pn >= b + len) pn = NULL;
  if (pn) { printable(s, sizeof s, pn, (size_t)(b + len - pn)); printf("  part number:     %s\n", s); }

  /* Board name: after the ATOM strings and a CR/LF. */
  if (b[OFF_NUM_STRINGS]) {
    const uint8_t *c = b + rd16(b, len, OFF_STRING_START), *end = b + len;
    for (int i = 0; i < b[OFF_NUM_STRINGS] && c < end; i++) { while (c < end && *c) c++; c++; }
    c += 2;
    if (c < end) { printable(s, sizeof s, c, (size_t)(end - c)); printf("  name:            %s\n", s); }
  }

  const uint8_t *ver = find(b, len, "ATOMBIOSBK-AMD", 3, hdr > 3 ? hdr : 1024);
  if (ver && ver + 18 + 32 <= b + len) { printable(s, sizeof s, ver + 18, 32); printf("  version:         %s\n", s); }

  const uint8_t *dt = b + OFF_DATE;
  printf("  build date:      20%c%c/%c%c/%c%c\n", dt[6], dt[7], dt[0], dt[1], dt[3], dt[4]);

  printf("  ATOM header:     @0x%04x rev %u.%u, subsystem %04x:%04x\n", hdr, b[hdr + 2], b[hdr + 3],
         rd16(b, len, hdr + 24), rd16(b, len, hdr + 26));
  printf("  command tables:  @0x%04x   data tables: @0x%04x\n", rd16(b, len, hdr + 30), rd16(b, len, hdr + 32));
  return 0;
}
