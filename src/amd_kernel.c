/* AMDGPU code object loader and linker (amd_kernel.h). */
#include "amd_kernel.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EM_AMDGPU 224
#define ELFOSABI_AMDGPU_HSA 64
#define ET_REL 1
#define ET_DYN 3
#define SHT_PROGBITS 1
#define SHT_SYMTAB 2
#define SHT_RELA 4
#define SHT_NOTE 7
#define SHT_NOBITS 8
#define SHT_REL 9
#define SHT_DYNSYM 11
#define SHF_ALLOC 2u
#define SHN_UNDEF 0
#define SHN_LORESERVE 0xff00
#define STT_OBJECT 1
#define STT_FUNC 2
#define KD_SIZE 64
#define NT_AMDGPU_METADATA 32

/* R_AMDGPU_* */
enum { R_NONE = 0, R_ABS32_LO = 1, R_ABS32_HI = 2, R_ABS64 = 3, R_REL32 = 4, R_REL64 = 5, R_ABS32 = 6, R_GOTPCREL = 7,
       R_GOTPCREL32_LO = 8, R_GOTPCREL32_HI = 9, R_REL32_LO = 10, R_REL32_HI = 11, R_RELATIVE64 = 13, R_REL16 = 14 };

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t le32(const uint8_t *p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t le64(const uint8_t *p) { return le32(p) | (uint64_t)le32(p + 4) << 32; }

typedef struct {
  uint32_t name, type, link, info;
  uint64_t flags, addr, off, size, align;
  bool placed;
  uint64_t img; /* image offset, if placed */
} shdr;

typedef struct {
  const uint8_t *d;
  size_t sz;
  const char *label;
  uint16_t etype, shnum;
  shdr *sh;
} elf;

static int bad(const char *label, const char *why) {
  fprintf(stderr, "rawgpu: %s: %s\n", label, why);
  return -EINVAL;
}

/* a NUL-terminated string at off in the string table section strtab, or NULL */
static const char *elf_str(const elf *e, uint32_t strtab, uint32_t off) {
  if (strtab >= e->shnum) return NULL;
  const shdr *s = &e->sh[strtab];
  if (off >= s->size) return NULL;
  const char *p = (const char *)e->d + s->off + off;
  return memchr(p, 0, s->size - off) ? p : NULL;
}

/* ---- msgpack, just enough for the metadata note ---- */

typedef struct { const uint8_t *p, *end; bool err; } mp;

static bool mp_need(mp *m, size_t n) {
  if (m->err || (size_t)(m->end - m->p) < n) { m->err = true; return false; }
  return true;
}
static uint64_t mp_be(mp *m, unsigned n) {
  if (!mp_need(m, n)) return 0;
  uint64_t v = 0;
  for (unsigned i = 0; i < n; i++) v = v << 8 | m->p[i];
  m->p += n;
  return v;
}
/* container header: returns element count of a map (pairs) or array, -1 if not that type */
static int64_t mp_container(mp *m, bool map) {
  if (!mp_need(m, 1)) return -1;
  uint8_t t = *m->p;
  if (map ? (t & 0xf0) == 0x80 : (t & 0xf0) == 0x90) { m->p++; return t & 0x0f; }
  if (t == (map ? 0xde : 0xdc)) { m->p++; return (int64_t)mp_be(m, 2); }
  if (t == (map ? 0xdf : 0xdd)) { m->p++; return (int64_t)mp_be(m, 4); }
  return -1;
}
static bool mp_str(mp *m, const char **s, uint32_t *len) {
  if (!mp_need(m, 1)) return false;
  uint8_t t = *m->p;
  uint64_t n;
  if ((t & 0xe0) == 0xa0) { m->p++; n = t & 0x1f; }
  else if (t == 0xd9) { m->p++; n = mp_be(m, 1); }
  else if (t == 0xda) { m->p++; n = mp_be(m, 2); }
  else if (t == 0xdb) { m->p++; n = mp_be(m, 4); }
  else return false;
  if (!mp_need(m, n)) return false;
  *s = (const char *)m->p;
  *len = (uint32_t)n;
  m->p += n;
  return true;
}
static bool mp_uint(mp *m, uint64_t *v) {
  if (!mp_need(m, 1)) return false;
  uint8_t t = *m->p;
  if (t <= 0x7f) { m->p++; *v = t; return true; }
  if (t >= 0xcc && t <= 0xcf) { m->p++; *v = mp_be(m, 1u << (t - 0xcc)); return !m->err; }
  if (t >= 0xd0 && t <= 0xd3) { m->p++; *v = mp_be(m, 1u << (t - 0xd0)); return !m->err; } /* signed, assumed >= 0 */
  return false;
}
static bool mp_bool(mp *m, bool *v) {
  if (!mp_need(m, 1)) return false;
  if (*m->p == 0xc2 || *m->p == 0xc3) { *v = *m->p++ == 0xc3; return true; }
  return false;
}
static void mp_skip(mp *m, int depth) {
  if (depth > 32) { m->err = true; return; }
  if (!mp_need(m, 1)) return;
  uint8_t t = *m->p;
  const char *s;
  uint32_t len;
  uint64_t v;
  int64_t n;
  if (mp_str(m, &s, &len) || mp_uint(m, &v) || m->err) return;
  if ((n = mp_container(m, true)) >= 0) { for (int64_t i = 0; i < 2 * n && !m->err; i++) mp_skip(m, depth + 1); return; }
  if ((n = mp_container(m, false)) >= 0) { for (int64_t i = 0; i < n && !m->err; i++) mp_skip(m, depth + 1); return; }
  m->p++;
  switch (t) {
  case 0xc0: case 0xc2: case 0xc3: return;                          /* nil, false, true */
  case 0xc4: m->p += mp_be(m, 1); break;                             /* bin */
  case 0xc5: m->p += mp_be(m, 2); break;
  case 0xc6: m->p += mp_be(m, 4); break;
  case 0xca: m->p += 4; break;                                       /* float */
  case 0xcb: m->p += 8; break;
  case 0xd4: m->p += 2; break; case 0xd5: m->p += 3; break;          /* fixext */
  case 0xd6: m->p += 5; break; case 0xd7: m->p += 9; break; case 0xd8: m->p += 17; break;
  case 0xc7: m->p += mp_be(m, 1) + 1; break;                         /* ext */
  case 0xc8: m->p += mp_be(m, 2) + 1; break;
  case 0xc9: m->p += mp_be(m, 4) + 1; break;
  default: if (t < 0xe0) m->err = true; return;                      /* negative fixint is fine */
  }
  if (m->p > m->end) m->err = true;
}
static bool mp_key(const char *s, uint32_t len, const char *want) { return strlen(want) == len && !memcmp(s, want, len); }
static void mp_copy(char *dst, size_t cap, const char *s, uint32_t len) {
  size_t n = len < cap - 1 ? len : cap - 1;
  memcpy(dst, s, n);
  dst[n] = 0;
}

static const struct { const char *name; amd_arg_kind kind; } kinds[] = {
  {"by_value", AMD_ARG_BY_VALUE}, {"global_buffer", AMD_ARG_GLOBAL_BUFFER},
  {"dynamic_shared_pointer", AMD_ARG_DYNAMIC_SHARED_POINTER}, {"image", AMD_ARG_IMAGE}, {"sampler", AMD_ARG_SAMPLER},
  {"pipe", AMD_ARG_PIPE}, {"queue", AMD_ARG_QUEUE},
  {"hidden_global_offset_x", AMD_ARG_HIDDEN_GLOBAL_OFFSET_X}, {"hidden_global_offset_y", AMD_ARG_HIDDEN_GLOBAL_OFFSET_Y},
  {"hidden_global_offset_z", AMD_ARG_HIDDEN_GLOBAL_OFFSET_Z}, {"hidden_block_count_x", AMD_ARG_HIDDEN_BLOCK_COUNT_X},
  {"hidden_block_count_y", AMD_ARG_HIDDEN_BLOCK_COUNT_Y}, {"hidden_block_count_z", AMD_ARG_HIDDEN_BLOCK_COUNT_Z},
  {"hidden_group_size_x", AMD_ARG_HIDDEN_GROUP_SIZE_X}, {"hidden_group_size_y", AMD_ARG_HIDDEN_GROUP_SIZE_Y},
  {"hidden_group_size_z", AMD_ARG_HIDDEN_GROUP_SIZE_Z}, {"hidden_remainder_x", AMD_ARG_HIDDEN_REMAINDER_X},
  {"hidden_remainder_y", AMD_ARG_HIDDEN_REMAINDER_Y}, {"hidden_remainder_z", AMD_ARG_HIDDEN_REMAINDER_Z},
  {"hidden_grid_dims", AMD_ARG_HIDDEN_GRID_DIMS}, {"hidden_dynamic_lds_size", AMD_ARG_HIDDEN_DYNAMIC_LDS_SIZE},
  {"hidden_private_base", AMD_ARG_HIDDEN_PRIVATE_BASE}, {"hidden_shared_base", AMD_ARG_HIDDEN_SHARED_BASE},
  {"hidden_queue_ptr", AMD_ARG_HIDDEN_QUEUE_PTR}, {"hidden_none", AMD_ARG_HIDDEN_NONE},
};

const char *amd_arg_kind_name(amd_arg_kind k) {
  for (size_t i = 0; i < sizeof kinds / sizeof kinds[0]; i++)
    if (kinds[i].kind == k) return kinds[i].name;
  return k == AMD_ARG_HIDDEN_OTHER ? "hidden (other)" : "unknown";
}

static int parse_arg(mp *m, amd_arg *a) {
  int64_t n = mp_container(m, true);
  if (n < 0) return -1;
  for (int64_t i = 0; i < n && !m->err; i++) {
    const char *k, *s;
    uint32_t kl, sl;
    uint64_t v;
    bool b;
    if (!mp_str(m, &k, &kl)) return -1;
    if (mp_key(k, kl, ".offset") && mp_uint(m, &v)) a->offset = (uint32_t)v;
    else if (mp_key(k, kl, ".size") && mp_uint(m, &v)) a->size = (uint32_t)v;
    else if (mp_key(k, kl, ".name") && mp_str(m, &s, &sl)) mp_copy(a->name, sizeof a->name, s, sl);
    else if (mp_key(k, kl, ".type_name") && mp_str(m, &s, &sl)) mp_copy(a->type_name, sizeof a->type_name, s, sl);
    else if (mp_key(k, kl, ".is_const") && mp_bool(m, &b)) a->is_const = b;
    else if (mp_key(k, kl, ".address_space") && mp_str(m, &s, &sl))
      a->address_space = mp_key(s, sl, "generic") ? 'G' : sl ? s[0] : 0;
    else if (mp_key(k, kl, ".value_kind") && mp_str(m, &s, &sl)) {
      a->kind = sl > 7 && !memcmp(s, "hidden_", 7) ? AMD_ARG_HIDDEN_OTHER : AMD_ARG_UNKNOWN;
      for (size_t j = 0; j < sizeof kinds / sizeof kinds[0]; j++)
        if (mp_key(s, sl, kinds[j].name)) a->kind = kinds[j].kind;
    } else mp_skip(m, 0);
  }
  return m->err ? -1 : 0;
}

/* one entry of amdhsa.kernels: fills the kernel whose descriptor symbol matches */
static int parse_kernel(mp *m, amd_program *p) {
  int64_t n = mp_container(m, true);
  if (n < 0) return -1;
  amd_kernel tmp = {0};
  char sym[80] = "", name[64] = "";
  for (int64_t i = 0; i < n && !m->err; i++) {
    const char *k, *s;
    uint32_t kl, sl;
    uint64_t v;
    bool b;
    if (!mp_str(m, &k, &kl)) return -1;
    if (mp_key(k, kl, ".symbol") && mp_str(m, &s, &sl)) mp_copy(sym, sizeof sym, s, sl);
    else if (mp_key(k, kl, ".name") && mp_str(m, &s, &sl)) mp_copy(name, sizeof name, s, sl);
    else if (mp_key(k, kl, ".kernarg_segment_align") && mp_uint(m, &v)) tmp.kernarg_align = (uint32_t)v;
    else if (mp_key(k, kl, ".max_flat_workgroup_size") && mp_uint(m, &v)) tmp.max_wg = (uint32_t)v;
    else if (mp_key(k, kl, ".sgpr_count") && mp_uint(m, &v)) tmp.sgprs = (uint32_t)v;
    else if (mp_key(k, kl, ".vgpr_count") && mp_uint(m, &v)) tmp.vgprs = (uint32_t)v;
    else if (mp_key(k, kl, ".sgpr_spill_count") && mp_uint(m, &v)) tmp.sgpr_spills = (uint32_t)v;
    else if (mp_key(k, kl, ".vgpr_spill_count") && mp_uint(m, &v)) tmp.vgpr_spills = (uint32_t)v;
    else if (mp_key(k, kl, ".uses_dynamic_stack") && mp_bool(m, &b)) tmp.dynamic_stack = b;
    else if (mp_key(k, kl, ".reqd_workgroup_size")) {
      int64_t c = mp_container(m, false);
      for (int64_t j = 0; j < c && !m->err; j++) {
        if (!mp_uint(m, &v)) return -1;
        if (j < 3) tmp.reqd_wg[j] = (uint32_t)v;
      }
    } else if (mp_key(k, kl, ".args")) {
      int64_t c = mp_container(m, false);
      if (c < 0 || c > 4096) return -1;
      free(tmp.args);
      tmp.args = calloc(c ? (size_t)c : 1, sizeof *tmp.args);
      if (!tmp.args) return -1;
      tmp.nargs = (uint32_t)c;
      for (int64_t j = 0; j < c; j++)
        if (parse_arg(m, &tmp.args[j])) { free(tmp.args); return -1; }
    } else mp_skip(m, 0);
  }
  if (m->err) { free(tmp.args); return -1; }
  for (uint32_t i = 0; i < p->nkernels; i++) {
    amd_kernel *kk = &p->kernels[i];
    char kd[80];
    snprintf(kd, sizeof kd, "%s.kd", kk->name);
    if ((sym[0] && !strcmp(sym, kd)) || (!sym[0] && !strcmp(name, kk->name))) {
      kk->has_metadata = true;
      kk->dynamic_stack = tmp.dynamic_stack;
      kk->kernarg_align = tmp.kernarg_align;
      kk->max_wg = tmp.max_wg;
      memcpy(kk->reqd_wg, tmp.reqd_wg, sizeof tmp.reqd_wg);
      kk->sgprs = tmp.sgprs; kk->vgprs = tmp.vgprs;
      kk->sgpr_spills = tmp.sgpr_spills; kk->vgpr_spills = tmp.vgpr_spills;
      free(kk->args);
      kk->args = tmp.args;
      kk->nargs = tmp.nargs;
      return 0;
    }
  }
  free(tmp.args);
  return 0;
}

static int parse_metadata(amd_program *p, const uint8_t *data, size_t len) {
  mp m = {data, data + len, false};
  int64_t n = mp_container(&m, true);
  if (n < 0) return -1;
  for (int64_t i = 0; i < n && !m.err; i++) {
    const char *k;
    uint32_t kl;
    if (!mp_str(&m, &k, &kl)) return -1;
    if (mp_key(k, kl, "amdhsa.kernels")) {
      int64_t c = mp_container(&m, false);
      if (c < 0) return -1;
      for (int64_t j = 0; j < c; j++)
        if (parse_kernel(&m, p)) return -1;
    } else mp_skip(&m, 0);
  }
  return m.err ? -1 : 0;
}

/* ---- the ELF file ---- */

static int read_notes(elf *e, amd_program *p) {
  for (uint16_t i = 0; i < e->shnum; i++) {
    if (e->sh[i].type != SHT_NOTE) continue;
    const uint8_t *q = e->d + e->sh[i].off, *end = q + e->sh[i].size;
    while (end - q >= 12) {
      uint32_t namesz = le32(q), descsz = le32(q + 4), type = le32(q + 8);
      size_t nalign = (namesz + 3) & ~3u, dalign = (descsz + 3) & ~(size_t)3;
      if ((size_t)(end - q - 12) < nalign || (size_t)(end - q - 12) - nalign < descsz) break;
      const uint8_t *name = q + 12, *desc = name + nalign;
      if (type == NT_AMDGPU_METADATA && namesz == 7 && !memcmp(name, "AMDGPU", 7) && parse_metadata(p, desc, descsz))
        return bad(e->label, "unreadable AMDGPU metadata note");
      if ((size_t)(end - desc) < dalign) break;
      q = desc + dalign;
    }
  }
  return 0;
}

typedef struct { uint32_t sec; uint64_t value; uint8_t type; const char *name; } esym;

/* the symbol table section: SYMTAB for relocatable objects, DYNSYM otherwise */
static int read_syms(const elf *e, uint32_t symsec, esym **out, uint32_t *count) {
  *out = NULL;
  *count = 0;
  if (symsec >= e->shnum) return 0;
  const shdr *s = &e->sh[symsec];
  uint64_t n = s->size / 24;
  if (!n) return 0;
  esym *v = calloc(n, sizeof *v);
  if (!v) return -ENOMEM;
  for (uint64_t i = 0; i < n; i++) {
    const uint8_t *q = e->d + s->off + 24 * i;
    v[i] = (esym){le16(q + 6), le64(q + 8), (uint8_t)(q[4] & 0xf), elf_str(e, s->link, le32(q))};
  }
  *out = v;
  *count = (uint32_t)n;
  return 0;
}

/* image offset of a symbol, or -1 if it is undefined or not in an allocated section */
static int64_t sym_img(const elf *e, const esym *s) {
  if (s->sec == SHN_UNDEF || s->sec >= SHN_LORESERVE || s->sec >= e->shnum || !e->sh[s->sec].placed) return -1;
  return e->etype == ET_DYN ? (int64_t)s->value : (int64_t)(e->sh[s->sec].img + s->value);
}

static int add_reloc(amd_program *p, uint32_t *cap, amd_reloc r) {
  if (p->nrelocs == *cap) {
    uint32_t nc = *cap ? *cap * 2 : 64;
    amd_reloc *nr = realloc(p->relocs, nc * sizeof *nr);
    if (!nr) return -ENOMEM;
    p->relocs = nr;
    *cap = nc;
  }
  p->relocs[p->nrelocs++] = r;
  return 0;
}

static unsigned reloc_width(uint32_t type) {
  switch (type) {
  case R_ABS64: case R_REL64: case R_RELATIVE64: return 8;
  case R_REL16: return 2;
  case R_NONE: return 0;
  default: return 4;
  }
}

int amd_program_load_mem(amd_program *p, const void *data, size_t sz, const char *label) {
  memset(p, 0, sizeof *p);
  snprintf(p->path, sizeof p->path, "%s", label);
  elf e = {.d = data, .sz = sz, .label = label};
  esym *syms = NULL;
  uint32_t nsyms = 0, relcap = 0;
  uint64_t *got = NULL; /* symbol index per GOT slot */
  uint32_t ngot = 0;
  char missing[512] = "";
  int r = -EINVAL;
  const uint8_t *d = data;

  if (sz < 64 || memcmp(d, "\x7f" "ELF", 4) || d[4] != 2 || d[5] != 1) return bad(label, "not a little-endian ELF64 file");
  if (d[7] != ELFOSABI_AMDGPU_HSA || le16(d + 18) != EM_AMDGPU) return bad(label, "not an AMDGPU HSA code object");
  e.etype = le16(d + 16);
  if (e.etype != ET_REL && e.etype != ET_DYN) return bad(label, "neither a relocatable nor a shared object");
  p->mach = le32(d + 48) & 0xff;
  uint64_t shoff = le64(d + 40);
  uint16_t shentsize = le16(d + 58);
  e.shnum = le16(d + 60);
  if (shentsize != 64 || !e.shnum || shoff > sz || (uint64_t)e.shnum * 64 > sz - shoff) return bad(label, "bad section table");
  if (!(e.sh = calloc(e.shnum, sizeof *e.sh))) return -ENOMEM;
  for (uint16_t i = 0; i < e.shnum; i++) {
    const uint8_t *q = d + shoff + 64ull * i;
    e.sh[i] = (shdr){le32(q), le32(q + 4), le32(q + 40), le32(q + 44), le64(q + 8), le64(q + 16), le64(q + 24), le64(q + 32),
                     le64(q + 48), false, 0};
    if (e.sh[i].type != SHT_NOBITS && (e.sh[i].off > sz || e.sh[i].size > sz - e.sh[i].off)) {
      r = bad(label, "section outside the file");
      goto out;
    }
  }

  /* layout: linked objects keep their addresses; relocatable ones are packed */
  uint64_t end = 0;
  for (uint16_t i = 0; i < e.shnum; i++) {
    shdr *s = &e.sh[i];
    if (!(s->flags & SHF_ALLOC) || (s->type != SHT_PROGBITS && s->type != SHT_NOBITS)) continue;
    uint64_t al = s->align > 256 ? s->align : 256;
    if (al & (al - 1) || al > (1u << 20)) { r = bad(label, "odd section alignment"); goto out; }
    s->img = e.etype == ET_DYN ? s->addr : (end + al - 1) & ~(al - 1);
    if (s->img > (256ull << 20) || s->size > (256ull << 20)) { r = bad(label, "image too large"); goto out; }
    s->placed = true;
    if (s->img + s->size > end) end = s->img + s->size;
  }

  /* symbols, and the relocations that will need a GOT */
  uint32_t symsec = UINT32_MAX;
  for (uint16_t i = 0; i < e.shnum; i++)
    if (e.sh[i].type == (e.etype == ET_DYN ? SHT_DYNSYM : SHT_SYMTAB)) symsec = i;
  if (e.etype == ET_DYN) /* kernel lookup wants the full table if there is one */
    for (uint16_t i = 0; i < e.shnum; i++)
      if (e.sh[i].type == SHT_SYMTAB) symsec = i;
  if ((r = read_syms(&e, symsec, &syms, &nsyms))) goto out;
  r = -EINVAL;

  for (uint16_t i = 0; i < e.shnum; i++) {
    const shdr *rs = &e.sh[i];
    if (rs->type == SHT_REL) { r = bad(label, "REL relocations (AMDGPU uses RELA)"); goto out; }
    if (rs->type != SHT_RELA || (e.etype == ET_REL && (rs->info >= e.shnum || !e.sh[rs->info].placed))) continue;
    for (uint64_t o = 0; o + 24 <= rs->size; o += 24) {
      uint32_t type = (uint32_t)le64(d + rs->off + o + 8), si = (uint32_t)(le64(d + rs->off + o + 8) >> 32);
      if (type != R_GOTPCREL && type != R_GOTPCREL32_LO && type != R_GOTPCREL32_HI) continue;
      bool have = false;
      for (uint32_t g = 0; g < ngot; g++) have |= got[g] == si;
      if (have) continue;
      uint64_t *ng = realloc(got, (ngot + 1) * sizeof *ng);
      if (!ng) { r = -ENOMEM; goto out; }
      got = ng;
      got[ngot++] = si;
    }
  }
  uint64_t got_off = (end + 255) & ~255ull;
  p->image_size = ngot ? got_off + 8ull * ngot : end;
  if (!p->image_size) { r = bad(label, "no code"); goto out; }
  if (!(p->image = calloc(1, p->image_size))) { r = -ENOMEM; goto out; }
  for (uint16_t i = 0; i < e.shnum; i++)
    if (e.sh[i].placed && e.sh[i].type == SHT_PROGBITS) memcpy(p->image + e.sh[i].img, d + e.sh[i].off, e.sh[i].size);

  /* relocations, resolved to image offsets now and to addresses by amd_program_relocate() */
  for (uint16_t i = 0; i < e.shnum; i++) {
    const shdr *rs = &e.sh[i];
    if (rs->type != SHT_RELA) continue;
    uint64_t base = 0;
    if (e.etype == ET_REL) {
      if (rs->info >= e.shnum || !e.sh[rs->info].placed) continue; /* debug info and the like */
      base = e.sh[rs->info].img;
    }
    const shdr *sy = rs->link < e.shnum ? &e.sh[rs->link] : NULL;
    esym *rsyms = syms;
    uint32_t nrsyms = nsyms;
    esym *own = NULL;
    if (sy && rs->link != symsec) { /* .rela.dyn uses .dynsym */
      if ((r = read_syms(&e, rs->link, &own, &nrsyms))) goto out;
      r = -EINVAL;
      rsyms = own;
    }
    for (uint64_t o = 0; o + 24 <= rs->size; o += 24) {
      const uint8_t *q = d + rs->off + o;
      uint64_t off = base + le64(q), info = le64(q + 8);
      int64_t addend = (int64_t)le64(q + 16);
      uint32_t type = (uint32_t)info, si = (uint32_t)(info >> 32);
      if (type == R_NONE) continue;
      unsigned w = reloc_width(type);
      if (type > R_REL16 || type == 12 || off > p->image_size || p->image_size - off < w) {
        free(own);
        r = bad(label, "unsupported or out-of-range relocation");
        goto out;
      }
      amd_reloc rel = {off, type, 0, addend};
      if (type == R_RELATIVE64) rel.type = R_ABS64; /* B + A */
      else {
        if (si >= nrsyms) { free(own); r = bad(label, "relocation symbol out of range"); goto out; }
        int64_t img = sym_img(&e, &rsyms[si]);
        if (img < 0 && si) {
          const char *nm = rsyms[si].name ? rsyms[si].name : "?";
          if (!strstr(missing, nm) && strlen(missing) + strlen(nm) + 3 < sizeof missing) {
            if (missing[0]) strcat(missing, ", ");
            strcat(missing, nm);
          }
          continue;
        }
        rel.sym = img < 0 ? 0 : (uint64_t)img;
        if (type == R_GOTPCREL || type == R_GOTPCREL32_LO || type == R_GOTPCREL32_HI) {
          uint32_t g = 0;
          while (got[g] != si) g++;
          rel.sym = got_off + 8ull * g;
          rel.type = type == R_GOTPCREL ? R_REL32 : type == R_GOTPCREL32_LO ? R_REL32_LO : R_REL32_HI;
        }
      }
      if ((r = add_reloc(p, &relcap, rel))) { free(own); goto out; }
      r = -EINVAL;
    }
    free(own);
  }
  for (uint32_t g = 0; g < ngot; g++) {
    int64_t img = got[g] < nsyms ? sym_img(&e, &syms[got[g]]) : -1;
    if (img < 0) continue; /* already listed as missing */
    if ((r = add_reloc(p, &relcap, (amd_reloc){got_off + 8ull * g, R_ABS64, (uint64_t)img, 0}))) goto out;
    r = -EINVAL;
  }
  if (missing[0]) {
    fprintf(stderr, "rawgpu: %s: the code calls or uses %s, which it does not define. Built-in functions come from "
                    "rawgpu's device header (device/rawgpu_cl.h); a code object for rawgpu must not need a device library.\n",
            label, missing);
    goto out;
  }

  /* kernels: a <name>.kd descriptor and a function <name> */
  uint32_t nk = 0;
  for (uint32_t i = 0; i < nsyms; i++) {
    const char *nm = syms[i].name;
    size_t l = nm ? strlen(nm) : 0;
    if (syms[i].type == STT_OBJECT && l > 3 && !strcmp(nm + l - 3, ".kd") && sym_img(&e, &syms[i]) >= 0) nk++;
  }
  if (!nk) { r = bad(label, "no kernels in the code object"); goto out; }
  if (!(p->kernels = calloc(nk, sizeof *p->kernels))) { r = -ENOMEM; goto out; }
  for (uint32_t i = 0; i < nsyms; i++) {
    const char *nm = syms[i].name;
    size_t l = nm ? strlen(nm) : 0;
    int64_t kd = syms[i].type == STT_OBJECT && l > 3 && !strcmp(nm + l - 3, ".kd") ? sym_img(&e, &syms[i]) : -1;
    if (kd < 0) continue;
    if (l - 3 >= sizeof p->kernels[0].name) { r = bad(label, "kernel name too long"); goto out; }
    if ((uint64_t)kd > p->image_size || p->image_size - (uint64_t)kd < KD_SIZE) { r = bad(label, "descriptor outside the image"); goto out; }
    amd_kernel *k = &p->kernels[p->nkernels];
    memcpy(k->name, nm, l - 3);
    k->name[l - 3] = 0;
    int64_t entry = -1;
    for (uint32_t j = 0; j < nsyms; j++)
      if (syms[j].type == STT_FUNC && syms[j].name && !strcmp(syms[j].name, k->name)) entry = sym_img(&e, &syms[j]);
    if (entry < 0) { fprintf(stderr, "rawgpu: %s: descriptor %s.kd has no code\n", label, k->name); goto out; }
    const uint8_t *q = p->image + kd;
    k->prog = p;
    k->entry = (uint64_t)entry;
    k->group_segment_size = le32(q + 0);
    k->private_segment_size = le32(q + 4);
    k->kernarg_size = le32(q + 8);
    k->rsrc3 = le32(q + 44);
    k->rsrc1 = le32(q + 48);
    k->rsrc2 = le32(q + 52);
    k->code_properties = le16(q + 56);
    p->nkernels++;
  }
  if ((r = read_notes(&e, p))) goto out;
  r = 0;
out:
  free(syms);
  free(got);
  free(e.sh);
  if (r) amd_program_free(p);
  return r;
}

int amd_program_load(amd_program *p, const char *path) {
  memset(p, 0, sizeof *p);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "rawgpu: code object %s not found (make kernels builds rawgpu's own)\n", path); return -ENOENT; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 64 || sz > (64l << 20)) { fclose(f); return bad(path, "not a code object (size)"); }
  uint8_t *d = malloc((size_t)sz);
  if (!d) { fclose(f); return -ENOMEM; }
  size_t n = fread(d, 1, (size_t)sz, f);
  fclose(f);
  int r = n == (size_t)sz ? amd_program_load_mem(p, d, (size_t)sz, path) : -EIO;
  free(d);
  return r;
}

void amd_program_relocate(amd_program *p, uint64_t base) {
  for (uint32_t i = 0; i < p->nrelocs; i++) {
    const amd_reloc *rl = &p->relocs[i];
    uint64_t S = base + rl->sym, P = base + rl->off, v;
    uint8_t *q = p->image + rl->off;
    switch (rl->type) {
    case R_ABS32_LO: case R_ABS32: v = (S + rl->addend) & 0xffffffffu; break;
    case R_ABS32_HI: v = (S + rl->addend) >> 32; break;
    case R_ABS64: v = S + rl->addend; break;
    case R_REL32: case R_REL32_LO: v = (S + rl->addend - P) & 0xffffffffu; break;
    case R_REL32_HI: v = (S + rl->addend - P) >> 32; break;
    case R_REL64: v = S + rl->addend - P; break;
    case R_REL16: v = (uint64_t)((int64_t)(S + rl->addend - P) - 4) / 4; break;
    default: continue;
    }
    for (unsigned b = 0; b < reloc_width(rl->type); b++) q[b] = (uint8_t)(v >> (8 * b));
  }
}

const amd_kernel *amd_program_kernel(const amd_program *p, const char *name) {
  for (uint32_t i = 0; i < p->nkernels; i++)
    if (!strcmp(p->kernels[i].name, name)) return &p->kernels[i];
  fprintf(stderr, "rawgpu: %s: no kernel named %s; it has", p->path, name);
  for (uint32_t i = 0; i < p->nkernels; i++) fprintf(stderr, "%s %s", i ? "," : "", p->kernels[i].name);
  fprintf(stderr, "\n");
  return NULL;
}

void amd_program_free(amd_program *p) {
  for (uint32_t i = 0; i < p->nkernels; i++) free(p->kernels[i].args);
  free(p->kernels);
  free(p->relocs);
  free(p->image);
  memset(p, 0, sizeof *p);
}

void amd_program_print(FILE *f, const amd_program *p) {
  fprintf(f, "%s: gfx%s code object, %zu-byte image, %u relocation%s, %u kernel%s\n", p->path,
          p->mach == EF_AMDGPU_MACH_AMDGCN_GFX803 ? "803" : "?", p->image_size, p->nrelocs, p->nrelocs == 1 ? "" : "s",
          p->nkernels, p->nkernels == 1 ? "" : "s");
  static const char *props[] = {"private segment buffer", "dispatch ptr", "queue ptr", "kernarg ptr", "dispatch id",
                                "flat scratch init", "private segment size"};
  for (uint32_t i = 0; i < p->nkernels; i++) {
    const amd_kernel *k = &p->kernels[i];
    fprintf(f, "\nkernel %s (entry +0x%llx)\n", k->name, (unsigned long long)k->entry);
    fprintf(f, "  LDS %u B, private %u B/work-item, kernargs %u B", k->group_segment_size, k->private_segment_size,
            k->kernarg_size);
    if (k->has_metadata)
      fprintf(f, ", %u SGPRs, %u VGPRs%s, max work-group %u", k->sgprs, k->vgprs,
              k->sgpr_spills || k->vgpr_spills ? " (spills)" : "", k->max_wg);
    if (k->reqd_wg[0]) fprintf(f, ", requires %ux%ux%u", k->reqd_wg[0], k->reqd_wg[1], k->reqd_wg[2]);
    fprintf(f, "\n  user SGPRs:");
    for (unsigned b = 0; b < 7; b++)
      if (k->code_properties & (1u << b)) fprintf(f, " [%s]", props[b]);
    fprintf(f, "%s\n", k->dynamic_stack ? "; dynamic stack" : "");
    for (uint32_t a = 0; a < k->nargs; a++) {
      const amd_arg *g = &k->args[a];
      fprintf(f, "  +%-4u %2u B  %-24s %s%s%s%s\n", g->offset, g->size, amd_arg_kind_name(g->kind),
              g->type_name[0] ? g->type_name : "", g->name[0] ? " " : "", g->name,
              g->address_space == 'l' ? " (__local)" : g->address_space == 'c' ? " (__constant)" : "");
    }
  }
}
