/* rawgpu: drive a PCIe GPU from userspace.
 *
 * Layering (each layer knows nothing about the ones above it):
 *   transport  - raw PCIe access: config space, BAR MMIO, DMA buffers (TinyGPU today)
 *   pci        - generic config-space decoding: IDs, capabilities, link, BARs
 *   family     - PCI ID -> chip family -> backend (and what supports it if we don't)
 *   backend    - per-family code: probe, VBIOS, and eventually bring-up
 */
#ifndef RAWGPU_RG_H
#define RAWGPU_RG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RG_MAX_BARS 6
#define RG_MAX_DMA_SEGS 32

typedef struct rg_dev rg_dev;

/* CPU memory the GPU can reach at the listed bus (IOVA) addresses. */
typedef struct {
  void *cpu;
  size_t size;
  int nseg;
  struct { uint64_t addr, len; } seg[RG_MAX_DMA_SEGS];
  int fd;
} rg_dma;

typedef struct {
  const char *name;
  int (*cfg_read)(rg_dev *, uint32_t off, uint32_t size, uint32_t *val);
  int (*cfg_write)(rg_dev *, uint32_t off, uint32_t size, uint32_t val);
  int (*bar_map)(rg_dev *, int bar, uint64_t *size);
  int (*mmio_read)(rg_dev *, int bar, uint64_t off, void *buf, size_t len);
  int (*mmio_write)(rg_dev *, int bar, uint64_t off, const void *buf, size_t len);
  int (*dma_alloc)(rg_dev *, size_t size, rg_dma *out);
  void (*dma_free)(rg_dev *, rg_dma *);
  void (*close)(rg_dev *);
} rg_transport_ops;

struct rg_dev {
  const rg_transport_ops *ops;
  void *priv;
  /* identity, filled by rg_pci_identify() */
  uint16_t vendor, device, subsys_vendor, subsys_device;
  uint8_t revision;
  uint32_t class_code; /* 24-bit base/sub/prog-if */
  uint64_t bar_size[RG_MAX_BARS]; /* filled by rg_bar_map(); 0 = not mapped */
};

/* ---- transport ---- */
int rg_open_tinygpu(rg_dev *d, const char *sock_path); /* NULL = default path */
void rg_close(rg_dev *d);

static inline int rg_cfg_read(rg_dev *d, uint32_t off, uint32_t size, uint32_t *v) { return d->ops->cfg_read(d, off, size, v); }
static inline int rg_cfg_write(rg_dev *d, uint32_t off, uint32_t size, uint32_t v) { return d->ops->cfg_write(d, off, size, v); }
int rg_bar_map(rg_dev *d, int bar, uint64_t *size);
static inline int rg_rreg32(rg_dev *d, int bar, uint32_t off, uint32_t *v) { return d->ops->mmio_read(d, bar, off, v, 4); }
static inline int rg_wreg32(rg_dev *d, int bar, uint32_t off, uint32_t v) { return d->ops->mmio_write(d, bar, off, &v, 4); }

/* ---- pci ---- */
enum { RG_PCI_CAP_PM = 0x01, RG_PCI_CAP_MSI = 0x05, RG_PCI_CAP_EXP = 0x10, RG_PCI_CAP_MSIX = 0x11 };

typedef struct {
  bool is_io, is_64, prefetch, upper_half;
  uint64_t raw; /* address as programmed by the host */
} rg_bar_desc;

int rg_pci_identify(rg_dev *d);
int rg_pci_find_cap(rg_dev *d, uint8_t id); /* config offset, or 0 if absent */
int rg_pci_link(rg_dev *d, int *cur_gen, int *cur_width, int *max_gen, int *max_width);
int rg_pci_bars(rg_dev *d, rg_bar_desc bars[RG_MAX_BARS]);
const char *rg_pci_class_name(uint32_t class_code);
const char *rg_pci_vendor_name(uint16_t vendor);

/* ---- family / backend ---- */
typedef enum {
  RG_GEN_UNKNOWN = 0,
  RG_GEN_AMD_SI,      /* GCN1: HD 7000 / R9 280 era */
  RG_GEN_AMD_CIK,     /* GCN2: R9 290 / R7 260 era */
  RG_GEN_AMD_VI,      /* GCN3/4: Tonga, Fiji, Polaris (RX 400/500) */
  RG_GEN_AMD_GFX9,    /* GCN5: Vega, Radeon VII, early CDNA */
  RG_GEN_AMD_NAVI1X,  /* RDNA1: RX 5000 */
  RG_GEN_AMD_NAVI2X,  /* RDNA2: RX 6000 */
  RG_GEN_AMD_IPDISC,  /* newer than amdgpu's static table: RDNA3/RDNA4/CDNA3+ */
  RG_GEN_NVIDIA,
} rg_gen;

#define RG_AMD_APU 1u
#define RG_AMD_MOBILITY 2u
#define RG_AMD_EXPERIMENTAL 4u

typedef struct { uint16_t device; const char *chip; uint32_t flags; } rg_amd_id;

typedef struct rg_backend rg_backend;

typedef struct {
  const char *trace_path; /* register trace file, or NULL */
  bool trace_ops;         /* include opcode-level ATOM debug output in the trace */
  bool force;             /* post even if the card already reports posted */
} rg_post_opts;

typedef struct {
  rg_gen gen;
  const char *chip;        /* e.g. "POLARIS10", or NULL */
  const char *gen_name;    /* e.g. "AMD GCN3/4 (VI)" */
  uint32_t flags;
  const rg_backend *backend; /* NULL if rawgpu has no backend for this family yet */
  const char *support;     /* one line: who can drive this card today */
} rg_ident;

struct rg_backend {
  const char *name;
  int reg_bar;                      /* BAR holding the register aperture */
  uint32_t safe_reg;                /* a side-effect-free register for latency tests */
  int (*probe)(rg_dev *, const rg_ident *);  /* read-only status report to stdout */
  int (*vbios_read)(rg_dev *, uint8_t **buf, size_t *len); /* caller frees *buf */
  int (*atom_info)(rg_dev *);                     /* parse the VBIOS tables; no hardware writes */
  int (*post)(rg_dev *, const rg_post_opts *);    /* run the VBIOS init: writes registers */
  int (*vram_test)(rg_dev *, bool enable_fb);     /* pattern test through the VRAM BAR */
};

void rg_identify_family(const rg_dev *d, rg_ident *out);

extern const rg_backend rg_backend_amd_vi;

/* ---- AMD ATOM BIOS (shared by GCN-era AMD backends) ---- */
int rg_atom_report(const uint8_t *bios, size_t len); /* parse and print the VBIOS header */

#endif
