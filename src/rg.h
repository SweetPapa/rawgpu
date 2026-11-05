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

typedef struct rg_dev rg_dev;

/* CPU memory the GPU can reach at the listed bus (IOVA) addresses. The
 * segments cover the buffer in order; together their lengths equal `size`. */
typedef struct { uint64_t addr, len; } rg_dma_seg;
typedef struct {
  void *cpu;
  size_t size;
  int nseg;
  rg_dma_seg *seg; /* malloc'd; freed by dma_free */
  int fd;
} rg_dma;

/* Bus address of byte `off` of the buffer, or 0 if off is out of range. */
uint64_t rg_dma_bus(const rg_dma *b, uint64_t off);

typedef struct {
  const char *name;
  int (*cfg_read)(rg_dev *, uint32_t off, uint32_t size, uint32_t *val);
  int (*cfg_write)(rg_dev *, uint32_t off, uint32_t size, uint32_t val);
  int (*bar_map)(rg_dev *, int bar, uint64_t *size);
  int (*mmio_read)(rg_dev *, int bar, uint64_t off, void *buf, size_t len);
  int (*mmio_write)(rg_dev *, int bar, uint64_t off, const void *buf, size_t len);
  int (*dma_alloc)(rg_dev *, size_t size, rg_dma *out);
  void (*dma_free)(rg_dev *, rg_dma *);
  int (*reset)(rg_dev *);  /* PCIe function-level reset, or hot reset if FLR is unsupported */
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
/* The server's socket path: sock_path, else $RAWGPU_TINYGPU_SOCK, else $TMPDIR/rawgpu-tinygpu.sock. */
const char *rg_tinygpu_sock_path(const char *sock_path, char *buf, size_t n);
/* Connect to a TinyGPU server that is already running, without starting one. If another client has the
 * card, this waits until it leaves. Returns 0 (and the server's pid); -ENOENT if there is no socket;
 * -ECONNREFUSED if connecting was refused (a stale socket, or a busy server's backlog is full); -EIO if
 * the server closed the connection. */
int rg_attach_tinygpu(rg_dev *d, const char *sock_path, int *server_pid);

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
int rg_pci_find_ext_cap(rg_dev *d, uint16_t id); /* extended config offset, or 0 */

typedef struct {
  uint32_t status, devsta;           /* PCI status (0x06), PCIe device status */
  bool has_aer;
  uint32_t uncor, cor, hdr_log[4];   /* AER uncorrectable/correctable status, header of the first error */
} rg_pci_err;
int rg_pci_errors(rg_dev *d, rg_pci_err *e);
int rg_pci_clear_errors(rg_dev *d);
void rg_pci_print_errors(const rg_pci_err *e);
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
  rg_gen gen;
  const char *chip;        /* e.g. "POLARIS10", or NULL */
  const char *gen_name;    /* e.g. "AMD GCN3/4 (VI)" */
  uint32_t flags;
  const rg_backend *backend; /* NULL if rawgpu has no backend for this family yet */
  const char *support;     /* one line: who can drive this card today */
} rg_ident;

/* A backend's own subcommand. `run` gets the arguments after the command name
 * and returns 0 or a negative errno; -EINVAL prints the usage line. */
typedef struct {
  const char *name;
  const char *args; /* argument synopsis, or "" */
  const char *help; /* one line; starts with "WRITES" if it changes hardware state */
  int (*run)(rg_dev *, const rg_ident *, int argc, char **argv);
} rg_cmd;

struct rg_backend {
  const char *name;
  int reg_bar;                      /* BAR holding the register aperture */
  uint32_t safe_reg;                /* byte offset of a side-effect-free register for latency tests */
  int (*probe)(rg_dev *, const rg_ident *);  /* read-only status report to stdout */
  int (*vbios_read)(rg_dev *, uint8_t **buf, size_t *len); /* caller frees *buf */
  /* Before the card is let go (rawgpu eject): wait for the engines to go idle and stop any that a
   * process left running, so nothing on the card reads or writes host memory afterwards. Prints what
   * it did. */
  int (*quiesce)(rg_dev *, const rg_ident *);
  const rg_cmd *cmds;               /* terminated by an entry with name NULL */
};

void rg_identify_family(const rg_dev *d, rg_ident *out);

extern const rg_backend rg_backend_amd_vi;

/* ---- AMD ATOM BIOS (shared by GCN-era AMD backends) ---- */
int rg_atom_report(const uint8_t *bios, size_t len); /* parse and print the VBIOS header */

#endif
