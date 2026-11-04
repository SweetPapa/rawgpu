/* shim for amdgpu.h: the parts of struct amdgpu_device and the amdgpu API that
 * the vendored powerplay code uses. rawgpu's glue (src/pp/rg_pp.c) fills the
 * struct and implements the functions. Constants are copied from Linux
 * 05441d697f7f; each names its source. */
#ifndef RG_PP_AMDGPU_H
#define RG_PP_AMDGPU_H

#include "rg_pp_kcompat.h"
#include <drm/amd_asic_type.h>
#include "amd_shared.h"

struct atom_context;
struct amdgpu_bo;
struct amdgpu_ih_ring;

/* include/uapi/drm/amdgpu_drm.h */
#define AMDGPU_GEM_DOMAIN_CPU  0x1
#define AMDGPU_GEM_DOMAIN_GTT  0x2
#define AMDGPU_GEM_DOMAIN_VRAM 0x4
#define AMDGPU_VRAM_TYPE_GDDR5 5
#define AMDGPU_FAMILY_CI 120
#define AMDGPU_FAMILY_VI 130
#define AMDGPU_FAMILY_CZ 135
#define AMDGPU_FAMILY_AI 141
#define AMDGPU_FAMILY_RV 142
/* amdgpu/amdgpu.h, amdgpu/amdgpu_irq.h, pm/inc/amdgpu_dpm.h */
#define AMDGPU_SWCTF_EXTRA_DELAY 50
#define AMDGPU_IRQ_CLIENTID_LEGACY 0
#define AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW 4
#define BACO_SUPPORT (1 << 0)
/* display/amdgpu_dm: only for a size check in amd_acpi.h */
#define MAX_LUMINANCE_DATA_POINTS 99
struct amdgpu_dm_luminance_data { u8 luminance; u8 input_signal; } __packed;

struct amdgpu_iv_entry { /* amdgpu_irq.h */
  struct amdgpu_ih_ring *ih;
  unsigned client_id, src_id, ring_id, vmid, vmid_src;
  uint64_t timestamp;
  unsigned timestamp_src, pasid, node_id;
  unsigned src_data[AMDGPU_IRQ_SRC_DATA_MAX_SIZE_DW];
  const uint32_t *iv_entry;
};
#include "soc15_ih_clientid.h"
struct amdgpu_irq_src;
enum amdgpu_interrupt_state { AMDGPU_IRQ_STATE_DISABLE, AMDGPU_IRQ_STATE_ENABLE }; /* amdgpu_irq.h */
struct amdgpu_device;
struct amdgpu_irq_src_funcs {
  int (*set)(struct amdgpu_device *adev, struct amdgpu_irq_src *source, unsigned type, enum amdgpu_interrupt_state state);
  int (*process)(struct amdgpu_device *adev, struct amdgpu_irq_src *source, struct amdgpu_iv_entry *entry);
};
struct amdgpu_irq_src { unsigned num_types; void *data; const struct amdgpu_irq_src_funcs *funcs; };

struct amdgpu_device {
  struct device *dev;
  struct pci_dev *pdev;
  enum amd_asic_type asic_type;
  uint32_t pg_flags, cg_flags;
  bool in_suspend;
  uint8_t *bios;        /* the VBIOS image */
  uint32_t bios_size;
  uint32_t ref_clock;   /* reference clock, 10 kHz units (FirmwareInfo usReferenceClock) */
  struct { struct atom_context *atom_context; } mode_info;
  struct {
    uint32_t pcie_gen_mask, pcie_mlw_mask, pp_feature;
    bool ac_power, no_fan, dpm_enabled;
    struct {
      struct {
        int min_temp, max_temp, sw_ctf_threshold, max_edge_emergency_temp;
        int min_hotspot_temp, max_hotspot_crit_temp, max_hotspot_emergency_temp;
        int min_mem_temp, max_mem_crit_temp, max_mem_emergency_temp;
      } thermal;
    } dpm;
  } pm;
  struct mutex grbm_idx_mutex;
  struct { uint32_t vram_width, vram_type; } gmc;
  struct { void *pp_handle; const void *pp_funcs; } powerplay;
  struct { struct { unsigned number; } cu_info; struct { unsigned max_shader_engines; } config; } gfx;
  uint32_t *reg_offset[32][8]; /* SOC15 register bases (common_baco.c); unused on VI */
  void *rg; /* rawgpu's context (struct rg_pp) */
};

/* amdgpu.h: kicker (revised) Polaris parts, by PCI device and revision */
#define ASICID_IS_P20(did, rid)		(((did == 0x67DF) && \
					 ((rid == 0xE3) || \
					  (rid == 0xE4) || \
					  (rid == 0xE5) || \
					  (rid == 0xE7) || \
					  (rid == 0xEF))) || \
					 ((did == 0x6FDF) && \
					 ((rid == 0xE7) || \
					  (rid == 0xEF) || \
					  (rid == 0xFF))))

#define ASICID_IS_P30(did, rid)		((did == 0x67DF) && \
					((rid == 0xE1) || \
					 (rid == 0xF7)))

#define ASICID_IS_P21(did, rid)		(((did == 0x67EF) && \
					 ((rid == 0xE0) || \
					  (rid == 0xE5))) || \
					 ((did == 0x67FF) && \
					 ((rid == 0xCF) || \
					  (rid == 0xEF) || \
					  (rid == 0xFF))))

#define ASICID_IS_P31(did, rid)		((did == 0x67EF) && \
					((rid == 0xE2)))

#define ASICID_IS_P23(did, rid)		(((did == 0x6987) && \
					 ((rid == 0xC0) || \
					  (rid == 0xC1) || \
					  (rid == 0xC3) || \
					  (rid == 0xC7))) || \
					 ((did == 0x6981) && \
					 ((rid == 0x00) || \
					  (rid == 0x01) || \
					  (rid == 0x10))))

/* amdgpu.h register access, through rawgpu's register helpers */
uint32_t rg_pp_rreg(struct amdgpu_device *adev, uint32_t reg);
void rg_pp_wreg(struct amdgpu_device *adev, uint32_t reg, uint32_t v);
#define RREG32(reg) rg_pp_rreg(adev, (reg))
#define WREG32(reg, v) rg_pp_wreg(adev, (reg), (v))

/* Implemented in src/pp/rg_pp.c */
uint32_t amdgpu_asic_get_xclk(struct amdgpu_device *adev);
int amdgpu_bo_create_kernel(struct amdgpu_device *adev, unsigned long size, int align, u32 domain,
                            struct amdgpu_bo **bo_ptr, u64 *gpu_addr, void **cpu_addr);
void amdgpu_bo_free_kernel(struct amdgpu_bo **bo, u64 *gpu_addr, void **cpu_addr);
int amdgpu_irq_add_id(struct amdgpu_device *adev, unsigned client_id, unsigned src_id, struct amdgpu_irq_src *source);
void amdgpu_gfx_rlc_enter_safe_mode(struct amdgpu_device *adev, int xcc_id);
void amdgpu_gfx_rlc_exit_safe_mode(struct amdgpu_device *adev, int xcc_id);
int amdgpu_device_ip_set_powergating_state(void *dev, enum amd_ip_block_type block_type, enum amd_powergating_state state);
int amdgpu_device_ip_set_clockgating_state(void *dev, enum amd_ip_block_type block_type, enum amd_clockgating_state state);
struct amdgpu_ip_block *amdgpu_device_ip_get_ip_block(struct amdgpu_device *adev, enum amd_ip_block_type type);
int amdgpu_ucode_init_bo(struct amdgpu_device *adev);
int amdgpu_acpi_pcie_performance_request(struct amdgpu_device *adev, u8 perf_req, bool advertise);
bool amdgpu_acpi_is_pcie_performance_request_supported(struct amdgpu_device *adev);
extern int amdgpu_dpm; /* module parameter: 1 = DPM on */
#define amdgpu_sriov_is_pp_one_vf(adev) false
#define amdgpu_passthrough(adev) false
#define amdgpu_sriov_vf(adev) false

#endif
