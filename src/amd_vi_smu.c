/* SMU7 on Polaris: start the SMC on its driver firmware, then have the SMC load
 * the engines' microcode (SDMA, RLC, CP) from VRAM.
 *
 * Polaris ignores microcode the host writes into an engine's ucode RAM
 * (docs/LOG.md, 2026-11-04), so this is the only way to get the engines
 * running. It follows Linux amdgpu at 05441d697f7f:
 *
 *   polaris10_start_smu()                       pm/powerplay/smumgr/polaris10_smumgr.c
 *     polaris10_start_smu_in_protection_mode()
 *     smu7_upload_smu_firmware_image()          pm/powerplay/smumgr/smu7_smumgr.c
 *   smu7_request_smu_load_fw()                  (same file)
 *     amdgpu_ucode_init_bo()                    amdgpu/amdgpu_ucode.c: where the images go
 *     amdgpu_cgs_get_firmware_info()            amdgpu/amdgpu_cgs.c: what each TOC entry says
 *
 * Where rawgpu differs:
 * - The images, the TOC and the SMU's work buffer all live in VRAM. That is
 *   amdgpu's debug_use_vram_fw_buf placement (bare-metal amdgpu uses GTT). The
 *   VBIOS leaves the memory controller in physical addressing mode, so the SMU
 *   reaches them without a GART, and nothing here needs host DMA.
 * - MEC2's image is left out. VI's TOC points both MEC jump tables at MEC1
 *   (fw_type_convert()), so the SMU never reads it.
 * - polaris10_avfs_event_mgr() is skipped. AVFS tunes voltages for DPM levels,
 *   and rawgpu runs at the VBIOS boot clocks with DPM off.
 * - Only protected mode (SMU_FIRMWARE.SMU_MODE = 1) is implemented, because
 *   that is how this card boots. The non-protected path would be untested.
 *
 * The SMC firmware keeps running until the card loses power, and the SMU may
 * keep using the VRAM above, so it stays reserved (below VI_VRAM_FREE).
 */
#include "amd_vi.h"

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* smu7_ppsmc.h */
#define PPSMC_MSG_Test             0x100
#define PPSMC_MSG_DRV_DRAM_ADDR_HI 0x250
#define PPSMC_MSG_DRV_DRAM_ADDR_LO 0x251
#define PPSMC_MSG_SMU_DRAM_ADDR_HI 0x252
#define PPSMC_MSG_SMU_DRAM_ADDR_LO 0x253
#define PPSMC_MSG_LoadUcodes       0x254
#define PPSMC_MSG_API_GetSclkFrequency 0x200
#define PPSMC_MSG_API_GetMclkFrequency 0x201
#define PPSMC_Result_OK            0x01

/* smu_ucode_xfer_vi.h */
enum {
  UCODE_ID_SDMA0 = 1, UCODE_ID_SDMA1 = 2, UCODE_ID_CP_CE = 3, UCODE_ID_CP_PFP = 4, UCODE_ID_CP_ME = 5,
  UCODE_ID_CP_MEC = 6, UCODE_ID_CP_MEC_JT1 = 7, UCODE_ID_CP_MEC_JT2 = 8, UCODE_ID_RLC_G = 10,
};
#define SMU_DRAMData_TOC_VERSION 1
#define SMU_MAX_ENTRIES 12

typedef struct { /* struct SMU_Entry, little-endian layout */
  uint16_t id, version;
  uint32_t image_addr_high, image_addr_low;
  uint32_t meta_data_addr_high, meta_data_addr_low;
  uint32_t data_size_byte;
  uint16_t flags, num_register_entries;
} smu_entry;

typedef struct { /* struct SMU_DRAMData_TOC */
  uint32_t structure_version, num_entries;
  smu_entry entry[SMU_MAX_ENTRIES];
} smu_toc;
_Static_assert(sizeof(smu_entry) == 28 && sizeof(smu_toc) == 344, "SMU_DRAMData_TOC layout");

/* smu74.h. The SMC's address space is little-endian dwords, as uploaded. */
#define SMU7_FIRMWARE_HEADER_LOCATION 0x20000
#define SMU7_SMC_SIZE                 0x20000 /* SMC RAM is 0x20000-0x3ffff */
#define SMC_RAM_END                   0x40000 /* the limit amdgpu gives smu7_read_smc_sram_dword() */
#define SMC_RAM_FW_START              0x20100 /* smu7_is_smc_ram_running(): PC at or above this */

#pragma pack(push, 1)
typedef struct { /* struct SMU74_Firmware_Header, up to the fields rawgpu reads */
  uint32_t Digest[5], Version, HeaderSize, Flags, EntryPoint, CodeSize, ImageSize, Rtos, SoftRegisters, DpmTable;
} smu74_firmware_header;

typedef struct { /* struct SMU74_SoftRegisters */
  uint32_t RefClockFrequency, PmTimerPeriod, FeatureEnables, PreVBlankGap, VBlankTimeout, TrainTimeGap;
  uint32_t MvddSwitchTime, LongestAcpiTrainTime, AcpiDelay, G5TrainTime, DelayMpllPwron, VoltageChangeTimeout;
  uint32_t HandshakeDisables;
  uint8_t DisplayPhyConfig[8];
  uint32_t AverageGraphicsActivity, AverageMemoryActivity, AverageGioActivity;
  uint8_t DpmEnabledLevels[8];
  uint32_t DRAM_LOG_ADDR_H, DRAM_LOG_ADDR_L, DRAM_LOG_PHY_ADDR_H, DRAM_LOG_PHY_ADDR_L, DRAM_LOG_BUFF_SIZE;
  uint32_t UlvEnterCount, UlvTime, UcodeLoadStatus, AllowMvddSwitch;
  uint8_t Activity_Weight, Reserved8[3];
} smu74_soft_registers;
#pragma pack(pop)
_Static_assert(offsetof(smu74_firmware_header, SoftRegisters) == 48, "SMU74_Firmware_Header layout");
_Static_assert(offsetof(smu74_soft_registers, UcodeLoadStatus) == 108, "SMU74_SoftRegisters layout");

#define SMU_BUF_BYTES (200u * 4096) /* smu7_init(): smu_buffer */
#define PAGE 4096u
#define ALIGN_PAGE(x) (((uint64_t)(x) + PAGE - 1) & ~(uint64_t)(PAGE - 1))
#define TIMEOUT_US 2000000u /* amdgpu polls ~1 s (usec_timeout); every read here is a socket round trip */

/* The images smu7_request_smu_load_fw() names, in amdgpu_ucode_init_bo()'s
 * layout order (enum AMDGPU_UCODE_ID). The CP images are the "_2" builds,
 * which gfx_v8_0_init_microcode() prefers on Polaris. */
enum { IMG_SDMA0, IMG_SDMA1, IMG_CE, IMG_PFP, IMG_ME, IMG_MEC, IMG_RLC, NIMG };
static const struct { const char *label, *file; uint16_t id; } img_desc[NIMG] = {
  [IMG_SDMA0] = {"SDMA0", "polaris10_sdma", UCODE_ID_SDMA0},
  [IMG_SDMA1] = {"SDMA1", "polaris10_sdma1", UCODE_ID_SDMA1},
  [IMG_CE] = {"CE", "polaris10_ce_2", UCODE_ID_CP_CE},
  [IMG_PFP] = {"PFP", "polaris10_pfp_2", UCODE_ID_CP_PFP},
  [IMG_ME] = {"ME", "polaris10_me_2", UCODE_ID_CP_ME},
  [IMG_MEC] = {"MEC", "polaris10_mec_2", UCODE_ID_CP_MEC},
  [IMG_RLC] = {"RLC", "polaris10_rlc", UCODE_ID_RLC_G},
};

/* smu7_request_smu_load_fw(), chips after Topaz */
#define FW_TO_LOAD (VI_UCODE_RLC_G | VI_UCODE_SDMA0 | VI_UCODE_SDMA1 | VI_UCODE_CP_CE | VI_UCODE_CP_ME | \
                    VI_UCODE_CP_PFP | VI_UCODE_CP_MEC)

/* ---- SMC access ---- */

#define SMC_WRITE_FIELD(v, reg, field, val) vi_wsmc(v, ix##reg, REG_SET_FIELD(vi_rsmc(v, ix##reg), reg, field, val))

/* Polls register `idx` until ((x & mask) == want) == eq, as phm_wait_on_register()
 * and phm_wait_for_register_unequal() do. */
static int poll(vi_dev *v, uint32_t idx, uint32_t mask, uint32_t want, bool eq, uint32_t *last) {
  double end = vi_now() + TIMEOUT_US / 1e6;
  uint32_t x;
  do {
    x = vi_rreg(v, idx);
    if (v->err) return v->err;
    if (((x & mask) == want) == eq) { if (last) *last = x; return 0; }
  } while (vi_now() < end);
  if (last) *last = x;
  return -ETIMEDOUT;
}

/* phm_wait_on_indirect_register(): select the SMC address once, then poll the data port. */
static int smc_poll(vi_dev *v, uint32_t addr, uint32_t mask, uint32_t want, bool eq, uint32_t *last) {
  vi_wreg(v, mmSMC_IND_INDEX_11, addr);
  return poll(v, mmSMC_IND_DATA_11, mask, want, eq, last);
}

/* smu7_send_msg_to_smc_with_parameter(), or smu7_send_msg_to_smc() when
 * has_arg is false. Returns the SMC's response: 1 = OK, 0xFE = not supported,
 * 0 = no answer within the timeout. */
static uint32_t smc_msg_x(vi_dev *v, uint16_t msg, bool has_arg, uint32_t arg) {
  poll(v, mmSMC_RESP_0, SMC_RESP_0__SMC_RESP_MASK, 0, false, NULL); /* previous message answered */
  if (has_arg) vi_wreg(v, mmSMC_MSG_ARG_0, arg);
  vi_wreg(v, mmSMC_RESP_0, 0);
  vi_wreg(v, mmSMC_MESSAGE_0, msg);
  uint32_t x = 0;
  poll(v, mmSMC_RESP_0, SMC_RESP_0__SMC_RESP_MASK, 0, false, &x);
  return v->err ? 0 : REG_GET_FIELD(x, SMC_RESP_0, SMC_RESP);
}

static uint32_t smc_msg(vi_dev *v, uint16_t msg, uint32_t arg) { return smc_msg_x(v, msg, true, arg); }

static uint32_t smc_msg_checked(vi_dev *v, const char *name, uint16_t msg, uint32_t arg) {
  uint32_t resp = smc_msg(v, msg, arg);
  if (resp != PPSMC_Result_OK && !v->err)
    fprintf(stderr, "rawgpu: SMC message %s (0x%x, arg 0x%08x) answered 0x%x%s\n", name, msg, arg, resp,
            resp == 0 ? " (no answer)" : resp == 0xFE ? " (not supported)" : "");
  return resp;
}

/* smu7_is_smc_ram_running() */
bool vi_smc_running(vi_dev *v) {
  uint32_t ck = vi_rsmc(v, ixSMC_SYSCON_CLOCK_CNTL_0), pc = vi_rsmc(v, ixSMC_PC_C);
  return !v->err && !REG_GET_FIELD(ck, SMC_SYSCON_CLOCK_CNTL_0, ck_disable) && pc >= SMC_RAM_FW_START;
}

static bool smc_firmware_up(vi_dev *v) {
  return vi_smc_running(v) && REG_GET_FIELD(vi_rsmc(v, ixFIRMWARE_FLAGS), FIRMWARE_FLAGS, INTERRUPTS_ENABLED);
}

/* The SMC address of SMU74_SoftRegisters, from the firmware header in SMC RAM. */
static uint32_t soft_regs_start(vi_dev *v) {
  return vi_rsmc(v, SMU7_FIRMWARE_HEADER_LOCATION + offsetof(smu74_firmware_header, SoftRegisters));
}

static bool soft_regs_ok(uint32_t s) {
  return s >= SMU7_FIRMWARE_HEADER_LOCATION && s % 4 == 0 && s + sizeof(smu74_soft_registers) <= SMC_RAM_END;
}

int vi_smu_ucode_status(vi_dev *v, uint32_t *status) {
  if (!smc_firmware_up(v)) return v->err ? v->err : -ENODEV;
  uint32_t s = soft_regs_start(v);
  if (!soft_regs_ok(s)) return v->err ? v->err : -EIO;
  *status = vi_rsmc(v, s + offsetof(smu74_soft_registers, UcodeLoadStatus));
  return v->err;
}

/* smu7_read_sensor() AMDGPU_PP_SENSOR_GFX_SCLK / _MCLK: the SMC answers in
 * SMC_MSG_ARG_0, in units of 10 kHz. */
int vi_smu_clocks(vi_dev *v, uint32_t *sclk_mhz, uint32_t *mclk_mhz) {
  if (!smc_firmware_up(v)) return v->err ? v->err : -ENODEV;
  if (smc_msg_x(v, PPSMC_MSG_API_GetSclkFrequency, false, 0) != PPSMC_Result_OK) return v->err ? v->err : -EIO;
  *sclk_mhz = vi_rreg(v, mmSMC_MSG_ARG_0) / 100;
  if (smc_msg_x(v, PPSMC_MSG_API_GetMclkFrequency, false, 0) != PPSMC_Result_OK) return v->err ? v->err : -EIO;
  *mclk_mhz = vi_rreg(v, mmSMC_MSG_ARG_0) / 100;
  return v->err;
}

/* smu7_thermal_get_temperature(), in degrees C */
int vi_temp_c(vi_dev *v) {
  uint32_t t = REG_GET_FIELD(vi_rsmc(v, ixCG_MULT_THERMAL_STATUS), CG_MULT_THERMAL_STATUS, CTF_TEMP);
  return t & 0x200 ? 255 : (int)(t & 0x1ff); /* upstream's "below the lowest usable value" check */
}

static void print_ucode_bits(uint32_t s) {
  static const char *names[16] = {
    [1] = "SDMA0", [2] = "SDMA1", [3] = "CE", [4] = "PFP", [5] = "ME", [6] = "MEC",
    [7] = "MEC_JT1", [8] = "MEC_JT2", [9] = "GMCON", [10] = "RLC", [11] = "IH", [12] = "VBIOS",
  };
  bool any = false;
  for (int i = 0; i < 16; i++)
    if (s & (1u << i)) { printf("%s%s", any ? " " : "", names[i] ? names[i] : "?"); any = true; }
  if (!any) printf("none");
}

void vi_smu_report(vi_dev *v) {
  uint32_t ck = vi_rsmc(v, ixSMC_SYSCON_CLOCK_CNTL_0), pc = vi_rsmc(v, ixSMC_PC_C);
  uint32_t flags = vi_rsmc(v, ixFIRMWARE_FLAGS), smufw = vi_rsmc(v, ixSMU_FIRMWARE);
  bool ram = vi_smc_running(v), up = ram && REG_GET_FIELD(flags, FIRMWARE_FLAGS, INTERRUPTS_ENABLED);
  printf("SMC                    clock %s, PC 0x%08x (%s), FIRMWARE_FLAGS 0x%08x (%s), SMU_FIRMWARE 0x%08x (%s, %s key)\n",
         REG_GET_FIELD(ck, SMC_SYSCON_CLOCK_CNTL_0, ck_disable) ? "off" : "on", pc,
         ram ? "running from RAM: driver firmware" : "boot ROM", flags, up ? "firmware up" : "no firmware", smufw,
         REG_GET_FIELD(smufw, SMU_FIRMWARE, SMU_MODE) ? "protected mode" : "non-protected mode",
         REG_GET_FIELD(smufw, SMU_FIRMWARE, SMU_SEL) ? "hard" : "soft");
  if (up) {
    uint32_t s = soft_regs_start(v);
    if (soft_regs_ok(s)) {
      uint32_t st = vi_rsmc(v, s + offsetof(smu74_soft_registers, UcodeLoadStatus));
      printf("SMU microcode loaded   UcodeLoadStatus 0x%08x: ", st);
      print_ucode_bits(st);
      printf("   (soft registers at SMC 0x%05x)\n", s);
    } else {
      printf("SMU microcode loaded   unknown: firmware header gives soft registers at 0x%08x\n", s);
    }
  }
  uint32_t sclk, mclk;
  if (up && !vi_smu_clocks(v, &sclk, &mclk))
    printf("clocks (SMC report)    engine %u MHz, memory %u MHz: not reliable before DPM is set up (bench-compute measures)\n",
           sclk, mclk);
  printf("temperature            %d C\n", vi_temp_c(v));
}

/* ---- starting the SMC firmware ---- */

/* smu7_upload_smc_firmware_data(): the image goes to SMC 0x20000 through the
 * auto-incrementing data port. Nothing may read SMC_IND_DATA_11 meanwhile. */
static int smc_upload(vi_dev *v, const amd_fw *fw) {
  if ((uint64_t)fw->ucode_dw * 4 > SMU7_SMC_SIZE) {
    fprintf(stderr, "rawgpu: %s: %u bytes do not fit in SMC RAM\n", fw->name, fw->ucode_dw * 4);
    return -EINVAL;
  }
  vi_wreg(v, mmSMC_IND_INDEX_11, SMU7_FIRMWARE_HEADER_LOCATION);
  uint32_t acc = vi_rreg(v, mmSMC_IND_ACCESS_CNTL);
  vi_wreg(v, mmSMC_IND_ACCESS_CNTL, REG_SET_FIELD(acc, SMC_IND_ACCESS_CNTL, AUTO_INCREMENT_IND_11, 1));
  for (uint32_t i = 0; i < fw->ucode_dw; i++) vi_wreg(v, mmSMC_IND_DATA_11, fw->ucode[i]);
  acc = vi_rreg(v, mmSMC_IND_ACCESS_CNTL);
  vi_wreg(v, mmSMC_IND_ACCESS_CNTL, REG_SET_FIELD(acc, SMC_IND_ACCESS_CNTL, AUTO_INCREMENT_IND_11, 0));
  return v->err;
}

/* Not in amdgpu: read the start and end of the uploaded image back from SMC
 * RAM before releasing reset, so a lost write shows up here and not as a
 * signature failure. */
static int smc_upload_check(vi_dev *v, const amd_fw *fw) {
  uint32_t idx[17], n = 0;
  for (uint32_t i = 0; i < 16 && i < fw->ucode_dw; i++) idx[n++] = i;
  idx[n++] = fw->ucode_dw - 1;
  for (uint32_t k = 0; k < n; k++) {
    uint32_t got = vi_rsmc(v, SMU7_FIRMWARE_HEADER_LOCATION + 4 * idx[k]);
    if (v->err) return v->err;
    if (got != fw->ucode[idx[k]]) {
      fprintf(stderr, "rawgpu: SMC RAM 0x%05x reads 0x%08x after upload, image has 0x%08x\n",
              SMU7_FIRMWARE_HEADER_LOCATION + 4 * idx[k], got, fw->ucode[idx[k]]);
      return -EIO;
    }
  }
  return 0;
}

/* polaris10_start_smu_in_protection_mode(). The boot ROM checks the image's
 * signature (SMU_STATUS.SMU_PASS) before it runs it. */
static int smc_start_protected(vi_dev *v, const amd_fw *fw) {
  uint32_t x;
  SMC_WRITE_FIELD(v, SMC_SYSCON_RESET_CNTL, rst_reg, 1);
  int r = smc_upload(v, fw);
  if (!r) r = smc_upload_check(v, fw);
  if (r) return r;
  printf("  uploaded %u bytes to SMC RAM at 0x%05x; start and end read back intact\n", fw->ucode_dw * 4,
         SMU7_FIRMWARE_HEADER_LOCATION);

  vi_wsmc(v, ixSMU_STATUS, 0);
  SMC_WRITE_FIELD(v, SMC_SYSCON_CLOCK_CNTL_0, ck_disable, 0);
  SMC_WRITE_FIELD(v, SMC_SYSCON_RESET_CNTL, rst_reg, 0);
  if ((r = smc_poll(v, ixRCU_UC_EVENTS, RCU_UC_EVENTS__INTERRUPTS_ENABLED_MASK, RCU_UC_EVENTS__INTERRUPTS_ENABLED_MASK,
                    true, &x)) == -ETIMEDOUT)
    fprintf(stderr, "rawgpu: warning: boot ROM never set RCU_UC_EVENTS.INTERRUPTS_ENABLED (0x%08x)\n", x);
  if (v->err) return v->err;

  /* smu7_send_msg_to_smc_offset(): "Test" with the image address starts the check */
  uint32_t resp = smc_msg(v, PPSMC_MSG_Test, SMU7_FIRMWARE_HEADER_LOCATION);
  r = smc_poll(v, ixSMU_STATUS, SMU_STATUS__SMU_DONE_MASK, 0, false, &x);
  if (v->err) return v->err;
  printf("  boot ROM: Test message answered 0x%x; SMU_STATUS 0x%08x (%s)\n", resp, x,
         r ? "never finished" : REG_GET_FIELD(x, SMU_STATUS, SMU_PASS) ? "signature accepted" : "REJECTED");
  if (r || !REG_GET_FIELD(x, SMU_STATUS, SMU_PASS)) {
    fprintf(stderr, "rawgpu: the SMC boot ROM did not accept %s.bin\n", fw->name);
    return -EIO;
  }

  vi_wsmc(v, ixFIRMWARE_FLAGS, 0);
  SMC_WRITE_FIELD(v, SMC_SYSCON_RESET_CNTL, rst_reg, 1);
  SMC_WRITE_FIELD(v, SMC_SYSCON_RESET_CNTL, rst_reg, 0);
  r = smc_poll(v, ixFIRMWARE_FLAGS, FIRMWARE_FLAGS__INTERRUPTS_ENABLED_MASK, FIRMWARE_FLAGS__INTERRUPTS_ENABLED_MASK,
               true, &x);
  if (v->err) return v->err;
  if (r) {
    fprintf(stderr, "rawgpu: SMC firmware did not come up (FIRMWARE_FLAGS 0x%08x, PC 0x%08x)\n", x,
            vi_rsmc(v, ixSMC_PC_C));
    return -EIO;
  }
  return 0;
}

/* The SMC firmware amdgpu_cgs_get_firmware_info() picks for a Polaris 10 board. */
static const char *smc_fw_name(const rg_dev *d, bool hard_key) {
  if (!hard_key) return "polaris10_smc_sk";
  uint16_t did = d->device, rid = d->revision;
  bool p20 = (did == 0x67DF && (rid == 0xE3 || rid == 0xE4 || rid == 0xE5 || rid == 0xE7 || rid == 0xEF)) ||
             (did == 0x6FDF && (rid == 0xE7 || rid == 0xEF || rid == 0xFF)); /* ASICID_IS_P20 */
  bool p30 = did == 0x67DF && (rid == 0xE1 || rid == 0xF7);                  /* ASICID_IS_P30 */
  return p20 ? "polaris10_k_smc" : p30 ? "polaris10_k2_smc" : "polaris10_smc";
}

/* ---- engine microcode ---- */

typedef struct {
  amd_fw fw[NIMG];
  uint64_t off[NIMG]; /* VRAM offset of each image; the MEC jump table follows its image */
  uint64_t jt_off, end;
} ucode_set;

static void ucode_free(ucode_set *u) {
  for (int i = 0; i < NIMG; i++) amd_fw_free(&u->fw[i]);
}

/* gfx_firmware_header_v1_0: jt_offset and jt_size, in dwords */
static uint32_t jt_offset(const amd_fw *f) { return amd_fw_hdr32(f, 36); }
static uint32_t jt_size(const amd_fw *f) { return amd_fw_hdr32(f, 40); }

/* amdgpu_ucode_init_bo(): each image page-aligned, with a copy of the MEC jump
 * table after the MEC image (amdgpu_ucode_patch_jt()). Writes them into VRAM. */
static int ucode_place(vi_dev *v, ucode_set *u, const char *dir) {
  memset(u, 0, sizeof *u);
  int r;
  for (int i = 0; i < NIMG; i++)
    if ((r = amd_fw_load(&u->fw[i], dir, img_desc[i].file))) return r;
  const amd_fw *mec = &u->fw[IMG_MEC];
  if (amd_fw_hdr32(mec, 4) < 44 || (uint64_t)jt_offset(mec) + jt_size(mec) > mec->ucode_dw || !jt_size(mec)) {
    fprintf(stderr, "rawgpu: %s: jump table %u+%u dwords is outside the %u-dword image\n", mec->name,
            jt_offset(mec), jt_size(mec), mec->ucode_dw);
    return -EINVAL;
  }
  uint64_t off = VI_VRAM_UCODE;
  for (int i = 0; i < NIMG; i++) {
    u->off[i] = off;
    off += ALIGN_PAGE(u->fw[i].ucode_dw * 4);
    if (i == IMG_MEC) { u->jt_off = off; off += ALIGN_PAGE(jt_size(mec) * 4); }
  }
  u->end = off;
  if (u->end > VI_VRAM_FREE) {
    fprintf(stderr, "rawgpu: microcode needs %llu KB of VRAM; only %llu KB are reserved\n",
            (unsigned long long)((u->end - VI_VRAM_UCODE) >> 10),
            (unsigned long long)((VI_VRAM_FREE - VI_VRAM_UCODE) >> 10));
    return -ENOSPC;
  }
  for (int i = 0; i < NIMG && !r; i++) r = vi_vram_write(v, u->off[i], u->fw[i].ucode, u->fw[i].ucode_dw * 4);
  if (!r) r = vi_vram_write(v, u->jt_off, mec->ucode + jt_offset(mec), jt_size(mec) * 4);
  return r;
}

/* smu7_populate_single_firmware_entry() + amdgpu_cgs_get_firmware_info() */
static smu_entry toc_entry(uint16_t id, const amd_fw *f, uint64_t mc, uint32_t bytes) {
  smu_entry e = {0};
  e.id = id;
  e.version = (uint16_t)f->ucode_version;
  e.image_addr_high = (uint32_t)(mc >> 32);
  e.image_addr_low = (uint32_t)mc;
  e.data_size_byte = bytes;
  e.flags = id == UCODE_ID_RLC_G || id == UCODE_ID_CP_MEC; /* UCODE_FLAG_UNHALT_MASK */
  return e;
}

/* smu7_request_smu_load_fw()'s TOC: every entry, in its order. */
static void build_toc(smu_toc *t, const ucode_set *u, uint64_t vram_mc) {
  memset(t, 0, sizeof *t);
  t->structure_version = SMU_DRAMData_TOC_VERSION;
  const amd_fw *mec = &u->fw[IMG_MEC];
  struct { int img; uint16_t id; } order[] = {
    {IMG_RLC, UCODE_ID_RLC_G}, {IMG_CE, UCODE_ID_CP_CE}, {IMG_PFP, UCODE_ID_CP_PFP}, {IMG_ME, UCODE_ID_CP_ME},
    {IMG_MEC, UCODE_ID_CP_MEC}, {IMG_MEC, UCODE_ID_CP_MEC_JT1}, {IMG_MEC, UCODE_ID_CP_MEC_JT2},
    {IMG_SDMA0, UCODE_ID_SDMA0}, {IMG_SDMA1, UCODE_ID_SDMA1},
  };
  for (size_t k = 0; k < sizeof order / sizeof order[0]; k++) {
    const amd_fw *f = &u->fw[order[k].img];
    uint64_t mc = vram_mc + u->off[order[k].img];
    uint32_t bytes = f->ucode_dw * 4;
    if (order[k].id == UCODE_ID_CP_MEC) bytes = jt_offset(mec) * 4; /* the code without its jump table */
    if (order[k].id == UCODE_ID_CP_MEC_JT1 || order[k].id == UCODE_ID_CP_MEC_JT2) {
      mc = vram_mc + u->jt_off;
      bytes = jt_size(mec) * 4;
    }
    t->entry[t->num_entries++] = toc_entry(order[k].id, f, mc, bytes);
  }
}

/* smu7_request_smu_load_fw() after "Setup SoftRegsStart": place the images
 * and the TOC, tell the SMU where they are, and wait until it reports every
 * image loaded. *status gets UcodeLoadStatus. */
static int load_ucodes(vi_dev *v, ucode_set *u, const char *fw_dir, uint32_t *status) {
  uint32_t soft = soft_regs_start(v);
  if (v->err) return v->err;
  if (!soft_regs_ok(soft)) {
    fprintf(stderr, "rawgpu: firmware header gives soft registers at 0x%08x, outside SMC RAM\n", soft);
    return -EIO;
  }
  uint32_t status_addr = soft + offsetof(smu74_soft_registers, UcodeLoadStatus);
  uint64_t vram_mc = (uint64_t)(vi_rreg(v, mmMC_VM_FB_LOCATION) & 0xFFFF) << 24; /* as vi_gmc_init() */

  int r = ucode_place(v, u, fw_dir);
  if (r) return r;
  smu_toc toc;
  build_toc(&toc, u, vram_mc);
  if ((r = vi_vram_write(v, VI_VRAM_SMU_TOC, &toc, sizeof toc)) || (r = vi_vram_zero(v, VI_VRAM_SMU_BUF, SMU_BUF_BYTES)))
    return r;
  vi_hdp_flush(v); /* the BAR writes must reach VRAM before the SMU reads them */
  printf("microcode:    %u TOC entries; images at VRAM +%llu MB (%llu KB), TOC at +%llu MB, SMU buffer at +%llu MB\n",
         toc.num_entries, (unsigned long long)(VI_VRAM_UCODE >> 20), (unsigned long long)((u->end - VI_VRAM_UCODE) >> 10),
         (unsigned long long)(VI_VRAM_SMU_TOC >> 20), (unsigned long long)(VI_VRAM_SMU_BUF >> 20));
  for (uint32_t k = 0; k < toc.num_entries; k++) {
    const smu_entry *e = &toc.entry[k];
    printf("  TOC[%u] id %2u v%-4u MC 0x%02x%08x %6u bytes%s\n", k, e->id, e->version, e->image_addr_high,
           e->image_addr_low, e->data_size_byte, e->flags ? "  unhalt" : "");
  }

  vi_wsmc(v, status_addr, 0);
  uint64_t buf_mc = vram_mc + VI_VRAM_SMU_BUF, toc_mc = vram_mc + VI_VRAM_SMU_TOC;
  smc_msg_checked(v, "SMU_DRAM_ADDR_HI", PPSMC_MSG_SMU_DRAM_ADDR_HI, (uint32_t)(buf_mc >> 32));
  smc_msg_checked(v, "SMU_DRAM_ADDR_LO", PPSMC_MSG_SMU_DRAM_ADDR_LO, (uint32_t)buf_mc);
  smc_msg_checked(v, "DRV_DRAM_ADDR_HI", PPSMC_MSG_DRV_DRAM_ADDR_HI, (uint32_t)(toc_mc >> 32));
  smc_msg_checked(v, "DRV_DRAM_ADDR_LO", PPSMC_MSG_DRV_DRAM_ADDR_LO, (uint32_t)toc_mc);
  double t0 = vi_now();
  uint32_t resp = smc_msg_checked(v, "LoadUcodes", PPSMC_MSG_LoadUcodes, FW_TO_LOAD);
  r = smc_poll(v, status_addr, FW_TO_LOAD, FW_TO_LOAD, true, status); /* smu7_check_fw_load_finish() */
  if (v->err) return v->err;
  printf("LoadUcodes:   answered 0x%x; UcodeLoadStatus 0x%08x after %.0f ms: ", resp, *status, (vi_now() - t0) * 1e3);
  print_ucode_bits(*status);
  printf("\n");
  if (r) {
    fprintf(stderr, "rawgpu: SMU load firmware failed: wanted 0x%08x, missing 0x%08x\n", FW_TO_LOAD,
            FW_TO_LOAD & ~*status);
    return -EIO;
  }
  return 0;
}

/* Not in amdgpu: read the SDMA microcode RAM back. Before the SMU loaded it, it
 * held its power-on contents whatever the host wrote. */
static int verify_sdma(vi_dev *v, const char *fw_dir) {
  int r = 0;
  for (int me = 0; me < 2; me++) {
    amd_fw f;
    int e = amd_fw_load(&f, fw_dir, img_desc[me ? IMG_SDMA1 : IMG_SDMA0].file);
    if (e) return e;
    uint32_t first_bad = 0;
    int bad = vi_sdma_check_ucode(v, me, &f, &first_bad);
    if (bad >= 0) {
      uint32_t n = f.ucode_dw - VI_SMU_DIGEST_DW;
      printf("SDMA%d ucode:  %u/%u dwords of microcode RAM match %s.bin (less its %u-byte digest)", me, n - bad, n,
             f.name, VI_SMU_DIGEST_DW * 4);
      if (bad) printf(" (first mismatch at dword %u)", first_bad);
      printf("\n");
    }
    amd_fw_free(&f);
    if (bad < 0) return bad;
    if (bad) r = -EIO;
  }
  return r;
}

/* ---- rawgpu smc-load ---- */

int vi_cmd_smc_load(rg_dev *d, const rg_ident *id, int argc, char **argv) {
  const char *fw_dir = NULL, *trace_path = NULL;
  bool force = false;
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--fw") && i + 1 < argc) fw_dir = argv[++i];
    else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
    else if (!strcmp(argv[i], "--force")) force = true;
    else return -EINVAL;
  }
  if (!id->chip || strcmp(id->chip, "POLARIS10")) {
    fprintf(stderr, "rawgpu: smc-load is only wired up for POLARIS10 so far (this is %s): the firmware\n"
                    "        names and SMU header layouts differ per chip.\n", id->chip ? id->chip : "unknown");
    return -ENOTSUP;
  }
  vi_dev v;
  int r = vi_open(&v, d);
  if (r) return r;
  uint32_t memsize;
  if (!vi_posted(&v, &memsize)) {
    if (!v.err) fprintf(stderr, "rawgpu: card is not posted; run `rawgpu post` first (the microcode goes in VRAM)\n");
    return v.err ? v.err : -EIO;
  }
  if (trace_path && !(v.trace = fopen(trace_path, "w"))) { perror(trace_path); return -EIO; }

  ucode_set u = {0};
  amd_fw smc = {0};
  uint32_t status = 0;
  bool loaded = false;
  printf("temperature:  %d C\n", vi_temp_c(&v));

  /* polaris10_start_smu(): only start the SMC if it is not already running its firmware */
  if (smc_firmware_up(&v)) {
    printf("SMC firmware: already running (PC 0x%08x)\n", vi_rsmc(&v, ixSMC_PC_C));
    if (!vi_smu_ucode_status(&v, &status) && (status & FW_TO_LOAD) == FW_TO_LOAD && !force) {
      printf("microcode:    already loaded (UcodeLoadStatus 0x%08x); --force loads it again\n", status);
      loaded = true;
    }
  } else {
    uint32_t smufw = vi_rsmc(&v, ixSMU_FIRMWARE);
    bool prot = REG_GET_FIELD(smufw, SMU_FIRMWARE, SMU_MODE), hard = REG_GET_FIELD(smufw, SMU_FIRMWARE, SMU_SEL);
    if (v.err) { r = v.err; goto out; }
    if (!prot) {
      fprintf(stderr, "rawgpu: SMU_FIRMWARE 0x%08x: the SMU is in non-protected mode, which rawgpu does not\n"
                      "        implement yet (polaris10_start_smu_in_non_protection_mode())\n", smufw);
      r = -ENOTSUP;
      goto out;
    }
    if ((r = amd_fw_load(&smc, fw_dir, smc_fw_name(d, hard)))) goto out;
    printf("SMC firmware: %s.bin v0x%x, %u bytes; SMU in protected mode, %s key\n", smc.name, smc.ucode_version,
           smc.ucode_dw * 4, hard ? "hard" : "soft");
    double t0 = vi_now();
    if ((r = smc_start_protected(&v, &smc))) goto out;
    printf("  firmware up after %.2f s: PC 0x%08x, FIRMWARE_FLAGS 0x%08x\n", vi_now() - t0, vi_rsmc(&v, ixSMC_PC_C),
           vi_rsmc(&v, ixFIRMWARE_FLAGS));
  }

  if (!loaded && (r = load_ucodes(&v, &u, fw_dir, &status))) goto out;
  r = verify_sdma(&v, fw_dir);
  printf("temperature:  %d C\n", vi_temp_c(&v));

out:
  ucode_free(&u);
  amd_fw_free(&smc);
  if (v.trace) { fclose(v.trace); printf("trace: %s\n", trace_path); }
  if (!r) r = v.err;
  if (r && r != -EIO) fprintf(stderr, "rawgpu: smc-load failed (%d)\n", r);
  return r;
}
