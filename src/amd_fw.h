/* AMD GPU firmware images (linux-firmware's amdgpu/<name>.bin) and their headers. */
#ifndef RAWGPU_AMD_FW_H
#define RAWGPU_AMD_FW_H

#include <stddef.h>
#include <stdint.h>

/* amdgpu_ucode.h: struct common_firmware_header, little-endian on disk. */
typedef struct {
  char name[64];       /* e.g. "polaris10_sdma" */
  uint8_t *data;
  size_t size;
  uint16_t hdr_major, hdr_minor, ip_major, ip_minor;
  uint32_t ucode_version;
  const uint32_t *ucode; /* points into data */
  uint32_t ucode_dw;
} amd_fw;

/* rawgpu's per-user data, firmware/ and device-libs/: $RAWGPU_HOME; else the local/ folder of the
 * checkout this was built in, if it is there; else ~/Library/Application Support/rawgpu (a release). */
const char *rg_data_dir(void);

/* Finds <dir>/amdgpu/<name>.bin (or <dir>/<name>.bin). dir NULL = $RAWGPU_FIRMWARE,
 * then rg_data_dir()/firmware. Checks the header against the file size. */
int amd_fw_load(amd_fw *fw, const char *dir, const char *name);
void amd_fw_free(amd_fw *fw);
uint32_t amd_fw_hdr32(const amd_fw *fw, size_t off); /* header dword at byte offset off */

#endif
