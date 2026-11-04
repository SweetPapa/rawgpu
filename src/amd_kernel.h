/* AMDGPU kernels from a code object (an ELF file LLVM writes for amdgcn-amd-amdhsa). */
#ifndef RAWGPU_AMD_KERNEL_H
#define RAWGPU_AMD_KERNEL_H

#include <stddef.h>
#include <stdint.h>

/* llvm AMDHSA kernel_code_properties: user SGPRs the kernel expects, in this order */
#define AMD_KCP_PRIVATE_SEGMENT_BUFFER 0x01u /* 4 SGPRs */
#define AMD_KCP_DISPATCH_PTR           0x02u /* 2 */
#define AMD_KCP_QUEUE_PTR              0x04u /* 2 */
#define AMD_KCP_KERNARG_SEGMENT_PTR    0x08u /* 2 */
#define AMD_KCP_DISPATCH_ID            0x10u /* 2 */
#define AMD_KCP_FLAT_SCRATCH_INIT      0x20u /* 2 */
#define AMD_KCP_PRIVATE_SEGMENT_SIZE   0x40u /* 1 */

typedef struct {
  char name[64];
  uint32_t mach;               /* EF_AMDGPU_MACH from e_flags, e.g. 0x2a = gfx803 */
  uint8_t *code;               /* the whole .text section the kernel is in */
  size_t code_size;
  uint64_t entry;              /* offset of the kernel's first instruction in code */
  /* from the 64-byte kernel descriptor (<name>.kd) */
  uint32_t group_segment_size, private_segment_size, kernarg_size;
  uint32_t rsrc1, rsrc2, rsrc3;
  uint16_t code_properties;
} amd_kernel;

/* Loads kernel `name` from the code object at `path`. A relocatable object
 * (clang -c) is fine as long as the code itself has no relocations; the one
 * relocation LLVM puts in the descriptor (its entry offset) is not needed,
 * because the caller points the hardware at the code directly. */
int amd_kernel_load(amd_kernel *k, const char *path, const char *name);
void amd_kernel_free(amd_kernel *k);

#endif
