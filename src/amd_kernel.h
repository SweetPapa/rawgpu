/* AMDGPU code objects: the ELF files LLVM writes for amdgcn-amd-amdhsa.
 *
 * A small loader and linker. It takes a relocatable object (clang -c) or a
 * linked one (ld.lld -shared), lays out the allocated sections in one image,
 * resolves the relocations for the image's load address, and reads each
 * kernel's descriptor (<name>.kd) and its metadata (the NT_AMDGPU_METADATA
 * note, msgpack).
 *
 * Layouts follow the ELF spec and LLVM's AMDGPUUsage ("Kernel Descriptor",
 * "Code Object V3+ Metadata", "Relocation Records"). A code object is input
 * data, so every read is bounds-checked.
 */
#ifndef RAWGPU_AMD_KERNEL_H
#define RAWGPU_AMD_KERNEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define EF_AMDGPU_MACH_AMDGCN_GFX803 0x2a /* e_flags machine: Polaris (and Fiji, Tonga) */

/* llvm AMDHSA kernel_code_properties: user SGPRs the kernel expects, in this order */
#define AMD_KCP_PRIVATE_SEGMENT_BUFFER 0x01u /* 4 SGPRs: the scratch buffer descriptor */
#define AMD_KCP_DISPATCH_PTR           0x02u /* 2: the hsa_kernel_dispatch_packet */
#define AMD_KCP_QUEUE_PTR              0x04u /* 2: amd_queue_t */
#define AMD_KCP_KERNARG_SEGMENT_PTR    0x08u /* 2 */
#define AMD_KCP_DISPATCH_ID            0x10u /* 2 */
#define AMD_KCP_FLAT_SCRATCH_INIT      0x20u /* 2: offset from the hidden private base, bytes per lane */
#define AMD_KCP_PRIVATE_SEGMENT_SIZE   0x40u /* 1 */

/* .value_kind of a kernel argument */
typedef enum {
  AMD_ARG_UNKNOWN,
  AMD_ARG_BY_VALUE,
  AMD_ARG_GLOBAL_BUFFER,
  AMD_ARG_DYNAMIC_SHARED_POINTER,
  AMD_ARG_IMAGE,
  AMD_ARG_SAMPLER,
  AMD_ARG_PIPE,
  AMD_ARG_QUEUE,
  /* hidden arguments, filled by the runtime */
  AMD_ARG_HIDDEN_GLOBAL_OFFSET_X,
  AMD_ARG_HIDDEN_GLOBAL_OFFSET_Y,
  AMD_ARG_HIDDEN_GLOBAL_OFFSET_Z,
  AMD_ARG_HIDDEN_BLOCK_COUNT_X,
  AMD_ARG_HIDDEN_BLOCK_COUNT_Y,
  AMD_ARG_HIDDEN_BLOCK_COUNT_Z,
  AMD_ARG_HIDDEN_GROUP_SIZE_X,
  AMD_ARG_HIDDEN_GROUP_SIZE_Y,
  AMD_ARG_HIDDEN_GROUP_SIZE_Z,
  AMD_ARG_HIDDEN_REMAINDER_X,
  AMD_ARG_HIDDEN_REMAINDER_Y,
  AMD_ARG_HIDDEN_REMAINDER_Z,
  AMD_ARG_HIDDEN_GRID_DIMS,
  AMD_ARG_HIDDEN_DYNAMIC_LDS_SIZE,
  AMD_ARG_HIDDEN_PRIVATE_BASE,
  AMD_ARG_HIDDEN_SHARED_BASE,
  AMD_ARG_HIDDEN_QUEUE_PTR,
  AMD_ARG_HIDDEN_NONE,
  AMD_ARG_HIDDEN_OTHER, /* printf/hostcall buffers, default queue, heap...: rawgpu leaves them zero */
} amd_arg_kind;

typedef struct {
  char name[48];      /* .name (empty if the metadata has none) */
  char type_name[48]; /* .type_name, e.g. "float*" */
  uint32_t offset, size;
  amd_arg_kind kind;
  char address_space; /* pointers: 'g'lobal, 'l'ocal, 'c'onstant, 'p'rivate, 'G'eneric; 0 otherwise */
  bool is_const;
} amd_arg;

typedef struct amd_program amd_program;

typedef struct {
  char name[64];
  const amd_program *prog;
  uint64_t entry;     /* offset of the kernel's first instruction in the program image */
  /* from the 64-byte kernel descriptor */
  uint32_t group_segment_size, private_segment_size, kernarg_size;
  uint32_t rsrc1, rsrc2, rsrc3;
  uint16_t code_properties;
  /* from the metadata (zero if the code object has none) */
  bool has_metadata, dynamic_stack;
  uint32_t kernarg_align, max_wg, reqd_wg[3];
  uint32_t sgprs, vgprs, sgpr_spills, vgpr_spills;
  uint32_t nargs;     /* explicit and hidden arguments, in metadata order */
  amd_arg *args;
} amd_kernel;

typedef struct { uint64_t off; uint32_t type; uint64_t sym; int64_t addend; } amd_reloc; /* sym: image offset */

struct amd_program {
  char path[256];
  uint32_t mach;      /* EF_AMDGPU_MACH from e_flags, e.g. 0x2a = gfx803 */
  uint8_t *image;     /* every allocated section at its image offset; NOBITS zeroed */
  size_t image_size;
  uint32_t nkernels;
  amd_kernel *kernels;
  uint32_t nrelocs;
  amd_reloc *relocs;
};

/* Loads a code object; `label` names it in messages. Fails, naming them, if
 * the code refers to symbols the object does not define (a built-in function
 * with no definition, for example). */
int amd_program_load(amd_program *p, const char *path);
int amd_program_load_mem(amd_program *p, const void *data, size_t size, const char *label);
/* Resolves the relocations for an image loaded at GPU address `base`
 * (256-byte aligned). Can be called again for another base. */
void amd_program_relocate(amd_program *p, uint64_t base);
/* NULL, with a message listing the kernels there are, if none is called `name` */
const amd_kernel *amd_program_kernel(const amd_program *p, const char *name);
void amd_program_free(amd_program *p);

const char *amd_arg_kind_name(amd_arg_kind k);
/* Human-readable summary of every kernel: resources, user SGPRs, arguments */
void amd_program_print(FILE *f, const amd_program *p);

#endif
