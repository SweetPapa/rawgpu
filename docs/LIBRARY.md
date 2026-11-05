# librawgpu: using the GPU from programs

`librawgpu` is the library behind the `rawgpu` CLI. Programs, in C or in
Python, use it to run compute kernels on the GPU: allocate memory, move
data, compile and launch OpenCL C kernels, and call the built-in SGEMM. It
drives the same RX 590 (Polaris 10) through TinyGPU as the rest of rawgpu.

## Building

```
make                 # build/rawgpu, build/librawgpu.a, build/librawgpu.dylib
make kernels         # rawgpu's own kernels; the first run fetches AMD's device libraries
make examples        # build/examples/saxpy, against the static library
```

Kernels are compiled by an LLVM that has the AMDGPU target: Homebrew's
(`brew install llvm`). Apple's clang does not have it.
`tools/fetch-device-libs.sh` downloads AMD's device libraries (OpenCL
built-ins as LLVM bitcode, pinned by sha256) into `local/device-libs/`.

## Bringing the card up

After power-on or a Mac sleep, the card needs `rawgpu post`,
`rawgpu mc-load` and `rawgpu dpm-enable` (README). The library can do this
itself: `rawgpu_open(&g, NULL, RAWGPU_OPEN_BRINGUP)` in C, or
`rawgpu.Device()` in Python, which defaults to `bringup=True`. It runs the
same three steps when the card is not set up. They take about two seconds
and print progress. A card that is already up is left as it is.

## C

```c
#include <rawgpu.h>

rawgpu *g;
if (rawgpu_open(&g, NULL, RAWGPU_OPEN_BRINGUP)) { puts(rawgpu_error(NULL)); return 1; }

rawgpu_buf *x = rawgpu_alloc(g, n * sizeof(float), RAWGPU_MEM_VRAM);
rawgpu_write(g, x, 0, host_x, n * sizeof(float));

rawgpu_program *p = rawgpu_program_build(g,
    "__kernel void scale(__global float *x, float s) { x[get_global_id(0)] *= s; }", NULL);
rawgpu_kernel *k = rawgpu_kernel_get(p, "scale");
float s = 2.0f;
rawgpu_set_arg_buf(k, 0, x);
rawgpu_set_arg(k, 1, sizeof s, &s);
size_t global = n;
rawgpu_launch(g, k, 1, &global, NULL);       /* asynchronous */

rawgpu_read(g, x, 0, host_x, n * sizeof(float));  /* waits for the launch */
rawgpu_close(g);                              /* frees everything still open */
```

`examples/saxpy.c` is a complete program. The header (`include/rawgpu.h`)
documents every call. The rules:

- **Order.** Launches are asynchronous and run in the order they were made,
  each one finishing before the next starts (a CS partial flush between them).
  `rawgpu_wait()` waits for one, `rawgpu_finish()` for all of them.
  Transfers (`write`, `read`, `copy`, `fill`) wait for every launch first,
  and are complete when they return.
- **Memory.** `RAWGPU_MEM_VRAM` is the card's 8 GB, the fast memory for
  kernels. `RAWGPU_MEM_HOST` is Mac memory the GPU reaches through the
  GART: a few GB/s over Thunderbolt, shared with the CPU through
  `rawgpu_buf_host()`. Read it from the CPU only once the launches that
  write it are finished.
- **Arguments** are set by index, as in the kernel's signature:
  `rawgpu_set_arg` for values (exact size), `rawgpu_set_arg_buf` for
  `__global`/`__constant` pointers, `rawgpu_set_arg_local` for a `__local`
  pointer's size per work-group. Every argument must be set before a
  launch. Sizes and kinds come from the code object's metadata, so a wrong
  size or kind is an error that names the argument.
- **Work-group size.** With `local = NULL`, rawgpu picks: the kernel's
  `reqd_work_group_size` if it has one, otherwise the largest power-of-two
  shape up to 256 that divides the global size. The global size must be a
  multiple of the local size (OpenCL 1.2 rules), so pad and bounds-check in
  the kernel when it is not.
- **Errors.** Calls return 0 or a negative errno (NULL, or event 0), and
  `rawgpu_error(g)` describes the failure. Lower-level diagnostics, such as
  a hung queue's register dump, go to stderr.
- **Threads.** A context is not thread-safe. One process at a time can
  have the GPU open; that is a TinyGPU limit.

## Python

`python/rawgpu` is a ctypes binding with numpy integration. No install is
needed:

```python
import sys; sys.path.insert(0, "python")
import numpy as np, rawgpu

gpu = rawgpu.Device()
x = gpu.array(np.arange(1 << 20, dtype=np.float32))
prog = gpu.program("""
    __kernel void scale(__global float *x, float s) { x[get_global_id(0)] *= s; }""")
prog.scale(x, 2.0, global_size=x.size)      # launches; arguments are converted by their declared types
print(x.numpy()[:4])                        # copies back, after waiting

c = gpu.matmul(gpu.array(a), gpu.array(b))  # SGEMM on 2-D float32 arrays
```

- `gpu.empty`, `gpu.zeros` and `gpu.array` create a `Buffer`. Pass
  `host=True` for host memory.
- Kernels are attributes of their program. Pass `rawgpu.Local(nbytes)` for a
  `__local` argument.
- `gpu.sgemm(...)` takes the full BLAS form: alpha, beta, transposes and
  leading dimensions.
- `gpu.info` reports the engine and memory clocks, the temperature, the fan's
  PWM duty (`fan_percent`, -1 before `dpm-enable` puts the SMC in charge of
  the fan) and free VRAM.
- Failures raise `rawgpu.Error` with the library's message.

`python/test_rawgpu.py` checks all of it on the card. It is also a good
example of the API.

## Kernels

Kernels are written in OpenCL C 1.2 by default; pass `-cl-std=CL2.0` in
`options` for 2.0 syntax. `rawgpu_program_build()` compiles them with clang
for gfx803. Clang links AMD's ROCm device libraries into the object: every
OpenCL built-in is available, and the math comes from ocml.
`rawgpu kernel-test` checks the accuracy of each function against the
CPU's double precision. Compiled objects are cached by content, under
`~/Library/Caches/rawgpu`.

Any self-contained AMDGPU code object for gfx803 loads too, through
`rawgpu_program_load()`. That includes clang `-c` objects and `ld.lld`
shared objects. `rawgpu kernel-info FILE` shows a code object's kernels,
resources and arguments without a GPU.

**What works:**

- work-items in 1 to 3 dimensions;
- `__local` memory, static or as arguments, up to 64 KB per work-group;
- barriers;
- atomics on VRAM;
- private arrays of any size (scratch memory, grown as needed);
- program-scope `__constant` data;
- double precision.

**What does not (yet):**

- `printf`;
- images and samplers;
- pipes;
- device-side enqueue;
- OpenCL 2.0 generic-address casts to `__local` or `__private` memory
  (on gfx8 they need the queue pointer);
- recursion (a dynamic stack).

The loader rejects a kernel that needs any of these, with a message.

**Performance notes for Polaris:**

- Fused `fma()` runs at a quarter of the rate of a non-fused multiply-add.
  Write `a * b + c` or `mad()`, which compile to full-rate `v_mad_f32`.
- Single-precision denormals are flushed to zero.
- A kernel gets at most 256 work-items per group unless it declares more
  with `reqd_work_group_size`.
- Two waves per SIMD need at most 128 VGPRs; `rawgpu kernel-info` shows a
  kernel's count.

## Measured (RX 590, DPM on, 2026-11-04)

| | |
|---|---|
| engine clock under load (`s_memtime`) | 1540 MHz |
| FP32 multiply-add (`v_mad_f32`) | 6961 GFLOPS (98% of 36 CUs x 128 x 1.545 GHz) |
| VRAM copy, a kernel | 186 GB/s (read + write) |
| host <-> VRAM, SDMA | 2.8-3.1 GB/s (the Thunderbolt link) |
| SGEMM 1024 / 2048 / 4096, `rawgpu_sgemm()` | 2.99 / 3.91 / 4.84 TFLOPS |
| launch to fence seen, empty work | about 100 us |
| first compile of a small kernel / cached | 70-270 ms / under 1 ms |

The SGEMM kernel (`kernels/sgemm.cl`) computes 8x8 outputs per work-item
in 128x128 tiles. It uses 124 VGPRs, so two waves fit per SIMD, and its
`KUNROLL=2` won the sweep in `python/tune_sgemm.py`. Unrolling all 8 k
steps needs 218 VGPRs and drops to 3.3 TFLOPS. Small matrices are limited
by parallelism: a 1024-square product has only 64 tiles for 36 compute
units.

Avoid leading dimensions that are multiples of a large power of two. A
tile's 128 rows then collide in the caches or memory channels, and the
`nt` form, where both operands are read along rows, suffers most. Pad each
row by 4 or more floats. Measured with M = 8192, except the square case:

| product | lda = ldb unpadded | padded by 32 floats |
|---|---|---|
| `nt`, N 512, K 2048 | 0.79 TFLOPS | 4.62 TFLOPS |
| `nt`, N 4096, K 512 | 2.74 | 4.27 |
| `nn`, N 512, K 2048 | 3.63 | 4.59 |
| `nt`, 3072 square | 3.40 | 4.32 |

ocml accuracy, from `rawgpu kernel-test`: the worst error in ULPs over
65536 inputs per function, against the CPU in double precision. Every
function is within its OpenCL 1.2 limit.

| function | ulp | function | ulp | function | ulp |
|---|---|---|---|---|---|
| sin / cos / tan | 1.46 / 1.50 / 1.87 | exp / exp2 / exp10 | 0.84 / 0.76 / 0.82 | sqrt / rsqrt / cbrt | 0.88 / 0.82 / 1.02 |
| asin / acos / atan | 2.46 / 1.39 / 1.73 | log / log2 / log10 | 1.83 / 1.00 / 1.70 | pow | 1.23 |
| atan2 | 2.30 | expm1 / log1p | 1.10 / 0.53 | erf / erfc | 1.32 / 2.51 |
| sinh / cosh / tanh | 0.80 / 0.55 / 1.31 | hypot / fmod | 1.36 / 0 | tgamma / lgamma | 9.07 / 3.70 |
| divide | 1.75 | | | | |

## Environment

| variable | default | what |
|---|---|---|
| `RAWGPU_CLANG` | Homebrew's `/opt/homebrew/opt/llvm/bin/clang` | the kernel compiler |
| `RAWGPU_DEVICE_LIBS` | this checkout's `local/device-libs` | the bitcode directory |
| `RAWGPU_CACHE` | `~/Library/Caches/rawgpu` | compiled kernels |
| `RAWGPU_FIRMWARE` | this checkout's `local/firmware` | firmware for bring-up |
| `RAWGPU_LIB` | this checkout's `build/librawgpu.dylib` | the library the Python binding loads |
| `RAWGPU_TINYGPU_SOCK` | `$TMPDIR/rawgpu-tinygpu.sock` | the TinyGPU server socket |
