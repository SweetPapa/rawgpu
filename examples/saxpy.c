/* The smallest complete rawgpu program: y = a*x + y on the GPU.
 *   make examples && build/examples/saxpy */
#include <rawgpu.h>

#include <stdio.h>
#include <stdlib.h>

static const char *source =
    "__kernel void saxpy(__global float *y, __global const float *x, float a) {\n"
    "  size_t i = get_global_id(0);\n"
    "  y[i] = a * x[i] + y[i];\n"
    "}\n";

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      fprintf(stderr, "saxpy: %s: %s\n", #cond, rawgpu_error(g));       \
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main(void) {
  rawgpu *g = NULL;
  CHECK(rawgpu_open(&g, NULL, RAWGPU_OPEN_BRINGUP) == 0);

  enum { N = 1 << 20 };
  float *x = malloc(N * sizeof *x), *y = malloc(N * sizeof *y);
  for (int i = 0; i < N; i++) { x[i] = (float)i; y[i] = 1.0f; }

  rawgpu_buf *gx = rawgpu_alloc(g, N * sizeof(float), RAWGPU_MEM_VRAM);
  rawgpu_buf *gy = rawgpu_alloc(g, N * sizeof(float), RAWGPU_MEM_VRAM);
  CHECK(gx && gy);
  CHECK(rawgpu_write(g, gx, 0, x, N * sizeof(float)) == 0);
  CHECK(rawgpu_write(g, gy, 0, y, N * sizeof(float)) == 0);

  rawgpu_program *p = rawgpu_program_build(g, source, NULL); /* compiled once, then cached */
  CHECK(p);
  rawgpu_kernel *k = rawgpu_kernel_get(p, "saxpy");
  CHECK(k);
  float a = 2.0f;
  CHECK(rawgpu_set_arg_buf(k, 0, gy) == 0);
  CHECK(rawgpu_set_arg_buf(k, 1, gx) == 0);
  CHECK(rawgpu_set_arg(k, 2, sizeof a, &a) == 0);
  size_t global = N;
  CHECK(rawgpu_launch(g, k, 1, &global, NULL) != 0);

  CHECK(rawgpu_read(g, gy, 0, y, N * sizeof(float)) == 0); /* waits for the kernel */
  int bad = 0;
  for (int i = 0; i < N; i++) bad += y[i] != 2.0f * (float)i + 1.0f;
  printf("saxpy: %d floats, %s\n", N, bad ? "WRONG" : "all correct");

  rawgpu_close(g); /* frees the buffers and the program too */
  free(x);
  free(y);
  return bad != 0;
}
