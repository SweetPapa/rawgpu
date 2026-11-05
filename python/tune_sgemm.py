#!/usr/bin/env python3
"""Time kernels/sgemm.cl's sgemm_nn under different compile-time settings,
through the library's runtime compiler: python3 python/tune_sgemm.py [N]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawgpu  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
src = open(os.path.join(os.path.dirname(__file__), "..", "kernels", "sgemm.cl")).read()
gpu = rawgpu.Device()
rng = np.random.default_rng(0)
A = (rng.integers(-8, 9, (N, N)) / 16).astype(np.float32)
B = (rng.integers(-8, 9, (N, N)) / 16).astype(np.float32)
gA, gB, gC = gpu.array(A), gpu.array(B), gpu.empty((N, N))
rows = slice(0, N, max(1, N // 16))
ref = (A[rows].astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

for opts in ["-DKUNROLL=1", "-DKUNROLL=2", "-DKUNROLL=4", "-DKUNROLL=8 -DWAVES_PER_EU=1", "-DKUNROLL=2 -DWAVES_PER_EU=1"]:
    prog = gpu.program(src, opts)
    k = prog.sgemm_nn
    g = (N + 127) // 128 * 16
    best = 1e9
    for _ in range(3):
        t = time.time()
        k(gA, gB, gC, N, N, N, N, N, N, 1.0, 0.0, global_size=(g, g), local_size=(16, 16), wait=True)
        best = min(best, time.time() - t)
    ok = np.array_equal(gC.numpy()[rows], ref)
    print(f"{opts:<30} {k.info['vgprs']:>3} VGPRs  {best * 1e3:8.2f} ms  {2 * N ** 3 / best / 1e9:6.0f} GFLOPS  "
          f"{'exact' if ok else 'WRONG'}")
    prog.free()
gpu.close()
