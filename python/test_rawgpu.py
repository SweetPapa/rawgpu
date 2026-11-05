#!/usr/bin/env python3
"""End-to-end checks of librawgpu through the Python binding: device info,
runtime compilation and its cache, argument handling, __local memory, host
buffers, error messages, and SGEMM in every transpose and edge shape against
numpy. Then SGEMM speed. Needs the GPU: python3 python/test_rawgpu.py [--quick]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawgpu  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{name:<22} {'ok' if ok else 'FAILED'}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def main():
    quick = "--quick" in sys.argv
    gpu = rawgpu.Device()
    i = gpu.info
    print(f"device: {i['name']}, {i['compute_units']} CUs, {i['vram_bytes'] >> 20} MB VRAM, "
          f"{i['engine_mhz']}/{i['memory_mhz']} MHz, {i['temperature_c']} C")

    # runtime compile, then the cache
    src = """
    __kernel void saxpy(__global float *y, __global const float *x, float a, uint n) {
      size_t i = get_global_id(0);
      if (i < n) y[i] = a * x[i] + y[i];
    }
    __kernel void block_sum(__global const float *in, __global float *out, __local float *tmp) {
      size_t l = get_local_id(0);
      tmp[l] = in[get_global_id(0)];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (size_t s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (l < s) tmp[l] += tmp[l + s];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (l == 0) out[get_group_id(0)] = tmp[0];
    }
    __kernel void gelu(__global float *x) {
      size_t i = get_global_id(0);
      float v = x[i];
      x[i] = 0.5f * v * (1.0f + erf(v * 0.70710678f));
    }"""
    t = time.time()
    prog = gpu.program(src, "-DUNUSED=1")
    t1 = time.time() - t
    t = time.time()
    prog2 = gpu.program(src, "-DUNUSED=1")
    t2 = time.time() - t
    check("compile + cache", prog.kernel_names == ["saxpy", "block_sum", "gelu"],
          f"first build {t1 * 1e3:.0f} ms, cached {t2 * 1e3:.0f} ms; kernels {prog.kernel_names}")
    prog2.free()

    n = 1 << 20
    x = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    y = np.random.default_rng(2).standard_normal(n).astype(np.float32)
    gx, gy = gpu.array(x), gpu.array(y)
    prog.saxpy(gy, gx, 3.0, n, global_size=n)
    got = gy.numpy()
    check("saxpy", np.allclose(got, 3.0 * x + y, rtol=1e-6, atol=1e-6), f"{n} floats, {prog.saxpy}")

    wg = 256
    out = gpu.empty(n // wg)
    prog.block_sum(gx, out, rawgpu.Local(wg * 4), global_size=n, local_size=wg)
    ref = x.reshape(-1, wg).sum(axis=1, dtype=np.float64)
    check("__local reduction", np.allclose(out.numpy(), ref, rtol=1e-5, atol=1e-4), f"{n // wg} blocks of {wg}")

    from math import erf
    v = gpu.array(x[:65536])
    prog.gelu(v, global_size=65536)
    ref = np.array([0.5 * a * (1 + erf(a / np.sqrt(2))) for a in x[:65536].astype(np.float64)])
    err = np.max(np.abs(v.numpy() - ref))
    check("gelu (ocml erf)", err < 1e-6, f"max abs error {err:.1e}")

    # host memory: the kernel writes the CPU's buffer directly
    h = gpu.empty(4096, np.float32, host=True)
    h.write(np.arange(4096, dtype=np.float32))
    prog.saxpy(h, gx, 0.0, 4096, global_size=4096)
    check("host buffer", np.array_equal(h.numpy(), np.arange(4096, dtype=np.float32)), "saxpy with a = 0 in host memory")

    # errors come back as exceptions with the library's message
    try:
        gpu.program("__kernel void broken(__global float *x) { x[0] = undefined_thing; }")
        check("compile error", False, "no exception")
    except rawgpu.Error as e:
        check("compile error", "undefined_thing" in str(e), str(e).splitlines()[0][:70])
    try:
        prog.saxpy(gy, gx, 3.0, global_size=n)
        check("argument count", False, "no exception")
    except rawgpu.Error as e:
        check("argument count", "takes 4 arguments" in str(e), str(e)[:70])

    # SGEMM: exact data (multiples of 1/16), every transpose, edge shapes
    rng = np.random.default_rng(7)
    shapes = [(1, 1, 1), (3, 5, 7), (128, 128, 8), (129, 127, 9), (257, 130, 77), (64, 300, 513), (512, 512, 512)]
    bad = []
    for (M, N, K) in shapes:
        for ta in (False, True):
            for tb in (False, True):
                A = (rng.integers(-8, 9, (K, M) if ta else (M, K)) / 16).astype(np.float32)
                B = (rng.integers(-8, 9, (N, K) if tb else (K, N)) / 16).astype(np.float32)
                C0 = (rng.integers(-8, 9, (M, N)) / 16).astype(np.float32)
                gA, gB, gC = gpu.array(A), gpu.array(B), gpu.array(C0)
                gpu.sgemm(gA, gB, gC, M, N, K, alpha=0.5, beta=2.0, trans_a=ta, trans_b=tb)
                opA = A.T if ta else A
                opB = B.T if tb else B
                ref = 0.5 * (opA.astype(np.float64) @ opB.astype(np.float64)) + 2.0 * C0
                if not np.array_equal(gC.numpy(), ref.astype(np.float32)):
                    bad.append(f"{M}x{N}x{K}{' tA' if ta else ''}{' tB' if tb else ''}")
                for b in (gA, gB, gC):
                    b.free()
    check("sgemm exact", not bad, f"{len(shapes) * 4} cases, alpha 0.5, beta 2" + (f"; wrong: {', '.join(bad)}" if bad else ""))

    A = rng.standard_normal((1000, 700)).astype(np.float32)
    B = rng.standard_normal((700, 900)).astype(np.float32)
    C = gpu.matmul(gpu.array(A), gpu.array(B)).numpy()
    ref = A.astype(np.float64) @ B.astype(np.float64)
    rel = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
    check("matmul random", rel < 1e-5, f"1000x700 @ 700x900, max error {rel:.1e} of the largest entry")

    # speed
    print("sgemm speed (sgemm_nn, best of 3):")
    for n in ([1024, 2048] if quick else [256, 512, 1024, 2048, 3072, 4096]):
        A = gpu.array(rng.standard_normal((n, n)).astype(np.float32))
        B = gpu.array(rng.standard_normal((n, n)).astype(np.float32))
        Cb = gpu.empty((n, n))
        best = 1e9
        for _ in range(3):
            t = time.time()
            gpu.wait(gpu.sgemm(A, B, Cb, n, n, n))
            best = min(best, time.time() - t)
        print(f"  {n:>5} x {n:<5} {best * 1e3:8.2f} ms  {2 * n ** 3 / best / 1e9:7.0f} GFLOPS")
        for b in (A, B, Cb):
            b.free()

    i = gpu.info
    print(f"after: {i['engine_mhz']}/{i['memory_mhz']} MHz, {i['temperature_c']} C, "
          f"{i['vram_free_bytes'] >> 20} MB VRAM free")
    gpu.close()
    print("all library checks passed" if not failures else f"FAILED: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
