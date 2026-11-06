#!/usr/bin/env python3
"""Check every rawgpu.nn op, forward and backward, against PyTorch (CPU, float64), on the GPU.

    local/eg2-venv/bin/python python/test_nn.py        # any Python with numpy and torch

Each case runs an op on random inputs, takes the sum of its output times a random tensor as the loss,
and compares the output and every input's gradient with torch's. The sizes are odd on purpose (edge
tiles of the matrix products), and some are multiples of 64, which get padded row pitches.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

from rawgpu import nn  # noqa: E402

rng = np.random.default_rng(0)
FAILED = []


def r(*shape, lo=None):
    a = rng.standard_normal(shape).astype(np.float32)
    if lo is not None:
        a = np.abs(a) + lo
    return a


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))


def check(name, ours, ref, inputs, tol=3e-4, grad_inputs=None):
    """ours(*tensors) -> Tensor, ref(*torch tensors) -> torch tensor of the same shape"""
    grad_inputs = range(len(inputs)) if grad_inputs is None else grad_inputs
    ts = [nn.tensor(a, requires_grad=i in grad_inputs) if a.dtype == np.float32 else nn.tensor(a)
          for i, a in enumerate(inputs)]
    out = ours(*ts)
    o = out.numpy()
    R = r(*o.shape)
    loss = nn.sum_all(nn.mul(out, nn.tensor(R)))
    loss.backward()
    tt = [torch.tensor(a.astype(np.float64) if a.dtype == np.float32 else a, requires_grad=(i in grad_inputs and
                                                                                            a.dtype == np.float32))
          for i, a in enumerate(inputs)]
    ro = ref(*tt)
    (ro * torch.tensor(R, dtype=torch.float64)).sum().backward()
    errs = [rel_err(o, ro.detach().numpy())]
    for i in grad_inputs:
        if inputs[i].dtype != np.float32:
            continue
        g = ts[i].grad.numpy() if ts[i].grad is not None else np.zeros_like(inputs[i])
        errs.append(rel_err(g, tt[i].grad.numpy()))
    worst = max(errs)
    ok = worst < tol and np.isfinite(worst)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<38} output {errs[0]:.1e}, gradients "
          + ", ".join(f"{e:.1e}" for e in errs[1:]))
    if not ok:
        FAILED.append(name)


def attn_ref(q, k, v, B, T, H, KV, causal, scale, lens=None, window=0, drop=None):
    hd = q.shape[1] // H
    q = q.view(B, T, H, hd).transpose(1, 2)
    k = k.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(H // KV, dim=1)
    v = v.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(H // KV, dim=1)
    s = q @ k.transpose(-1, -2) * scale
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    m = torch.ones(T, T, dtype=torch.bool)
    if causal:
        m &= j <= i
    if window:
        m &= (i - j).abs() < window
    m = m[None, None].expand(B, 1, T, T).clone()
    if lens is not None:
        m &= (j[None, None] < torch.as_tensor(lens).view(B, 1, 1, 1))
    s = s.masked_fill(~m, float("-inf"))
    p = torch.nan_to_num(torch.softmax(s, -1), nan=0.0)
    if drop is not None:  # [B * H * T, T]: the kept weights' scale, 0 where dropped
        p = p * torch.as_tensor(drop, dtype=torch.float64).view(B, H, T, T)
    return (p @ v).transpose(1, 2).reshape(B * T, H * hd)


def mask_of(shape, p, seed):
    """The multiplier nn.dropout applies with this seed: 1 / (1 - p) where kept, 0 where dropped"""
    return nn.dropout(nn.ones(shape), p, seed=seed).numpy()


def report(name, ok, text):
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<38} {text}")
    if not ok:
        FAILED.append(name)


def dropout_checks():
    print("dropout")
    p = 0.3
    m = mask_of((777, 1283), p, 7)
    kept = m != 0
    frac = kept.mean()
    vals = np.unique(m[kept])
    report("dropout keeps 1 - p, scaled 1/(1-p)", abs(frac - (1 - p)) < 0.003 and len(vals) == 1 and
           abs(vals[0] - 1 / (1 - p)) < 1e-6, f"kept {frac:.4f} (want {1 - p}), value {vals[0]:.6f}")
    a, b = kept[:, :-1].reshape(-1).astype(np.float64), kept[:, 1:].reshape(-1).astype(np.float64)
    c_next = abs(np.corrcoef(a, b)[0, 1])
    c_seed = abs(np.corrcoef(kept.reshape(-1), (mask_of((777, 1283), p, 8) != 0).reshape(-1))[0, 1])
    c_row = abs(np.corrcoef(kept[:-1].reshape(-1), kept[1:].reshape(-1))[0, 1])
    report("masks uncorrelated", max(c_next, c_seed, c_row) < 0.005,
           f"next element {c_next:.4f}, next row {c_row:.4f}, next seed {c_seed:.4f}")
    # the mask follows the logical index, not the row pitch: a column view (ld > cols) gets the same one
    x = r(300, 64 * 5)                       # padded pitch
    big = nn.tensor(np.concatenate([x, r(300, 7)], 1))
    view = nn.slice_cols(big, 0, 320)
    same = np.array_equal(nn.dropout(view, 0.5, seed=3).numpy(), nn.dropout(nn.tensor(x), 0.5, seed=3).numpy())
    report("mask independent of the row pitch", same, f"ld {view.ld} vs {nn.tensor(x).ld}")
    nn.manual_seed(11)
    d1, d2 = nn.dropout(nn.ones((64, 64)), 0.5).numpy(), nn.dropout(nn.ones((64, 64)), 0.5).numpy()
    nn.manual_seed(11)
    d3 = nn.dropout(nn.ones((64, 64)), 0.5).numpy()
    report("seed sequence, manual_seed", not np.array_equal(d1, d2) and np.array_equal(d1, d3),
           "successive calls differ, a restart repeats")
    for shape, q in (((37, 130), 0.1), ((128, 256), 0.5), ((5, 3), 0.9)):
        M = mask_of(shape, q, 21)
        check(f"dropout {shape[0]}x{shape[1]} p={q}", lambda a, q=q: nn.dropout(a, q, seed=21),
              lambda a, M=M: a * torch.as_tensor(M, dtype=torch.float64), [r(*shape)])
    report("dropout p=0 is the identity", nn.dropout(big, 0.0) is big, "returns its input")

    # attention-weight dropout: torch applies the same mask (read back from nn.dropout) after the softmax
    for (B, T, H, KV, hd, causal, lens) in [(2, 40, 4, 4, 32, True, None), (2, 70, 4, 2, 32, False, [70, 33])]:
        D, Dk = H * hd, KV * hd
        M = mask_of((B * H * T, T), 0.2, 99)
        check(f"attention dropout B{B} T{T} H{H}/{KV}{' causal' if causal else ''}{' lens' if lens else ''}",
              lambda a, B=B, T=T, H=H, KV=KV, D=D, Dk=Dk, causal=causal, lens=lens:
              nn.attention(nn.slice_cols(a, 0, D), nn.slice_cols(a, D, D + Dk), nn.slice_cols(a, D + Dk, D + 2 * Dk),
                           B, T, H, KV, causal, 1 / np.sqrt(hd), lens, dropout=0.2, seed=99),
              lambda a, B=B, T=T, H=H, KV=KV, D=D, Dk=Dk, causal=causal, lens=lens, M=M:
              attn_ref(a[:, :D], a[:, D:D + Dk], a[:, D + Dk:], B, T, H, KV, causal, 1 / np.sqrt(hd), lens, drop=M),
              [r(B * T, D + 2 * Dk)], tol=5e-4)

    # a checkpointed block recomputes with the same masks it drew in the forward pass
    x0, w0, R = r(48, 96), r(96, 96), r(48, 96)

    def f(x, w):
        return nn.dropout(nn.gelu(nn.dropout(nn.linear(x, w), 0.4)), 0.3)
    grads = []
    for ck in (False, True):
        nn.manual_seed(5)
        x, w = nn.tensor(x0, requires_grad=True), nn.tensor(w0, requires_grad=True)
        y = nn.checkpoint(f, [x, w]) if ck else f(x, w)
        y2 = nn.dropout(y, 0.5)  # a draw after the checkpoint: the recompute must not shift it
        nn.sum_all(nn.mul(y2, nn.tensor(R))).backward()
        grads.append((y2.numpy(), x.grad.numpy(), w.grad.numpy()))
    e = max(rel_err(a, b) for a, b in zip(*grads))
    report("checkpoint replays dropout masks", e < 1e-6, f"checkpointed vs not {e:.1e}")

    model = nn.GPT(vocab=64, dim=64, heads=2, layers=2, seq_len=16, seed=2, dropout=0.3)
    ids = (np.arange(16 * 4) % 64).reshape(4, 16)
    t1, t2 = model(ids).numpy(), model(ids).numpy()
    model.eval()
    e1, e2 = model(ids).numpy(), model(ids).numpy()
    model.train()
    report("eval mode turns dropout off", not np.array_equal(t1, t2) and np.array_equal(e1, e2),
           "training passes differ, eval passes repeat")


def rope_ref(x, freq, T, groups):
    hd = x.shape[1] // groups
    half = hd // 2
    pos = (torch.arange(x.shape[0]) % T).double()[:, None]
    ang = pos * torch.as_tensor(freq, dtype=torch.float64)[None, :]
    c, s = torch.cos(ang), torch.sin(ang)
    outs = []
    for g in range(groups):
        a, b = x[:, g * hd:g * hd + half], x[:, g * hd + half:(g + 1) * hd]
        outs += [a * c - b * s, b * c + a * s]
    return torch.cat(outs, 1)


def main():
    t0 = time.time()
    nn.ctx()
    print(f"rawgpu.nn on {nn.ctx().dev.info['name']}, kernels compiled in {time.time() - t0:.1f} s\n")
    T_ = torch

    print("products")
    for ta in (False, True):
        for tb in (False, True):
            M, N, K = 37, 129, 70
            a = r(K, M) if ta else r(M, K)
            b = r(N, K) if tb else r(K, N)
            check(f"matmul {'t' if ta else 'n'}{'t' if tb else 'n'} 37x129x70",
                  lambda x, y, ta=ta, tb=tb: nn.matmul(x, y, ta, tb, alpha=0.5),
                  lambda x, y, ta=ta, tb=tb: 0.5 * (x.T if ta else x) @ (y.T if tb else y), [a, b])
    check("matmul nn 200x256x384 (padded rows)", lambda x, y: nn.matmul(x, y), lambda x, y: x @ y,
          [r(200, 384), r(384, 256)])
    check("linear with bias, 300x256 -> 192", lambda x, w, b: nn.linear(x, w, b),
          lambda x, w, b: x @ w.T + b, [r(300, 256), r(192, 256), r(192)])

    print("element-wise")
    x, y = r(70, 130), r(70, 130)
    check("add", lambda a, b: a + b, lambda a, b: a + b, [x, y])
    check("add, the same tensor twice", lambda a: nn.add(a, a), lambda a: a + a, [x])
    check("add a row (bias)", lambda a, b: a + b, lambda a, b: a + b, [x, r(1, 130)])
    check("add a column", lambda a, b: a + b, lambda a, b: a + b, [x, r(70, 1)])
    check("add a scalar tensor", lambda a, b: a + b, lambda a, b: a + b, [x, r(1, 1)])
    check("sub, alpha/beta", lambda a, b: nn.add(a, b, 2.0, -0.5), lambda a, b: 2 * a - 0.5 * b, [x, y])
    check("mul", lambda a, b: a * b, lambda a, b: a * b, [x, y])
    check("mul a row", lambda a, b: a * b, lambda a, b: a * b, [x, r(1, 130)])
    check("mul a column", lambda a, b: a * b, lambda a, b: a * b, [x, r(70, 1)])
    check("div", lambda a, b: a / b, lambda a, b: a / b, [x, r(70, 130, lo=0.5)])
    check("div by a column", lambda a, b: a / b, lambda a, b: a / b, [x, r(70, 1, lo=0.5)])
    check("affine 3x - 1, then 1 - x", lambda a: 1 - (a * 3 - 1), lambda a: 1 - (a * 3 - 1), [x])
    acts = [("relu", nn.relu, T_.relu), ("gelu", nn.gelu, lambda a: torch.nn.functional.gelu(a, approximate="tanh")),
            ("silu", nn.silu, torch.nn.functional.silu), ("tanh", nn.tanh, T_.tanh),
            ("sigmoid", nn.sigmoid, T_.sigmoid), ("exp", nn.exp, T_.exp), ("square", nn.square, T_.square),
            ("softplus", nn.softplus, torch.nn.functional.softplus), ("abs", nn.abs, T_.abs)]
    for name, f, g in acts:
        check(name, f, g, [x])
    check("log", nn.log, T_.log, [r(70, 130, lo=0.1)])
    check("sqrt", nn.sqrt, T_.sqrt, [r(70, 130, lo=0.1)])

    print("reductions and views")
    check("sum_all", lambda a: nn.sum_all(a), lambda a: a.sum().view(1, 1), [r(333, 77)])
    check("mean_all", lambda a: nn.mean_all(a), lambda a: a.mean().view(1, 1), [r(5, 300)])
    check("sum_rows (mean), 1000 rows", lambda a: nn.sum_rows(a, mean=True), lambda a: a.mean(0, keepdim=True),
          [r(1000, 96)])
    check("sum_cols", lambda a: nn.sum_cols(a), lambda a: a.sum(1, keepdim=True), [r(50, 300)])
    check("slice_cols, then concat", lambda a: nn.concat_cols([nn.slice_cols(a, 64, 128), nn.slice_cols(a, 0, 10)]),
          lambda a: torch.cat([a[:, 64:128], a[:, :10]], 1), [r(40, 192)])
    check("slice_rows, then concat_rows", lambda a: nn.concat_rows([nn.slice_rows(a, 5, 9), nn.slice_rows(a, 0, 2)]),
          lambda a: torch.cat([a[5:9], a[:2]], 0), [r(12, 130)])
    check("reshape (padded rows)", lambda a: nn.reshape(a, 16, 64 * 8), lambda a: a.reshape(16, 512), [r(128, 64)])
    check("reshape [8, 6] -> [4, 12]", lambda a: nn.reshape(a, 4, 12) * 2, lambda a: a.reshape(4, 12) * 2, [r(8, 6)])

    print("normalization")
    check("rmsnorm, weight", lambda a, w: nn.rmsnorm(a, w), lambda a, w: a * torch.rsqrt(
        (a * a).mean(1, keepdim=True) + 1e-6) * w, [r(77, 512), r(512)])
    check("rmsnorm, 1 + weight", lambda a, w: nn.rmsnorm(a, w, plus_one=True), lambda a, w: a * torch.rsqrt(
        (a * a).mean(1, keepdim=True) + 1e-6) * (1 + w), [r(30, 96), r(96)])

    def grouped(a, w, G):
        n, c = a.shape
        x = a.reshape(n, G, c // G)
        y = x * torch.rsqrt((x * x).mean(2, keepdim=True) + 1e-6)
        return (y * w if w is not None else y).reshape(n, c)
    check("rmsnorm, 4 groups (per head)", lambda a, w: nn.rmsnorm(a, w, groups=4), lambda a, w: grouped(a, w, 4),
          [r(40, 4 * 256), r(256)])
    check("rmsnorm, no weight, on a column view",
          lambda a: nn.rmsnorm(nn.slice_cols(a, 128, 384), None, groups=2),
          lambda a: grouped(a[:, 128:384], None, 2), [r(20, 512)])
    check("layernorm", lambda a, w, b: nn.layernorm(a, w, b),
          lambda a, w, b: torch.nn.functional.layer_norm(a, (a.shape[1],), w, b, 1e-5), [r(90, 384), r(384), r(384)])
    check("l2norm", nn.l2norm, lambda a: torch.nn.functional.normalize(a, dim=1), [r(50, 768)])
    check("softmax", lambda a: nn.softmax(a, 0.5), lambda a: torch.softmax(0.5 * a, 1), [r(33, 1000)])

    print("losses")
    lab = rng.integers(0, 101, 64).astype(np.int32)
    lab[[3, 7]] = -1
    keep = torch.as_tensor(lab >= 0)

    def ce_ref(a, smoothing=0.0):
        lt = torch.as_tensor(np.where(lab < 0, 0, lab)).long()
        lp = torch.log_softmax(a, 1)
        nll = -lp.gather(1, lt[:, None])[:, 0]
        loss = (1 - smoothing) * nll - smoothing * lp.mean(1)
        return (loss * keep).sum().view(1, 1) / keep.sum()
    check("cross_entropy, 2 ignored rows", lambda a: nn.cross_entropy(a, lab), ce_ref, [r(64, 101)])
    check("cross_entropy, label smoothing 0.1", lambda a: nn.cross_entropy(a, lab, smoothing=0.1),
          lambda a: ce_ref(a, 0.1), [r(64, 101)])
    tgt = rng.random((40, 3)).astype(np.float32)
    check("bce_with_logits", lambda a: nn.bce_with_logits(a, tgt),
          lambda a: torch.nn.functional.binary_cross_entropy_with_logits(a, torch.as_tensor(tgt, dtype=torch.float64))
          .view(1, 1), [r(40, 3)])
    check("mse_loss", lambda a, b: nn.mse_loss(a, b), lambda a, b: ((a - b) ** 2).mean().view(1, 1),
          [r(31, 70), r(31, 70)])

    print("attention")
    for (B, T, H, KV, hd, causal, lens, window) in [
            (2, 40, 4, 4, 32, True, None, 0),
            (3, 130, 2, 2, 64, False, [130, 77, 5], 0),
            (2, 64, 4, 2, 32, False, [64, 50], 0),
            (2, 200, 4, 1, 32, False, [200, 150], 33),
            (1, 260, 3, 3, 64, True, None, 0)]:
        D, Dk = H * hd, KV * hd
        qkv = r(B * T, D + 2 * Dk)
        sc = 1.0 / np.sqrt(hd)
        check(f"attention B{B} T{T} H{H}/{KV} hd{hd}{' causal' if causal else ''}"
              f"{' lens' if lens else ''}{' window' if window else ''}",
              lambda a, B=B, T=T, H=H, KV=KV, D=D, Dk=Dk, causal=causal, lens=lens, window=window, sc=sc:
              nn.attention(nn.slice_cols(a, 0, D), nn.slice_cols(a, D, D + Dk), nn.slice_cols(a, D + Dk, D + 2 * Dk),
                           B, T, H, KV, causal, sc, lens, window),
              lambda a, B=B, T=T, H=H, KV=KV, D=D, Dk=Dk, causal=causal, lens=lens, window=window, sc=sc:
              attn_ref(a[:, :D], a[:, D:D + Dk], a[:, D + Dk:], B, T, H, KV, causal, sc, lens, window),
              [qkv], tol=5e-4)
    freq = nn.rope_frequencies(64)
    ft = nn.tensor(freq)
    check("rope, 3 heads of 64", lambda a: nn.rope(a, ft, 50, 3), lambda a: rope_ref(a, freq, 50, 3), [r(100, 192)])

    print("gathers and pooling")
    ids = rng.integers(0, 50, (7, 9)).astype(np.int32)
    ids[0, :20 % 9] = 3  # repeated ids: atomic adds
    check("embedding (repeated ids)", lambda w: nn.embedding(ids, w),
          lambda w: w[torch.as_tensor(ids).long().reshape(-1)].reshape(7, 9, 130), [r(50, 130)])
    lens = np.array([9, 4, 1, 7], np.int32)

    def pool_ref(a):
        a = a.view(4, 9, -1)
        m = (torch.arange(9)[None, :] < torch.as_tensor(lens)[:, None]).double()[..., None]
        return (a * m).sum(1) / torch.as_tensor(lens).double()[:, None]
    check("pool_mean", lambda a: nn.pool_mean(a, lens, 4, 9), pool_ref, [r(36, 200)])

    dropout_checks()

    print("optimizer")
    w0, g0 = r(130, 70), r(130, 70)
    b0, gb = r(70), r(70)
    p = nn.tensor(w0, requires_grad=True)
    pb = nn.tensor(b0, requires_grad=True)
    opt = nn.AdamW([p, pb], lr=1e-2, weight_decay=0.1, max_grad_norm=1.0)
    tp = torch.tensor(w0.astype(np.float64), requires_grad=True)
    tb = torch.tensor(b0.astype(np.float64), requires_grad=True)
    topt = torch.optim.AdamW([{"params": [tp], "weight_decay": 0.1}, {"params": [tb], "weight_decay": 0.0}], lr=1e-2)
    for step in range(3):
        p.grad, pb.grad = nn.tensor(g0 * (step + 1)), nn.tensor(gb * (step + 1))
        opt.step()
        tp.grad = torch.tensor(g0.astype(np.float64) * (step + 1))
        tb.grad = torch.tensor(gb.astype(np.float64) * (step + 1))
        torch.nn.utils.clip_grad_norm_([tp, tb], 1.0)
        topt.step()
    e = max(rel_err(p.numpy(), tp.detach().numpy()), rel_err(pb.numpy(), tb.detach().numpy()))
    ok = e < 1e-5
    print(f"  {'ok  ' if ok else 'FAIL'} {'AdamW, 3 steps, clipping, decay':<38} weights {e:.1e}, "
          f"grad norm {opt.grad_norm():.3f}")
    if not ok:
        FAILED.append("adamw")

    print("end to end")
    model = nn.GPT(vocab=64, dim=128, heads=4, layers=2, seq_len=32, seed=1)
    opt = nn.AdamW(model.parameters(), lr=3e-3, max_grad_norm=1.0)
    seq = (np.arange(33 * 16) * 7 % 64).reshape(16, 33)  # a learnable pattern
    first = None
    t0 = time.time()
    for step in range(60):
        logits = model(seq[:, :-1])
        loss = nn.cross_entropy(logits, seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    last = loss.item()
    dt = time.time() - t0
    ok = last < 0.25 * first
    print(f"  {'ok  ' if ok else 'FAIL'} {'tiny GPT learns a pattern':<38} loss {first:.3f} -> {last:.3f} "
          f"in 60 steps ({dt / 60 * 1000:.0f} ms/step, {nn.ctx().launches} launches)")
    if not ok:
        FAILED.append("gpt")
    mem = nn.ctx().memory()
    print(f"\nVRAM: peak {mem['peak'] >> 20} MB through the pool")
    print("ALL OK" if not FAILED else f"FAILED: {', '.join(FAILED)}")
    nn.close()
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
