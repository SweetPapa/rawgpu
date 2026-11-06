"""Modules: parameters, layers, and the transformer pieces small models are made of."""
import math

import numpy as np

from . import core as F
from .core import Tensor


class Parameter:
    """Marks a tensor a Module owns and trains. Use Module.param() to make one."""


def _param(a, name=None):
    t = F.tensor(np.asarray(a, np.float32), requires_grad=True, name=name)
    return t


class Module:
    """Holds parameters (tensors with requires_grad) and submodules as attributes, lists or dicts."""

    training = True

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    def named_parameters(self, prefix=""):
        seen = set()
        for name, v in self._children(prefix):
            if isinstance(v, Tensor):
                if v.requires_grad and id(v) not in seen:
                    seen.add(id(v))
                    yield name, v
            elif isinstance(v, Module):
                for n, p in v.named_parameters(name + "."):
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield n, p

    def _children(self, prefix):
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    yield f"{prefix}{k}.{i}", x
            elif isinstance(v, dict):
                for i, x in v.items():
                    yield f"{prefix}{k}.{i}", x
            else:
                yield prefix + k, v

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for _, v in self._children(""):
            if isinstance(v, Module):
                yield from v.modules()

    def train(self, on=True):
        for m in self.modules():
            m.training = on
        return self

    def eval(self):
        return self.train(False)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        return self

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def state_dict(self):
        return {n: p.numpy() for n, p in self._all_tensors()}

    def _all_tensors(self, prefix=""):
        for name, v in self._children(prefix):
            if isinstance(v, Tensor) and v.dtype == F.F32:
                yield name, v
            elif isinstance(v, Module):
                yield from v._all_tensors(name + ".")

    def load_state_dict(self, d, strict=True):
        own = dict(self._all_tensors())
        missing = [n for n in own if n not in d]
        if strict and missing:
            raise KeyError(f"missing from the checkpoint: {missing[:5]}")
        for n, t in own.items():
            if n in d:
                a = np.asarray(d[n], np.float32)
                if a.size != t.size:
                    raise ValueError(f"{n}: checkpoint has {a.shape}, the model {t.shape}")
                F.upload(a.reshape(t.shape), t)

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        with np.load(path) as d:
            self.load_state_dict({k: d[k] for k in d.files})
        return self


def _init(rng, fan_in, shape, std=None):
    std = std if std is not None else 1.0 / math.sqrt(fan_in)
    return (rng.standard_normal(shape) * std).astype(np.float32)


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, rng=None, std=None, zero=False):
        rng = rng or np.random.default_rng()
        w = np.zeros((n_out, n_in), np.float32) if zero else _init(rng, n_in, (n_out, n_in), std)
        self.weight = _param(w)
        self.bias = _param(np.zeros(n_out, np.float32)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Embedding(Module):
    def __init__(self, n, dim, rng=None, std=0.02):
        rng = rng or np.random.default_rng()
        self.weight = _param(_init(rng, dim, (n, dim), std))

    def forward(self, ids):
        return F.embedding(ids, self.weight)


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, bias=True):
        self.weight = _param(np.ones(dim, np.float32))
        self.bias = _param(np.zeros(dim, np.float32)) if bias else None
        self.eps = eps

    def forward(self, x):
        return F.layernorm(x, self.weight, self.bias, self.eps)


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = _param(np.ones(dim, np.float32))
        self.eps = eps

    def forward(self, x):
        return F.rmsnorm(x, self.weight, self.eps)


ACTS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu, "tanh": F.tanh, "none": lambda x: x}


class MLP(Module):
    """Linear layers with an activation between them (and LayerNorm, with norm=True)"""

    def __init__(self, sizes, act="gelu", norm=False, rng=None, final_act=False):
        rng = rng or np.random.default_rng()
        self.layers = [Linear(a, b, rng=rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.norms = [LayerNorm(b) for b in sizes[1:-1]] if norm else []
        self.act, self.final_act = act, final_act

    def forward(self, x):
        n = len(self.layers)
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < n - 1:
                if self.norms:
                    x = self.norms[i](x)
                x = ACTS[self.act](x)
            elif self.final_act:
                x = ACTS[self.act](x)
        return x


def rope_frequencies(hd, theta=10000.0):
    return (1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))).astype(np.float32)


class SelfAttention(Module):
    """Multi-head self-attention over [batch * T, dim], one fused q/k/v projection, rotary positions"""

    def __init__(self, dim, heads, causal=True, rng=None, rope=True, out_std=None):
        rng = rng or np.random.default_rng()
        self.heads, self.dim, self.causal = heads, dim, causal
        self.hd = dim // heads
        self.qkv = Linear(dim, 3 * dim, bias=False, rng=rng)
        self.proj = Linear(dim, dim, bias=False, rng=rng, std=out_std)
        self._freq = F.tensor(rope_frequencies(self.hd)) if rope else None

    def forward(self, x, batch, T, lens=None):
        qkv = self.qkv(x)
        D = self.dim
        q, k, v = F.slice_cols(qkv, 0, D), F.slice_cols(qkv, D, 2 * D), F.slice_cols(qkv, 2 * D, 3 * D)
        if self._freq is not None:
            q, k = F.rope(q, self._freq, T, self.heads), F.rope(k, self._freq, T, self.heads)
        a = F.attention(q, k, v, batch, T, self.heads, causal=self.causal, lens=lens)
        return self.proj(a)


class Block(Module):
    """A pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))"""

    def __init__(self, dim, heads, mlp_ratio=4, causal=True, rng=None, n_layers=1):
        rng = rng or np.random.default_rng()
        out_std = 1.0 / math.sqrt(dim) / math.sqrt(2 * n_layers)  # GPT-2's scaled residual projections
        self.ln1, self.ln2 = LayerNorm(dim), LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, causal, rng, out_std=out_std)
        self.fc = Linear(dim, mlp_ratio * dim, rng=rng)
        self.out = Linear(mlp_ratio * dim, dim, rng=rng, std=out_std / math.sqrt(mlp_ratio))

    def forward(self, x, batch, T, lens=None):
        x = x + self.attn(self.ln1(x), batch, T, lens)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class Transformer(Module):
    """A stack of blocks over token ids or vectors, with a final LayerNorm"""

    def __init__(self, dim, heads, layers, causal=True, mlp_ratio=4, rng=None):
        rng = rng or np.random.default_rng()
        self.blocks = [Block(dim, heads, mlp_ratio, causal, rng, layers) for _ in range(layers)]
        self.ln_f = LayerNorm(dim)

    def forward(self, x, batch, T, lens=None):
        for b in self.blocks:
            x = b(x, batch, T, lens)
        return self.ln_f(x)


class GPT(Module):
    """A small decoder-only language model over a byte or token vocabulary, the head tied to the embedding"""

    def __init__(self, vocab, dim=256, heads=4, layers=4, seq_len=256, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab, self.dim, self.seq_len = vocab, dim, seq_len
        self.tok = Embedding(vocab, dim, rng)
        self.body = Transformer(dim, heads, layers, True, rng=rng)

    def forward(self, ids):
        """ids [batch, T] -> logits [batch * T, vocab]"""
        ids = np.asarray(ids)
        B, T = ids.shape
        h = self.body(self.tok(ids), B, T)
        return F.matmul(h, self.tok.weight, tb=True)

    def generate(self, prompt_ids, n, temperature=0.8, top_k=40, rng=None):
        rng = rng or np.random.default_rng()
        ids = list(prompt_ids)
        with F.no_grad():
            for _ in range(n):
                ctx = ids[-self.seq_len:]
                logits = self.forward(np.array([ctx])).numpy()[-1].astype(np.float64)
                logits /= max(temperature, 1e-4)
                if top_k:
                    cut = np.sort(logits)[-min(top_k, len(logits))]
                    logits[logits < cut] = -np.inf
                p = np.exp(logits - logits.max())
                p /= p.sum()
                ids.append(int(rng.choice(len(p), p=p)))
        return ids
