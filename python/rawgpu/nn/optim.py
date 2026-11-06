"""Optimizers, gradient clipping, learning-rate schedules and weight averaging, all on the GPU."""
import math

from . import core as F

PARTS = 64  # work-groups per tensor for the gradient norm's partial sums


class _Clip:
    """The global gradient norm and the clipping factor, computed on the GPU: [factor, norm]"""

    def __init__(self, n):
        self.parts = F.empty((n * PARTS, 1))
        self.out = F.zeros((1, 2))

    def run(self, params, max_norm):
        k = F.ctx().k
        live = [p for p in params if p.grad is not None]
        for i, p in enumerate(live):
            g = p.grad
            k["sumsq_parts"](g.arg, g.ld, g.rows, g.cols, (self.parts.mem.h, i * PARTS * 4), g=PARTS * 256, l=256)
        k["clip_factor"](self.parts.arg, len(live) * PARTS, float(max_norm or 0.0), self.out.arg, g=256, l=256)


class Optimizer:
    def __init__(self, params, lr, max_grad_norm=None):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.max_grad_norm, self.t = lr, max_grad_norm, 0
        self._clip = _Clip(len(self.params))

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def grad_norm(self):
        """The last step's gradient norm, before clipping (reads the GPU, so it waits)"""
        return float(self._clip.out.numpy().reshape(-1)[1])

    def _prepare(self):
        self.t += 1
        self._clip.run(self.params, self.max_grad_norm)
        return self._clip.out.arg


class AdamW(Optimizer):
    """Adam with decoupled weight decay. Weight decay applies to matrices only (not biases or norm scales),
    unless decay_vectors. A step whose gradient norm is not finite is skipped on the GPU."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, max_grad_norm=None,
                 decay_vectors=False):
        super().__init__(params, lr, max_grad_norm)
        self.b1, self.b2 = betas
        self.eps, self.wd, self.decay_vectors = eps, weight_decay, decay_vectors
        self.m = [F.zeros(p.shape) for p in self.params]
        self.v = [F.zeros(p.shape) for p in self.params]

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        gs = self._prepare()
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        k = F.ctx().k["adamw"]
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if g is None:
                continue
            wd = self.wd if (p.rows > 1 or self.decay_vectors) else 0.0
            k(p.arg, g.arg, m.arg, v.arg, p.ld, g.ld, p.rows, p.cols, float(lr), float(self.b1), float(self.b2),
              float(self.eps), float(wd), float(bc1), float(bc2), gs, g=F._grid(p.size), l=256)

    def state(self):
        return {"t": self.t, "m": [m.numpy() for m in self.m], "v": [v.numpy() for v in self.v]}

    def load_state(self, s):
        self.t = int(s["t"])
        for t, a in zip(self.m, s["m"]):
            F.upload(a, t)
        for t, a in zip(self.v, s["v"]):
            F.upload(a, t)


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0, max_grad_norm=None):
        super().__init__(params, lr, max_grad_norm)
        self.momentum, self.wd = momentum, weight_decay
        self.m = [F.zeros(p.shape) for p in self.params] if momentum else [None] * len(self.params)

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        gs = self._prepare()
        k = F.ctx().k["sgd"]
        for p, m in zip(self.params, self.m):
            g = p.grad
            if g is None:
                continue
            k(p.arg, g.arg, m.arg if m is not None else None, p.ld, g.ld, p.rows, p.cols, float(lr),
              float(self.momentum), float(self.wd), gs, g=F._grid(p.size), l=256)


def cosine_lr(step, total, peak, warmup=0, floor=0.1):
    """Linear warmup to `peak`, then cosine decay to floor * peak at `total`"""
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def ema_update(target, source, decay):
    """target = decay * target + (1 - decay) * source, parameter by parameter (modules or lists)"""
    tp = target.parameters_all() if hasattr(target, "parameters_all") else _tensors(target)
    sp = source.parameters_all() if hasattr(source, "parameters_all") else _tensors(source)
    k = F.ctx().k["ema"]
    for t, s in zip(tp, sp):
        k(t.arg, t.ld, s.arg, s.ld, t.rows, t.cols, float(decay), g=F._grid(t.size), l=256)


def _tensors(m):
    if isinstance(m, (list, tuple)):
        return list(m)
    return [t for _, t in m._all_tensors()]


def copy_weights(target, source):
    """target's weights = source's (for an EMA target encoder's start)"""
    for t, s in zip(_tensors(target), _tensors(source)):
        F._axpby(s, t, 1.0, 0.0)
