"""Tensors with reverse-mode autograd on the GPU: the device context, memory, and every op.

A Tensor is 2-D in memory: `rows` x `cols` float32 (or int32), rows `ld` elements apart, at element
`off` of a buffer. Its `shape` may have more dimensions; they fold into the rows (the last one is the
columns). Column slices are views (same memory, a different offset), which is how attention reads its
q, k and v out of one fused projection.

Ops run asynchronously and in order on the GPU. Nothing waits for the GPU until a tensor is read back
(.numpy(), .item()) or uploaded with a blocking write, so a training step queues all its work and the
host is free to prepare the next batch.
"""
import contextlib
import ctypes
import math
import os

import numpy as np

import rawgpu
from rawgpu import _L

_HERE = os.path.dirname(os.path.abspath(__file__))
F32, I32 = np.dtype(np.float32), np.dtype(np.int32)

# op codes of kernels.cl's unary()
U_AFFINE, U_RELU, U_GELU, U_SILU, U_TANH, U_SIGMOID, U_EXP, U_LOG, U_SQUARE, U_SQRT, U_SOFTPLUS, U_ABS = range(12)


def _sgemm_source():
    for p in (os.path.join(_HERE, "sgemm.cl"),                                  # a release copies it here
              os.path.join(_HERE, "..", "..", "..", "kernels", "sgemm.cl")):    # a checkout
        if os.path.exists(p):
            return open(p).read()
    raise rawgpu.Error("kernels/sgemm.cl not found")


def pitch(cols):
    """The row pitch for a new tensor of `cols` columns: rows a multiple of 64 floats wide get 32 more, since
    the matrix products run several times slower on rows a large power of two apart."""
    return cols + 32 if cols >= 128 and cols % 64 == 0 else cols


# ---------------------------------------------------------------------------------------------- kernels

class _Kernel:
    """A kernel with its argument setters prepared once (rawgpu.Kernel checks every argument's type on every
    call, which is slow for the thousands of small launches of a training step)."""
    _ctypes = {"float": ctypes.c_float, "uint": ctypes.c_uint32, "int": ctypes.c_int32, "ulong": ctypes.c_uint64,
               "long": ctypes.c_int64}

    def __init__(self, ctx, k):
        self.ctx, self.h, self.name = ctx, k._h, k.name
        self.L = _L()
        self.spec = []
        for i, (name, tname, size, kind) in enumerate(k.args):
            if kind == rawgpu.ARG_BUFFER:
                self.spec.append((i, None, 0))
            else:
                t = tname.replace("unsigned ", "u").strip()
                self.spec.append((i, self._ctypes[t], size))
        self.n = len(self.spec)

    def __call__(self, *args, g, l=None):
        L, h = self.L, self.h
        if len(args) != self.n:
            raise rawgpu.Error(f"{self.name} takes {self.n} arguments, got {len(args)}")
        for (i, ct, size), a in zip(self.spec, args):
            if ct is None:
                if a is None:
                    r = L.rawgpu_set_arg_buf_offset(h, i, None, 0)
                else:
                    r = L.rawgpu_set_arg_buf_offset(h, i, a[0], a[1])
            else:
                r = L.rawgpu_set_arg(h, i, size, ctypes.byref(ct(a)))
            if r:
                raise self.ctx.dev._err()
        if isinstance(g, int):
            g = (g,)
        dims = len(g)
        G = (ctypes.c_size_t * dims)(*g)
        Lp = (ctypes.c_size_t * dims)(*((l,) if isinstance(l, int) else l)) if l is not None else None
        e = L.rawgpu_launch(self.ctx.dev._h, h, dims, G, Lp)
        if not e:
            raise self.ctx.dev._err()
        self.ctx.last_event = e
        self.ctx.launches += 1
        return e


class _Mem:
    """A block of VRAM inside a chunk, back to the pool when the last tensor using it goes."""
    __slots__ = ("pool", "chunk", "h", "base", "n", "__weakref__")

    def __init__(self, pool, chunk, base, n):
        self.pool, self.chunk, self.h, self.base, self.n = pool, chunk, chunk.h, base, n

    def __del__(self):
        try:
            self.pool.give_back(self)
        except Exception:
            pass


class _Chunk:
    """One large rawgpu allocation, carved into blocks: a free list of (offset, size) extents, sorted and
    coalesced."""
    __slots__ = ("h", "size", "free", "used")

    def __init__(self, h, size):
        self.h, self.size, self.free, self.used = h, size, [[0, size]], 0

    def take(self, n):
        for i, e in enumerate(self.free):
            if e[1] >= n:
                off = e[0]
                if e[1] == n:
                    del self.free[i]
                else:
                    e[0] += n
                    e[1] -= n
                self.used += n
                return off
        return None

    def put(self, off, n):
        import bisect
        f = self.free
        i = bisect.bisect_left(f, [off, 0])
        self.used -= n
        if i > 0 and f[i - 1][0] + f[i - 1][1] == off:
            f[i - 1][1] += n
            if i < len(f) and f[i - 1][0] + f[i - 1][1] == f[i][0]:
                f[i - 1][1] += f[i][1]
                del f[i]
        elif i < len(f) and off + n == f[i][0]:
            f[i][0], f[i][1] = off, f[i][1] + n
        else:
            f.insert(i, [off, n])


class _Pool:
    """VRAM for tensors. rawgpu's own allocator aligns every buffer to 64 KB and tracks a few hundred free
    extents, which suits a model's weights but not the thousands of small tensors of a training step, so
    the pool takes VRAM in chunks of CHUNK bytes and carves them itself.

    Freed blocks go to a cache by exact size first: training repeats the same shapes every step, so after
    the first step nearly every tensor comes from there. When a chunk is full the cache is folded back into
    the chunks' free lists (coalescing), and only then is a new chunk allocated. Reuse needs no waiting:
    launches run in order, so a kernel that writes a reused block runs after every kernel that read it."""
    CHUNK = 256 << 20

    def __init__(self, dev):
        self.dev, self.chunks, self.cache = dev, [], {}
        self.in_use = self.cached = self.peak = 0

    def alloc(self, nbytes):
        n = max(256, (int(nbytes) + 255) & ~255)
        lst = self.cache.get(n)
        if lst:
            chunk, off = lst.pop()
            self.cached -= n
        else:
            chunk, off = self._carve(n)
        self.in_use += n
        self.peak = max(self.peak, self.in_use)
        return _Mem(self, chunk, off, n)

    def _carve(self, n):
        for c in self.chunks:
            off = c.take(n)
            if off is not None:
                return c, off
        if self.cached:
            self.flush()
            for c in self.chunks:
                off = c.take(n)
                if off is not None:
                    return c, off
        size = max(self.CHUNK, (n + 0xffff) & ~0xffff)
        L = _L()
        h = L.rawgpu_alloc(self.dev._h, size, rawgpu.MEM_VRAM)
        if not h:
            self.release()
            h = L.rawgpu_alloc(self.dev._h, size, rawgpu.MEM_VRAM)
            if not h and size > n:  # the last of the card: a chunk just big enough
                size = (n + 0xffff) & ~0xffff
                h = L.rawgpu_alloc(self.dev._h, size, rawgpu.MEM_VRAM)
            if not h:
                raise MemoryError(f"out of VRAM allocating {n / 2**20:.1f} MB "
                                  f"({self.in_use >> 20} MB in use by tensors)")
        c = _Chunk(h, size)
        self.chunks.append(c)
        return c, c.take(n)

    def give_back(self, mem):
        if self.dev._h is None:
            return
        self.cache.setdefault(mem.n, []).append((mem.chunk, mem.base))
        self.in_use -= mem.n
        self.cached += mem.n

    def flush(self):
        """Fold the cache back into the chunks' free lists"""
        for n, lst in self.cache.items():
            for chunk, off in lst:
                chunk.put(off, n)
        self.cache.clear()
        self.cached = 0

    def release(self):
        """Return chunks with no tensors in them to the driver"""
        self.flush()
        L = _L()
        keep = []
        for c in self.chunks:
            if c.used == 0:
                L.rawgpu_free(self.dev._h, c.h)
            else:
                keep.append(c)
        self.chunks = keep

    def reserved(self):
        return sum(c.size for c in self.chunks)


class _Staging:
    """Uploads without waiting for the GPU: a ring of host buffers the GPU copies from (a blocking write,
    rawgpu_write, would first wait for every queued launch). Slots are SLOT bytes; a larger upload goes
    through the ring in pieces of whole rows. The GPU reaches host memory through a window of limited
    size, so the ring stays small."""
    SLOT = 8 << 20

    def __init__(self, ctx, slots=6):
        self.ctx, self.slots, self.i = ctx, [[None, 0] for _ in range(slots)], 0  # buffer, last event

    def _slot(self):
        slot = self.slots[self.i]
        self.i = (self.i + 1) % len(self.slots)
        if slot[1]:
            self.ctx.dev.wait(slot[1])
        if slot[0] is None:
            slot[0] = rawgpu.Buffer(self.ctx.dev, self.SLOT, np.uint8, host=True)
        return slot

    def upload(self, a, dst):
        rows, cols = dst.rows, dst.cols
        a = np.ascontiguousarray(a).reshape(rows, cols)
        row_bytes = cols * 4
        if row_bytes > self.SLOT:  # rows too wide for a slot: blocking writes, row by row
            L = _L()
            self.ctx.dev.finish()
            for r in range(rows):
                if L.rawgpu_write(self.ctx.dev._h, dst.mem.h, dst.mem.base + (dst.off + r * dst.ld) * 4,
                                  a[r].ctypes.data, row_bytes):
                    raise self.ctx.dev._err()
            return
        per = max(1, self.SLOT // row_bytes)
        for r0 in range(0, rows, per):
            n = min(per, rows - r0)
            slot = self._slot()
            view = slot[0].host_view()
            view[:n * row_bytes] = a[r0:r0 + n].reshape(-1).view(np.uint8)
            slot[1] = self.ctx.k["axpby"]((slot[0]._h, 0), cols, (dst.mem.h, dst.mem.base + (dst.off + r0 * dst.ld) * 4),
                                          dst.ld, n, cols, 1.0, 0.0, 1, g=_grid(n * cols), l=256)


class Context:
    """The GPU, its compiled kernels, the memory pool and the staging ring."""

    def __init__(self, dev=None):
        self.dev = dev or rawgpu.Device()
        src = _sgemm_source() + "\n" + open(os.path.join(_HERE, "kernels.cl")).read()
        self.prog = self.dev.program(src)
        self.k = {n: _Kernel(self, self.prog.kernel(n)) for n in self.prog.kernel_names}
        self.pool = _Pool(self.dev)
        self.staging = _Staging(self)
        self.last_event = 0
        self.launches = 0

    def sync(self):
        self.dev.finish()

    def memory(self):
        """VRAM held by tensors, cached for reuse, the peak, and free on the card (bytes)."""
        return dict(in_use=self.pool.in_use, cached=self.pool.cached, peak=self.pool.peak,
                    reserved=self.pool.reserved(), free=self.dev.info["vram_free_bytes"])

    def close(self):
        self.pool.release()
        self.dev.close()


_ctx = None


def ctx(dev=None):
    """The training context, opening the GPU on first use (or using `dev`, a rawgpu.Device)."""
    global _ctx
    if _ctx is None:
        _ctx = Context(dev)
    return _ctx


def close():
    global _ctx
    if _ctx is not None:
        _ctx.close()
        _ctx = None


def _grid(n):
    return max(256, (int(n) + 255) // 256 * 256)


def _K(name):
    return ctx().k[name]


# ------------------------------------------------------------------------------------------------ tensor

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Run ops without recording them for backward (evaluation, inference, EMA targets)."""
    global _grad_enabled
    old, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = old


@contextlib.contextmanager
def enable_grad():
    global _grad_enabled
    old, _grad_enabled = _grad_enabled, True
    try:
        yield
    finally:
        _grad_enabled = old


class Tensor:
    __slots__ = ("mem", "off", "rows", "cols", "ld", "shape", "dtype", "requires_grad", "grad", "_back",
                 "_parents", "name", "__weakref__")

    def __init__(self, mem, rows, cols, ld=None, off=0, shape=None, dtype=F32, requires_grad=False, name=None):
        self.mem, self.rows, self.cols, self.off = mem, int(rows), int(cols), int(off)
        self.ld = int(ld if ld is not None else cols)
        self.shape = tuple(shape) if shape is not None else (self.rows, self.cols)
        self.dtype = np.dtype(dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self._back = None
        self._parents = ()
        self.name = name

    # -- basics
    @property
    def arg(self):
        return (self.mem.h, self.mem.base + self.off * 4)

    @property
    def size(self):
        return self.rows * self.cols

    def __repr__(self):
        g = ", requires_grad" if self.requires_grad else ""
        return f"<Tensor {self.shape} {self.dtype.name}{g}{' ' + self.name if self.name else ''}>"

    def numpy(self):
        """Copy to the host (waits for the GPU)."""
        if not self.rows or not self.cols:
            return np.empty(self.shape, self.dtype)
        n = (self.rows - 1) * self.ld + self.cols
        out = np.empty(self.rows * self.ld, self.dtype)
        if _L().rawgpu_read(_ctx.dev._h, self.mem.h, self.mem.base + self.off * 4, out.ctypes.data, n * 4):
            raise _ctx.dev._err()
        return out.reshape(self.rows, self.ld)[:, :self.cols].reshape(self.shape).copy()

    def item(self):
        return float(self.numpy().reshape(-1)[0])

    def detach(self):
        return Tensor(self.mem, self.rows, self.cols, self.ld, self.off, self.shape, self.dtype)

    def zero_grad(self):
        self.grad = None

    # -- autograd
    def backward(self, grad=None):
        """Gradients of this tensor (a scalar unless `grad` is given) into every tensor that requires one.
        Intermediate gradients and the graph are freed as they are used; leaves keep .grad."""
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            t, done = stack.pop()
            if done:
                topo.append(t)
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for p in t._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        if grad is None:
            if self.size != 1:
                raise rawgpu.Error("backward() without a gradient needs a scalar")
            grad = full((1, 1), 1.0)
        self.grad = grad
        for t in reversed(topo):
            if t._back is not None:
                if t.grad is not None:
                    t._back(t.grad)
                t._back, t._parents, t.grad = None, (), None

    # -- operators
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(self, o)
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return affine(self, -1.0, o) if _is_num(o) else sub(o, self)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(self, o)
    def __truediv__(self, o): return div(self, o)
    def __neg__(self): return affine(self, -1.0, 0.0)
    def __matmul__(self, o): return matmul(self, o)

    @property
    def T(self):
        raise rawgpu.Error("no transposed views: use matmul(a, b, ta=..., tb=...)")

    def reshape(self, *shape): return reshape(self, *shape)
    def sum(self): return sum_all(self)
    def mean(self): return mean_all(self)


def _is_num(x):
    return isinstance(x, (int, float, np.floating, np.integer))


def _shape2(shape):
    shape = (int(shape),) if _is_num(shape) else tuple(int(s) for s in shape)
    if len(shape) == 1:
        return shape, 1, shape[0]
    return shape, int(np.prod(shape[:-1])), shape[-1]


def empty(shape, dtype=F32, ld=None, name=None):
    shape, rows, cols = _shape2(shape)
    ld = ld if ld is not None else (pitch(cols) if np.dtype(dtype) == F32 else cols)
    mem = ctx().pool.alloc(max(rows, 1) * ld * 4)
    return Tensor(mem, rows, cols, ld, 0, shape, dtype, name=name)


def empty_like(t):
    return empty(t.shape, t.dtype)


def full(shape, value, dtype=F32):
    t = empty(shape, dtype)
    if np.dtype(dtype) == F32:
        _K("fill")(t.arg, t.ld, t.rows, t.cols, float(value), g=_grid(t.size), l=256)
    else:
        upload(np.full(t.shape, value, dtype), t)
    return t


def zeros(shape, dtype=F32):
    return full(shape, 0, dtype)


def zeros_like(t):
    return zeros(t.shape, t.dtype)


def ones(shape):
    return full(shape, 1.0)


def upload(a, dst):
    """Copy numpy array `a` into tensor `dst` without waiting for the GPU (through the staging ring)."""
    a = np.ascontiguousarray(a, dst.dtype).reshape(dst.rows, dst.cols)
    ctx().staging.upload(a, dst)
    return dst


def tensor(a, requires_grad=False, dtype=None, name=None):
    """A GPU tensor holding numpy array (or list) `a`: float32 unless it is an integer array (int32)."""
    a = np.asarray(a)
    if dtype is None:
        dtype = I32 if a.dtype.kind in "iub" else F32
    a = a.astype(dtype, copy=False)
    t = empty(a.shape if a.ndim else (1,), dtype, name=name)
    upload(a.reshape(t.shape), t)
    t.requires_grad = requires_grad
    return t


def _node(out, parents, back):
    if _grad_enabled and any(p is not None and p.requires_grad for p in parents):
        out.requires_grad = True
        out._parents = tuple(p for p in parents if p is not None and p.requires_grad)
        out._back = back
    return out


def _target(t):
    """Where to write t's gradient: (tensor, beta). beta 0 on a fresh gradient (kernels then overwrite it)."""
    if t.grad is None:
        t.grad = empty_like(t)
        return t.grad, 0.0
    return t.grad, 1.0


def _give(t, g, own=True):
    """Add gradient g to t.grad. If t has none yet and `own`, t takes g itself (the caller must not hand the
    same memory to another tensor). Returns whether it took g."""
    if not t.requires_grad:
        return False
    if t.grad is None:
        if own and g.rows == t.rows and g.cols == t.cols:
            t.grad = g if g.shape == t.shape else Tensor(g.mem, g.rows, g.cols, g.ld, g.off, t.shape)
            return True
        t.grad = empty_like(t)
        _axpby(g, t.grad, 1.0, 0.0)
        return False
    _axpby(g, t.grad, 1.0, 1.0)
    return False


def _axpby(x, y, alpha, beta):
    _K("axpby")(x.arg, x.ld, y.arg, y.ld, x.rows, x.cols, float(alpha), float(beta), 0, g=_grid(x.size), l=256)


def _bcast_mode(x, y):
    if y.rows == x.rows and y.cols == x.cols:
        return 0
    if y.rows == 1 and y.cols == 1:
        return 3
    if y.rows == 1 and y.cols == x.cols:
        return 1
    if y.cols == 1 and y.rows == x.rows:
        return 2
    raise rawgpu.Error(f"cannot broadcast {y.shape} to {x.shape}")


def _bcast(src, mode, dst, alpha, beta):
    _K("broadcast")(src.arg, src.ld, mode, dst.arg, dst.ld, dst.rows, dst.cols, float(alpha), float(beta),
                    g=_grid(dst.size), l=256)


def _colsum(x, out, alpha=1.0, beta=0.0):
    """out[1, cols] = beta * out + alpha * (sum of x's rows)"""
    if x.rows <= 256:
        _K("colsum")(x.arg, x.ld, x.rows, x.cols, x.rows, out.arg, out.ld, float(alpha), float(beta),
                     g=(_grid(x.cols), 1), l=(256, 1))
        return
    per = 64
    chunks = (x.rows + per - 1) // per
    tmp = empty((chunks, x.cols))
    _K("colsum")(x.arg, x.ld, x.rows, x.cols, per, tmp.arg, tmp.ld, 1.0, 0.0, g=(_grid(x.cols), chunks), l=(256, 1))
    _colsum(tmp, out, alpha, beta)


def _rowsum(x, out, alpha=1.0, beta=0.0, squares=False):
    """out[rows, 1] = beta * out + alpha * (sum of each row)"""
    _K("rowsum")(x.arg, x.ld, x.cols, out.arg, out.ld, float(alpha), float(beta), int(squares), g=x.rows * 64, l=64)


def _reduce_to(g, mode, like):
    """g summed down to the shape a broadcast operand had (mode as in _bcast_mode)"""
    if mode == 0:
        return g
    out = empty(like.shape)
    if mode == 1:
        _colsum(g, out)
    elif mode == 2:
        _rowsum(g, out)
    else:
        col = empty((g.rows, 1))
        _rowsum(g, col)
        _colsum(col, out)
    return out


# --------------------------------------------------------------------------------------------- products

def _gemm(ta, tb, M, N, K, alpha, A, B, beta, C):
    if M == 0 or N == 0:
        return
    if K == 0:
        if beta == 0:
            _K("fill")(C.arg, C.ld, C.rows, C.cols, 0.0, g=_grid(C.size), l=256)
        return
    name = "sgemm_" + ("t" if ta else "n") + ("t" if tb else "n")
    _K(name)(A.arg, B.arg, C.arg, M, N, K, A.ld, B.ld, C.ld, float(alpha), float(beta),
             g=((N + 127) // 128 * 16, (M + 127) // 128 * 16), l=(16, 16))


def matmul(a, b, ta=False, tb=False, alpha=1.0):
    """alpha * op(a) @ op(b), op = transpose where ta / tb"""
    M, K = (a.cols, a.rows) if ta else (a.rows, a.cols)
    K2, N = (b.cols, b.rows) if tb else (b.rows, b.cols)
    if K != K2:
        raise rawgpu.Error(f"matmul: inner dimensions differ ({K} and {K2}); shapes {a.shape} {b.shape}")
    out = empty((M, N))
    _gemm(ta, tb, M, N, K, alpha, a, b, 0.0, out)

    def back(g):
        if a.requires_grad:
            da, beta = _target(a)
            if not ta:
                _gemm(False, not tb, M, K, N, alpha, g, b, beta, da)
            else:
                _gemm(tb, True, K, M, N, alpha, b, g, beta, da)
        if b.requires_grad:
            db, beta = _target(b)
            if not tb:
                _gemm(not ta, False, K, N, M, alpha, a, g, beta, db)
            else:
                _gemm(True, ta, N, K, M, alpha, g, a, beta, db)
    return _node(out, (a, b), back)


def linear(x, w, b=None, alpha=1.0):
    """alpha * x @ w^T + b: x [n, in], w [out, in], b [out] (or None)"""
    if x.cols != w.cols:
        raise rawgpu.Error(f"linear: input has {x.cols} features, the weight takes {w.cols}")
    shape = x.shape[:-1] + (w.rows,)
    out = empty(shape)
    _gemm(False, True, x.rows, w.rows, x.cols, alpha, x, w, 0.0, out)
    if b is not None:
        _bcast(b, 1, out, 1.0, 1.0)

    def back(g):
        if x.requires_grad:
            dx, beta = _target(x)
            _gemm(False, False, x.rows, x.cols, w.rows, alpha, g, w, beta, dx)
        if w.requires_grad:
            dw, beta = _target(w)
            _gemm(True, False, w.rows, w.cols, x.rows, alpha, g, x, beta, dw)
        if b is not None and b.requires_grad:
            db, beta = _target(b)
            _colsum(g, db, 1.0, beta)
    return _node(out, (x, w, b), back)


# ------------------------------------------------------------------------------------------ element-wise

def _unary(x, op, a=1.0, b=0.0):
    out = empty(x.shape)
    _K("unary")(x.arg, x.ld, out.arg, out.ld, x.rows, x.cols, op, float(a), float(b), g=_grid(x.size), l=256)

    def back(g):
        dx, beta = _target(x)
        _K("unary_bwd")(x.arg, x.ld, out.arg, out.ld, g.arg, g.ld, dx.arg, dx.ld, x.rows, x.cols, op, float(a),
                        beta, g=_grid(x.size), l=256)
    return _node(out, (x,), back)


def affine(x, a=1.0, b=0.0): return _unary(x, U_AFFINE, a, b)
def scale(x, a): return _unary(x, U_AFFINE, a, 0.0)
def relu(x): return _unary(x, U_RELU)
def gelu(x): return _unary(x, U_GELU)
def silu(x): return _unary(x, U_SILU)
def tanh(x): return _unary(x, U_TANH)
def sigmoid(x): return _unary(x, U_SIGMOID)
def exp(x): return _unary(x, U_EXP)
def log(x): return _unary(x, U_LOG)
def square(x): return _unary(x, U_SQUARE)
def sqrt(x): return _unary(x, U_SQRT)
def softplus(x): return _unary(x, U_SOFTPLUS)
def abs(x): return _unary(x, U_ABS)  # noqa: A001


# ------------------------------------------------------------------------------------------------- dropout

_drop = [0x5EED, 0]  # the seed of the mask sequence, and how many masks it has given


def manual_seed(seed):
    """Restart dropout's sequence of masks from `seed` (each dropout call takes the next mask)."""
    _drop[0], _drop[1] = int(seed) & 0xFFFFFFFF, 0


def _next_seed():
    s = (_drop[0] * 0x9E3779B1 + _drop[1] * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
    _drop[1] += 1
    return s


def _dropout_into(x, y, p, seed, beta=0.0):
    """y = beta * y + x with the mask of (p, seed) applied (y may be x)"""
    cut = min(int(round(p * 2.0 ** 32)), 0xFFFFFFFF)
    scale_ = 0.0 if p >= 1 else 1.0 / (1.0 - p)
    _K("dropout")(x.arg, x.ld, y.arg, y.ld, x.rows, x.cols, cut, scale_, seed, float(beta), g=_grid(x.size), l=256)


def dropout(x, p, seed=None):
    """Inverted dropout: each element zeroed with probability p, the rest scaled by 1 / (1 - p), so the
    expected value is unchanged and evaluation needs no rescaling. Applies whenever called: modules skip it
    in eval mode. The mask is a hash of (seed, element) that the backward pass regenerates rather than
    stores; `seed` defaults to the next of nn's sequence (manual_seed restarts it)."""
    if not 0.0 <= p <= 1.0:
        raise rawgpu.Error(f"dropout: p must be in [0, 1], got {p}")
    if p == 0:
        return x
    seed = _next_seed() if seed is None else int(seed) & 0xFFFFFFFF
    out = empty(x.shape)
    _dropout_into(x, out, p, seed)

    def back(g):
        dx, beta = _target(x)
        _dropout_into(g, dx, p, seed, beta)
    return _node(out, (x,), back)


def _binary(x, y, op, alpha=1.0, beta=1.0):
    mode = _bcast_mode(x, y)
    out = empty(x.shape)
    _K("binary")(x.arg, x.ld, y.arg, y.ld, out.arg, out.ld, x.rows, x.cols, op, mode, float(alpha), float(beta), 0.0,
                 g=_grid(x.size), l=256)
    return out, mode


def add(x, y, alpha=1.0, beta=1.0):
    """alpha * x + beta * y; y may be a row [1, cols], a column [rows, 1], a scalar tensor or a number"""
    if _is_num(y):
        return affine(x, alpha, float(y) * beta)
    if _is_num(x):
        return affine(y, beta, float(x) * alpha)
    if y.size > x.size:
        return add(y, x, beta, alpha)
    out, mode = _binary(x, y, 0, alpha, beta)

    def back(g):
        took = False
        if x.requires_grad:
            if alpha == 1.0:
                took = _give(x, g)
            else:
                dx, b0 = _target(x)
                _axpby(g, dx, alpha, b0)
        if y.requires_grad:
            r = _reduce_to(g, mode, y)
            if beta == 1.0:
                _give(y, r, own=not (took and r is g))
            else:
                dy, b0 = _target(y)
                _axpby(r, dy, beta, b0)
    return _node(out, (x, y), back)


def sub(x, y):
    if _is_num(y):
        return affine(x, 1.0, -float(y))
    return add(x, y, 1.0, -1.0)


def mul(x, y):
    """x * y element-wise, y broadcast as in add()"""
    if _is_num(y):
        return scale(x, float(y))
    if _is_num(x):
        return scale(y, float(x))
    if y.size > x.size:
        x, y = y, x
    out, mode = _binary(x, y, 1)

    def back(g):
        if x.requires_grad:
            dx, b0 = _target(x)
            _K("binary")(g.arg, g.ld, y.arg, y.ld, dx.arg, dx.ld, x.rows, x.cols, 1, mode, 1.0, 0.0, b0,
                         g=_grid(x.size), l=256)
        if y.requires_grad:
            t, _ = _binary(g, x, 1)
            _give(y, _reduce_to(t, mode, y))
    return _node(out, (x, y), back)


def div(x, y):
    """x / y element-wise, y broadcast as in add()"""
    if _is_num(y):
        return scale(x, 1.0 / float(y))
    out, mode = _binary(x, y, 2)

    def back(g):
        if x.requires_grad:
            dx, b0 = _target(x)
            _K("binary")(g.arg, g.ld, y.arg, y.ld, dx.arg, dx.ld, x.rows, x.cols, 2, mode, 1.0, 0.0, b0,
                         g=_grid(x.size), l=256)
        if y.requires_grad:  # d/dy = -g * out / y
            t, _ = _binary(g, out, 1)
            t2, _ = _binary(t, y, 2, -1.0)
            _give(y, _reduce_to(t2, mode, y))
    return _node(out, (x, y), back)


# ----------------------------------------------------------------------------------------------- reductions

def sum_all(x):
    """The sum of every element: a [1, 1] tensor"""
    out = empty((1, 1))
    if x.rows == 1:
        _rowsum(x, out)
    else:
        col = empty((x.rows, 1))
        _rowsum(x, col)
        _colsum(col, out)

    def back(g):
        dx, beta = _target(x)
        _bcast(g, 3, dx, 1.0, beta)
    return _node(out, (x,), back)


def mean_all(x):
    return scale(sum_all(x), 1.0 / max(x.size, 1))


def sum_rows(x, mean=False):
    """Sum (or mean) over the rows: [1, cols]"""
    out = empty((1, x.cols))
    a = 1.0 / x.rows if mean else 1.0
    _colsum(x, out, a)

    def back(g):
        dx, beta = _target(x)
        _bcast(g, 1, dx, a, beta)
    return _node(out, (x,), back)


def sum_cols(x, mean=False):
    """Sum (or mean) over the columns: [rows, 1]"""
    out = empty((x.rows, 1))
    a = 1.0 / x.cols if mean else 1.0
    _rowsum(x, out, a)

    def back(g):
        dx, beta = _target(x)
        _bcast(g, 2, dx, a, beta)
    return _node(out, (x,), back)


# ----------------------------------------------------------------------------------------------- views

def slice_cols(x, c0, c1):
    """Columns [c0, c1) of x, sharing its memory"""
    out = Tensor(x.mem, x.rows, c1 - c0, x.ld, x.off + c0, x.shape[:-1] + (c1 - c0,), x.dtype)

    def back(g):
        if x.grad is None:
            x.grad = zeros_like(x)
        dst = Tensor(x.grad.mem, x.rows, c1 - c0, x.grad.ld, x.grad.off + c0)
        _axpby(g, dst, 1.0, 1.0)
    return _node(out, (x,), back)


def slice_rows(x, r0, r1):
    """Rows [r0, r1) of x, sharing its memory"""
    out = Tensor(x.mem, r1 - r0, x.cols, x.ld, x.off + r0 * x.ld, (r1 - r0, x.cols), x.dtype)

    def back(g):
        if x.grad is None:
            x.grad = zeros_like(x)
        dst = Tensor(x.grad.mem, r1 - r0, x.cols, x.grad.ld, x.grad.off + r0 * x.grad.ld)
        _axpby(g, dst, 1.0, 1.0)
    return _node(out, (x,), back)


def contiguous(x):
    """A copy with rows exactly `cols` apart"""
    out = empty(x.shape, x.dtype, ld=x.cols)
    _K("axpby")(x.arg, x.ld, out.arg, out.ld, x.rows, x.cols, 1.0, 0.0, int(x.dtype != F32), g=_grid(x.size), l=256)

    def back(g):
        _give(x, g)
    return _node(out, (x,), back)


def reshape(x, *shape):
    """The same elements in a new shape (copies first if the rows are padded)"""
    if len(shape) == 1 and not _is_num(shape[0]):
        shape = tuple(shape[0])
    shape = tuple(int(s) for s in shape)
    if -1 in shape:
        known = int(np.prod([s for s in shape if s != -1]))
        shape = tuple(x.size // known if s == -1 else s for s in shape)
    shape, rows, cols = _shape2(shape)
    if rows * cols != x.size:
        raise rawgpu.Error(f"cannot reshape {x.shape} to {shape}")
    if cols == x.cols:
        out = Tensor(x.mem, rows, cols, x.ld, x.off, shape, x.dtype)
    else:
        src = x if x.ld == x.cols else contiguous(x)  # a node of its own when it copies
        out = Tensor(src.mem, rows, cols, cols, src.off, shape, x.dtype)

        def back_flat(g):
            gc = g if g.ld == g.cols else contiguous(g)
            _give(src, Tensor(gc.mem, src.rows, src.cols, src.cols, gc.off, src.shape))
        return _node(out, (src,), back_flat)

    def back(g):
        _give(x, Tensor(g.mem, g.rows, g.cols, g.ld, g.off, x.shape))
    return _node(out, (x,), back)


def concat_cols(xs):
    """Join tensors with the same rows side by side"""
    rows = xs[0].rows
    cols = sum(t.cols for t in xs)
    out = empty(xs[0].shape[:-1] + (cols,))
    c = 0
    for t in xs:
        _axpby(t, Tensor(out.mem, rows, t.cols, out.ld, out.off + c), 1.0, 0.0)
        c += t.cols

    def back(g):
        c = 0
        for t in xs:
            if t.requires_grad:
                _give(t, Tensor(g.mem, rows, t.cols, g.ld, g.off + c))
            c += t.cols
    return _node(out, tuple(xs), back)


def concat_rows(xs):
    """Stack tensors with the same columns on top of each other"""
    cols = xs[0].cols
    rows = sum(t.rows for t in xs)
    out = empty((rows, cols))
    r = 0
    for t in xs:
        _axpby(t, Tensor(out.mem, t.rows, cols, out.ld, out.off + r * out.ld), 1.0, 0.0)
        r += t.rows

    def back(g):
        r = 0
        for t in xs:
            if t.requires_grad:
                _give(t, Tensor(g.mem, t.rows, cols, g.ld, g.off + r * g.ld, t.shape))
            r += t.rows
    return _node(out, tuple(xs), back)


# -------------------------------------------------------------------------------------------- normalization

def rmsnorm(x, w=None, eps=1e-6, groups=1, plus_one=False):
    """RMSNorm over each row, or over each of `groups` equal parts of a row (per-head norms); w [cols/groups]
    or None. plus_one scales by 1 + w (Gemma 1-3)."""
    dim = x.cols // groups
    out = empty(x.shape)
    n = x.rows * groups
    rstd = empty((n, 1))
    _K("rmsnorm_fwd")(x.arg, x.ld, w.arg if w is not None else None, out.arg, out.ld, rstd.arg, groups, dim,
                      float(eps), int(plus_one), g=n * 64, l=64)

    def back(g):
        dwp = empty((n, dim), ld=dim) if (w is not None and w.requires_grad) else None
        if x.requires_grad:
            dx, beta = _target(x)
            _K("rmsnorm_bwd")(x.arg, x.ld, w.arg if w is not None else None, rstd.arg, g.arg, g.ld, dx.arg, dx.ld,
                              dwp.arg if dwp is not None else None, groups, dim, int(plus_one), beta, g=n * 64, l=64)
        elif dwp is not None:
            tmp = empty(x.shape)
            _K("rmsnorm_bwd")(x.arg, x.ld, w.arg, rstd.arg, g.arg, g.ld, tmp.arg, tmp.ld, dwp.arg, groups, dim,
                              int(plus_one), 0.0, g=n * 64, l=64)
        if dwp is not None:
            dw, beta = _target(w)
            _colsum(dwp, dw, 1.0, beta)
    return _node(out, (x, w), back)


def layernorm(x, w=None, b=None, eps=1e-5):
    out = empty(x.shape)
    stats = empty((x.rows, 2), ld=2)
    _K("layernorm_fwd")(x.arg, x.ld, w.arg if w is not None else None, b.arg if b is not None else None, out.arg,
                        out.ld, stats.arg, x.cols, float(eps), g=x.rows * 64, l=64)

    def back(g):
        need_w = w is not None and w.requires_grad
        dwp = empty((x.rows, x.cols), ld=x.cols) if need_w else None
        if x.requires_grad or need_w:
            dx, beta = _target(x) if x.requires_grad else (empty(x.shape), 0.0)
            _K("layernorm_bwd")(x.arg, x.ld, w.arg if w is not None else None, stats.arg, g.arg, g.ld, dx.arg, dx.ld,
                                dwp.arg if dwp is not None else None, x.cols, beta, g=x.rows * 64, l=64)
        if need_w:
            dw, beta = _target(w)
            _colsum(dwp, dw, 1.0, beta)
        if b is not None and b.requires_grad:
            db, beta = _target(b)
            _colsum(g, db, 1.0, beta)
    return _node(out, (x, w, b), back)


def l2norm(x, eps=1e-12):
    """Each row scaled to unit length"""
    out = empty(x.shape)
    nrm = empty((x.rows, 1))
    _K("l2norm_fwd")(x.arg, x.ld, out.arg, out.ld, nrm.arg, x.cols, float(eps), g=x.rows * 64, l=64)

    def back(g):
        dx, beta = _target(x)
        _K("l2norm_bwd")(out.arg, out.ld, nrm.arg, g.arg, g.ld, dx.arg, dx.ld, x.cols, beta, g=x.rows * 64, l=64)
    return _node(out, (x,), back)


def softmax(x, scale_=1.0):
    """Softmax of each row (of scale_ * x)"""
    src = x if scale_ == 1.0 else scale(x, scale_)
    out = empty(x.shape)
    _K("softmax_fwd")(src.arg, src.ld, out.arg, out.ld, x.cols, g=x.rows * 64, l=64)

    def back(g):
        dx, beta = _target(src)
        _K("softmax_bwd")(out.arg, out.ld, g.arg, g.ld, dx.arg, dx.ld, x.cols, 1.0, beta, g=x.rows * 64, l=64)
    return _node(out, (src,), back)


# ------------------------------------------------------------------------------------------------- losses

def _labels(labels, rows):
    if isinstance(labels, Tensor):
        return labels, None
    a = np.asarray(labels, np.int32).reshape(rows, 1)
    return tensor(a, dtype=I32), a


def cross_entropy(logits, labels, smoothing=0.0, reduction="mean"):
    """Softmax cross entropy of each row against an integer label (-1: ignore the row). labels: a numpy
    array / list (counted on the host for the mean) or an int32 Tensor (then the mean is over all rows).
    reduction "mean", "sum", or "none" (per-row losses [rows, 1], no gradient)."""
    lab, host = _labels(labels, logits.rows)
    loss_r = empty((logits.rows, 1))
    lse = empty((logits.rows, 1))
    _K("ce_fwd")(logits.arg, logits.ld, lab.arg, logits.cols, loss_r.arg, lse.arg, float(smoothing),
                 g=logits.rows * 64, l=64)
    if reduction == "none":
        return loss_r
    valid = int((host >= 0).sum()) if host is not None else logits.rows
    s = 1.0 / max(valid, 1) if reduction == "mean" else 1.0
    out = empty((1, 1))
    _colsum(loss_r, out, s)

    def back(g):
        dx, beta = _target(logits)
        _K("ce_bwd")(logits.arg, logits.ld, lab.arg, lse.arg, g.arg, s, dx.arg, dx.ld, logits.rows, logits.cols,
                     float(smoothing), beta, g=_grid(logits.size), l=256)
    return _node(out, (logits,), back)


def bce_with_logits(logits, targets, reduction="mean"):
    """Binary cross entropy of sigmoid(logits) against targets in [0, 1] (same shape; -1 masks an element)"""
    if not isinstance(targets, Tensor):
        host = np.asarray(targets, np.float32).reshape(logits.rows, logits.cols)
        targets = tensor(host)
        valid = int((host >= 0).sum())
    else:
        valid = logits.size
    loss_e = empty(logits.shape)
    _K("bce_fwd")(logits.arg, logits.ld, targets.arg, targets.ld, loss_e.arg, loss_e.ld, logits.rows, logits.cols,
                  g=_grid(logits.size), l=256)
    s = 1.0 / max(valid, 1) if reduction == "mean" else 1.0
    with no_grad():
        out = sum_all(loss_e)
    if s != 1.0:
        _axpby(out, out, s, 0.0)

    def back(g):
        dx, beta = _target(logits)
        _K("bce_bwd")(logits.arg, logits.ld, targets.arg, targets.ld, g.arg, s, dx.arg, dx.ld, logits.rows,
                      logits.cols, beta, g=_grid(logits.size), l=256)
    return _node(out, (logits,), back)


def mse_loss(a, b):
    return mean_all(square(sub(a, b)))


# ---------------------------------------------------------------------------------------------- attention

def attention(q, k, v, batch, T, heads, kv_heads=None, causal=False, scale_=None, lens=None, window=0, dropout=0.0,
              seed=None):
    """Multi-head attention. q [batch * T, heads * hd], k and v [batch * T, kv_heads * hd] (column views of a
    fused projection work). Query head h reads key/value head h / (heads / kv_heads). `lens`: each
    sequence's length (an int32 Tensor or array; keys past it are masked out); `window`: keys `window` or
    more positions from the query are masked. `dropout`: the probability of dropping each attention
    weight after the softmax (the mask is that of nn.dropout on the [batch * heads * T, T] weights).
    Returns [batch * T, heads * hd]."""
    if not 0.0 <= dropout < 1.0:
        raise rawgpu.Error(f"attention: dropout must be in [0, 1), got {dropout}")
    if dropout > 0:
        seed = _next_seed() if seed is None else int(seed) & 0xFFFFFFFF
    kv_heads = kv_heads or heads
    hd = q.cols // heads
    group = heads // kv_heads
    scale_ = (1.0 / math.sqrt(hd)) if scale_ is None else scale_
    if lens is not None and not isinstance(lens, Tensor):
        lens = tensor(np.asarray(lens, np.int32))
    sp = pitch(T)
    S = empty((batch * heads * T, T), ld=sp)
    tiles = lambda n: (n + 127) // 128 * 16
    Tq = T * q.ld
    # scores
    _K("bgemm_nt")(q.arg, k.arg, S.arg, T, T, hd, q.ld, k.ld, sp, 1.0, 0.0, heads, group,
                   Tq, hd, T * k.ld, hd, heads * T * sp, T * sp,
                   g=(tiles(T), tiles(T), batch * heads), l=(16, 16, 1))
    _K("attn_softmax_fwd")(S.arg, sp, lens.arg if lens is not None else None, heads, T, float(scale_), int(causal),
                           int(window), g=batch * heads * T * 64, l=64)
    P = S  # the weights the values are mixed with: S, or S with dropout (which backward regenerates)
    if dropout > 0:
        P = empty((batch * heads * T, T), ld=sp)
        _dropout_into(S, P, dropout, seed)
    out = empty((batch * T, heads * hd))
    _K("bgemm_nn")(P.arg, v.arg, out.arg, T, hd, T, sp, v.ld, out.ld, 1.0, 0.0, heads, group,
                   heads * T * sp, T * sp, T * v.ld, hd, T * out.ld, hd,
                   g=(tiles(hd), tiles(T), batch * heads), l=(16, 16, 1))

    def back(g):
        dS = empty((batch * heads * T, T), ld=sp)
        # dP = dO V^T
        _K("bgemm_nt")(g.arg, v.arg, dS.arg, T, T, hd, g.ld, v.ld, sp, 1.0, 0.0, heads, group,
                       T * g.ld, hd, T * v.ld, hd, heads * T * sp, T * sp,
                       g=(tiles(T), tiles(T), batch * heads), l=(16, 16, 1))
        if dropout > 0:  # through the dropout: the same mask on the gradient
            _dropout_into(dS, dS, dropout, seed)
        _K("attn_softmax_bwd")(S.arg, sp, dS.arg, sp, T, float(scale_), g=batch * heads * T * 64, l=64)
        if q.requires_grad:  # dQ = dS K
            dq, beta = _target(q)
            _K("bgemm_nn")(dS.arg, k.arg, dq.arg, T, hd, T, sp, k.ld, dq.ld, 1.0, beta, heads, group,
                           heads * T * sp, T * sp, T * k.ld, hd, T * dq.ld, hd,
                           g=(tiles(hd), tiles(T), batch * heads), l=(16, 16, 1))
        # dK = dS^T Q and dV = P^T dO, summed over the query heads that share each kv head
        Pd = S
        if dropout > 0 and v.requires_grad:
            Pd = empty((batch * heads * T, T), ld=sp)
            _dropout_into(S, Pd, dropout, seed)
        for src, other, dst_t in ((dS, q, k), (Pd, g, v)):
            if not dst_t.requires_grad:
                continue
            dst, beta = _target(dst_t)
            for gi in range(group):
                # A: the scores of query head kvh * group + gi, transposed; B: that head's q (or dO) columns
                A = (src.mem.h, src.mem.base + (src.off + gi * T * sp) * 4)
                Bm = (other.mem.h, other.mem.base + (other.off + gi * hd) * 4)
                _K("bgemm_tn")(A, Bm, dst.arg, T, hd, T, sp, other.ld, dst.ld, 1.0, beta if gi == 0 else 1.0,
                               kv_heads, 1, heads * T * sp, group * T * sp, T * other.ld, group * hd,
                               T * dst.ld, hd, g=(tiles(hd), tiles(T), batch * kv_heads), l=(16, 16, 1))
    return _node(out, (q, k, v), back)


def rope(x, inv_freq, T, groups):
    """Rotary position embedding (rotate-half) on each of `groups` heads of x's rows; position = row % T.
    inv_freq: a Tensor [hd / 2]."""
    hd = x.cols // groups
    out = empty(x.shape)
    n = x.rows * groups * (hd // 2)
    _K("rope")(x.arg, x.ld, out.arg, out.ld, inv_freq.arg, x.rows, groups, hd, T, 1.0, 0.0, g=_grid(n), l=256)

    def back(g):
        dx, beta = _target(x)
        _K("rope")(g.arg, g.ld, dx.arg, dx.ld, inv_freq.arg, x.rows, groups, hd, T, -1.0, beta, g=_grid(n), l=256)
    return _node(out, (x,), back)


# ----------------------------------------------------------------------------------------- gathers, pooling

def _ids(ids):
    if isinstance(ids, Tensor):
        return ids
    a = np.asarray(ids, np.int32).reshape(-1, 1)
    return tensor(a, dtype=I32)


def embedding(ids, table):
    """table[ids]: rows of `table` [vocab, dim] for each id (an int array/Tensor of any shape)"""
    shape = np.shape(ids) if not isinstance(ids, Tensor) else (ids.size,)
    idt = _ids(ids)
    out = empty(tuple(shape) + (table.cols,))
    _K("gather_rows")(idt.arg, table.arg, table.ld, out.arg, out.ld, out.rows, out.cols, g=_grid(out.size), l=256)

    def back(g):
        if table.grad is None:
            table.grad = zeros_like(table)
        _K("scatter_add_rows")(idt.arg, g.arg, g.ld, table.grad.arg, table.grad.ld, out.rows, out.cols,
                               g=_grid(out.size), l=256)
    return _node(out, (table,), back)


def take_rows(x, idx):
    """x[idx]: selected rows (gradients scatter back)"""
    return embedding(idx, x)


def pool_mean(x, lens, batch, T):
    """Mean over each sequence's first lens[b] rows: x [batch * T, cols] -> [batch, cols]"""
    if not isinstance(lens, Tensor):
        lens = tensor(np.asarray(lens, np.int32))
    out = empty((batch, x.cols))
    _K("pool_mean_fwd")(x.arg, x.ld, lens.arg, T, x.cols, out.arg, out.ld, g=(_grid(x.cols), batch), l=(256, 1))

    def back(g):
        dx, beta = _target(x)
        _K("pool_mean_bwd")(g.arg, g.ld, lens.arg, T, dx.arg, dx.ld, x.rows, x.cols, beta, g=_grid(x.size), l=256)
    return _node(out, (x,), back)


def checkpoint(fn, inputs, params=()):
    """fn(*inputs) -> Tensor, without keeping its intermediate results: the backward pass runs fn again.
    Trades a forward pass of compute for the memory of fn's activations (a transformer layer's are tens
    of times its input's). `params`: the trainable tensors fn uses (so the gradient reaches them even when
    no input needs one)."""
    inputs = tuple(inputs)
    drop = list(_drop)  # the recompute must draw the same dropout masks
    with no_grad():
        out = fn(*inputs)

    def back(g):
        ins = []
        for x in inputs:
            d = x.detach()
            d.requires_grad = x.requires_grad
            ins.append(d)
        now = list(_drop)
        _drop[:] = drop
        try:
            with enable_grad():
                o = fn(*ins)
        finally:
            _drop[:] = now
        if o.requires_grad:
            o.backward(g)
        for d, x in zip(ins, inputs):
            if x.requires_grad and d.grad is not None:
                _give(x, d.grad)
    return _node(out, inputs + tuple(params), back)


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("contextlib", "ctypes", "math", "os", "np",
                                                                     "rawgpu")]
