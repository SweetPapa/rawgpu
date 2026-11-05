"""rawgpu from Python: OpenCL C kernels and SGEMM on the GPU, with numpy arrays.

    import numpy as np, rawgpu
    gpu = rawgpu.Device()                      # bring-up runs if the card needs it
    x = gpu.array(np.arange(1 << 20, dtype=np.float32))
    prog = gpu.program('''
        __kernel void scale(__global float *x, float s) { x[get_global_id(0)] *= s; }''')
    prog.scale(x, 2.0, global_size=x.size)
    print(x.numpy()[:4])                       # waits for the kernel

    c = gpu.matmul(gpu.array(a), gpu.array(b)) # SGEMM, float32, 2-D

A thin ctypes layer over librawgpu (include/rawgpu.h); its rules hold here:
launches are asynchronous and in order, transfers wait for them, errors raise
rawgpu.Error with the library's message. Set RAWGPU_LIB to use a specific
librawgpu.dylib; by default the one in this checkout's build/ directory.
"""
import atexit
import ctypes
import ctypes.util
import os
import struct
import weakref

import numpy as np

__all__ = ["Device", "Buffer", "BufferView", "Program", "Kernel", "Local", "Error"]

MEM_VRAM, MEM_HOST = 0, 1
OPEN_BRINGUP = 1
ARG_VALUE, ARG_BUFFER, ARG_LOCAL, ARG_OTHER = range(4)


class Error(RuntimeError):
    """A rawgpu call failed; the message is the library's."""


def _find_lib():
    env = os.environ.get("RAWGPU_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "..", "build", "librawgpu.dylib"),
                 os.path.join(here, "librawgpu.dylib")):
        if os.path.exists(cand):
            return os.path.normpath(cand)
    found = ctypes.util.find_library("rawgpu")
    if found:
        return found
    raise Error("librawgpu.dylib not found: run `make` in the rawgpu checkout, or set RAWGPU_LIB")


class _Info(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 64), ("arch", ctypes.c_char * 16), ("compute_units", ctypes.c_uint),
                ("wavefront_size", ctypes.c_uint), ("max_workgroup_size", ctypes.c_uint),
                ("local_mem_bytes", ctypes.c_uint), ("vram_bytes", ctypes.c_uint64),
                ("vram_free_bytes", ctypes.c_uint64), ("vram_largest_free", ctypes.c_uint64),
                ("engine_mhz", ctypes.c_int), ("memory_mhz", ctypes.c_int), ("temperature_c", ctypes.c_int)]


class _ArgInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("type_name", ctypes.c_char_p), ("size", ctypes.c_uint),
                ("kind", ctypes.c_int), ("is_const", ctypes.c_int)]


class _KernelInfo(ctypes.Structure):
    _fields_ = [("num_args", ctypes.c_uint), ("max_workgroup_size", ctypes.c_uint),
                ("reqd_workgroup_size", ctypes.c_uint * 3), ("local_mem_bytes", ctypes.c_uint),
                ("private_mem_bytes", ctypes.c_uint), ("sgprs", ctypes.c_uint), ("vgprs", ctypes.c_uint)]


_lib = None
_open_devices = weakref.WeakSet()


@atexit.register
def _close_open_devices():
    # Interpreter shutdown does not reliably run __del__ (it does not once torch is loaded, for one), and a
    # device left open leaves its compute queue active on the card for the next process to retire.
    for dev in list(_open_devices):
        try:
            dev.close()
        except Exception:
            pass


def _L():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(_find_lib())
    P, I, U, Z, U64, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_float
    sig = {
        "rawgpu_open": (I, [ctypes.POINTER(P), ctypes.c_char_p, U]), "rawgpu_close": (None, [P]),
        "rawgpu_error": (ctypes.c_char_p, [P]), "rawgpu_get_info": (I, [P, ctypes.POINTER(_Info)]),
        "rawgpu_alloc": (P, [P, Z, U]), "rawgpu_free": (None, [P, P]), "rawgpu_buf_size": (Z, [P]),
        "rawgpu_buf_gpu_addr": (U64, [P]), "rawgpu_buf_host": (P, [P]),
        "rawgpu_write": (I, [P, P, Z, P, Z]), "rawgpu_read": (I, [P, P, Z, P, Z]),
        "rawgpu_copy": (I, [P, P, Z, P, Z, Z]), "rawgpu_fill": (I, [P, P, Z, Z, ctypes.c_uint32]),
        "rawgpu_program_build": (P, [P, ctypes.c_char_p, ctypes.c_char_p]),
        "rawgpu_program_load": (P, [P, ctypes.c_char_p]), "rawgpu_program_load_mem": (P, [P, P, Z]),
        "rawgpu_program_free": (None, [P, P]), "rawgpu_program_num_kernels": (U, [P]),
        "rawgpu_program_kernel_name": (ctypes.c_char_p, [P, U]), "rawgpu_kernel_get": (P, [P, ctypes.c_char_p]),
        "rawgpu_kernel_get_info": (I, [P, ctypes.POINTER(_KernelInfo)]),
        "rawgpu_kernel_arg_info": (I, [P, U, ctypes.POINTER(_ArgInfo)]),
        "rawgpu_set_arg": (I, [P, U, Z, P]), "rawgpu_set_arg_buf_offset": (I, [P, U, P, Z]),
        "rawgpu_set_arg_local": (I, [P, U, Z]),
        "rawgpu_launch": (U64, [P, P, U, ctypes.POINTER(Z), ctypes.POINTER(Z)]),
        "rawgpu_wait": (I, [P, U64]), "rawgpu_is_done": (I, [P, U64]), "rawgpu_finish": (I, [P]),
        "rawgpu_sgemm": (U64, [P, I, I, Z, Z, Z, F, P, Z, P, Z, F, P, Z]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(L, name)
        fn.restype, fn.argtypes = res, args
    _lib = L
    return L


class BufferView:
    """A buffer argument starting `offset` bytes in: Buffer.at(offset)."""

    def __init__(self, buf, offset):
        self.buf, self.offset = buf, int(offset)


class Local:
    """A __local pointer argument: `nbytes` of LDS per work-group."""

    def __init__(self, nbytes):
        self.nbytes = int(nbytes)


class Device:
    """The GPU. One process at a time can open it."""

    def __init__(self, socket=None, bringup=True):
        L = _L()
        h = ctypes.c_void_p()
        r = L.rawgpu_open(ctypes.byref(h), socket.encode() if socket else None, OPEN_BRINGUP if bringup else 0)
        if r:
            raise Error(L.rawgpu_error(None).decode(errors="replace"))
        self._h = h
        _open_devices.add(self)

    def _err(self):
        return Error(_L().rawgpu_error(self._h).decode(errors="replace"))

    def close(self):
        """Release the GPU. Buffers and programs of this device are invalid afterwards."""
        if getattr(self, "_h", None):
            _L().rawgpu_close(self._h)  # frees every buffer and program still open
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def info(self):
        i = _Info()
        if _L().rawgpu_get_info(self._h, ctypes.byref(i)):
            raise self._err()
        return {f: (getattr(i, f).decode() if isinstance(getattr(i, f), bytes) else getattr(i, f)) for f, _ in i._fields_}

    # ---- memory ----
    def empty(self, shape, dtype=np.float32, host=False):
        """An uninitialized GPU array (VRAM unless host=True)."""
        return Buffer(self, shape, dtype, host)

    def zeros(self, shape, dtype=np.float32, host=False):
        b = Buffer(self, shape, dtype, host)
        b.fill(0)
        return b

    def array(self, a, host=False):
        """A GPU copy of a numpy array (made C-contiguous)."""
        a = np.ascontiguousarray(a)
        b = Buffer(self, a.shape, a.dtype, host)
        b.write(a)
        return b

    # ---- code ----
    def program(self, source, options=None):
        """Compile OpenCL C source (cached by content)."""
        p = _L().rawgpu_program_build(self._h, source.encode(), options.encode() if options else None)
        if not p:
            raise self._err()
        return Program(self, p)

    def load(self, path):
        """Load a gfx803 code object file."""
        p = _L().rawgpu_program_load(self._h, os.fsencode(path))
        if not p:
            raise self._err()
        return Program(self, p)

    def finish(self):
        if _L().rawgpu_finish(self._h):
            raise self._err()

    def wait(self, event):
        if _L().rawgpu_wait(self._h, event):
            raise self._err()

    # ---- built-ins ----
    def sgemm(self, A, B, C, M, N, K, alpha=1.0, beta=0.0, trans_a=False, trans_b=False, lda=None, ldb=None, ldc=None):
        """C = alpha * op(A) @ op(B) + beta * C on float32 buffers (row-major). Asynchronous; returns the event."""
        lda = lda or (M if trans_a else K)
        ldb = ldb or (K if trans_b else N)
        ldc = ldc or N
        e = _L().rawgpu_sgemm(self._h, int(trans_a), int(trans_b), M, N, K, alpha, A._h, lda, B._h, ldb, beta, C._h, ldc)
        if not e:
            raise self._err()
        return e

    def matmul(self, A, B, out=None):
        """A @ B for 2-D float32 GPU arrays; returns a new (or the given) GPU array."""
        if A.dtype != np.float32 or B.dtype != np.float32 or len(A.shape) != 2 or len(B.shape) != 2:
            raise Error("matmul takes 2-D float32 arrays")
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise Error(f"matmul: inner dimensions differ ({K} and {K2})")
        C = out if out is not None else self.empty((M, N), np.float32)
        self.sgemm(A, B, C, M, N, K)
        return C


class Buffer:
    """GPU memory holding an array of `shape` and `dtype`."""

    def __init__(self, dev, shape, dtype, host=False):
        self.dev, self.dtype = dev, np.dtype(dtype)
        self.shape = (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        h = _L().rawgpu_alloc(dev._h, max(self.nbytes, 1), MEM_HOST if host else MEM_VRAM)
        if not h:
            raise dev._err()
        self._h = h
        self.host = host

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def gpu_addr(self):
        return _L().rawgpu_buf_gpu_addr(self._h)

    def write(self, a):
        a = np.ascontiguousarray(a, dtype=self.dtype)
        if a.nbytes > self.nbytes:
            raise Error(f"{a.nbytes} bytes do not fit in a {self.nbytes}-byte buffer")
        if _L().rawgpu_write(self.dev._h, self._h, 0, a.ctypes.data, a.nbytes):
            raise self.dev._err()

    def at(self, offset):
        """This buffer, `offset` bytes in, as a kernel argument."""
        return BufferView(self, offset)

    def host_view(self):
        """A numpy array that *is* this host buffer's memory (no copy). The GPU writes it
        directly: read it only after the launches that write it are done."""
        if not self.host:
            raise Error("host_view() needs a host buffer (host=True)")
        ptr = _L().rawgpu_buf_host(self._h)
        raw = (ctypes.c_uint8 * self.nbytes).from_address(ptr)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.shape)

    def numpy(self):
        """Copy to a new numpy array (waits for pending launches)."""
        out = np.empty(self.shape, self.dtype)
        if _L().rawgpu_read(self.dev._h, self._h, 0, out.ctypes.data, self.nbytes):
            raise self.dev._err()
        return out

    def fill(self, value):
        """Every element = value (by SDMA, for 1/2/4-byte types whose pattern repeats in 32 bits)."""
        word = np.array([value], self.dtype).tobytes()
        if len(word) not in (1, 2, 4):
            if value == 0:
                word = b"\0\0\0\0"
            else:
                self.write(np.full(self.shape, value, self.dtype))
                return
        pattern = struct.unpack("<I", (word * (4 // len(word)))[:4])[0]
        n = (self.nbytes + 3) & ~3
        if _L().rawgpu_fill(self.dev._h, self._h, 0, n, pattern):
            raise self.dev._err()

    def free(self):
        if getattr(self, "_h", None) and getattr(self.dev, "_h", None):
            _L().rawgpu_free(self.dev._h, self._h)
        self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return f"<rawgpu.Buffer {self.shape} {self.dtype} {'host' if self.host else 'VRAM'}>"


class Kernel:
    def __init__(self, prog, h, name):
        self.prog, self._h, self.name = prog, h, name
        ki = _KernelInfo()
        _L().rawgpu_kernel_get_info(h, ctypes.byref(ki))
        self.info = {"max_workgroup_size": ki.max_workgroup_size, "reqd_workgroup_size": tuple(ki.reqd_workgroup_size),
                     "local_mem_bytes": ki.local_mem_bytes, "private_mem_bytes": ki.private_mem_bytes,
                     "sgprs": ki.sgprs, "vgprs": ki.vgprs}
        self.args = []
        for i in range(ki.num_args):
            ai = _ArgInfo()
            _L().rawgpu_kernel_arg_info(h, i, ctypes.byref(ai))
            self.args.append((ai.name.decode(), ai.type_name.decode(), ai.size, ai.kind))

    def _pack(self, i, value):
        name, tname, size, kind = self.args[i]
        dev = self.prog.dev
        if kind == ARG_BUFFER:
            if isinstance(value, Buffer):
                r = _L().rawgpu_set_arg_buf_offset(self._h, i, value._h, 0)
            elif isinstance(value, BufferView):
                r = _L().rawgpu_set_arg_buf_offset(self._h, i, value.buf._h, value.offset)
            elif value is None:
                r = _L().rawgpu_set_arg_buf_offset(self._h, i, None, 0)
            else:
                raise Error(f"{self.name} argument {i} ({tname} {name}) takes a rawgpu.Buffer")
        elif kind == ARG_LOCAL:
            if not isinstance(value, Local):
                raise Error(f"{self.name} argument {i} ({tname} {name}) takes rawgpu.Local(nbytes)")
            r = _L().rawgpu_set_arg_local(self._h, i, value.nbytes)
        else:
            if isinstance(value, np.generic) or isinstance(value, np.ndarray):
                raw = np.asarray(value).tobytes()
            else:
                t = tname.replace("unsigned ", "u").strip()
                if t in ("float",):
                    raw = struct.pack("<f", value)
                elif t in ("double",):
                    raw = struct.pack("<d", value)
                elif t in ("int", "uint", "long", "ulong", "short", "ushort", "char", "uchar", "size_t", "bool"):
                    fmt = {1: "b", 2: "h", 4: "i", 8: "q"}[size]
                    if t.startswith("u") or t == "size_t":
                        fmt = fmt.upper()
                    raw = struct.pack("<" + fmt, int(value))
                else:
                    raise Error(f"{self.name} argument {i} ({tname} {name}): pass a numpy value or bytes")
            if len(raw) != size:
                raise Error(f"{self.name} argument {i} ({tname} {name}) takes {size} bytes, got {len(raw)}")
            buf = ctypes.create_string_buffer(raw, len(raw))
            r = _L().rawgpu_set_arg(self._h, i, size, buf)
        if r:
            raise dev._err()

    def __call__(self, *args, global_size, local_size=None, wait=False):
        """Set the arguments and launch. Returns the event (or waits, with wait=True)."""
        if len(args) != len(self.args):
            raise Error(f"{self.name} takes {len(self.args)} arguments ({', '.join(a[1] + ' ' + a[0] for a in self.args)})")
        for i, v in enumerate(args):
            self._pack(i, v)
        g = (global_size,) if isinstance(global_size, int) else tuple(global_size)
        dims = len(g)
        G = (ctypes.c_size_t * dims)(*g)
        Lp = None
        if local_size is not None:
            l = (local_size,) if isinstance(local_size, int) else tuple(local_size)
            Lp = (ctypes.c_size_t * dims)(*l)
        dev = self.prog.dev
        e = _L().rawgpu_launch(dev._h, self._h, dims, G, Lp)
        if not e:
            raise dev._err()
        if wait:
            dev.wait(e)
        return e

    def __repr__(self):
        return f"<rawgpu.Kernel {self.name}({', '.join(a[1] + ' ' + a[0] for a in self.args)})>"


class Program:
    """A loaded code object; its kernels are attributes: prog.name(args..., global_size=...)."""

    def __init__(self, dev, h):
        self.dev, self._h = dev, h
        n = _L().rawgpu_program_num_kernels(h)
        self.kernel_names = [_L().rawgpu_program_kernel_name(h, i).decode() for i in range(n)]
        self._kernels = {}

    def kernel(self, name):
        if name not in self._kernels:
            k = _L().rawgpu_kernel_get(self._h, name.encode())
            if not k:
                raise self.dev._err()
            self._kernels[name] = Kernel(self, k, name)
        return self._kernels[name]

    def __getattr__(self, name):
        if name.startswith("_") or name not in self.__dict__.get("kernel_names", []):
            raise AttributeError(name)
        return self.kernel(name)

    def free(self):
        if getattr(self, "_h", None) and getattr(self.dev, "_h", None):
            _L().rawgpu_program_free(self.dev._h, self._h)
        self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def __repr__(self):
        return f"<rawgpu.Program kernels={self.kernel_names}>"
