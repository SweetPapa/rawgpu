/* Transport: tinygrad's TinyGPU driver extension, via its Unix-socket server.
 *
 * TinyGPU.app ships an Apple-signed DriverKit extension that binds any display-
 * class PCI device from AMD (0x1002) or NVIDIA (0x10de). Only an app signed with
 * tinygrad's user-client entitlement may open the extension directly, so the
 * app runs a small server (`TinyGPU server <sock>`) that brokers access. This
 * file speaks that server's wire protocol; see tinygrad's
 * extra/usbgpu/tbgpu/installer/Shared/server.c for the other side.
 *
 * The server handles one client at a time. While tinygrad is connected, we get
 * refused, and vice versa.
 */
#include "rg.h"

#include <errno.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define TINYGPU_APP "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"
#define TG_BULK_MAX (64u << 20) /* server's bulk buffer */

enum {
  TG_PROBE = 0, TG_MAP_BAR = 1, TG_MAP_SYSMEM_FD = 2, TG_CFG_READ = 3, TG_CFG_WRITE = 4,
  TG_RESET = 5, TG_MMIO_READ = 6, TG_MMIO_WRITE = 7,
};

typedef struct __attribute__((packed)) { uint8_t cmd; uint32_t dev_id, bar; uint64_t arg0, arg1, arg2; } tg_req;
typedef struct __attribute__((packed)) { uint8_t status; uint64_t resp0, resp1; } tg_resp;
_Static_assert(sizeof(tg_req) == 33, "TinyGPU request wire format");
_Static_assert(sizeof(tg_resp) == 17, "TinyGPU response wire format");

typedef struct { int fd; } tg_priv;

static int send_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = buf;
  while (len) {
    ssize_t n = send(fd, p, len, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -EIO;
    p += n; len -= (size_t)n;
  }
  return 0;
}

static int recv_all(int fd, void *buf, size_t len) {
  uint8_t *p = buf;
  while (len) {
    ssize_t n = recv(fd, p, len, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -EIO;
    p += n; len -= (size_t)n;
  }
  return 0;
}

/* After a failed status, the server may send an error string of resp0 bytes. */
static int read_error(int fd, const tg_resp *rs, const char *what) {
  char msg[512] = "";
  if (rs->resp0 > 0 && rs->resp0 < 4096) {
    size_t n = rs->resp0 < sizeof(msg) - 1 ? (size_t)rs->resp0 : sizeof(msg) - 1;
    if (recv_all(fd, msg, n) == 0) msg[n] = 0;
    for (size_t left = (size_t)rs->resp0 - n; left; ) { /* drain the rest */
      char junk[256]; size_t k = left < sizeof junk ? left : sizeof junk;
      if (recv_all(fd, junk, k)) break;
      left -= k;
    }
  }
  fprintf(stderr, "rawgpu: TinyGPU %s failed%s%s\n", what, msg[0] ? ": " : "", msg);
  return -EIO;
}

static int tg_rpc(rg_dev *d, uint8_t cmd, uint32_t bar, uint64_t a0, uint64_t a1, uint64_t a2,
                  tg_resp *out, const char *what) {
  tg_priv *p = d->priv;
  tg_req rq = {.cmd = cmd, .dev_id = 0, .bar = bar, .arg0 = a0, .arg1 = a1, .arg2 = a2};
  tg_resp rs;
  if (send_all(p->fd, &rq, sizeof rq) || recv_all(p->fd, &rs, sizeof rs)) {
    fprintf(stderr, "rawgpu: TinyGPU server closed the connection during %s "
                    "(is another client, e.g. tinygrad, connected?)\n", what);
    return -EIO;
  }
  if (rs.status != 0) return read_error(p->fd, &rs, what);
  if (out) *out = rs;
  return 0;
}

static int tg_cfg_read(rg_dev *d, uint32_t off, uint32_t size, uint32_t *val) {
  tg_resp rs;
  int r = tg_rpc(d, TG_CFG_READ, 0, off, size, 0, &rs, "config read");
  if (!r) *val = (uint32_t)rs.resp0;
  return r;
}

static int tg_cfg_write(rg_dev *d, uint32_t off, uint32_t size, uint32_t val) {
  return tg_rpc(d, TG_CFG_WRITE, 0, off, size, val, NULL, "config write");
}

static int tg_bar_map(rg_dev *d, int bar, uint64_t *size) {
  tg_resp rs;
  int r = tg_rpc(d, TG_MAP_BAR, (uint32_t)bar, 0, 0, 0, &rs, "BAR map");
  if (!r) *size = rs.resp1;
  return r;
}

/* The server silently drops writes to a BAR that is not mapped or out of range
 * (writes get no response), so check here and fail loudly instead. */
static int check_bar(rg_dev *d, int bar, uint64_t off, size_t len, const char *what) {
  if (bar < 0 || bar >= RG_MAX_BARS || !d->bar_size[bar] || off + len > d->bar_size[bar] || len > TG_BULK_MAX) {
    fprintf(stderr, "rawgpu: MMIO %s of %zu bytes at BAR%d+0x%llx: BAR %s\n", what, len, bar, (unsigned long long)off,
            bar >= 0 && bar < RG_MAX_BARS && d->bar_size[bar] ? "too small" : "not mapped (call rg_bar_map first)");
    return -EINVAL;
  }
  return 0;
}

static int tg_mmio_read(rg_dev *d, int bar, uint64_t off, void *buf, size_t len) {
  tg_priv *p = d->priv;
  if (check_bar(d, bar, off, len, "read")) return -EINVAL;
  tg_req rq = {.cmd = TG_MMIO_READ, .bar = (uint32_t)bar, .arg0 = off, .arg1 = len};
  tg_resp rs;
  if (send_all(p->fd, &rq, sizeof rq) || recv_all(p->fd, &rs, sizeof rs)) return -EIO;
  if (rs.status != 0) return read_error(p->fd, &rs, "MMIO read (BAR not mapped or out of range?)");
  return recv_all(p->fd, buf, len);
}

/* Fire-and-forget on the server side: no response. Ordering with later reads
 * is preserved because the server handles requests strictly in order. */
static int tg_mmio_write(rg_dev *d, int bar, uint64_t off, const void *buf, size_t len) {
  tg_priv *p = d->priv;
  if (check_bar(d, bar, off, len, "write")) return -EINVAL;
  tg_req rq = {.cmd = TG_MMIO_WRITE, .bar = (uint32_t)bar, .arg0 = off, .arg1 = len};
  if (send_all(p->fd, &rq, sizeof rq) || send_all(p->fd, buf, len)) return -EIO;
  return 0;
}

/* The server allocates shared memory, has the extension map it for DMA, and
 * writes the (bus addr, len) segment list, terminated by (0,0), into the first
 * bytes of the buffer before handing us the fd. */
static int tg_dma_alloc(rg_dev *d, size_t size, rg_dma *out) {
  tg_priv *p = d->priv;
  tg_req rq = {.cmd = TG_MAP_SYSMEM_FD, .arg0 = size, .arg1 = 0};
  if (send_all(p->fd, &rq, sizeof rq)) return -EIO;

  tg_resp rs;
  char cbuf[CMSG_SPACE(sizeof(int))];
  struct iovec iov = {&rs, sizeof rs};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof cbuf};
  ssize_t n = recvmsg(p->fd, &msg, MSG_WAITALL);
  if (n != (ssize_t)sizeof rs) return -EIO;
  if (rs.status != 0) { fprintf(stderr, "rawgpu: TinyGPU DMA allocation failed\n"); return -ENOMEM; }

  int fd = -1;
  for (struct cmsghdr *c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
    if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(c), sizeof fd);
  if (fd < 0) return -EIO;

  void *cpu = mmap(NULL, rs.resp0, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (cpu == MAP_FAILED) { close(fd); return -errno; }

  memset(out, 0, sizeof *out);
  out->cpu = cpu; out->size = rs.resp0; out->fd = fd;
  /* The list ends at the first (0,0) pair, and the segments must cover the buffer. */
  const uint64_t *segs = cpu;
  size_t max = out->size / 16, ns = 0;
  uint64_t covered = 0;
  while (ns < max && segs[2 * ns + 1] != 0) covered += segs[2 * ns++ + 1];
  out->seg = malloc((ns ? ns : 1) * sizeof *out->seg);
  if (!out->seg || covered < out->size) {
    fprintf(stderr, "rawgpu: TinyGPU DMA segment list covers %llu of %zu bytes\n", (unsigned long long)covered,
            out->size);
    munmap(cpu, out->size); close(fd); free(out->seg);
    memset(out, 0, sizeof *out); out->fd = -1;
    return -EIO;
  }
  for (size_t i = 0; i < ns; i++) { out->seg[i].addr = segs[2 * i]; out->seg[i].len = segs[2 * i + 1]; }
  out->nseg = (int)ns;
  memset(cpu, 0, out->size); /* the segment list has been consumed */
  return 0;
}

/* The server keeps the DMA mapping until we disconnect; this only unmaps our view. */
static void tg_dma_free(rg_dev *d, rg_dma *b) {
  (void)d;
  if (b->cpu) munmap(b->cpu, b->size);
  if (b->fd >= 0) close(b->fd);
  free(b->seg);
  memset(b, 0, sizeof *b);
  b->fd = -1;
}

/* The extension tries a function-level reset and falls back to a hot reset of the link. */
static int tg_reset(rg_dev *d) {
  int r = tg_rpc(d, TG_RESET, 0, 0, 0, 0, NULL, "reset");
  memset(d->bar_size, 0, sizeof d->bar_size); /* map BARs again afterwards */
  return r;
}

static void tg_close(rg_dev *d) {
  tg_priv *p = d->priv;
  if (p) { close(p->fd); free(p); }
  d->priv = NULL;
}

static const rg_transport_ops tg_ops = {
  .name = "tinygpu", .cfg_read = tg_cfg_read, .cfg_write = tg_cfg_write, .bar_map = tg_bar_map,
  .mmio_read = tg_mmio_read, .mmio_write = tg_mmio_write, .dma_alloc = tg_dma_alloc,
  .dma_free = tg_dma_free, .reset = tg_reset, .close = tg_close,
};

static int try_connect(const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  strncpy(a.sun_path, path, sizeof a.sun_path - 1);
  if (connect(fd, (struct sockaddr *)&a, sizeof a) == 0) return fd;
  close(fd);
  return -1;
}

static int spawn_server(const char *sock_path) {
  char log_path[1024];
  const char *tmp = getenv("TMPDIR");
  snprintf(log_path, sizeof log_path, "%s/rawgpu-tinygpu-server.log", tmp ? tmp : "/tmp");

  posix_spawn_file_actions_t fa;
  posix_spawnattr_t attr;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
  posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
  posix_spawnattr_init(&attr);
#ifdef POSIX_SPAWN_SETSID
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID); /* survive our exit and Ctrl-C */
#endif
  char *argv[] = {"TinyGPU", "server", (char *)sock_path, NULL};
  pid_t pid;
  int r = posix_spawn(&pid, TINYGPU_APP, &fa, &attr, argv, environ);
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&attr);
  return r;
}

int rg_open_tinygpu(rg_dev *d, const char *sock_path) {
  char def[1024];
  if (!sock_path) sock_path = getenv("RAWGPU_TINYGPU_SOCK");
  if (!sock_path) {
    const char *tmp = getenv("TMPDIR");
    snprintf(def, sizeof def, "%s/rawgpu-tinygpu.sock", tmp ? tmp : "/tmp");
    sock_path = def;
  }
  if (strlen(sock_path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
    fprintf(stderr, "rawgpu: socket path too long: %s\n", sock_path);
    return -ENAMETOOLONG;
  }

  int fd = try_connect(sock_path);
  if (fd < 0) {
    if (access(TINYGPU_APP, X_OK) != 0) {
      fprintf(stderr, "rawgpu: %s not found. Run tools/install-tinygpu.sh first.\n", TINYGPU_APP);
      return -ENOENT;
    }
    int r = spawn_server(sock_path);
    if (r) { fprintf(stderr, "rawgpu: could not start TinyGPU server: %s\n", strerror(r)); return -r; }
    for (int i = 0; i < 100 && fd < 0; i++) {
      nanosleep(&(struct timespec){0, 50 * 1000 * 1000}, NULL);
      fd = try_connect(sock_path);
    }
    if (fd < 0) { fprintf(stderr, "rawgpu: TinyGPU server did not come up at %s\n", sock_path); return -ETIMEDOUT; }
  }

  int bufsz = 8 << 20;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsz, sizeof bufsz);
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsz, sizeof bufsz);

  tg_priv *p = calloc(1, sizeof *p);
  if (!p) { close(fd); return -ENOMEM; }
  p->fd = fd;
  memset(d, 0, sizeof *d);
  d->ops = &tg_ops;
  d->priv = p;
  return 0;
}

void rg_close(rg_dev *d) {
  if (d->ops && d->ops->close) d->ops->close(d);
}

uint64_t rg_dma_bus(const rg_dma *b, uint64_t off) {
  for (int i = 0; i < b->nseg; i++) {
    if (off < b->seg[i].len) return b->seg[i].addr + off;
    off -= b->seg[i].len;
  }
  return 0;
}

int rg_bar_map(rg_dev *d, int bar, uint64_t *size) {
  if (bar < 0 || bar >= RG_MAX_BARS) return -EINVAL;
  if (d->bar_size[bar]) { if (size) *size = d->bar_size[bar]; return 0; }
  uint64_t sz = 0;
  int r = d->ops->bar_map(d, bar, &sz);
  if (!r) { d->bar_size[bar] = sz; if (size) *size = sz; }
  return r;
}
