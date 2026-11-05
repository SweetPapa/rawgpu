/* rawgpu CLI. Every command is read-only unless noted. */
#include "amd_kernel.h"
#include "rg.h"

#include <errno.h>
#include <libproc.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

static const char *usage =
  "usage: rawgpu [--sock PATH] <command>\n"
  "\n"
  "  probe            identify the card, link and BARs, then run its family's probe (default)\n"
  "  cfg [--clear-errors]\n"
  "                   hex-dump PCI config space and decode PCI/PCIe/AER error status\n"
  "  vbios -o FILE    save the VBIOS image (writes the ROM index register, as amdgpu does)\n"
  "  bench [N]        round-trip latency of config and register reads\n"
  "  dma-test [KB]    allocate a DMA buffer and print its bus addresses (no GPU activity)\n"
  "  reset            WRITES: PCIe reset through TinyGPU (FLR, else hot reset); a no-op on the RX 590\n"
  "  eject [--timeout SECONDS]\n"
  "                   wait for the program using the card to finish, stop the card's engines, close it,\n"
  "                   and stop the TinyGPU server; then says it is safe to unplug\n"
  "  kernel-info FILE describe the kernels in a code object (no GPU needed)\n";

static void print_backend_usage(FILE *f, const rg_backend *b) {
  fprintf(f, "\n%s commands:\n", b->name);
  for (const rg_cmd *c = b->cmds; c && c->name; c++) {
    char head[128];
    snprintf(head, sizeof head, "%s%s%s", c->name, c->args[0] ? " " : "", c->args);
    if (strlen(head) <= 16) fprintf(f, "  %-16s %s\n", head, c->help);
    else fprintf(f, "  %s\n  %-16s %s\n", head, "", c->help);
  }
}

static void print_usage(FILE *f) {
  fputs(usage, f);
  print_backend_usage(f, &rg_backend_amd_vi);
}

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static int cmd_probe(rg_dev *d, const rg_ident *id) {
  printf("device:            %04x:%04x rev %02x  (%s)\n", d->vendor, d->device, d->revision,
         rg_pci_vendor_name(d->vendor));
  printf("  class:           %06x %s\n", d->class_code, rg_pci_class_name(d->class_code));
  printf("  board:           %04x:%04x  (%s)\n", d->subsys_vendor, d->subsys_device,
         rg_pci_vendor_name(d->subsys_vendor));
  printf("  family:          %s%s%s\n", id->gen_name, id->chip ? " / " : "", id->chip ? id->chip : "");
  printf("  support:         %s\n", id->support);

  int cg, cw, mg, mw;
  if (!rg_pci_link(d, &cg, &cw, &mg, &mw))
    printf("  PCIe link:       Gen%d x%d (card supports Gen%d x%d)\n", cg, cw, mg, mw);

  rg_bar_desc bars[RG_MAX_BARS];
  if (!rg_pci_bars(d, bars)) {
    for (int i = 0; i < RG_MAX_BARS; i++) {
      if (bars[i].upper_half || (!bars[i].raw && !bars[i].is_io)) continue;
      uint64_t sz = 0;
      if (!bars[i].is_io) rg_bar_map(d, i, &sz);
      printf("  BAR%d:            %s%s%s", i, bars[i].is_io ? "I/O" : "mem",
             bars[i].is_64 ? " 64-bit" : "", bars[i].prefetch ? " prefetchable" : "");
      if (sz) printf(", %llu %s", sz >= (1u << 20) ? (unsigned long long)(sz >> 20) : (unsigned long long)(sz >> 10),
                     sz >= (1u << 20) ? "MB" : "KB");
      else if (bars[i].is_io) printf(" (not usable: Apple Silicon has no I/O port space)");
      printf("\n");
    }
  }

  if (!id->backend) { printf("  backend:         none\n"); return 0; }
  printf("  backend:         %s\n", id->backend->name);
  return id->backend->probe(d, id);
}

static int cmd_cfg(rg_dev *d, bool clear) {
  for (uint32_t off = 0; off < 0x100; off += 16) {
    printf("%03x:", off);
    for (uint32_t i = 0; i < 16; i += 4) {
      uint32_t v = 0;
      if (rg_cfg_read(d, off + i, 4, &v)) return -EIO;
      printf(" %02x %02x %02x %02x", v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, v >> 24);
    }
    printf("\n");
  }
  rg_pci_err e;
  if (rg_pci_errors(d, &e)) return -EIO;
  rg_pci_print_errors(&e);
  if (clear) {
    if (rg_pci_clear_errors(d) || rg_pci_errors(d, &e)) return -EIO;
    printf("after clearing:\n");
    rg_pci_print_errors(&e);
  }
  return 0;
}

static int cmd_vbios(rg_dev *d, const rg_ident *id, const char *path) {
  if (!id->backend || !id->backend->vbios_read) { fprintf(stderr, "rawgpu: no VBIOS reader for this family\n"); return -ENOTSUP; }
  uint8_t *buf = NULL;
  size_t len = 0;
  int r = id->backend->vbios_read(d, &buf, &len);
  if (r) return r;
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(buf, 1, len, f) != len) { perror(path); free(buf); if (f) fclose(f); return -EIO; }
  fclose(f);
  printf("wrote %zu bytes to %s\n", len, path);
  rg_atom_report(buf, len);
  free(buf);
  return 0;
}

static int cmd_bench(rg_dev *d, const rg_ident *id, int n) {
  uint32_t v;
  double t0 = now_us();
  for (int i = 0; i < n; i++) if (rg_cfg_read(d, 0, 4, &v)) return -EIO;
  double cfg = (now_us() - t0) / n;
  printf("config read:   %7.1f us/op over %d ops\n", cfg, n);
  if (id->backend) {
    if (rg_bar_map(d, id->backend->reg_bar, NULL)) return -EIO;
    t0 = now_us();
    for (int i = 0; i < n; i++) if (rg_rreg32(d, id->backend->reg_bar, id->backend->safe_reg, &v)) return -EIO;
    printf("register read: %7.1f us/op over %d ops (BAR%d +0x%x)\n", (now_us() - t0) / n, n,
           id->backend->reg_bar, id->backend->safe_reg);
  }
  printf("(each op is a socket round trip to the TinyGPU server plus one PCIe read)\n");
  return 0;
}

static int cmd_dma_test(rg_dev *d, size_t kb) {
  rg_dma b;
  int r = d->ops->dma_alloc(d, kb << 10, &b);
  if (r) return r;
  printf("DMA buffer: %zu KB mapped at %p, %d segment(s)\n", b.size >> 10, b.cpu, b.nseg);
  for (int i = 0; i < b.nseg; i++)
    printf("  bus 0x%010llx  len 0x%llx\n", (unsigned long long)b.seg[i].addr, (unsigned long long)b.seg[i].len);
  ((volatile uint32_t *)b.cpu)[0] = 0xC0FFEE;
  printf("CPU write/read of the buffer: %s\n", ((volatile uint32_t *)b.cpu)[0] == 0xC0FFEE ? "OK" : "FAILED");
  d->ops->dma_free(d, &b);
  return 0;
}

/* ---- rawgpu eject ---- */

#define TINYGPU_EXE "/TinyGPU.app/Contents/MacOS/TinyGPU"

static bool is_tinygpu_process(int pid) {
  char path[PROC_PIDPATHINFO_MAXSIZE];
  if (pid <= 0 || proc_pidpath(pid, path, sizeof path) <= 0) return false;
  size_t n = strlen(path), m = strlen(TINYGPU_EXE);
  return n >= m && !strcmp(path + n - m, TINYGPU_EXE);
}

/* a and b name the same path, ignoring repeated slashes ($TMPDIR ends in one) */
static bool same_path(const char *a, const char *b) {
  for (;;) {
    while (a[0] == '/' && a[1] == '/') a++;
    while (b[0] == '/' && b[1] == '/') b++;
    if (*a != *b) return false;
    if (!*a) return true;
    a++, b++;
  }
}

/* Is pid `TinyGPU server <sock>`? Reads its arguments (KERN_PROCARGS2: argc, the executable path, then
 * the argument strings). */
static bool serves_socket(int pid, const char *sock) {
  if (!is_tinygpu_process(pid)) return false;
  int mib[3] = {CTL_KERN, KERN_PROCARGS2, pid};
  char buf[16384];
  size_t size = sizeof buf;
  if (sysctl(mib, 3, buf, &size, NULL, 0) || size < sizeof(int)) return false;
  int argc;
  memcpy(&argc, buf, sizeof argc);
  char *p = buf + sizeof argc, *end = buf + size;
  p += strnlen(p, (size_t)(end - p)); /* the executable path, then padding */
  while (p < end && !*p) p++;
  const char *argv[3] = {0};
  for (int i = 0; i < argc && i < 3 && p < end; i++) {
    argv[i] = p;
    p += strnlen(p, (size_t)(end - p)) + 1;
  }
  return argc >= 3 && argv[1] && !strcmp(argv[1], "server") && argv[2] && same_path(argv[2], sock);
}

/* Processes running TinyGPU.app's executable: the server for `sock` (if any) in *server, the count of
 * others (a server some other program started, e.g. tinygrad's) returned, the first of them in *other. */
static int tinygpu_processes(const char *sock, int *server, int *other) {
  *server = *other = 0;
  int n = proc_listallpids(NULL, 0);
  if (n <= 0) return 0;
  size_t cap = (size_t)n + 64;
  pid_t *all = calloc(cap, sizeof *all);
  if (!all) return 0;
  n = proc_listallpids(all, (int)(cap * sizeof *all));
  int others = 0;
  for (int i = 0; i < n; i++) {
    if (all[i] == getpid() || !is_tinygpu_process(all[i])) continue;
    if (!*server && serves_socket(all[i], sock)) *server = all[i];
    else if (!others++) *other = all[i];
  }
  free(all);
  return others;
}

/* How many clients the server is connected to: its AF_UNIX sockets other than the listening one. A
 * look at its file descriptors, so it neither queues behind the client nor disturbs it. -1 if unknown. */
static int server_clients(int pid) {
  int bytes = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, NULL, 0);
  if (bytes <= 0) return -1;
  struct proc_fdinfo *fds = malloc((size_t)bytes);
  if (!fds) return -1;
  bytes = proc_pidinfo(pid, PROC_PIDLISTFDS, 0, fds, bytes);
  int n = bytes > 0 ? bytes / (int)sizeof *fds : 0, clients = 0;
  for (int i = 0; i < n; i++) {
    if (fds[i].proc_fdtype != PROX_FDTYPE_SOCKET) continue;
    struct socket_fdinfo si;
    if (proc_pidfdinfo(pid, fds[i].proc_fd, PROC_PIDFDSOCKETINFO, &si, sizeof si) != sizeof si) continue;
    if (si.psi.soi_family == AF_UNIX && !(si.psi.soi_options & SO_ACCEPTCONN)) clients++;
  }
  free(fds);
  return bytes > 0 ? clients : -1;
}

static bool process_gone(int pid, double seconds) {
  for (double t = 0; t < seconds; t += 0.1) {
    if (kill(pid, 0) && errno == ESRCH) return true;
    nanosleep(&(struct timespec){0, 100 * 1000 * 1000}, NULL);
  }
  return kill(pid, 0) && errno == ESRCH;
}

static int not_safe_others(int n, int pid) {
  printf("But %d other TinyGPU process%s run%s (pid %d), so another program may have the card.\n"
         "NOT safe to unplug until %s.\n", n, n > 1 ? "es" : "", n > 1 ? "" : "s", pid, n > 1 ? "they exit" : "it exits");
  return -EBUSY;
}

/* `rawgpu eject`: release the card so the enclosure can be unplugged. The server takes one client at a
 * time, so while another program has the card this waits. Then it attaches, has the backend stop the
 * engines (nothing left reading or writing host memory), disconnects (the server unmaps the BARs and
 * DMA buffers), and stops the server. The server has no shutdown request or signal handler; SIGTERM
 * ends it, and the kernel releases its hold on the driver extension. */
static int cmd_eject(const char *sock, int argc, char **argv) {
  double timeout = 0; /* 0: wait as long as it takes */
  for (int i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--timeout") && i + 1 < argc) timeout = atof(argv[++i]);
    else { fprintf(stderr, "usage: rawgpu eject [--timeout SECONDS]\n"); return -EINVAL; }
  }
  char buf[1024];
  const char *path = rg_tinygpu_sock_path(sock, buf, sizeof buf);
  rg_dev d;
  int pid, other, nothers, r;
  double t0 = now_us(), next_note = 60;
  bool waiting = false;
  for (;;) {
    nothers = tinygpu_processes(path, &pid, &other);
    if (!pid) {
      struct stat st;
      if (!lstat(path, &st) && S_ISSOCK(st.st_mode)) unlink(path); /* stale: nothing serves it */
      printf("No TinyGPU server is running for rawgpu.\n");
      if (nothers) return not_safe_others(nothers, other);
      printf("Nothing has the card. Safe to unplug.\n");
      return 0;
    }
    int clients = server_clients(pid);
    if (clients == 0) {
      r = rg_attach_tinygpu(&d, path, NULL); /* if another client got in first, this waits for it */
      if (r != -ECONNREFUSED) break;
    }
    double waited = (now_us() - t0) / 1e6;
    if (!waiting) {
      printf("Another program is using the card (TinyGPU server %d is busy). Waiting for it to finish;\n"
             "Ctrl-C stops waiting.\n", pid);
      waiting = true;
    } else if (waited >= next_note) {
      printf("  still waiting after %.0f min\n", waited / 60);
      next_note += 60;
    }
    fflush(stdout);
    if (timeout > 0 && waited >= timeout) {
      printf("Gave up after %.0f s. The card is still in use: NOT safe to unplug.\n", waited);
      return -EBUSY;
    }
    sleep(5);
  }
  if (r) { fprintf(stderr, "rawgpu: could not attach to TinyGPU server %d at %s (%d)\n", pid, path, r); return r; }
  if (waiting) printf("The card is free (waited %.0f s).\n", (now_us() - t0) / 1e6);
  printf("TinyGPU server: pid %d\n", pid);

  if (!(r = rg_pci_identify(&d))) {
    rg_ident id;
    rg_identify_family(&d, &id);
    if (id.backend && id.backend->quiesce) r = id.backend->quiesce(&d, &id);
  } else {
    printf("card:          no answer to config reads, so nothing on it is running\n");
    r = 0;
  }
  rg_close(&d);
  if (r) {
    fprintf(stderr, "rawgpu: stopping the card's engines failed (%d). The server is left running: NOT safe to unplug.\n", r);
    return r;
  }
  nanosleep(&(struct timespec){0, 300 * 1000 * 1000}, NULL); /* let the server finish its cleanup */

  kill(pid, SIGTERM);
  if (!process_gone(pid, 5)) {
    kill(pid, SIGKILL);
    if (!process_gone(pid, 3)) {
      fprintf(stderr, "rawgpu: TinyGPU server %d did not exit. NOT safe to unplug.\n", pid);
      return -EBUSY;
    }
  }
  struct stat st;
  if (!lstat(path, &st) && S_ISSOCK(st.st_mode)) unlink(path); /* it does not remove its socket on a signal */
  printf("TinyGPU server: stopped\n");
  if (nothers) return not_safe_others(nothers, other);
  printf("\nSafe to unplug. The next rawgpu command starts a new server.\n");
  return 0;
}

int main(int argc, char **argv) {
  const char *sock = NULL;
  int i = 1;
  if (i + 1 < argc && !strcmp(argv[i], "--sock")) { sock = argv[i + 1]; i += 2; }
  const char *cmd = i < argc ? argv[i++] : "probe";
  if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help") || !strcmp(cmd, "help")) { print_usage(stdout); return 0; }

  if (!strcmp(cmd, "kernel-info")) { /* needs no device */
    if (i >= argc) { print_usage(stderr); return 1; }
    amd_program p;
    if (amd_program_load(&p, argv[i])) return 1;
    amd_program_print(stdout, &p);
    amd_program_free(&p);
    return 0;
  }

  if (!strcmp(cmd, "eject")) return cmd_eject(sock, argc - i, argv + i) ? 1 : 0; /* must not start a server */

  rg_dev d;
  int r = rg_open_tinygpu(&d, sock);
  if (r) return 1;
  if (!strcmp(cmd, "reset")) { /* must work on a card that no longer answers config reads */
    r = d.ops->reset(&d);
    uint32_t id0 = 0;
    int rr = rg_cfg_read(&d, 0, 4, &id0);
    printf("reset %s; config space now reads %08x%s\n", r ? "FAILED" : "done", id0, rr ? " (read failed)" : "");
    rg_close(&d);
    return r || rr || id0 == 0xFFFFFFFFu ? 1 : 0;
  }
  if ((r = rg_pci_identify(&d))) { fprintf(stderr, "rawgpu: no device answered config reads (%d)\n", r); rg_close(&d); return 1; }
  rg_ident id;
  rg_identify_family(&d, &id);

  if (!strcmp(cmd, "probe")) r = cmd_probe(&d, &id);
  else if (!strcmp(cmd, "cfg")) r = cmd_cfg(&d, i < argc && !strcmp(argv[i], "--clear-errors"));
  else if (!strcmp(cmd, "vbios")) {
    if (i + 1 < argc && !strcmp(argv[i], "-o")) r = cmd_vbios(&d, &id, argv[i + 1]);
    else { print_usage(stderr); r = -EINVAL; }
  } else if (!strcmp(cmd, "bench")) r = cmd_bench(&d, &id, i < argc ? atoi(argv[i]) : 2000);
  else if (!strcmp(cmd, "dma-test")) r = cmd_dma_test(&d, i < argc ? (size_t)atoi(argv[i]) : 1024);
  else if (!strcmp(cmd, "reset")) {
    r = d.ops->reset(&d);
    uint32_t id0 = 0;
    if (!r) r = rg_cfg_read(&d, 0, 4, &id0);
    printf("reset %s; config space now reads %04x:%04x\n", r ? "FAILED" : "done", id0 & 0xFFFF, id0 >> 16);
  }
  else {
    const rg_cmd *c = NULL;
    for (const rg_cmd *k = id.backend ? id.backend->cmds : NULL; k && k->name && !c; k++)
      if (!strcmp(k->name, cmd)) c = k;
    if (!c) {
      fprintf(stderr, "rawgpu: unknown command '%s'%s\n", cmd, id.backend ? "" : " (and this card has no backend)");
      print_usage(stderr);
      r = -EINVAL;
    } else {
      r = c->run(&d, &id, argc - i, argv + i);
      if (r == -EINVAL) fprintf(stderr, "usage: rawgpu %s %s\n", c->name, c->args);
    }
  }

  rg_close(&d);
  return r ? 1 : 0;
}
