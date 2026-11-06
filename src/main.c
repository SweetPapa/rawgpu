/* rawgpu CLI. Every command is read-only unless noted. */
#include "amd_fw.h"
#include "amd_kernel.h"
#include "rawgpu.h"
#include "rg.h"

#include <errno.h>
#include <limits.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#ifndef RAWGPU_DEFAULT_CLANG
#define RAWGPU_DEFAULT_CLANG "/opt/homebrew/opt/llvm/bin/clang"
#endif
#define RAWGPU_DEFAULT_CLANG_PATH RAWGPU_DEFAULT_CLANG
#ifndef RAWGPU_DEFAULT_LOCAL
#define RAWGPU_DEFAULT_LOCAL ""
#endif
#define RAWGPU_DEFAULT_LOCAL_DIR RAWGPU_DEFAULT_LOCAL

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
  "  doctor [--json]  check the setup: TinyGPU, the card, firmware, the kernel compiler (no GPU access)\n"
  "  up               WRITES: bring the card up if it needs it (post, mc-load, dpm-enable), as the library does\n"
  "  who [--json]     the TinyGPU server and the program using the card (never connects, never starts a server)\n"
  "  eject [--timeout SECONDS]\n"
  "                   wait for the program using the card to finish, stop the card's engines, close it,\n"
  "                   and stop the TinyGPU server; then says it is safe to unplug\n"
  "  kernel-info FILE describe the kernels in a code object (no GPU needed)\n"
  "  --version        print the version\n";

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

/* How many clients the server is connected to: its AF_UNIX sockets other than the listening one. A
 * look at its file descriptors, so it neither queues behind the client nor disturbs it. -1 if unknown.
 * With peers, also the kernel's handles for the other ends of up to max of those connections. */
static int server_connections(int pid, uint64_t *peers, int max) {
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
    if (si.psi.soi_family != AF_UNIX || (si.psi.soi_options & SO_ACCEPTCONN)) continue;
    if (peers && clients < max) peers[clients] = si.psi.soi_proto.pri_un.unsi_conn_pcb;
    clients++;
  }
  free(fds);
  return bytes > 0 ? clients : -1;
}

static int server_clients(int pid) { return server_connections(pid, NULL, 0); }

/* The processes at the other end of the server's connections: whose socket's control block is the one
 * the server's socket is connected to (lsof matches them the same way). Only this user's processes can
 * be inspected, which covers anything that could have opened the card. */
static int server_peers(int server, int *pids, int max) {
  uint64_t pcb[16];
  int npcb = server_connections(server, pcb, 16), found = 0;
  if (npcb > 16) npcb = 16;
  if (npcb <= 0) return 0;
  int n = proc_listallpids(NULL, 0);
  if (n <= 0) return 0;
  size_t cap = (size_t)n + 64;
  pid_t *all = calloc(cap, sizeof *all);
  if (!all) return 0;
  n = proc_listallpids(all, (int)(cap * sizeof *all));
  for (int i = 0; i < n && found < max; i++) {
    if (all[i] == server || all[i] == getpid()) continue;
    int bytes = proc_pidinfo(all[i], PROC_PIDLISTFDS, 0, NULL, 0);
    if (bytes <= 0) continue;
    struct proc_fdinfo *fds = malloc((size_t)bytes);
    if (!fds) continue;
    bytes = proc_pidinfo(all[i], PROC_PIDLISTFDS, 0, fds, bytes);
    bool hit = false;
    for (int j = 0; bytes > 0 && j < bytes / (int)sizeof *fds && !hit; j++) {
      if (fds[j].proc_fdtype != PROX_FDTYPE_SOCKET) continue;
      struct socket_fdinfo si;
      if (proc_pidfdinfo(all[i], fds[j].proc_fd, PROC_PIDFDSOCKETINFO, &si, sizeof si) != sizeof si) continue;
      if (si.psi.soi_family != AF_UNIX) continue;
      for (int k = 0; k < npcb && !hit; k++) hit = si.psi.soi_pcb == pcb[k];
    }
    free(fds);
    if (hit) pids[found++] = all[i];
  }
  free(all);
  return found;
}

/* pid's command line, its arguments joined by spaces (the executable's name if they are unreadable) */
static void proc_cmdline(int pid, char *out, size_t n) {
  char *buf;
  const char *argv[64];
  int argc = rg_proc_args(pid, &buf, argv, 64);
  out[0] = 0;
  if (argc > 0) {
    size_t len = 0;
    for (int i = 0; i < argc && len + 1 < n; i++)
      len += (size_t)snprintf(out + len, n - len, "%s%s", i ? " " : "", argv[i]);
  } else if (proc_name(pid, out, (uint32_t)n) <= 0) {
    snprintf(out, n, "?");
  }
  free(buf);
}

static void json_str(const char *s) {
  putchar('"');
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') printf("\\%c", c);
    else if (c < 0x20) printf("\\u%04x", c);
    else putchar(c);
  }
  putchar('"');
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
  sigset_t stop, unblocked;
  sigemptyset(&stop);
  sigaddset(&stop, SIGINT);
  sigaddset(&stop, SIGTERM);
  sigaddset(&stop, SIGHUP);
  for (;;) {
    nothers = rg_tinygpu_processes(path, &pid, &other);
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
      /* From here a stop request must not cut the conversation short: the server would later answer
       * a closed socket and die of SIGPIPE, mid-cleanup. So Ctrl-C and SIGTERM wait until the end. */
      sigprocmask(SIG_BLOCK, &stop, &unblocked);
      r = rg_attach_tinygpu(&d, path, NULL); /* if another client got in first, this waits for it */
      if (r != -ECONNREFUSED) break;
      sigprocmask(SIG_SETMASK, &unblocked, NULL);
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
  sigset_t pending;
  if (!sigpending(&pending) && (sigismember(&pending, SIGINT) || sigismember(&pending, SIGTERM) || sigismember(&pending, SIGHUP)))
    printf("(A stop request came while the card was being closed, so the eject finished first.)\n");
  if (nothers) return not_safe_others(nothers, other);
  printf("\nSafe to unplug. The next rawgpu command starts a new server.\n");
  return 0;
}

/* `rawgpu who`: is a server running, and which program has the card? Only looks at processes, so it
 * can run at any time: it never queues behind a client and never starts a server. */
static int cmd_who(const char *sock, int argc, char **argv) {
  bool json = argc == 1 && !strcmp(argv[0], "--json");
  if (argc && !json) { fprintf(stderr, "usage: rawgpu who [--json]\n"); return -EINVAL; }
  char buf[1024];
  const char *path = rg_tinygpu_sock_path(sock, buf, sizeof buf);
  int pid, other, nothers = rg_tinygpu_processes(path, &pid, &other);
  int clients = pid ? server_clients(pid) : 0, peers[8], npeers = clients > 0 ? server_peers(pid, peers, 8) : 0;
  char cmdline[4096];
  if (json) {
    printf("{\"socket\":");
    json_str(path);
    printf(",\"server_pid\":%d,\"client_count\":%d,\"clients\":[", pid, clients);
    for (int i = 0; i < npeers; i++) {
      proc_cmdline(peers[i], cmdline, sizeof cmdline);
      struct proc_bsdinfo bi;
      long started = proc_pidinfo(peers[i], PROC_PIDTBSDINFO, 0, &bi, sizeof bi) == sizeof bi ? (long)bi.pbi_start_tvsec : 0;
      printf("%s{\"pid\":%d,\"started\":%ld,\"command\":", i ? "," : "", peers[i], started);
      json_str(cmdline);
      printf("}");
    }
    printf("],\"other_tinygpu\":%d}\n", nothers);
    return 0;
  }
  if (!pid) printf("No TinyGPU server is running for rawgpu (%s), so nothing has the card through it.\n", path);
  else if (clients == 0) printf("TinyGPU server %d is running, and nothing has the card.\n", pid);
  else {
    printf("TinyGPU server %d is running. The card is in use by:\n", pid);
    for (int i = 0; i < npeers; i++) {
      proc_cmdline(peers[i], cmdline, sizeof cmdline);
      printf("  pid %d: %s\n", peers[i], cmdline);
    }
    if (npeers < clients) printf("  %d connection%s whose process could not be found\n", clients - npeers, clients - npeers > 1 ? "s" : "");
  }
  if (nothers) printf("%d other TinyGPU process%s running (pid %d), e.g. another program's server.\n", nothers,
                      nothers > 1 ? "es are" : " is", other);
  return 0;
}

/* ---- rawgpu doctor ---- */

/* Does a command's output contain needle? (for systemextensionsctl and ioreg) */
static bool output_has(const char *cmd, const char *needle, const char *also) {
  FILE *p = popen(cmd, "r");
  if (!p) return false;
  char line[2048];
  bool hit = false;
  while (fgets(line, sizeof line, p))
    if (strstr(line, needle) && (!also || strstr(line, also))) hit = true;
  pclose(p);
  return hit;
}

static bool file_ok(const char *dir, const char *rel) {
  char path[1400];
  snprintf(path, sizeof path, "%s/%s", dir, rel);
  struct stat st;
  return !stat(path, &st) && S_ISREG(st.st_mode) && st.st_size > 0;
}

/* Where the fetch and install scripts are: a checkout's tools/ for a build from one; beside a release
 * binary, ../share/rawgpu (the tarball) or ../Resources/scripts (inside the menu bar app). */
static void script_dir(char *out, size_t n) {
  char exe[1024], cand[1200], real[PATH_MAX];
  uint32_t sz = sizeof exe;
  struct stat st;
  out[0] = 0;
  if (RAWGPU_DEFAULT_LOCAL_DIR[0]) {
    snprintf(cand, sizeof cand, "%s/../tools", RAWGPU_DEFAULT_LOCAL_DIR);
    if (!stat(cand, &st) && realpath(cand, real)) { snprintf(out, n, "%s", real); return; }
  }
  if (_NSGetExecutablePath(exe, &sz)) return;
  char *slash = strrchr(exe, '/');
  if (!slash) return;
  *slash = 0;
  const char *rel[] = {"../share/rawgpu", "../Resources/scripts", "scripts"};
  for (int i = 0; i < 3; i++) {
    snprintf(cand, sizeof cand, "%s/%s", exe, rel[i]);
    if (!stat(cand, &st) && realpath(cand, real)) { snprintf(out, n, "%s", real); return; }
  }
}

/* `rawgpu doctor`: is everything rawgpu needs in place? Only looks: never opens the card or starts a server. */
static int cmd_doctor(const char *sock, int argc, char **argv) {
  bool json = argc == 1 && !strcmp(argv[0], "--json");
  if (argc && !json) { fprintf(stderr, "usage: rawgpu doctor [--json]\n"); return -EINVAL; }
  int arm = 0;
  size_t al = sizeof arm;
  if (sysctlbyname("hw.optional.arm64", &arm, &al, NULL, 0)) arm = 0;
  bool tg_app = access("/Applications/TinyGPU.app/Contents/MacOS/TinyGPU", X_OK) == 0;
  bool tg_on = tg_app && output_has("systemextensionsctl list 2>/dev/null", "org.tinygrad.tinygpu", "[activated enabled]");
  bool bound = output_has("ioreg -r -n tinygpu -d 1 2>/dev/null", "tinygpu", NULL);
  char buf[1024];
  const char *path = rg_tinygpu_sock_path(sock, buf, sizeof buf);
  int server, other, others = rg_tinygpu_processes(path, &server, &other);
  int clients = server ? server_clients(server) : 0;

  const char *data = rg_data_dir();
  char fw[1100], libs[1100];
  const char *fwenv = getenv("RAWGPU_FIRMWARE"), *libenv = getenv("RAWGPU_DEVICE_LIBS");
  snprintf(fw, sizeof fw, "%s", fwenv ? fwenv : "");
  if (!fwenv) snprintf(fw, sizeof fw, "%s/firmware", data);
  snprintf(libs, sizeof libs, "%s", libenv ? libenv : "");
  if (!libenv) snprintf(libs, sizeof libs, "%s/device-libs", data);
  static const char *common[] = {"sdma", "sdma1", "rlc", "mec", "mec_2", "mec2", "mec2_2", "ce_2", "pfp_2", "me_2"};
  static const char *smc[] = {"smc", "k_smc", "k2_smc", "smc_sk"}, *mc[] = {"mc", "k_mc"};
  char missing[512] = "";
  for (size_t i = 0; i < sizeof common / sizeof *common; i++) {
    char rel[64];
    snprintf(rel, sizeof rel, "amdgpu/polaris10_%s.bin", common[i]);
    if (!file_ok(fw, rel)) snprintf(missing + strlen(missing), sizeof missing - strlen(missing), "%s%s", missing[0] ? " " : "", rel + 7);
  }
  bool any_smc = false, any_mc = false;
  for (size_t i = 0; i < 4; i++) { char rel[64]; snprintf(rel, sizeof rel, "amdgpu/polaris10_%s.bin", smc[i]); any_smc |= file_ok(fw, rel); }
  for (size_t i = 0; i < 2; i++) { char rel[64]; snprintf(rel, sizeof rel, "amdgpu/polaris10_%s.bin", mc[i]); any_mc |= file_ok(fw, rel); }
  bool fw_ok = !missing[0] && any_smc && any_mc;
  bool libs_ok = file_ok(libs, "opencl.bc") && file_ok(libs, "ocml.bc") && file_ok(libs, "ockl.bc");
  const char *clang = getenv("RAWGPU_CLANG") ? getenv("RAWGPU_CLANG") : RAWGPU_DEFAULT_CLANG_PATH;
  bool clang_ok = access(clang, X_OK) == 0;
  char scripts[1200];
  script_dir(scripts, sizeof scripts);
  if (!scripts[0]) snprintf(scripts, sizeof scripts, "<rawgpu>/tools");

  if (json) {
    printf("{\"version\":");
    json_str(rawgpu_version());
    printf(",\"apple_silicon\":%s,\"tinygpu_installed\":%s,\"tinygpu_enabled\":%s,\"card_bound\":%s,"
           "\"server_pid\":%d,\"client_count\":%d,\"other_tinygpu\":%d,\"data_dir\":",
           arm ? "true" : "false", tg_app ? "true" : "false", tg_on ? "true" : "false", bound ? "true" : "false",
           server, clients, others);
    json_str(data);
    printf(",\"firmware_dir\":");
    json_str(fw);
    printf(",\"firmware_ok\":%s,\"firmware_missing\":", fw_ok ? "true" : "false");
    json_str(missing[0] ? missing : any_smc && any_mc ? "" : "smc/mc");
    printf(",\"device_libs_dir\":");
    json_str(libs);
    printf(",\"device_libs_ok\":%s,\"clang\":", libs_ok ? "true" : "false");
    json_str(clang);
    printf(",\"clang_ok\":%s,\"scripts\":", clang_ok ? "true" : "false");
    json_str(scripts);
    printf("}\n");
    return 0;
  }
  printf("rawgpu %s\n\n", rawgpu_version());
  char fix_tg[1400], fix_fw[1400], fix_libs[1400];
  snprintf(fix_tg, sizeof fix_tg, "run %s/install-tinygpu.sh", scripts);
  snprintf(fix_fw, sizeof fix_fw, "run %s/fetch-firmware.sh (into %s)", scripts, fw);
  snprintf(fix_libs, sizeof fix_libs, "run %s/fetch-device-libs.sh (into %s)", scripts, libs);
  const struct { bool ok; const char *what, *fix; } c[7] = {
    {arm, "Apple Silicon Mac", "rawgpu needs an Apple Silicon Mac (TinyGPU is arm64 only)"},
    {tg_app, "TinyGPU installed (/Applications/TinyGPU.app)", fix_tg},
    {tg_on, "TinyGPU driver extension approved and enabled",
     "approve it: System Settings > General > Login Items & Extensions > Driver Extensions > TinyGPU"},
    {bound, "a card is attached and TinyGPU has bound to it",
     "plug in the enclosure with an AMD card and power it on; it binds about 10 s after the driver is approved"},
    {fw_ok, "AMD firmware", fix_fw},
    {libs_ok, "device libraries (only to compile OpenCL kernels)", fix_libs},
    {clang_ok, "kernel compiler: LLVM with the AMDGPU target (only to compile kernels)",
     "brew install llvm, or set RAWGPU_CLANG"},
  };
  int bad = 0;
  for (int i = 0; i < 7; i++) {
    printf("  %s  %s\n", c[i].ok ? "ok " : "NO ", c[i].what);
    if (!c[i].ok) { printf("       -> %s\n", c[i].fix); bad++; }
  }
  printf("\n  data folder:   %s\n", data);
  if (server) printf("  TinyGPU server: pid %d, %s\n", server, clients ? "a program has the card (rawgpu who)" : "idle");
  else printf("  TinyGPU server: not running (the next rawgpu command starts it)\n");
  if (others) printf("  note: %d other TinyGPU process%s running, e.g. tinygrad's own server\n", others, others > 1 ? "es are" : " is");
  printf("\n%s\n", bad ? "Fix the NO lines above, then run `rawgpu up`." : "Everything is in place. `rawgpu up` brings the card up.");
  return 0;
}

/* `rawgpu up`: what rawgpu_open(RAWGPU_OPEN_BRINGUP) does. A card that is already up is left alone. */
static int cmd_up(const char *sock) {
  rawgpu *g;
  if (rawgpu_open(&g, sock, RAWGPU_OPEN_BRINGUP)) { fprintf(stderr, "rawgpu: %s\n", rawgpu_error(NULL)); return -EIO; }
  rawgpu_info info;
  if (!rawgpu_get_info(g, &info))
    printf("The card is up: %s; engine %d MHz, memory %d MHz, %d C, fan %d%%, %llu of %llu MB VRAM free.\n", info.name,
           info.engine_mhz, info.memory_mhz, info.temperature_c, info.fan_percent,
           (unsigned long long)(info.vram_free_bytes >> 20), (unsigned long long)(info.vram_bytes >> 20));
  rawgpu_close(g);
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
  if (!strcmp(cmd, "who")) return cmd_who(sock, argc - i, argv + i) ? 1 : 0;     /* nor this */
  if (!strcmp(cmd, "doctor")) return cmd_doctor(sock, argc - i, argv + i) ? 1 : 0; /* nor this */
  if (!strcmp(cmd, "--version") || !strcmp(cmd, "version")) { printf("rawgpu %s\n", rawgpu_version()); return 0; }
  if (!strcmp(cmd, "up")) return cmd_up(sock) ? 1 : 0;

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
