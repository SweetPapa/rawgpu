#!/usr/bin/env python3
"""rawgpu visual demos: every pixel and every physics step is computed by the
RX 590 through librawgpu, then shown in a window on the Mac.

  python3 demos/rawgpu_demo.py                  # bouncing balls; keys 1/2/3 switch scenes
  python3 demos/rawgpu_demo.py --scene valley   # fly over a procedural valley
  python3 demos/rawgpu_demo.py --scene galaxy   # two galaxies colliding (N-body)
  python3 demos/rawgpu_demo.py --bench          # each scene for 15 s, then a summary
  python3 demos/rawgpu_demo.py --record out.mp4 --seconds 30

Keys: 1 balls, 2 valley, 3 galaxy, space kick the balls / restart the galaxies,
H hide the overlay, S screenshot, Esc or Q quit.

The GPU renders each frame straight into Mac memory (a GART-mapped host
buffer). Two of them alternate: while the GPU draws one, the window shows
the other.
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import rawgpu  # noqa: E402


def f4(*v):
    return np.array(list(v) + [0.0] * (4 - len(v)), np.float32)


def up16(x):
    return (x + 15) // 16 * 16


# ---------------------------------------------------------------- scenes

class Balls:
    name = "Bouncing balls"
    BOX = 6.0

    def __init__(self, gpu, w, h, n=512):
        self.gpu, self.w, self.h = gpu, w, h
        self.prog = gpu.program(open(os.path.join(HERE, "balls.cl")).read())
        self.setup(n)

    def setup(self, n):
        self.n = max(64, n // 64 * 64)
        rng = np.random.default_rng(3)
        n = self.n
        r = rng.uniform(0.22, 0.42, n).astype(np.float32)
        side = int(math.ceil(math.sqrt(n / 6)))
        pos = np.zeros((n, 4), np.float32)
        sp = 2 * self.BOX / (side + 1)
        for i in range(n):
            layer, k = divmod(i, side * side)
            pos[i, 0] = -self.BOX + sp * (1 + k % side) + rng.uniform(-0.05, 0.05)
            pos[i, 2] = -self.BOX + sp * (1 + k // side) + rng.uniform(-0.05, 0.05)
            pos[i, 1] = 1.0 + layer * 0.95
        pos[:, 3] = r
        vel = np.zeros((n, 4), np.float32)
        vel[:, :3] = rng.normal(0, 0.6, (n, 3))
        hue = rng.uniform(0, 1, n)
        mat = np.zeros((n, 4), np.float32)
        for i in range(n):  # bright colours from a hue wheel, a third of them mirror-like
            hh = hue[i] * 6
            c = [abs(hh - 3) - 1, 2 - abs(hh - 2), 2 - abs(hh - 4)]
            mat[i, :3] = np.clip(c, 0.05, 1) * 0.85
            mat[i, 3] = 0.55 if rng.uniform() < 0.33 else 0.06
        self.pos = [self.gpu.array(pos), self.gpu.empty((n, 4))]
        self.vel = [self.gpu.array(vel), self.gpu.empty((n, 4))]
        self.mat = self.gpu.array(mat)
        self.cur, self.last_kick, self.kicks = 0, -3.0, 0

    def kick(self, t):
        self.kicks += 1
        self.prog.balls_kick(self.vel[self.cur], self.n, self.kicks * 7919, 9.0, global_size=self.n, local_size=64)
        self.last_kick = t

    def frame(self, out, t, dt):
        sub = 8
        sdt = min(dt, 1 / 20) / sub
        if t - self.last_kick > 6.0:
            self.kick(t)
        for _ in range(sub):
            a, b = self.cur, 1 - self.cur
            self.prog.balls_step(self.pos[a], self.vel[a], self.pos[b], self.vel[b], self.n, sdt, self.BOX, 0.78,
                                 global_size=self.n, local_size=64)
            self.cur = b
        ang = 0.12 * t
        cam = f4(15 * math.cos(ang), 6.5 + 1.5 * math.sin(0.2 * t), 15 * math.sin(ang))
        return self.prog.balls_render(out, self.w, self.h, self.pos[self.cur], self.mat, self.n, cam, f4(0, 1.2, 0),
                                      self.BOX, global_size=(up16(self.w), up16(self.h)), local_size=(16, 16))

    def stats(self, fps):
        return [f"{self.n} balls, {8 * self.n * self.n * fps / 1e6:.0f} M collision checks/s",
                f"ray tracing: {self.w * self.h * fps / 1e6:.0f} Mpixel/s; each pixel tests every ball 3-13 times "
                f"(shadow, occlusion, reflections)"]

    def key(self, k, t):
        if k == "space":
            self.kick(t)
        elif k in ("+", "="):
            self.setup(self.n * 2)
        elif k == "-":
            self.setup(self.n // 2)


class Valley:
    name = "Valley flyover"
    SPEED = 85.0  # m/s

    def __init__(self, gpu, w, h):
        self.gpu, self.w, self.h = gpu, w, h
        self.prog = gpu.program(open(os.path.join(HERE, "valley.cl")).read())
        # The camera flies along a valley floor: sample the terrain under and around the path on the GPU,
        # then fly a smoothed altitude above the highest ground nearby.
        self.step = 10.0
        s = np.arange(0, 200_000, self.step, dtype=np.float32)
        xs, zs = self.path(s)
        pts = []
        for dx in (-60, 0, 60):
            pts.append(np.stack([xs + dx, zs], 1))
        pts = np.concatenate(pts).astype(np.float32)
        bxz, bh = gpu.array(pts), gpu.empty(len(pts), np.float32)
        npad = (len(pts) + 255) // 256 * 256
        self.prog.valley_heights(bh, bxz, len(pts), global_size=npad, local_size=256)
        hts = bh.numpy().reshape(3, -1).max(axis=0)
        bxz.free()
        bh.free()
        win = int(300 / self.step)  # look this far ahead for high ground
        ahead = np.array([hts[i:i + win].max() for i in range(len(hts))], np.float32)
        alt = ahead + 38.0
        for _ in range(3):  # smooth it so the flight glides
            alt = np.convolve(np.pad(alt, 40, mode="edge"), np.ones(81) / 81, mode="valid")
        self.alt = np.maximum(alt, hts + 30.0)

    @staticmethod
    def path(s):
        return -545.0 * np.sin(0.0005 * s) + 70.0 * np.sin(0.0037 * s), s

    def at(self, s):
        i = min(int(s / self.step), len(self.alt) - 2)
        f = s / self.step - i
        y = self.alt[i] * (1 - f) + self.alt[i + 1] * f
        x, z = self.path(np.float32(s))
        return float(x), float(y), float(z)

    def frame(self, out, t, dt):
        s = self.SPEED * t + 2000.0
        x, y, z = self.at(s)
        tx, ty, tz = self.at(s + 160.0)
        bx, _, _ = self.at(s + 320.0)
        roll = max(-0.35, min(0.35, (bx - 2 * tx + x) * -0.004))  # bank into the turns
        return self.prog.valley(out, self.w, self.h, f4(x, y, z, roll), f4(tx, ty - 22.0, tz), t,
                                global_size=(up16(self.w), up16(self.h)), local_size=(16, 16))

    def stats(self, fps):
        return [f"{self.w * self.h * fps / 1e6:.0f} Mpixel/s ray-marched; up to 260 steps x 7 noise octaves per pixel",
                "plus soft shadows, water reflections and clouds"]

    def key(self, k, t):
        pass


class Galaxy:
    name = "Galaxy collision"
    STEPS = 2

    def __init__(self, gpu, w, h, n=32768):
        self.gpu, self.w, self.h = gpu, w, h
        self.prog = gpu.program(open(os.path.join(HERE, "galaxy.cl")).read())
        self.n = max(512, n // 512 * 512)
        self.accum = gpu.zeros(w * h * 3, np.uint32)
        self.reset()

    def reset(self):
        rng = np.random.default_rng(11)
        n, half = self.n, self.n // 2
        pos = np.zeros((n, 4), np.float32)
        vel = np.zeros((n, 4), np.float32)
        core, disk = 1000.0, 160.0
        setups = [((-30.0, 0.0, -8.0), (2.6, 0.0, 0.75), (0.0, 1.0, 0.0)),
                  ((30.0, 0.0, 8.0), (-2.6, 0.0, -0.75), (0.35, 0.86, 0.36))]
        for g, (c, v, nrm) in enumerate(setups):
            sl = slice(g * half, (g + 1) * half)
            nrm = np.array(nrm) / np.linalg.norm(nrm)
            e1 = np.cross(nrm, [0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.9 else np.cross(nrm, [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(nrm, e1)
            u = rng.uniform(0, 1, half)
            r = 1.2 + 13.0 * u ** 0.85
            arm = rng.integers(0, 2, half)
            th = arm * math.pi + 1.6 * np.log(r) + rng.normal(0, 0.45, half)
            hgt = rng.normal(0, 0.18, half)
            p = (np.outer(np.cos(th) * r, e1) + np.outer(np.sin(th) * r, e2) + np.outer(hgt, nrm))
            menc = core + disk * np.argsort(np.argsort(r)) / half  # mass inside each star's radius
            vc = np.sqrt(menc / r)
            tang = np.cross(nrm, p)
            tang /= np.linalg.norm(tang, axis=1, keepdims=True)
            pv = tang * vc[:, None] + rng.normal(0, 0.15, (half, 3))
            pos[sl, :3] = p + c
            pos[sl, 3] = disk / half
            vel[sl, :3] = pv + v
            vel[sl, 3] = g
            pos[g * half, :3], pos[g * half, 3] = c, core  # the central mass
            vel[g * half, :3] = v
        self.pos = [self.gpu.array(pos), self.gpu.empty((n, 4))]
        self.vel = [self.gpu.array(vel), self.gpu.empty((n, 4))]
        self.cur, self.started = 0, None

    def frame(self, out, t, dt):
        if self.started is None:
            self.started = t
        if t - self.started > 60.0:  # the merger has settled: start over
            self.reset()
            self.started = t
        for _ in range(self.STEPS):
            a, b = self.cur, 1 - self.cur
            self.prog.nbody_step(self.pos[a], self.vel[a], self.pos[b], self.vel[b], self.n, 0.0025, 0.09,
                                 global_size=self.n, local_size=256)
            self.cur = b
        ang = 0.04 * t + 0.6
        cam = f4(56 * math.cos(ang), 24 + 5 * math.sin(0.11 * t), 56 * math.sin(ang))
        tgt = f4(0, 0, 0)
        exposure = 2.2 * 32768 / self.n
        self.prog.galaxy_splat(self.pos[self.cur], self.vel[self.cur], self.n, self.accum, self.w, self.h, cam, tgt,
                               exposure, global_size=self.n, local_size=256)
        self.prog.galaxy_resolve(self.accum, out, self.w, self.h, global_size=(up16(self.w), up16(self.h)),
                                 local_size=(16, 16))
        m = self.w * self.h * 3
        return self.prog.galaxy_clear(self.accum, m, global_size=(m + 255) // 256 * 256, local_size=256)

    def stats(self, fps):
        inter = self.n * self.n * self.STEPS * fps
        return [f"{self.n} stars, {self.STEPS} gravity steps/frame: {inter / 1e9:.1f} G interactions/s",
                f">= {20 * inter / 1e9:.0f} GFLOPS sustained in the gravity kernel (20 FLOPs each; frames include drawing)"]

    def key(self, k, t):
        if k == "space":
            self.reset()


SCENES = {"balls": Balls, "valley": Valley, "galaxy": Galaxy}
ORDER = ["balls", "valley", "galaxy"]


# ---------------------------------------------------------------- the window

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=ORDER, default="balls")
    ap.add_argument("--size", default="1280x720", help="render size, WxH")
    ap.add_argument("--window", default=None, help="window size if different (frames are scaled)")
    ap.add_argument("--seconds", type=float, default=0, help="quit after this long (per scene with --bench)")
    ap.add_argument("--bench", action="store_true", help="run every scene, then print a summary")
    ap.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    ap.add_argument("--record", metavar="FILE.mp4", help="also write a video (60 fps of simulated time, via ffmpeg)")
    ap.add_argument("--shots", metavar="DIR", help="save a screenshot every --shot-every seconds")
    ap.add_argument("--shot-every", type=float, default=3.0)
    ap.add_argument("--balls", type=int, default=512)
    ap.add_argument("--stars", type=int, default=32768)
    args = ap.parse_args()
    if args.bench and not args.seconds:
        args.seconds = 15.0
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    w, h = (int(v) for v in args.size.lower().split("x"))
    ww, wh = (int(v) for v in args.window.lower().split("x")) if args.window else (w, h)
    gpu = rawgpu.Device()
    pygame.init()
    screen = pygame.display.set_mode((ww, wh))
    pygame.display.set_caption("rawgpu: RX 590 over Thunderbolt")
    font = pygame.font.Font(None, 22)
    big = pygame.font.Font(None, 34)

    slots = [gpu.empty((h, w), np.uint32, host=True) for _ in range(2)]
    views = [s.host_view() for s in slots]
    scenes = {}

    def get(name):
        if name not in scenes:
            msg = big.render(f"compiling {SCENES[name].name}...", True, (230, 230, 230))
            screen.fill((10, 10, 14))
            screen.blit(msg, (30, 30))
            pygame.display.flip()
            scenes[name] = SCENES[name](gpu, w, h, **({"n": args.balls} if name == "balls" else
                                                       {"n": args.stars} if name == "galaxy" else {}))
        return scenes[name]

    ffmpeg = None
    if args.record:
        ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                                   f"{ww}x{wh}", "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                   "-crf", "18", "-pix_fmt", "yuv420p", args.record], stdin=subprocess.PIPE)
    if args.shots:
        os.makedirs(args.shots, exist_ok=True)

    queue = ORDER[ORDER.index(args.scene):] + ORDER[:ORDER.index(args.scene)] if args.bench else [args.scene]
    results = []
    show_hud, running = True, True
    info, info_at = gpu.info, 0.0
    while queue and running:
        name = queue.pop(0)
        scene = get(name)
        gpu.finish()
        t_sim, frames, start = 0.0, 0, time.time()
        fps, last, pending, k = 0.0, time.time(), None, 0
        temps, clocks, next_shot = [], [], args.shot_every
        fixed = bool(args.record or args.bench)
        while running:
            now = time.time()
            real_dt = now - last
            last = now
            dt = 1 / 60 if fixed else min(real_dt, 0.05)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    key = pygame.key.name(ev.key)
                    if key in ("escape", "q"):
                        running = False
                    elif key in ("1", "2", "3") and not args.bench:
                        queue = [ORDER[int(key) - 1]]
                        break
                    elif key == "h":
                        show_hud = not show_hud
                    elif key == "s":
                        path = f"rawgpu-{name}-{int(time.time())}.png"
                        pygame.image.save(screen, path)
                        print("saved", path)
                    else:
                        scene.key(key, t_sim)
            if queue and not args.bench and queue[0] != name:
                break
            event = scene.frame(slots[k], t_sim, dt)  # the GPU starts on this frame...
            if pending is not None:  # ...while the window shows the previous one
                gpu.wait(pending[0])
                surf = pygame.image.frombuffer(views[pending[1]], (w, h), "RGBA")
                if (ww, wh) != (w, h):
                    surf = pygame.transform.smoothscale(surf, (ww, wh))
                screen.blit(surf, (0, 0))
                if now - info_at > 0.5:
                    info, info_at = gpu.info, now
                    temps.append(info["temperature_c"])
                    clocks.append(info["engine_mhz"])
                if real_dt > 0:
                    fps = 0.9 * fps + 0.1 / max(real_dt, 1e-4) if fps else 1 / max(real_dt, 1e-4)
                if show_hud:
                    lines = [f"{scene.name}   {fps:5.1f} fps   {w}x{h}",
                             *scene.stats(fps),
                             f"RX 590: engine {info['engine_mhz']} MHz, memory {info['memory_mhz']} MHz, "
                             f"{info['temperature_c']} C",
                             "1 balls  2 valley  3 galaxy  space kick/reset  H overlay  S screenshot  Q quit"]
                    pad = 8
                    surfs = [font.render(s, True, (240, 240, 240)) for s in lines]
                    bw = max(s.get_width() for s in surfs) + 2 * pad
                    bh = sum(s.get_height() for s in surfs) + 2 * pad
                    box = pygame.Surface((bw, bh), pygame.SRCALPHA)
                    box.fill((0, 0, 0, 140))
                    screen.blit(box, (10, 10))
                    yy = 10 + pad
                    for s in surfs:
                        screen.blit(s, (10 + pad, yy))
                        yy += s.get_height()
                pygame.display.flip()
                frames += 1
                if ffmpeg:
                    ffmpeg.stdin.write(pygame.image.tobytes(screen, "RGB"))
                if args.shots and t_sim >= next_shot:
                    pygame.image.save(screen, os.path.join(args.shots, f"{name}-{int(t_sim):03d}s.png"))
                    next_shot += args.shot_every
            pending = (event, k)
            k ^= 1
            t_sim += dt
            if args.seconds and (t_sim if fixed else time.time() - start) >= args.seconds:
                break
        gpu.wait(pending[0]) if pending else None
        wall = time.time() - start
        if frames:
            avg = frames / wall
            results.append((scene.name, frames, avg, scene.stats(avg), max(temps or [0]),
                            (min(clocks or [0]), max(clocks or [0]))))
    if ffmpeg:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print("video:", args.record)
    for name, frames, avg, st, tmax, (cmin, cmax) in results:
        print(f"{name:<18} {frames:5d} frames  {avg:6.1f} fps  engine {cmin}-{cmax} MHz  max {tmax} C")
        for s in st:
            print(f"{'':<18} {s}")
    pygame.quit()
    gpu.close()


if __name__ == "__main__":
    main()
