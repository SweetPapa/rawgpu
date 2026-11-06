"""A JEPA world model: learn to predict the future in latent space, then plan by searching imagined rollouts.

The world is a small arena (32 x 32 pixels): an agent the actions move, and a ball that bounces on its
own. An observation is the current frame plus the ball's previous position (so its velocity is visible).

    encoder E        observation -> latent z (an MLP)
    target E'        an exponential moving average of E: the targets, with no gradient
    predictor P      (z, action) -> the next z, applied again and again: a latent rollout

The loss is the distance between P's rollout from E(o_0) and E'(o_1..o_H), plus VICReg's variance and
covariance terms on E's latents so they cannot collapse to a constant, plus an inverse-dynamics term: a
small head must tell which action was taken from (z_t, z_t+1). Without it the latents keep the ball, whose
motion is the hard part to predict, and drop the agent; with it they keep what the actions control.
Nothing ever reconstructs pixels.

To see what the latents know, a linear probe (ridge regression, fit on E' latents) reads the agent's and
the ball's positions out of them. Applied to a rollout, it shows how far ahead the imagination stays
right. Planning uses the same pieces: random-shooting model-predictive control, scoring candidate action
sequences by where the probe says the agent ends up.
"""
import math
import time

import numpy as np

META = {
    "name": "JEPA world model",
    "group": "From scratch",
    "tagline": "Predict the future in latent space, then plan with it",
    "description": "A joint-embedding predictive architecture on a toy world. An encoder, its moving-average "
                   "twin, and an action-conditioned predictor learn to roll the world forward in latent space, "
                   "never reconstructing pixels. A linear probe decodes positions from imagined latents, and a "
                   "planner searches latent rollouts to steer the agent to goals.",
    "params": [
        {"key": "latent", "label": "Latent size", "type": "choice", "default": 128, "choices": [16, 32, 64, 128]},
        {"key": "hidden", "label": "Hidden width", "type": "choice", "default": 512, "choices": [256, 512, 1024]},
        {"key": "horizon", "label": "Rollout horizon", "type": "int", "default": 8, "min": 1, "max": 32,
         "unit": "steps", "help": "How many steps ahead the predictor is trained to imagine."},
        {"key": "batch", "label": "Batch", "type": "int", "default": 256, "min": 16, "max": 2048},
        {"key": "steps", "label": "Steps", "type": "int", "default": 4000, "min": 10},
        {"key": "lr", "label": "Learning rate", "type": "float", "default": 1e-3},
        {"key": "ema", "label": "Target EMA", "type": "float", "default": 0.996, "advanced": True,
         "help": "How slowly the target encoder follows the encoder (rises to 1 over the run)."},
        {"key": "var_weight", "label": "Variance weight", "type": "float", "default": 1.0, "advanced": True},
        {"key": "cov_weight", "label": "Covariance weight", "type": "float", "default": 0.04, "advanced": True},
        {"key": "idm_weight", "label": "Inverse-dynamics weight", "type": "float", "default": 0.3, "advanced": True,
         "help": "Keeps what the actions control in the latents (0: off, and the agent tends to vanish)."},
        {"key": "eval_every", "label": "Evaluate every", "type": "int", "default": 500, "unit": "steps",
         "advanced": True},
    ],
    "tiles": [
        {"key": "agent_px_8", "label": "Agent error at 8 steps", "fmt": ".2f", "best": "min", "unit": "px"},
        {"key": "ball_px_8", "label": "Ball error at 8 steps", "fmt": ".2f", "best": "min", "unit": "px"},
        {"key": "plan_success", "label": "Planner reaches goal", "fmt": ".0%", "best": "max"},
        {"key": "z_std", "label": "Latent spread", "fmt": ".3f", "help": "about 1 is healthy; near 0, collapse"},
    ],
    "charts": [
        {"title": "Loss", "series": [{"key": "pred_loss", "label": "Prediction"}, {"key": "var_loss", "label": "Variance"},
                                     {"key": "cov_loss", "label": "Covariance"}, {"key": "idm_loss", "label": "Inverse dynamics"}],
         "y": "log"},
        {"title": "Decoded position error at 8 steps", "series": [{"key": "agent_px_8", "label": "Agent"},
                                                                  {"key": "ball_px_8", "label": "Ball"}], "unit": "px"},
        {"title": "Latent spread", "series": [{"key": "z_std", "label": "Std of z"}], "small": True},
    ],
    "panels": ["rollout", "planning"],
}

S = 32          # arena size in pixels
SIGMA = 1.2     # blob radius
ACTS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], np.float32)  # stay, up, down, left, right
ACT_NAMES = ["stay", "up", "down", "left", "right"]
SPEED = 2.0


class Arena:
    """N arenas stepped together (numpy, on the host)"""

    def __init__(self, n, rng):
        self.rng = rng
        self.agent = rng.uniform(3, S - 4, (n, 2)).astype(np.float32)
        self.ball = rng.uniform(3, S - 4, (n, 2)).astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(0.7, 1.6, n)
        self.vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        self.prev_ball = self.ball - self.vel

    def step(self, a):
        self.agent = np.clip(self.agent + ACTS[a] * SPEED, 2, S - 3)
        self.prev_ball = self.ball.copy()
        b = self.ball + self.vel
        for lo_hi in range(2):
            lo, hi = 1.0, S - 2.0
            under, over = b[:, lo_hi] < lo, b[:, lo_hi] > hi
            b[under, lo_hi] = 2 * lo - b[under, lo_hi]
            b[over, lo_hi] = 2 * hi - b[over, lo_hi]
            self.vel[under | over, lo_hi] *= -1
        self.ball = b

    def state(self):
        return np.concatenate([self.agent, self.ball], 1)

    def obs(self):
        return render(self.agent, self.ball, self.prev_ball)


_g = np.arange(S, dtype=np.float32)


def _blob(p):
    gx = np.exp(-(_g[None, :] - p[:, 0:1]) ** 2 / (2 * SIGMA ** 2))
    gy = np.exp(-(_g[None, :] - p[:, 1:2]) ** 2 / (2 * SIGMA ** 2))
    return (gy[:, :, None] * gx[:, None, :]).reshape(len(p), S * S)


def render(agent, ball, prev_ball):
    """[n, 3 * S * S]: the agent, the ball, and the ball one step ago"""
    return np.concatenate([_blob(agent), _blob(ball), _blob(prev_ball)], 1).astype(np.float32)


def episodes(n, H, rng, persist=0.75):
    """n random episodes of H actions: observations [H + 1, n, D], actions [H, n], states [H + 1, n, 4]"""
    env = Arena(n, rng)
    obs, states, acts = [env.obs()], [env.state()], []
    a = rng.integers(0, 5, n)
    for _ in range(H):
        change = rng.random(n) > persist
        a = np.where(change, rng.integers(0, 5, n), a)
        acts.append(a)
        env.step(a)
        obs.append(env.obs())
        states.append(env.state())
    return np.stack(obs), np.stack(acts), np.stack(states)


def build(p, rng):
    from rawgpu import nn
    D, Z, Hd = 3 * S * S, int(p["latent"]), int(p["hidden"])

    class Encoder(nn.Module):
        def __init__(self):
            self.mlp = nn.MLP([D, Hd, Hd // 2, Z], act="gelu", norm=True, rng=rng)

        def forward(self, x):
            return self.mlp(x)

    class Predictor(nn.Module):
        """z' = z + MLP([z, onehot(a)])"""

        def __init__(self):
            self.mlp = nn.MLP([Z + 5, Hd // 2, Hd // 2, Z], act="gelu", norm=True, rng=rng)

        def forward(self, z, a_onehot):
            return z + self.mlp(nn.concat_cols([z, a_onehot]))

    class InverseDynamics(nn.Module):
        """which action led from z_t to z_t+1"""

        def __init__(self):
            self.mlp = nn.MLP([2 * Z, Hd // 4, 5], act="gelu", rng=rng)

        def forward(self, z0, z1):
            return self.mlp(nn.concat_cols([z0, z1]))

    return Encoder(), Encoder(), Predictor(), InverseDynamics()


def onehot(a):
    o = np.zeros((len(a), 5), np.float32)
    o[np.arange(len(a)), a] = 1
    return o


def vicreg(z, var_w, cov_w):
    """VICReg's variance hinge (each dimension's std at least 1) and covariance (decorrelated dimensions)"""
    from rawgpu import nn
    B, Z = z.rows, z.cols
    m = nn.sum_rows(z, mean=True)
    c = nn.add(z, m, 1.0, -1.0)
    std = nn.sqrt(nn.affine(nn.sum_rows(nn.square(c), mean=True), 1.0, 1e-4))
    var_loss = nn.mean_all(nn.relu(nn.affine(std, -1.0, 1.0)))
    cov = nn.matmul(c, c, ta=True, alpha=1.0 / (B - 1))
    off = nn.tensor(1.0 - np.eye(Z, dtype=np.float32))
    cov_loss = nn.scale(nn.sum_all(nn.square(cov * off)), 1.0 / Z)
    return var_loss, cov_loss, nn.add(nn.scale(var_loss, var_w), cov_loss, 1.0, cov_w)


def encode(E, obs, batch=4096):
    from rawgpu import nn
    out = []
    with nn.no_grad():
        for i in range(0, len(obs), batch):
            out.append(E(nn.tensor(obs[i:i + batch])).numpy())
    return np.concatenate(out)


def rollout(P, z0, acts):
    """Latents [len(acts), n, Z] from z0 [n, Z] (a Tensor) under actions [k, n]"""
    from rawgpu import nn
    zs, z = [], z0
    with nn.no_grad():
        for a in acts:
            z = P(z, nn.tensor(onehot(a)))
            zs.append(z)
    return zs


class Probe:
    """Ridge regression from latents to the true state [agent x, y, ball x, y]"""

    def fit(self, Z, Y, lam=1e-2):
        X = np.concatenate([Z, np.ones((len(Z), 1), np.float32)], 1).astype(np.float64)
        self.W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y.astype(np.float64)).astype(np.float32)
        return self

    def __call__(self, Z):
        return np.concatenate([Z, np.ones((len(Z), 1), np.float32)], 1) @ self.W


def evaluate(E, Et, P, run, step, rng_eval, H):
    """Rollout error by horizon, decoded positions, and the planner"""
    from rawgpu import nn
    Hl = 2 * H  # beyond the training horizon too
    # the probe, fit on target-encoder latents of random frames
    o_fit, _, s_fit = episodes(4096, 1, rng_eval)
    probe = Probe().fit(encode(Et, o_fit.reshape(-1, o_fit.shape[-1])), s_fit.reshape(-1, 4))
    obs, acts, states = episodes(512, Hl, rng_eval)
    zt = encode(Et, obs.reshape(-1, obs.shape[-1])).reshape(Hl + 1, 512, -1)
    floor = np.abs(probe(zt[1:].reshape(-1, zt.shape[-1])) - states[1:].reshape(-1, 4))
    with nn.no_grad():
        z0 = E(nn.tensor(obs[0]))
    zr = np.stack([z.numpy() for z in rollout(P, z0, acts)])  # [Hl, 512, Z]
    var = zt[1:].var(axis=(1,)).mean() + 1e-8
    lat = ((zr - zt[1:]) ** 2).mean(axis=(1, 2)) / var
    copy = ((zt[0][None] - zt[1:]) ** 2).mean(axis=(1, 2)) / var
    dec = probe(zr.reshape(-1, zr.shape[-1])).reshape(Hl, 512, 4)
    err = np.sqrt(((dec - states[1:]) ** 2).reshape(Hl, 512, 2, 2).sum(-1)).mean(1)  # [Hl, 2] agent, ball
    k8 = min(8, Hl) - 1
    ex = 0  # one episode to draw
    run.event("rollout", step, horizons=list(range(1, Hl + 1)), trained_horizon=H,
              latent_err=lat.round(4).tolist(), copy_err=copy.round(4).tolist(),
              agent_px=err[:, 0].round(3).tolist(), ball_px=err[:, 1].round(3).tolist(),
              floor_agent_px=float(np.sqrt((floor.reshape(-1, 2, 2) ** 2).sum(-1)).mean(0)[0]),
              floor_ball_px=float(np.sqrt((floor.reshape(-1, 2, 2) ** 2).sum(-1)).mean(0)[1]),
              arena=S, example={"true": states[:, ex].round(2).tolist(), "imagined": dec[:, ex].round(2).tolist(),
                                "actions": [ACT_NAMES[a] for a in acts[:, ex]]})
    plan = planning(E, P, probe, rng_eval)
    run.event("planning", step, **plan)
    return {"agent_px_8": float(err[k8, 0]), "ball_px_8": float(err[k8, 1]), "rollout_err_8": float(lat[k8]),
            "plan_success": plan["success"], "plan_random": plan["random_success"]}


def planning(E, P, probe, rng, episodes_n=64, cand=256, L=8, replan=4, budget=16):
    """Random-shooting MPC: from each start, sample `cand` action sequences, roll them out in latent space,
    keep the one whose decoded final agent position is nearest the goal, execute `replan` of its actions in
    the real arena, repeat until `budget` steps. Success: within 3 px of the goal at the end."""
    from rawgpu import nn
    env = Arena(episodes_n, rng)
    goal = rng.uniform(4, S - 5, (episodes_n, 2)).astype(np.float32)
    start = env.agent.copy()
    paths = [env.agent.copy()]
    for t in range(0, budget, replan):
        seqs = rng.integers(0, 5, (episodes_n, cand, L))
        seqs[:, :, :] = np.where(rng.random((episodes_n, cand, 1)) < 0.5, seqs[:, :, :1], seqs)  # some straight runs
        with nn.no_grad():
            z0 = E(nn.tensor(env.obs())).numpy()
            z = nn.tensor(np.repeat(z0, cand, 0))
            for k in range(L):
                z = P(z, nn.tensor(onehot(seqs[:, :, k].reshape(-1))))
            zf = z.numpy()
        pos = probe(zf)[:, :2].reshape(episodes_n, cand, 2)
        best = ((pos - goal[:, None]) ** 2).sum(-1).argmin(1)
        chosen = seqs[np.arange(episodes_n), best]
        for k in range(replan):
            env.step(chosen[:, k])
            paths.append(env.agent.copy())
    dist = np.sqrt(((env.agent - goal) ** 2).sum(1))
    # the baseline: the same number of random actions
    benv = Arena(episodes_n, rng)
    benv.agent = start.copy()
    for _ in range(budget):
        benv.step(rng.integers(0, 5, episodes_n))
    bdist = np.sqrt(((benv.agent - goal) ** 2).sum(1))
    path = np.stack(paths)[:, 0]
    return {"success": float((dist < 3).mean()), "random_success": float((bdist < 3).mean()),
            "mean_dist": float(dist.mean()), "random_dist": float(bdist.mean()), "episodes": episodes_n,
            "candidates": cand, "plan_len": L, "arena": S,
            "example": {"start": start[0].round(2).tolist(), "goal": goal[0].round(2).tolist(),
                        "path": path.round(2).tolist()}}


def train(run):
    from rawgpu import nn
    p = run.params
    seed = int(p.get("seed", 0))
    rng = np.random.default_rng(seed)
    rng_eval = np.random.default_rng(seed + 1)
    E, Et, P, IDM = build(p, rng)
    nn.copy_weights(Et, E)
    Et.freeze()
    params = E.parameters() + P.parameters() + IDM.parameters()
    total, B, H = int(p["steps"]), int(p["batch"]), int(p["horizon"])
    opt = nn.AdamW(params, lr=float(p["lr"]), weight_decay=0.01, max_grad_norm=1.0)
    start = 0
    if run.resume:
        extra = run.load(model=_Both(E, P, Et, IDM), opt=opt)
        if extra:
            start = int(extra["step"]) + 1
    run.saver = lambda: run.save(model=_Both(E, P, Et, IDM), opt=opt, step=run._step)
    n = E.num_parameters() + P.num_parameters()
    run.event("info", start, text=f"encoder and predictor {n / 1e6:.2f}M parameters; latent {p['latent']}; "
                                  f"rollouts of {H} steps; {B} episodes per step")
    run.summary(params=n)
    var_w, cov_w, idm_w = float(p["var_weight"]), float(p["cov_weight"]), float(p.get("idm_weight", 1.0))
    t_log, acc_steps = time.time(), 0
    for step in run.steps(start, total):
        lr = nn.cosine_lr(step, total, float(p["lr"]), 100)
        tau = 1 - (1 - float(p["ema"])) * (math.cos(math.pi * step / total) + 1) / 2
        obs, acts, _ = episodes(B, H, rng)
        with nn.no_grad():
            targets = Et(nn.tensor(obs[1:].reshape(-1, obs.shape[-1])))          # [H * B, Z]
        z01 = E(nn.tensor(obs[:2].reshape(-1, obs.shape[-1])))                # o_0 and o_1: [2B, Z]
        z0 = nn.slice_rows(z01, 0, B)
        z, pred = z0, None
        for k in range(H):
            z = P(z, nn.tensor(onehot(acts[k])))
            d = nn.mse_loss(z, nn.slice_rows(targets, k * B, (k + 1) * B))
            pred = d if pred is None else pred + d
        pred = nn.scale(pred, 1.0 / H)
        var_loss, cov_loss, reg = vicreg(z01, var_w, cov_w)
        loss = pred + reg
        idm_loss = None
        if idm_w > 0:
            idm_loss = nn.cross_entropy(IDM(z0, nn.slice_rows(z01, B, 2 * B)), acts[0])
            loss = loss + nn.scale(idm_loss, idm_w)
        opt.zero_grad()
        loss.backward()
        opt.step(lr)
        nn.ema_update(Et, E, tau)
        acc_steps += 1
        if run.every(25) or step == total - 1:
            with nn.no_grad():
                zs = float(np.sqrt(np.maximum(z0.numpy().var(0), 0)).mean())
            dt = time.time() - t_log
            run.log(step, loss=loss.item(), pred_loss=pred.item(), var_loss=var_loss.item(),
                    cov_loss=cov_loss.item(), idm_loss=idm_loss.item() if idm_loss is not None else 0.0, z_std=zs,
                    lr=lr, ema=tau, episodes_s=acc_steps * B / dt)
            t_log, acc_steps = time.time(), 0
        if run.every(int(p["eval_every"])) or step == total - 1:
            m = evaluate(E, Et, P, run, step, np.random.default_rng(seed + 1), H)
            run.log(step, **m)
            run.summary(**m)
            t_log = time.time()
        if run.every(1000):
            run.save(model=_Both(E, P, Et, IDM), opt=opt, step=step)
            t_log = time.time()
    run.save(model=_Both(E, P, Et, IDM), opt=opt, step=total - 1)


def _Both(E, P, Et, IDM):
    from rawgpu import nn

    class Both(nn.Module):
        def __init__(self):
            self.encoder, self.predictor, self.target, self.idm = E, P, Et, IDM
    return Both()
