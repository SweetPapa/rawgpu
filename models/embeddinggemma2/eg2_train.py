"""EmbeddingGemma 2 as a trainable rawgpu.nn module: fine-tune the text encoder on the RX 590.

    from eg2_train import EG2Train
    m = EG2Train(train_layers=8)                 # the top 8 of 24 layers, the final norm and projection
    q = m(*m.batch(["how do I eject the card"], "code"))     # unit embeddings [1, 768], a Tensor
    d = m(*m.batch(["def eject(): ..."], "document"))
    loss = nn.cross_entropy(nn.matmul(q, d, tb=True, alpha=20.0), [0])
    loss.backward()
    m.export("finetuned/")                       # a model directory eg2.EmbeddingGemma2(path=...) loads

The forward pass is eg2.py's math, built from rawgpu.nn ops so that autograd can run it backward:
float32 throughout, the 262144-row token embedding table frozen on the host. check_train.py compares
the embeddings with eg2.py's and the gradients with PyTorch's.
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, HERE)
from rawgpu import nn  # noqa: E402
from eg2 import PROMPTS, _Safetensors, _snapshot  # noqa: E402

PRE = "language_model."


class EG2Train(nn.Module):
    def __init__(self, path=None, repo="google/embeddinggemma-2", train_layers="all", train_ple=False,
                 max_len=512, verbose=True):
        """train_layers: "all", 0 (nothing but the projection), or N for the top N layers. The final norm
        and the output projection train unless train_layers is None (everything frozen)."""
        self.path = path or _snapshot(repo)
        cfg = json.load(open(os.path.join(self.path, "config.json")))["text_config"]
        self.cfg = cfg
        self.D, self.L, self.H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
        self.I, self.P, self.E = cfg["intermediate_size"], cfg["hidden_size_per_layer_input"], cfg["embedding_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"] + 1
        self.max_len = max_len
        st = _Safetensors(os.path.join(self.path, "model.safetensors"))
        self._st = st
        self._embed = st.bf16_rows(PRE + "embed_tokens.weight")  # frozen, on the host
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(os.path.join(self.path, "tokenizer.json"))
        self.bos, self.eos = cfg["bos_token_id"], cfg["eos_token_id"]

        if train_layers == "all":
            first = 0
        elif train_layers is None:
            first = self.L + 1
        else:
            first = self.L - int(train_layers)
        up = lambda a, train: self._p(np.asarray(a, np.float32), train)
        w = lambda n: st.fp32(PRE + n)
        self.w_ple = up(w("ple.per_layer_model_projection.weight"), train_ple)
        self.w_ple_norm = up(w("ple.per_layer_projection_norm.weight"), train_ple)
        self.layers = []
        self._meta = []
        for i in range(self.L):
            lc = cfg["per_layer_config"].get(f"{i:02d}", {})
            hd = lc.get("head_dim", cfg["head_dim"])
            kv = lc.get("num_key_value_heads", cfg["num_key_value_heads"])
            kind = cfg["layer_types"][i]
            theta = cfg["rope_parameters"][kind]["rope_theta"]
            inv = (1.0 / (np.float32(theta) ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)))).astype(np.float32)
            t = i >= first
            lw = lambda n: w(f"layers.{i}." + n)
            self.layers.append(dict(
                qkv=up(np.concatenate([lw("self_attn.q_proj.weight"), lw("self_attn.k_proj.weight"),
                                       lw("self_attn.v_proj.weight")]), t),
                q_norm=up(lw("self_attn.q_norm.weight"), t), k_norm=up(lw("self_attn.k_norm.weight"), t),
                o=up(lw("self_attn.o_proj.weight"), t),
                in_ln=up(lw("input_layernorm.weight"), t), post_attn=up(lw("post_attention_layernorm.weight"), t),
                pre_ff=up(lw("pre_feedforward_layernorm.weight"), t),
                post_ff=up(lw("post_feedforward_layernorm.weight"), t),
                gu=up(np.concatenate([lw("mlp.gate_proj.weight"), lw("mlp.up_proj.weight")]), t),
                down=up(lw("mlp.down_proj.weight"), t),
                ple_gate=up(lw("ple_block.per_layer_input_gate.weight"), t),
                ple_proj=up(lw("ple_block.per_layer_projection.weight"), t),
                ple_norm=up(lw("ple_block.post_per_layer_input_norm.weight"), t)))
            self._meta.append(dict(hd=hd, kv=kv, sliding=kind == "sliding_attention", scalar=float(lw("layer_scalar")[0]),
                                   freq=nn.tensor(inv)))
        top = train_layers is not None
        self.w_norm = up(w("norm.weight"), top)
        self.w_proj = up(w("embedding_projection.weight"), top)
        if verbose:
            n_all = sum(p.size for _, p in self._all_tensors())
            print(f"EmbeddingGemma 2 for training: {n_all / 1e6:.0f}M parameters on the GPU, "
                  f"{self.num_parameters() / 1e6:.1f}M trainable"
                  f" ({'all layers' if first == 0 else f'layers {first}-{self.L - 1}' if first < self.L else 'none'})")

    @staticmethod
    def _p(a, train):
        t = nn.tensor(a)
        t.requires_grad = bool(train)
        return t

    # ---- inputs
    def tokenize(self, texts, prompt=None):
        pre = PROMPTS.get(prompt, prompt) if prompt else ""
        enc = self._tok.encode_batch([pre + t for t in texts], add_special_tokens=False)
        return [[self.bos] + e.ids[:self.max_len - 2] + [self.eos] for e in enc]

    def batch(self, texts, prompt=None):
        """(ids [B, T], lens [B]) for forward(), T rounded up to a multiple of 8"""
        seqs = self.tokenize(texts, prompt)
        T = (max(len(s) for s in seqs) + 7) // 8 * 8
        ids = np.zeros((len(seqs), T), np.int64)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
        return ids, np.array([len(s) for s in seqs], np.int32)

    # ---- the model
    def forward(self, ids, lens, checkpoint=True):
        """Unit-length embeddings [B, 768] of token ids [B, T] with lengths lens [B]. With checkpoint (and
        gradients on), each layer keeps only its input and is recomputed in the backward pass: about 15x
        less memory for activations, for a third more compute."""
        B, T = ids.shape
        N, D = B * T, self.D
        x = (self._embed[ids.reshape(-1)].astype(np.uint32) << 16).view(np.float32) * np.float32(D ** 0.5)
        x = nn.tensor(x.reshape(N, D))
        lens_t = nn.tensor(np.asarray(lens, np.int32))
        ple = nn.rmsnorm(nn.linear(x, self.w_ple, alpha=D ** -0.5), self.w_ple_norm, self.eps, groups=self.L)
        P = self.P
        for i in range(self.L):
            fn = lambda x, p, i=i: self._layer(i, x, p, B, T, lens_t)
            p_i = nn.slice_cols(ple, i * P, (i + 1) * P)
            l = self.layers[i]
            if checkpoint and nn.core._grad_enabled and any(t.requires_grad for t in l.values()):
                x = nn.checkpoint(fn, (x, p_i), [t for t in l.values() if t.requires_grad])
            else:
                x = fn(x, p_i)
        e = nn.linear(nn.rmsnorm(x, self.w_norm, self.eps), self.w_proj)
        return nn.l2norm(nn.pool_mean(e, lens_t, B, T))

    def _layer(self, i, x, ple_i, B, T, lens_t):
        l, m, H = self.layers[i], self._meta[i], self.H
        hd, kv, eps = m["hd"], m["kv"], self.eps
        h = nn.rmsnorm(x, l["in_ln"], eps)
        qkv = nn.linear(h, l["qkv"])
        q = nn.rope(nn.rmsnorm(nn.slice_cols(qkv, 0, H * hd), l["q_norm"], eps, groups=H), m["freq"], T, H)
        k = nn.rope(nn.rmsnorm(nn.slice_cols(qkv, H * hd, (H + kv) * hd), l["k_norm"], eps, groups=kv),
                    m["freq"], T, kv)
        v = nn.rmsnorm(nn.slice_cols(qkv, (H + kv) * hd, (H + 2 * kv) * hd), None, eps, groups=kv)
        att = nn.attention(q, k, v, B, T, H, kv, causal=False, scale_=1.0, lens=lens_t,
                           window=self.window if m["sliding"] else 0)
        x = x + nn.rmsnorm(nn.linear(att, l["o"]), l["post_attn"], eps)
        gu = nn.linear(nn.rmsnorm(x, l["pre_ff"], eps), l["gu"])
        act = nn.gelu(nn.slice_cols(gu, 0, self.I)) * nn.slice_cols(gu, self.I, 2 * self.I)
        x = x + nn.rmsnorm(nn.linear(act, l["down"]), l["post_ff"], eps)
        g = nn.gelu(nn.linear(x, l["ple_gate"])) * ple_i
        x = x + nn.rmsnorm(nn.linear(g, l["ple_proj"]), l["ple_norm"], eps)
        return nn.scale(x, m["scalar"]) if m["scalar"] != 1.0 else x

    def encode(self, texts, prompt=None, batch=32):
        """Embeddings as numpy, without gradients"""
        out = []
        with nn.no_grad():
            for i in range(0, len(texts), batch):
                out.append(self.forward(*self.batch(texts[i:i + batch], prompt)).numpy())
        return np.concatenate(out) if out else np.zeros((0, self.E), np.float32)

    # ---- saving
    def _named_weights(self):
        """(checkpoint name, tensor, column range or None) for every weight, the fused ones split"""
        yield "ple.per_layer_model_projection.weight", self.w_ple, None
        yield "ple.per_layer_projection_norm.weight", self.w_ple_norm, None
        yield "norm.weight", self.w_norm, None
        yield "embedding_projection.weight", self.w_proj, None
        H = self.H
        for i, (l, m) in enumerate(zip(self.layers, self._meta)):
            hd, kv, p = m["hd"], m["kv"], f"layers.{i}."
            yield p + "self_attn.q_proj.weight", l["qkv"], (0, H * hd)
            yield p + "self_attn.k_proj.weight", l["qkv"], (H * hd, (H + kv) * hd)
            yield p + "self_attn.v_proj.weight", l["qkv"], ((H + kv) * hd, (H + 2 * kv) * hd)
            yield p + "mlp.gate_proj.weight", l["gu"], (0, self.I)
            yield p + "mlp.up_proj.weight", l["gu"], (self.I, 2 * self.I)
            for n, k in (("self_attn.q_norm.weight", "q_norm"), ("self_attn.k_norm.weight", "k_norm"),
                         ("self_attn.o_proj.weight", "o"), ("input_layernorm.weight", "in_ln"),
                         ("post_attention_layernorm.weight", "post_attn"), ("pre_feedforward_layernorm.weight", "pre_ff"),
                         ("post_feedforward_layernorm.weight", "post_ff"), ("mlp.down_proj.weight", "down"),
                         ("ple_block.per_layer_input_gate.weight", "ple_gate"),
                         ("ple_block.per_layer_projection.weight", "ple_proj"),
                         ("ple_block.post_per_layer_input_norm.weight", "ple_norm")):
                yield p + n, l[k], None

    def trained_weights(self):
        """{checkpoint name: float32 array} of the weights that train"""
        out, cache = {}, {}
        for name, t, rng in self._named_weights():
            if not t.requires_grad:
                continue
            if id(t) not in cache:
                cache[id(t)] = t.numpy()
            a = cache[id(t)]
            out[name] = a[rng[0]:rng[1]] if rng else a
        return out

    def load_trained(self, weights):
        """Put weights (from trained_weights() or a .npz of it) back"""
        full = {}
        for name, t, rng in self._named_weights():
            if name not in weights:
                continue
            if rng is None:
                nn.upload(np.asarray(weights[name], np.float32).reshape(t.shape), t)
            else:
                full.setdefault(id(t), [t, t.numpy()])[1][rng[0]:rng[1]] = weights[name]
        for t, a in full.values():
            nn.upload(a, t)

    def export(self, out_dir):
        """A model directory for eg2.EmbeddingGemma2(path=out_dir): config, tokenizer, and the text model's
        weights (the trained ones in float32, the rest as they were)."""
        os.makedirs(out_dir, exist_ok=True)
        for f in ("config.json", "tokenizer.json"):
            shutil.copy(os.path.join(self.path, f), os.path.join(out_dir, f))
        trained = self.trained_weights()
        st = self._st
        tensors = []
        for name, m in st.meta.items():
            if name == "__metadata__" or not name.startswith(PRE):
                continue
            short = name[len(PRE):]
            if short in trained:
                a = np.ascontiguousarray(trained[short], np.float32)
                tensors.append((name, "F32", list(a.shape), a.tobytes()))
            else:
                a, b = m["data_offsets"]
                tensors.append((name, m["dtype"], m["shape"], st.f[st.base + a:st.base + b].tobytes()))
        header, off = {}, 0
        for name, dt, shape, raw in tensors:
            header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
            off += len(raw)
        h = json.dumps(header).encode()
        h += b" " * ((8 - len(h) % 8) % 8)
        with open(os.path.join(out_dir, "model.safetensors"), "wb") as f:
            f.write(len(h).to_bytes(8, "little"))
            f.write(h)
            for *_, raw in tensors:
                f.write(raw)
        return out_dir
