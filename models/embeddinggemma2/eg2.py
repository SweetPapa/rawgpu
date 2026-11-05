"""EmbeddingGemma 2 (text) on the RX 590 through rawgpu.

    from eg2 import EmbeddingGemma2
    model = EmbeddingGemma2()                              # loads google/embeddinggemma-2 into VRAM
    q = model.encode(["what is a GPU?"], prompt="query")
    d = model.encode(docs, prompt="document")
    scores = q @ d.T                                       # unit vectors: dot product = cosine

The whole text forward pass runs on the GPU in float32. The weights (bf16 in
the checkpoint, exact in fp32) stay in VRAM, except the 262144-row token
embedding table, which stays in Mac memory: gathering a batch's rows there
costs less than keeping the table on the card.

The math follows transformers' modeling_embedding_gemma2.py; compare.py
checks it against that implementation. The model's prompts are in PROMPTS;
use "query" and "document" for retrieval.

Every matrix a product reads has its rows padded by PAD floats: at the
model's power-of-two widths (512, 1024, 2048) the SGEMM runs 1.5-6x slower
unpadded. Sliding-window layers compute only the attention scores within
the window (in 128 x 128 tiles) and multiply only those by V.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
import rawgpu  # noqa: E402

PAD = 32  # floats past the end of each row of a product's operands

PROMPTS = {  # config_sentence_transformers.json
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "similarity": "task: sentence similarity | query: ",
    "code": "task: code retrieval | query: ",
    "qa": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
}


def _snapshot(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["config.json", "model.safetensors", "tokenizer.json"])


class _Safetensors:
    """Reads tensors from a .safetensors file without the safetensors package (numpy has no bf16)."""

    def __init__(self, path):
        self.f = np.memmap(path, dtype=np.uint8, mode="r")
        n = int(np.frombuffer(self.f[:8].tobytes(), "<u8")[0])
        self.meta = json.loads(self.f[8:8 + n].tobytes())
        self.base = 8 + n

    def fp32(self, name):
        m = self.meta[name]
        a, b = m["data_offsets"]
        raw = self.f[self.base + a:self.base + b]
        if m["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
        if m["dtype"] == "F32":
            return raw.view(np.float32).reshape(m["shape"]).copy()
        raise ValueError(f"{name}: dtype {m['dtype']}")

    def bf16_rows(self, name):
        m = self.meta[name]
        a, b = m["data_offsets"]
        return self.f[self.base + a:self.base + b].view(np.uint16).reshape(m["shape"])


class _Weight:
    """A linear layer's weight [n, k] in VRAM, its rows PAD floats longer than k."""

    def __init__(self, gpu, w):
        self.n, self.k = w.shape
        self.ld = self.k + PAD
        a = np.zeros((self.n, self.ld), np.float32)
        a[:, :self.k] = w
        self.buf = gpu.array(a)
        self.nbytes = self.buf.nbytes


class EmbeddingGemma2:
    def __init__(self, gpu=None, repo="google/embeddinggemma-2", path=None, max_tokens=16384, max_seq=8192,
                 verbose=True):
        self.gpu = gpu or rawgpu.Device()
        path = path or _snapshot(repo)
        cfg = json.load(open(os.path.join(path, "config.json")))["text_config"]
        self.cfg = cfg
        self.D = cfg["hidden_size"]                    # 512
        self.L = cfg["num_hidden_layers"]              # 24
        self.H = cfg["num_attention_heads"]            # 4
        self.I = cfg["intermediate_size"]              # 2048
        self.P = cfg["hidden_size_per_layer_input"]    # 512
        self.E = cfg["embedding_dim"]                  # 768
        self.eps = cfg["rms_norm_eps"]
        self.window = cfg["sliding_window"] + 1        # visible when |q - k| <= sliding_window
        self.max_tokens, self.max_seq = max_tokens, max_seq
        st = _Safetensors(os.path.join(path, "model.safetensors"))
        pre = "language_model."
        self.embed = st.bf16_rows(pre + "embed_tokens.weight")  # stays on the host
        self.embed_scale = np.float32(self.D ** 0.5)

        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        self.bos, self.eos = cfg["bos_token_id"], cfg["eos_token_id"]

        g = self.gpu
        up = lambda a: g.array(np.ascontiguousarray(a, np.float32))
        lin = lambda a: _Weight(g, a)
        self.w_ple = lin(st.fp32(pre + "ple.per_layer_model_projection.weight"))  # [L*P, D]
        self.w_ple_norm = up(st.fp32(pre + "ple.per_layer_projection_norm.weight"))
        self.w_norm = up(st.fp32(pre + "norm.weight"))
        self.w_proj = lin(st.fp32(pre + "embedding_projection.weight"))           # [E, D]
        self.layers = []
        for i in range(self.L):
            lp = f"{pre}layers.{i}."
            lc = cfg["per_layer_config"].get(f"{i:02d}", {})
            hd = lc.get("head_dim", cfg["head_dim"])
            kv = lc.get("num_key_value_heads", cfg["num_key_value_heads"])
            kind = cfg["layer_types"][i]
            theta = cfg["rope_parameters"][kind]["rope_theta"]
            # torch: 1 / base ** (arange(0, dim, 2, float32) / dim)
            inv_freq = (1.0 / (np.float32(theta) ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)))).astype(np.float32)
            w = lambda n: st.fp32(lp + n)
            self.layers.append(dict(
                hd=hd, kv=kv, sliding=kind == "sliding_attention",
                qp=(self.H + 2 * kv) * hd + PAD, ap=self.H * hd + PAD,  # row pitches of qkv and att
                qkv=lin(np.concatenate([w("self_attn.q_proj.weight"), w("self_attn.k_proj.weight"),
                                        w("self_attn.v_proj.weight")])),
                q_norm=up(w("self_attn.q_norm.weight")), k_norm=up(w("self_attn.k_norm.weight")),
                o=lin(w("self_attn.o_proj.weight")), inv_freq=up(inv_freq),
                in_ln=up(w("input_layernorm.weight")), post_attn=up(w("post_attention_layernorm.weight")),
                pre_ff=up(w("pre_feedforward_layernorm.weight")), post_ff=up(w("post_feedforward_layernorm.weight")),
                gu=lin(np.concatenate([w("mlp.gate_proj.weight"), w("mlp.up_proj.weight")])),
                down=lin(w("mlp.down_proj.weight")),
                ple_gate=lin(w("ple_block.per_layer_input_gate.weight")),
                ple_proj=lin(w("ple_block.per_layer_projection.weight")),
                ple_norm=up(w("ple_block.post_per_layer_input_norm.weight")),
                scalar=float(w("layer_scalar")[0])))

        src = open(os.path.join(HERE, "..", "..", "kernels", "sgemm.cl")).read() + "\n" + open(os.path.join(HERE, "eg2.cl")).read()
        self.k = g.program(src)

        # activations, sized for max_tokens tokens at once; the ones products read have padded rows
        n = max_tokens
        self.xp, self.hp, self.actp = self.D + PAD, max(self.D, self.P) + PAD, self.I + PAD
        self.x = g.empty((n, self.xp))
        self.h = g.empty((n, self.hp))
        self.qkv = g.empty((n, max(l["qp"] for l in self.layers)))
        self.att = g.empty((n, max(l["ap"] for l in self.layers)))
        self.o = g.empty((n, self.D))
        self.gu = g.empty((n, 2 * self.I))
        self.act = g.empty((n, self.actp))
        self.ple = g.empty((n, self.L * self.P))
        self.e = g.empty((n, self.E))
        self.scores_bytes = 4 * self.H * max_seq * (max_seq + PAD)  # one sequence of max_seq, every head
        self.S = g.empty(self.scores_bytes // 4)
        self.max_batch = 1024
        self.lens = g.empty(self.max_batch, np.uint32)
        self.out = g.empty((self.max_batch, self.E))
        if verbose:
            mb = sum(b.nbytes for l in self.layers for b in l.values() if isinstance(b, (rawgpu.Buffer, _Weight))) >> 20
            print(f"EmbeddingGemma 2: {self.L} layers, {mb} MB of layer weights in VRAM "
                  f"(+{(self.w_ple.nbytes + self.w_proj.nbytes) >> 20} MB PLE and projection); "
                  f"{g.info['vram_free_bytes'] >> 20} MB VRAM free")

    # ---- the pieces ----
    def _linear(self, A, lda, W, C, ldc, M, alpha=1.0):
        """C[M, W.n] = alpha * A[M, W.k] @ W^T, rows lda and ldc floats apart"""
        self.k.sgemm_nt(A, W.buf, C, M, W.n, W.k, lda, W.ld, ldc, alpha, 0.0,
                        global_size=((W.n + 127) // 128 * 16, (M + 127) // 128 * 16), local_size=(16, 16))

    def _rows(self, fn, rows, *args):
        fn(*args, global_size=rows * 64, local_size=64)

    def _elem(self, fn, total, *args):
        fn(*args, global_size=(total + 255) // 256 * 256, local_size=256)

    def _forward(self, ids, lens, T, tap=None):
        """ids [B, T] int, lens [B]: queue the forward pass; returns the event of the last launch.
        tap(name, buffer, rows, cols, pitch), if given, is called at each step (for checking against the
        reference): the step's result is rows x cols, rows pitch floats apart."""
        import time
        t_prep = time.perf_counter()
        tap = tap or (lambda *a: None)
        k, D, H, P = self.k, self.D, self.H, self.P
        B = ids.shape[0]
        N = B * T
        x = np.zeros((N, self.xp), np.float32)
        x[:, :D] = (self.embed[ids.reshape(-1)].astype(np.uint32) << 16).view(np.float32)  # bf16 rows to fp32
        x[:, :D] *= self.embed_scale
        self.x.write(x)
        self.lens.write(np.asarray(lens, np.uint32))
        self.prep_s = time.perf_counter() - t_prep  # host work: the embedding rows, gathered and uploaded
        tap("embed", self.x, N, D, self.xp)
        # per-layer embeddings: rmsnorm((x @ W_ple^T) * D^-0.5) over each layer's P-wide slice
        self._linear(self.x, self.xp, self.w_ple, self.ple, self.L * P, N, alpha=D ** -0.5)
        self._rows(k.rmsnorm, N * self.L, self.ple, self.ple, self.w_ple_norm, P, P, P, self.eps)
        tap("ple", self.ple, N, self.L * P, self.L * P)
        sp = T + PAD  # the scores' row pitch
        tiles = lambda n: (n + 127) // 128 * 16
        # h = the next block's normed input. The residual adds that precede a norm compute it.
        self._rows(k.rmsnorm, N, self.x, self.h, self.layers[0]["in_ln"], D, self.xp, self.hp, self.eps)
        for i, l in enumerate(self.layers):
            hd, kv, qp, ap = l["hd"], l["kv"], l["qp"], l["ap"]
            slots = H + 2 * kv
            window = self.window if l["sliding"] else 0
            # attention
            self._linear(self.h, self.hp, l["qkv"], self.qkv, qp, N)
            tap(f"{i}.qkv_proj", self.qkv, N, slots * hd, qp)
            self._rows(k.qkv_post, N * slots, self.qkv, l["q_norm"], l["k_norm"], l["inv_freq"], H, kv, hd, qp, T,
                       self.eps)
            tap(f"{i}.qkv_rope", self.qkv, N, slots * hd, qp)
            # scores[b, h] = Q[b, h] @ K[b, h / group]^T, one product per (sequence, head); with a window,
            # only the tiles that hold visible scores
            k.sgemm_batched_nt(self.qkv, self.qkv.at(H * hd * 4), self.S, T, T, hd, qp, qp, sp, 1.0, 0.0,
                               H, H // kv, T * qp, hd, T * qp, hd, H * T * sp, T * sp, window,
                               global_size=(tiles(T), tiles(T), B * H), local_size=(16, 16, 1))
            self._rows(k.softmax_mask, B * H * T, self.S, self.lens, H, T, sp, window)
            # att[b, t, h] = P[b, h] @ V[b, h / group], over the keys each tile of queries can see
            k.sgemm_batched_nn(self.S, self.qkv.at((H + kv) * hd * 4), self.att, T, hd, T, sp, qp, ap, 1.0, 0.0,
                               H, H // kv, H * T * sp, T * sp, T * qp, hd, T * ap, hd, window,
                               global_size=(tiles(hd), tiles(T), B * H), local_size=(16, 16, 1))
            tap(f"{i}.att", self.att, N, H * hd, ap)
            self._linear(self.att, ap, l["o"], self.o, D, N)
            self._rows(k.add_rmsnorm, N, self.x, self.o, l["post_attn"], D, self.xp, D, self.eps, 1.0,
                       self.h, l["pre_ff"], self.hp)
            tap(f"{i}.after_attn", self.x, N, D, self.xp)
            # MLP
            self._linear(self.h, self.hp, l["gu"], self.gu, 2 * self.I, N)
            self._elem(k.gelu_mul, N * self.I, self.gu, self.act, self.I, self.actp, N * self.I)
            self._linear(self.act, self.actp, l["down"], self.o, D, N)
            self._rows(k.add_rmsnorm, N, self.x, self.o, l["post_ff"], D, self.xp, D, self.eps, 1.0, None, None, 0)
            tap(f"{i}.after_mlp", self.x, N, D, self.xp)
            # per-layer-embedding block, then the layer scalar
            self._linear(self.x, self.xp, l["ple_gate"], self.h, self.hp, N)
            self._elem(k.ple_gate, N * P, self.h, self.ple, P, self.hp, self.L, i, N * P)
            self._linear(self.h, self.hp, l["ple_proj"], self.o, D, N)
            next_norm = self.layers[i + 1]["in_ln"] if i + 1 < self.L else self.w_norm
            self._rows(k.add_rmsnorm, N, self.x, self.o, l["ple_norm"], D, self.xp, D, self.eps, l["scalar"],
                       self.h, next_norm, self.hp)
            tap(f"{i}.out", self.x, N, D, self.xp)
        self._linear(self.h, self.hp, self.w_proj, self.e, self.E, N)
        tap("proj", self.e, N, self.E, self.E)
        # mean over each sequence's tokens; encode() normalizes
        return k.pool_mean(self.e, self.out, self.lens, T, self.E, global_size=((self.E + 63) // 64 * 64, B),
                           local_size=(64, 1))

    # ---- the API ----
    def tokenize(self, texts, prompt=None):
        pre = PROMPTS.get(prompt, prompt) if prompt else ""
        enc = self.tok.encode_batch([pre + t for t in texts], add_special_tokens=False)
        out = []
        for e in enc:
            ids = [self.bos] + e.ids[:self.max_seq - 2] + [self.eos]
            out.append(ids)
        return out

    def encode(self, texts, prompt=None, dim=None, return_stats=False):
        """Unit-length embeddings [len(texts), dim or 768] (float32). `prompt`: a PROMPTS key or the
        prompt text itself. `dim` 512/256/128 truncates (Matryoshka) and renormalizes."""
        if isinstance(texts, str):
            texts = [texts]
        return self.encode_tokens(self.tokenize(texts, prompt), dim, return_stats)

    def encode_tokens(self, seqs, dim=None, return_stats=False):
        """encode() for texts already tokenized by tokenize(). The stats (return_stats) split the time:
        host_s batching and uploading the embedding rows, gpu_s launching (which waits whenever the
        command ring is full) and waiting for the results."""
        import time
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))  # similar lengths batch together
        out = np.zeros((len(seqs), self.E), np.float32)
        pos, tokens, padded, batches, host_s, gpu_s = 0, 0, 0, 0, 0.0, 0.0
        while pos < len(order):
            t0 = time.perf_counter()
            T = len(seqs[order[pos]])
            T = (T + 7) // 8 * 8
            cap = min(self.max_tokens // T, self.scores_bytes // (4 * self.H * T * (T + PAD)), self.max_batch)
            idx = order[pos:pos + max(cap, 1)]
            ids = np.zeros((len(idx), T), np.int64)
            lens = []
            for r, i in enumerate(idx):
                ids[r, :len(seqs[i])] = seqs[i]
                lens.append(len(seqs[i]))
            t1 = time.perf_counter()
            self._forward(ids, lens, T)
            out[idx] = self.out.numpy()[:len(idx)]  # waits for the batch
            t2 = time.perf_counter()
            host_s += t1 - t0 + self.prep_s
            gpu_s += t2 - t1 - self.prep_s
            tokens += sum(lens)
            padded += len(idx) * T
            batches += 1
            pos += len(idx)
        if dim and dim < self.E:
            out = out[:, :dim]
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        if return_stats:
            return out, dict(tokens=tokens, padded_tokens=padded, batches=batches, host_s=host_s, gpu_s=gpu_s)
        return out
