# EmbeddingGemma 2 on the RX 590

Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
text encoder, run entirely on the RX 590 through librawgpu. It's a 24-layer,
512-wide Gemma encoder with per-layer embeddings, sliding-window and full
attention, and 768-dimensional output (Matryoshka down to 128). It's released
under Apache 2.0 and isn't gated.

```python
import sys; sys.path.insert(0, "models/embeddinggemma2")
from eg2 import EmbeddingGemma2
model = EmbeddingGemma2()                       # downloads the weights once, then loads them into VRAM
q = model.encode(["who wrote the Iliad?"], prompt="query")
d = model.encode(docs, prompt="document")       # unit vectors: q @ d.T is cosine similarity
```

- **Precision:** the whole forward pass runs in float32 on the GPU.
- **Kernels:** the matrix products are `kernels/sgemm.cl`'s register-blocked
  SGEMM, plus batched variants for attention. The row and element-wise steps
  are in `eg2.cl`: RMSNorm, per-head norms with RoPE, masked softmax, GELU
  gates, and pooling.
- **Padded rows:** every matrix a product reads has its rows padded by 32
  floats. At the model's power-of-two widths the SGEMM runs 1.5-6x slower
  otherwise (`docs/LIBRARY.md`).
- **Sliding-window attention:** on the 20 sliding layers, the score product
  skips the 128 × 128 tiles that lie wholly outside the ±512-token window.
  Softmax touches only the keys each query's tile can see, and P·V
  multiplies only over those keys. At 2048 tokens that's 9 of 16 tiles.
  Skipping zeros changes nothing, so the results are the same as computing
  the full matrix.
- **VRAM:** the layer weights take 520 MB. Activations are sized for 16384
  tokens per batch, and the scores for one 8192-token sequence (1.1 GB).
- **Token embedding table:** the 262144 × 512 table stays in Mac memory as
  bf16. Only a batch's rows are sent.
- **Sequence length:** up to 8192 tokens (`max_seq`); longer inputs are
  truncated. The 4 full-attention layers compute the whole score matrix, so
  their cost grows with the square of the length.
- **Prompts:** the model's own prompts are in `PROMPTS`. For retrieval, use
  `"query"` and `"document"`.

## Checking it

`compare.py` runs nine texts in English, German, Japanese and code. One is
1209 tokens, crossing the 512-token sliding window, and one is 5998 tokens
(Python's argparse source). It runs them through this engine and through
the reference (sentence-transformers, float32, CPU). It checks two things:

- the token ids are identical;
- the embeddings agree to cosine ≥ 0.9999.

```sh
python3 -m venv --system-site-packages local/eg2-venv
local/eg2-venv/bin/pip install "transformers>=5.22" "sentence-transformers>=6.2" torchvision
local/eg2-venv/bin/python models/embeddinggemma2/compare.py
```

## Benchmark

`bench.py` uses MuSiQue's multi-hop questions over the Wikipedia paragraphs
that come with them. It runs the RX 590 against the same model on the M4
Pro's GPU (PyTorch MPS, float32), each in its own process:

```sh
local/eg2-venv/bin/python models/embeddinggemma2/bench.py --questions 500
```

Results from 2026-11-05 (500 questions, 7029 paragraphs averaging 119
tokens, 64 joined documents averaging 1926 tokens):

| | paragraphs | questions | 2K-token documents | recall@1 / 5 / 10 / 20 |
|---|---|---|---|---|
| RX 590 (rawgpu, fp32) | **89/s**, 10.6k tok/s | **277/s** | **3.8/s**, 7.3k tok/s | 0.390 / 0.668 / 0.734 / 0.800 |
| M4 Pro GPU (MPS, fp32) | 36/s, 4.3k tok/s | 92/s | 1.5/s, 2.9k tok/s | 0.390 / 0.668 / 0.734 / 0.800 |

- **Total time:** the RX 590 took 98 s and the M4 Pro about 245 s, so the
  RX 590 ran about 2.5x as fast. On 2026-11-04 the RX 590 took 152 s,
  before rows were padded and sliding layers were windowed.
- **Agreement:** the two embeddings of every text agree to cosine 1.000000.
- **Card state:** the engine was at 1287 MHz and 73 C at the end.
- **M4 Pro variance:** its time depends on what else is using the Mac's
  GPU. Two runs took 243 s and 246 s, and a third, with a browser busy,
  took 304 s. Its question rate ranged from 89 to 160 per second.
- **Batch sizes on the Mac:** MPS ran with batch 32 for short texts and 4 for
  long documents. Batch 64 ran out of MPS memory on the long documents.

Where the time goes in one forward pass. Each launch was waited on and timed
separately, so the small kernels' shares include some launch overhead:

| | 64 × 128 tokens | 8 × 2048 tokens | 1 × 8192 tokens |
|---|---|---|---|
| tokens per second (normal, pipelined) | 11.6k | 8.1k | 5.2k |
| linear layers (SGEMM, ~4 TFLOPS) | 73% | 53% | 37% |
| attention products (3.7-4.5 TFLOPS) | 7% | 29% | 47% |
| norms, softmax, GELU, RoPE, pooling | 20% | 18% | 17% |

The linear layers now run near this SGEMM's ceiling of 4.6-4.8 TFLOPS.
Further gains would need a faster SGEMM, or fewer passes over memory for
the row-wise steps, by fusing them into the products.

## Indexing a code folder

`index_code.py` embeds a folder of code (by default `~/code`) on the card
and searches it. The rules it follows are in its docstring.

```sh
PY=local/eg2-venv/bin/python; IX=models/embeddinggemma2/index_code.py
$PY $IX dry-run                      # files, chunks, tokens, time and size; what it skips and why
caffeinate -i $PY $IX run            # resumable; a rerun embeds only what changed
$PY $IX search "how do we post the card" -k 5 [--project rawgpu]
```

- **Read-only:** it only reads the folder, and writes to `~/rawgpu-index`
  (`--out`), which must be outside it. It makes no network requests.
- **Your code only:** clones, submodules, vendored folders and known
  open-source projects are skipped, by their git remotes, package
  manifests, names and folder names. Your accounts go in
  `<out>/config.json` as `{"own_remotes": [regex, ...]}`. `dry-run` lists
  every folder it treats as someone else's.
- **Also skipped:** VCS folders, dependencies, build output, virtualenvs,
  caches, binaries, minified files, lockfiles, generated files, files over
  1 MB, `.env` files, keys, and any file whose text looks like it holds a
  credential.
- **Chunks:** 200-500 tokens, cut at blank lines, top-level definitions
  and headings. A chunk overlaps the next one by up to 40 tokens only when
  the cut falls inside a block.
- **Storage:** one SQLite file holds each chunk's path, project, lines,
  SHA-256, text and 768-float vector. `search` builds a memory-mapped copy
  of the vectors next to it on first use.
- **Resuming:** work is committed every 1024 chunks. Unchanged files
  (same size and mtime) are not reread, and a chunk whose hash is already
  indexed reuses its vector.
- **Self-checks:** every 10 minutes it embeds a fixed text and compares it
  with the CPU reference's vector. It stops on a mismatch, a GPU error, a
  non-finite vector, or 85 C. It logs throughput, clocks, temperature, fan,
  and host against GPU time to `<out>/logs/`.

The full run on 2026-11-05:

| | |
|---|---|
| indexed | 48,706 files in 308 projects, 435,243 chunks, 163M tokens |
| time | 4.59 h, 26.0 chunks/s, 9.7k tokens/s |
| host share | 4% (file reading, chunking, tokenizing, embedding lookup); GPU 96% |
| size | 2.00 GB SQLite, plus a 1.34 GB search cache |
| card | engine 997-1543 MHz, memory 400-2000 MHz, 49-75 C, fan up to 57% |
| canary | 28 checks, all cosine 1.0000001, max difference 1.7e-7 |
| errors | none |
| search | about 1 s per run of queries, including loading the model |
