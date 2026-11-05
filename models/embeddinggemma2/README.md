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
- **VRAM:** the layer weights take 496 MB, and activations are sized for
  16384 tokens per batch.
- **Token embedding table:** the 262144 × 512 table stays in Mac memory as
  bf16. Only a batch's rows are sent.
- **Sequence length:** up to 2048 tokens. Attention materializes the full
  score matrix, so longer inputs are truncated.
- **Prompts:** the model's own prompts are in `PROMPTS`. For retrieval, use
  `"query"` and `"document"`.

## Checking it

`compare.py` runs eight texts in English, German, Japanese and code,
including one of 1209 tokens that crosses the 512-token sliding window. It
runs them through this engine and through the reference
(sentence-transformers, float32, CPU). It checks two things:

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

Results from 2026-11-04 (500 questions, 7029 paragraphs averaging 119
tokens, 64 joined documents averaging 1926 tokens):

| | paragraphs | questions | 2K-token documents | recall@1 / 5 / 10 / 20 |
|---|---|---|---|---|
| RX 590 (rawgpu, fp32) | **62/s**, 7.4k tok/s | **184/s** | **1.8/s**, 3.4k tok/s | 0.390 / 0.668 / 0.734 / 0.800 |
| M4 Pro GPU (MPS, fp32) | 35/s, 4.2k tok/s | 160/s | 1.5/s, 2.9k tok/s | 0.390 / 0.668 / 0.734 / 0.800 |

- **Total time:** the RX 590 took 152 s and the M4 Pro 246 s, so the RX 590
  ran 1.61x as fast.
- **Agreement:** the two embeddings of every text agree to cosine 1.000000.
- **Card state:** the engine ran at about 1420-1450 MHz and reached 76 C.
- **Batch sizes on the Mac:** MPS ran with batch 32 for short texts and 4 for
  long documents. Batch 64 ran out of MPS memory on the long documents.

**Where the time goes on the RX 590:**
- **Short texts:** the GEMMs dominate. About 280 MFLOP per token makes this
  roughly 2 TFLOPS sustained.
- **Long documents:** attention computes and masks the whole 2048 × 2048
  score matrix. On sliding layers fewer than half of those scores are
  visible, so a windowed or flash-style attention kernel is the obvious next
  speedup.
