# Training on the RX 590

rawgpu trains models on the card itself: **rawgpu.nn**, a small autograd library whose every kernel
has a backward pass checked against PyTorch, and **rawgpu Studio**, a local web app that runs training
recipes and shows them as they go.

```
tools/studio.sh                       # http://127.0.0.1:8765, opened in the browser
```

Studio needs a Python with numpy. The EmbeddingGemma 2 recipes also need `tokenizers` and
`huggingface_hub`; this checkout's `local/eg2-venv` has all three
([models/embeddinggemma2/README.md](../models/embeddinggemma2/README.md) sets it up), and
`tools/studio.sh` uses it when it's there.

## rawgpu.nn

```python
import numpy as np
from rawgpu import nn

model = nn.MLP([768, 256, 10])
opt = nn.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, max_grad_norm=1.0)
for x, y in batches:                          # numpy arrays
    loss = nn.cross_entropy(model(nn.tensor(x)), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(loss.item())                            # reading a value waits for the GPU; nothing else does
```

- **Tensors:** float32 (int32 for ids and labels), 2-D in memory: the leading dimensions fold into
  the rows. Column and row slices are views of the same memory, which is how attention reads q, k and
  v out of one fused projection.
- **Ops**, each with its backward pass:
  - matrix products in all four transposes, and `linear`;
  - element-wise arithmetic with broadcasting of rows, columns and scalars;
  - ReLU, GELU, SiLU, tanh, sigmoid, exp, log, square, sqrt, softplus and abs;
  - sums and means;
  - RMSNorm (also per head), LayerNorm and L2 normalization;
  - softmax, cross entropy (with ignored rows and label smoothing), BCE with logits, MSE;
  - multi-head attention: causal, padding masks, sliding windows, and grouped key/value heads;
  - RoPE, embedding gathers (the backward is an atomic scatter-add), mean pooling, and concatenation.
- **Layers:**
  - `Linear`, `Embedding`, `LayerNorm`, `RMSNorm` and `MLP`;
  - `SelfAttention`, `Block` and `Transformer`;
  - `GPT`, a small decoder with tied embeddings and sampling.
- **Optimizers:** `AdamW` and `SGD`, with the global gradient norm and clipping computed on the GPU:
  a step never waits for the host, and a non-finite gradient skips the step. There is also
  `cosine_lr`, and `ema_update` for moving-average weights.
- **Memory:** `nn.checkpoint(fn, inputs, params)` recomputes `fn` in the backward pass instead of
  keeping its activations.
- **Checked:** `python/test_nn.py` runs every op on random inputs and compares the output and every
  gradient with PyTorch in float64. They agree to about 1e-7, AdamW over three clipped steps to 1e-7,
  and a tiny GPT learns a pattern.

  ```
  local/eg2-venv/bin/python python/test_nn.py
  ```

How it stays fast enough:
- **No host round trips.** Launches are asynchronous and in order, so a step queues all its work and
  only a `.numpy()` or `.item()` waits. Uploads go through a small ring of host buffers that the GPU
  copies from, because a plain write would wait for every queued launch first.
- **Its own VRAM pool.** It takes VRAM in 256 MB chunks and carves them itself. The library's
  allocator aligns every buffer to 64 KB and tracks 256 free extents, which is too few for a training
  step's thousands of small tensors. Freed blocks are reused by size, safely, because a kernel that
  writes a reused block runs after every kernel that read it.
- **Padded rows.** Rows a multiple of 64 floats wide get 32 more, because the SGEMM runs several times
  slower on rows a large power of two apart (see `docs/LIBRARY.md`).

Limits:
- **Precision:** float32 only. Polaris runs half precision no faster, so it would only save memory.
- **Missing ops:** no convolutions and no dropout yet.
- **One program:** only one process at a time can use the card (Studio queues its runs).

## EmbeddingGemma 2 fine-tuning

`models/embeddinggemma2/eg2_train.py` is `eg2.py`'s forward pass rebuilt from rawgpu.nn ops:
- **What trains:** any number of the top layers train, and the rest are frozen. The 262144-row token
  table stays frozen on the host.
- **Memory:** each layer is checkpointed. Without that, training holds about 2 MB of activations per
  token; with it, a 4-text batch up to 903 tokens peaks at 2.3 GB.
- **Export:** `export(dir)` writes a model directory that `eg2.EmbeddingGemma2(path=dir)` loads.

`check_train.py` checks it against the reference:
- **Embeddings:** cosine 1.000000 with eg2.py and with sentence-transformers.
- **Gradients:** all 388 weights' gradients agree with PyTorch's. The median relative difference is
  7e-6, and the largest 8e-5.
- **Speed:** forward and backward over all 24 layers took 1.8 s on the card, and 8.4 s on the M4 Pro's
  CPU.

```
local/eg2-venv/bin/python models/embeddinggemma2/check_train.py
```

## Studio's recipes

| Recipe | In the diagram | What it trains |
|---|---|---|
| Tiny GPT | (from scratch) | A byte-level transformer on any text; by default this repo's source. |
| JEPA world model | World model, Planner | An encoder, its EMA target, and an action-conditioned latent predictor; a planner searches imagined rollouts. |
| Salience probes | Salience filters | One probe per concept on EmbeddingGemma 2 embeddings, against cosine-to-prototype. |
| Jev-style critic | Critic | State plus a typed question (Choice, Score, yes/no) to calibrated probabilities over options written in text. |
| Fine-tune EmbeddingGemma 2 | Encoder | Contrastive (InfoNCE) fine-tuning of the encoder's top layers. |

Every run is a directory under `<rawgpu data>/runs`, which is this checkout's `local/runs`, or
`~/Library/Application Support/rawgpu/runs` for a release. It holds:
- its parameters (`config.json`) and progress, including the card's readings (`status.json`);
- metrics (`metrics.jsonl`) and evaluations and samples (`events.jsonl`);
- the recipe's output (`log.txt`) and `checkpoints/`.

Runs behave the same way across recipes:
- **Stop** saves a checkpoint, and **Resume** continues from it.
- A run pauses while the card is at its temperature limit (85 °C by default).
- **Try it** runs a finished model on your input (GPT, probes, critic).

### The JEPA world model

**The world** is a 32 × 32 arena: an agent the actions move, and a ball that bounces on its own. An
observation is the frame plus the ball's previous position.

**The model** learns to predict future latents and never reconstructs pixels. Its loss has three terms:
- **Prediction:** the distance between the predictor's rollout from E(o₀) and the EMA target's
  E′(o₁…o_H).
- **Anti-collapse:** VICReg's variance and covariance terms keep the latents from collapsing.
- **Inverse dynamics:** a small head must tell which action was taken from (zₜ, zₜ₊₁).

The last term matters:
- **Without it,** the latents kept the ball, the hard part to predict, and dropped the agent entirely.
  A probe could not find it (10 px error, about chance), and the planner never reached a goal.
- **With weight 1.0,** they kept the agent and lost the ball.
- **The default, 0.3 with a 128-dim latent,** keeps both.

**To see what the latents know,** a ridge-regression probe reads positions out of them. Applied to an
imagined rollout, it shows how far ahead the imagination stays right.

**The planner** uses random-shooting model-predictive control:
- it imagines 256 action sequences of 8 steps in latent space;
- it keeps the one whose decoded end is nearest the goal;
- it takes 4 of those steps in the real arena, then plans again.

Measured with the defaults (4000 steps, about 4.5 minutes):

| | 1 step ahead | 16 steps ahead | Encoded directly |
|---|---|---|---|
| Agent position error | 0.74 px | 2.08 px | 0.68 px |
| Ball position error | 1.36 px | 5.32 px | 2.86 px |

- **Latent prediction error:** 0.02–0.05 of the latents' variance up to 8 steps ahead. Predicting no
  change scores 0.2–2.0.
- **Planning:** within 3 px of a random goal after 16 steps in 53% of 64 episodes. Random actions get
  there 5% of the time. The mean distance left is 3.0 px against 13.4.

### The Jev-style critic

[Jev](https://codelabs.developers.google.com/codelabs/discriminative-models) (TypeSafe AI) answers
typed questions about a state with probabilities, not text. This is the same contract at small scale:

**Encoding:** EmbeddingGemma 2 embeds the state, the question and each option's description.

**Scoring:** a two-tower head maps (state, question) and (option, question) into one space. Their dot
products, softmaxed over the question's options, are the probabilities.

**Training:** it trains on several phrasings of each question, and on each option as "name:
description", the bare name, and the description alone. Options are text, so you can ask it a new
question; it answers best near what it was trained on.

**Calibration:** temperature scaling on the validation split calibrates it.

**Measurement:** the test split measures ECE, Brier, NLL and reliability before and after. Confidence
is Jev's (J·p_max − 1)/(J − 1).

The built-in data asks four questions about chunks of your code index:
- what language it is in;
- what kind of file it is part of;
- how large that file is (a Score);
- whether it is a test (yes or no).

The answers come from file paths. The index embedded every chunk as `title: <path> | text: <chunk>`,
so the paths, and with them the answers, are partly readable from the index's own vectors. The
default is therefore **content only**: the critic and the probes re-embed the chunks' text alone on
the card (the first 256 tokens, cached in `<rawgpu data>/studio-cache.sqlite`). "As indexed" is
there to compare.

### Data and privacy

Everything stays on the Mac:
- **Inputs:** recipes read the code index read-only, or your files and JSONL.
- **Outputs:** they write only to the run directories and the cache.
- **Network:** the server listens on 127.0.0.1 only. It accepts changes only as JSON from its own
  page, and checks the Host header against DNS rebinding.

## Writing a recipe

A recipe is a module in `studio/recipes/`, listed in `ORDER` in `studio/recipes/__init__.py`. It
defines:
- **`META`:** its name, its parameters with defaults, and the tiles, charts and panels the page shows.
- **`train(run)`:** the training loop.
- **`try_it(run, request)`** (optional): the playground.

`studio/run.py` documents what `train` gets:

```python
def train(run):
    model, opt = ...
    run.saver = lambda: run.save(model=model, opt=opt, step=run._step)   # what Stop saves
    for step in run.steps(0, int(run.p("steps"))):     # stops cleanly, pauses when hot
        ...
        if run.every(25): run.log(step, loss=loss.item())
        if run.every(500): run.event("sample", step, text=...)
```
