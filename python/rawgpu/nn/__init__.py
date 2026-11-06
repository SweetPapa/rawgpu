"""rawgpu.nn: train models on the GPU, with reverse-mode autograd over rawgpu kernels.

    import numpy as np
    from rawgpu import nn
    model = nn.MLP([784, 256, 10])
    opt = nn.AdamW(model.parameters(), lr=1e-3, max_grad_norm=1.0)
    for x, y in batches:
        loss = nn.cross_entropy(model(nn.tensor(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(loss.item())                 # reading a value waits for the GPU; everything else is queued

Everything is float32. The matrix products are rawgpu's register-blocked SGEMM; the rest are the kernels
in kernels.cl, each with its backward pass, checked against PyTorch by python/test_nn.py.
"""
from .core import *  # noqa: F401,F403
from .core import Context, Tensor, close, ctx, no_grad, pitch  # noqa: F401
from .layers import (GPT, MLP, Block, Dropout, Embedding, LayerNorm, Linear, Module, RMSNorm,  # noqa: F401
                     SelfAttention, Transformer, rope_frequencies)
from .optim import SGD, AdamW, copy_weights, cosine_lr, ema_update  # noqa: F401
