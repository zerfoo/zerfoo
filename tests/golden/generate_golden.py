#!/usr/bin/env python3
"""
Generate golden test data for Zerfoo layer parity testing.

For each layer, generates deterministic inputs and weights using a fixed seed,
runs the forward pass (and backward where applicable) through PyTorch, and
saves the results as JSON files in tests/golden/layers/.

Usage:
    pip install torch numpy
    python tests/golden/generate_golden.py
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "layers")
SEED = 42


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_list(t):
    """Flatten tensor to list of Python floats."""
    return t.detach().float().contiguous().view(-1).tolist()


def save_case(name, case):
    """Save a test case to JSON."""
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(case, f, indent=2)
    print(f"  {name}.json")


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def gen_relu():
    set_seed()
    x = torch.randn(2, 8)
    y = F.relu(x)
    # Backward
    dy = torch.randn_like(y)
    x.requires_grad_(True)
    y2 = F.relu(x)
    y2.backward(dy)
    save_case("activation_relu", {
        "layer": "relu",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x.grad),
        "tolerance": 1e-6,
    })


def gen_gelu():
    set_seed()
    x = torch.randn(2, 8)
    # Zerfoo uses tanh approximation
    y = F.gelu(x, approximate="tanh")
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    y2 = F.gelu(x_g, approximate="tanh")
    y2.backward(dy)
    save_case("activation_gelu", {
        "layer": "gelu",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-5,
    })


def gen_sigmoid():
    set_seed()
    x = torch.randn(2, 8)
    y = torch.sigmoid(x)
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    y2 = torch.sigmoid(x_g)
    y2.backward(dy)
    save_case("activation_sigmoid", {
        "layer": "sigmoid",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-6,
    })


def gen_tanh():
    set_seed()
    x = torch.randn(2, 8)
    y = torch.tanh(x)
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    y2 = torch.tanh(x_g)
    y2.backward(dy)
    save_case("activation_tanh", {
        "layer": "tanh",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-6,
    })


def gen_silu():
    set_seed()
    x = torch.randn(2, 8)
    y = F.silu(x)
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    y2 = F.silu(x_g)
    y2.backward(dy)
    save_case("activation_silu", {
        "layer": "silu",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-6,
    })


def gen_leaky_relu():
    set_seed()
    x = torch.randn(2, 8)
    alpha = 0.01
    y = F.leaky_relu(x, negative_slope=alpha)
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    y2 = F.leaky_relu(x_g, negative_slope=alpha)
    y2.backward(dy)
    save_case("activation_leaky_relu", {
        "layer": "leaky_relu",
        "alpha": alpha,
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-6,
    })


def gen_softmax():
    set_seed()
    x = torch.randn(2, 8)
    y = F.softmax(x, dim=-1)
    save_case("activation_softmax", {
        "layer": "softmax",
        "axis": -1,
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-6,
    })


def gen_erf():
    set_seed()
    x = torch.randn(2, 8)
    y = torch.erf(x)
    save_case("activation_erf", {
        "layer": "erf",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-6,
    })


def gen_swiglu():
    """SwiGLU: input is [batch, 2*dim], splits into [x1, x2], output = SiLU(x1) * x2."""
    set_seed()
    dim = 8
    x = torch.randn(2, 2 * dim)
    x1, x2 = x.chunk(2, dim=-1)
    y = F.silu(x1) * x2

    dy = torch.randn(2, dim)
    x_g = x.clone().requires_grad_(True)
    x1g, x2g = x_g.chunk(2, dim=-1)
    y2 = F.silu(x1g) * x2g
    y2.backward(dy)
    save_case("activation_swiglu", {
        "layer": "swiglu",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def gen_layer_norm():
    set_seed()
    batch, features = 2, 8
    x = torch.randn(batch, features)
    gamma = torch.randn(features)
    beta = torch.randn(features)
    eps = 1e-5

    # Manual LayerNorm matching Zerfoo (unbiased=False, i.e. divide by N)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = gamma * (x - mean) / torch.sqrt(var + eps) + beta

    # Also verify against PyTorch's LayerNorm
    ln = nn.LayerNorm(features, eps=eps)
    ln.weight.data.copy_(gamma)
    ln.bias.data.copy_(beta)
    y_pt = ln(x)
    assert torch.allclose(y, y_pt, atol=1e-6), "LayerNorm formula mismatch"

    # Backward
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    ln2 = nn.LayerNorm(features, eps=eps)
    ln2.weight.data.copy_(gamma)
    ln2.bias.data.copy_(beta)
    y2 = ln2(x_g)
    y2.backward(dy)

    save_case("norm_layer_norm", {
        "layer": "layer_norm",
        "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "gamma": to_list(gamma), "gamma_shape": list(gamma.shape),
        "beta": to_list(beta), "beta_shape": list(beta.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-5,
    })


def gen_rms_norm():
    set_seed()
    batch, dim = 2, 8
    x = torch.randn(batch, dim)
    gain = torch.randn(dim)
    eps = 1e-6

    # RMSNorm: y = gain * x / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = gain * (x / rms)

    # Backward
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    rms2 = torch.sqrt(x_g.pow(2).mean(dim=-1, keepdim=True) + eps)
    y2 = gain * (x_g / rms2)
    y2.backward(dy)

    save_case("norm_rms_norm", {
        "layer": "rms_norm",
        "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "gain": to_list(gain), "gain_shape": list(gain.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-5,
    })


def gen_batch_norm():
    set_seed()
    # BatchNorm inference mode: y = scale * (x - mean) / sqrt(var + eps) + bias
    n, c, h, w = 1, 4, 2, 2
    x = torch.randn(n, c, h, w)
    running_mean = torch.randn(c)
    running_var = torch.abs(torch.randn(c)) + 0.1  # positive
    scale = torch.randn(c)
    bias = torch.randn(c)
    eps = 1e-5

    bn = nn.BatchNorm2d(c, eps=eps)
    bn.weight.data.copy_(scale)
    bn.bias.data.copy_(bias)
    bn.running_mean.data.copy_(running_mean)
    bn.running_var.data.copy_(running_var)
    bn.eval()
    y = bn(x)

    save_case("norm_batch_norm", {
        "layer": "batch_norm",
        "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "scale": to_list(scale), "scale_shape": list(scale.shape),
        "bias": to_list(bias), "bias_shape": list(bias.shape),
        "running_mean": to_list(running_mean),
        "running_var": to_list(running_var),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# Core layers
# ---------------------------------------------------------------------------

def gen_linear():
    """Zerfoo Linear: y = x @ W where W is [in_features, out_features]. No bias."""
    set_seed()
    in_f, out_f = 8, 4
    x = torch.randn(2, in_f)
    # Zerfoo stores W as [in, out], so generate it that way
    w = torch.randn(in_f, out_f)
    y = x @ w

    # Backward
    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    w_g = w.clone().requires_grad_(True)
    y2 = x_g @ w_g
    y2.backward(dy)

    save_case("core_linear", {
        "layer": "linear",
        "input": to_list(x), "input_shape": list(x.shape),
        "weight": to_list(w), "weight_shape": list(w.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "expected_grad_weight": to_list(w_g.grad),
        "tolerance": 1e-5,
    })


def gen_dense():
    """Dense = Linear + Bias."""
    set_seed()
    in_f, out_f = 8, 4
    x = torch.randn(2, in_f)
    w = torch.randn(in_f, out_f)
    b = torch.randn(out_f)
    y = x @ w + b

    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    w_g = w.clone().requires_grad_(True)
    b_g = b.clone().requires_grad_(True)
    y2 = x_g @ w_g + b_g
    y2.backward(dy)

    save_case("core_dense", {
        "layer": "dense",
        "input": to_list(x), "input_shape": list(x.shape),
        "weight": to_list(w), "weight_shape": list(w.shape),
        "bias": to_list(b), "bias_shape": list(b.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "expected_grad_weight": to_list(w_g.grad),
        "expected_grad_bias": to_list(b_g.grad),
        "tolerance": 1e-5,
    })


def gen_conv1d():
    set_seed()
    batch, in_ch, length = 1, 2, 8
    out_ch, kernel = 3, 3
    stride, padding = 1, 1

    x = torch.randn(batch, in_ch, length)
    conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    # Zerfoo Conv1D uses cross-correlation like PyTorch
    y = conv(x)

    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    conv2 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
    conv2.weight.data.copy_(conv.weight.data)
    conv2.bias.data.copy_(conv.bias.data)
    y2 = conv2(x_g)
    y2.backward(dy)

    save_case("core_conv1d", {
        "layer": "conv1d",
        "in_channels": in_ch, "out_channels": out_ch,
        "kernel_size": kernel, "stride": stride, "padding": padding,
        "input": to_list(x), "input_shape": list(x.shape),
        "weight": to_list(conv.weight), "weight_shape": list(conv.weight.shape),
        "bias": to_list(conv.bias), "bias_shape": list(conv.bias.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-4,
    })


def gen_conv2d():
    set_seed()
    batch, in_ch, h, w = 1, 1, 4, 4
    out_ch, kh, kw = 2, 3, 3
    stride, pad = [1, 1], [1, 1, 1, 1]

    x = torch.randn(batch, in_ch, h, w)
    conv = nn.Conv2d(in_ch, out_ch, (kh, kw), stride=tuple(stride), padding=(pad[0], pad[1]))
    y = conv(x)

    dy = torch.randn_like(y)
    x_g = x.clone().requires_grad_(True)
    conv2 = nn.Conv2d(in_ch, out_ch, (kh, kw), stride=tuple(stride), padding=(pad[0], pad[1]))
    conv2.weight.data.copy_(conv.weight.data)
    conv2.bias.data.copy_(conv.bias.data)
    y2 = conv2(x_g)
    y2.backward(dy)

    save_case("core_conv2d", {
        "layer": "conv2d",
        "in_channels": in_ch, "out_channels": out_ch,
        "kernel_size": [kh, kw], "stride": stride,
        "padding": pad,
        "input": to_list(x), "input_shape": list(x.shape),
        "weight": to_list(conv.weight), "weight_shape": list(conv.weight.shape),
        "bias": to_list(conv.bias), "bias_shape": list(conv.bias.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_input": to_list(x_g.grad),
        "tolerance": 1e-4,
    })


def gen_matmul():
    set_seed()
    a = torch.randn(2, 4)
    b = torch.randn(4, 3)
    y = a @ b

    dy = torch.randn_like(y)
    a_g = a.clone().requires_grad_(True)
    b_g = b.clone().requires_grad_(True)
    y2 = a_g @ b_g
    y2.backward(dy)

    save_case("core_matmul", {
        "layer": "matmul",
        "input_a": to_list(a), "input_a_shape": list(a.shape),
        "input_b": to_list(b), "input_b_shape": list(b.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "grad_output": to_list(dy),
        "expected_grad_a": to_list(a_g.grad),
        "expected_grad_b": to_list(b_g.grad),
        "tolerance": 1e-5,
    })


def gen_ffn():
    """FFN: gate-up-swiglu-down pattern.
    gate = x @ w1 (dim -> hidden)
    up   = x @ w3 (dim -> hidden)
    mid  = SiLU(gate) * up
    out  = mid @ w2 (hidden -> dim)
    """
    set_seed()
    batch, dim, hidden = 2, 8, 16
    x = torch.randn(batch, dim)
    # Zerfoo stores weights as [in, out]
    w1 = torch.randn(dim, hidden)  # gate
    w3 = torch.randn(dim, hidden)  # up
    w2 = torch.randn(hidden, dim)  # down

    gate = x @ w1
    up = x @ w3
    mid = F.silu(gate) * up
    y = mid @ w2

    save_case("core_ffn", {
        "layer": "ffn",
        "input": to_list(x), "input_shape": list(x.shape),
        "w1": to_list(w1), "w1_shape": list(w1.shape),
        "w2": to_list(w2), "w2_shape": list(w2.shape),
        "w3": to_list(w3), "w3_shape": list(w3.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def gen_scaled_dot_product_attention():
    """SDPA: causal, scale = 1/sqrt(d_k)."""
    set_seed()
    batch, seq_len, d_k = 1, 4, 8
    q = torch.randn(batch, seq_len, d_k)
    k = torch.randn(batch, seq_len, d_k)
    v = torch.randn(batch, seq_len, d_k)

    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    y = torch.matmul(attn_weights, v)

    save_case("attention_sdpa_causal", {
        "layer": "scaled_dot_product_attention",
        "causal": True,
        "query": to_list(q), "query_shape": list(q.shape),
        "key": to_list(k), "key_shape": list(k.shape),
        "value": to_list(v), "value_shape": list(v.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "expected_attn_weights": to_list(attn_weights),
        "tolerance": 1e-5,
    })

    # Bidirectional (no mask)
    set_seed(43)
    q2 = torch.randn(batch, seq_len, d_k)
    k2 = torch.randn(batch, seq_len, d_k)
    v2 = torch.randn(batch, seq_len, d_k)
    scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale
    attn_weights2 = F.softmax(scores2, dim=-1)
    y2 = torch.matmul(attn_weights2, v2)

    save_case("attention_sdpa_bidirectional", {
        "layer": "scaled_dot_product_attention",
        "causal": False,
        "query": to_list(q2), "query_shape": list(q2.shape),
        "key": to_list(k2), "key_shape": list(k2.shape),
        "value": to_list(v2), "value_shape": list(v2.shape),
        "expected_output": to_list(y2), "output_shape": list(y2.shape),
        "expected_attn_weights": to_list(attn_weights2),
        "tolerance": 1e-5,
    })


def gen_multi_head_attention():
    """Multi-head attention: split heads, run SDPA per head, concat."""
    set_seed()
    seq_len, d_model = 4, 16
    n_heads = 2
    d_k = d_model // n_heads

    q = torch.randn(seq_len, d_model)
    k = torch.randn(seq_len, d_model)
    v = torch.randn(seq_len, d_model)

    # Split into heads: [seq, d_model] -> [n_heads, seq, d_k]
    q_heads = q.view(seq_len, n_heads, d_k).transpose(0, 1)
    k_heads = k.view(seq_len, n_heads, d_k).transpose(0, 1)
    v_heads = v.view(seq_len, n_heads, d_k).transpose(0, 1)

    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
    # No causal mask - functional.MultiHeadAttention is bidirectional
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v_heads)

    # Concat heads: [n_heads, seq, d_k] -> [seq, d_model]
    y = attn_out.transpose(0, 1).contiguous().view(seq_len, d_model)

    save_case("attention_multi_head", {
        "layer": "multi_head_attention",
        "n_heads": n_heads,
        "query": to_list(q), "query_shape": list(q.shape),
        "key": to_list(k), "key_shape": list(k.shape),
        "value": to_list(v), "value_shape": list(v.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def gen_token_embedding():
    set_seed()
    vocab, dim = 16, 8
    table = torch.randn(vocab, dim)
    indices = torch.tensor([0, 3, 7, 15, 1])
    y = table[indices]

    save_case("embedding_token", {
        "layer": "token_embedding",
        "vocab_size": vocab, "embedding_dim": dim,
        "table": to_list(table), "table_shape": list(table.shape),
        "indices": indices.tolist(),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-7,
    })


def gen_rotary_embedding():
    """RoPE: interleaved pairs rotation."""
    set_seed()
    batch, seq_len, head_dim = 1, 4, 8
    base = 10000.0
    x = torch.randn(batch, seq_len, head_dim)

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    # Position indices
    pos = torch.arange(seq_len, dtype=torch.float32)
    # Outer product: [seq_len, head_dim/2]
    angles = torch.outer(pos, inv_freq)

    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # Apply rotation to split halves (GPT-NeoX / Zerfoo style)
    # First half and second half of head_dim
    half = head_dim // 2
    x1 = x[..., :half]   # first half
    x2 = x[..., half:]   # second half
    y = torch.zeros_like(x)
    y[..., :half] = x1 * cos_vals - x2 * sin_vals
    y[..., half:] = x2 * cos_vals + x1 * sin_vals

    save_case("embedding_rotary", {
        "layer": "rotary_embedding",
        "base": base,
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "cos_values": to_list(cos_vals), "cos_shape": list(cos_vals.shape),
        "sin_values": to_list(sin_vals), "sin_shape": list(sin_vals.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def gen_mse_loss():
    set_seed()
    pred = torch.randn(2, 4)
    target = torch.randn(2, 4)
    loss = F.mse_loss(pred, target, reduction="mean")

    # Backward
    pred_g = pred.clone().requires_grad_(True)
    loss2 = F.mse_loss(pred_g, target, reduction="mean")
    loss2.backward()

    save_case("loss_mse", {
        "layer": "mse_loss",
        "predictions": to_list(pred), "predictions_shape": list(pred.shape),
        "targets": to_list(target), "targets_shape": list(target.shape),
        "expected_loss": loss.item(),
        "expected_grad": to_list(pred_g.grad),
        "tolerance": 1e-6,
    })


def gen_bce_loss():
    set_seed()
    # BCE expects probabilities, not logits
    pred = torch.sigmoid(torch.randn(2, 4))  # ensure in (0,1)
    target = torch.randint(0, 2, (2, 4)).float()
    loss = F.binary_cross_entropy(pred, target, reduction="mean")

    pred_g = pred.clone().requires_grad_(True)
    loss2 = F.binary_cross_entropy(pred_g, target, reduction="mean")
    loss2.backward()

    save_case("loss_bce", {
        "layer": "bce_loss",
        "predictions": to_list(pred), "predictions_shape": list(pred.shape),
        "targets": to_list(target), "targets_shape": list(target.shape),
        "expected_loss": loss.item(),
        "expected_grad": to_list(pred_g.grad),
        "tolerance": 1e-5,
    })


def gen_cross_entropy_loss():
    set_seed()
    batch, classes = 4, 5
    logits = torch.randn(batch, classes)
    # CrossEntropy expects class indices
    targets = torch.randint(0, classes, (batch,))

    loss = F.cross_entropy(logits, targets, reduction="mean")

    logits_g = logits.clone().requires_grad_(True)
    loss2 = F.cross_entropy(logits_g, targets, reduction="mean")
    loss2.backward()

    save_case("loss_cross_entropy", {
        "layer": "cross_entropy_loss",
        "logits": to_list(logits), "logits_shape": list(logits.shape),
        "targets": targets.tolist(),
        "expected_loss": loss.item(),
        "expected_grad": to_list(logits_g.grad),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# SSM / Recurrent
# ---------------------------------------------------------------------------

def gen_simple_rnn():
    """SimpleRNN: h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b)."""
    set_seed()
    batch, seq_len, input_dim, hidden_dim = 1, 4, 8, 6
    x = torch.randn(batch, seq_len, input_dim)

    rnn = nn.RNN(input_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
    h0 = torch.zeros(1, batch, hidden_dim)
    y, hn = rnn(x, h0)

    # Extract weights for golden file (PyTorch RNN stores them transposed)
    # weight_ih: [hidden, input], weight_hh: [hidden, hidden]
    # bias_ih + bias_hh combined
    w_ih = rnn.weight_ih_l0.detach()  # [hidden, input]
    w_hh = rnn.weight_hh_l0.detach()  # [hidden, hidden]
    b_ih = rnn.bias_ih_l0.detach()    # [hidden]
    b_hh = rnn.bias_hh_l0.detach()    # [hidden]

    save_case("recurrent_simple_rnn", {
        "layer": "simple_rnn",
        "input_dim": input_dim, "hidden_dim": hidden_dim,
        "input": to_list(x), "input_shape": list(x.shape),
        "weight_ih": to_list(w_ih), "weight_ih_shape": list(w_ih.shape),
        "weight_hh": to_list(w_hh), "weight_hh_shape": list(w_hh.shape),
        "bias_ih": to_list(b_ih), "bias_hh": to_list(b_hh),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "expected_final_hidden": to_list(hn),
        "tolerance": 1e-5,
    })


def gen_s4():
    """S4 diagonal SSM: x_k = a_disc * x_{k-1} + b * u_k, y_k = c^T * x_k + d * u_k.
    a_disc = exp(-exp(a_log))
    """
    set_seed()
    batch, seq_len = 1, 8
    input_dim, state_dim = 4, 6

    u = torch.randn(batch, seq_len, input_dim)

    # Learnable parameters
    a_log = torch.randn(input_dim, state_dim) * 0.1
    b = torch.randn(input_dim, state_dim) * 0.1
    c = torch.randn(input_dim, state_dim) * 0.1
    d = torch.randn(input_dim) * 0.1

    # Discretize
    a_disc = torch.exp(-torch.exp(a_log))  # stable in (0, 1)

    # Scan
    outputs = []
    state = torch.zeros(batch, input_dim, state_dim)
    for t in range(seq_len):
        u_t = u[:, t, :]  # [batch, input_dim]
        u_expanded = u_t.unsqueeze(-1)  # [batch, input_dim, 1]
        state = a_disc.unsqueeze(0) * state + b.unsqueeze(0) * u_expanded
        y_t = (state * c.unsqueeze(0)).sum(dim=-1) + d.unsqueeze(0) * u_t
        outputs.append(y_t)
    y = torch.stack(outputs, dim=1)

    save_case("ssm_s4", {
        "layer": "s4",
        "input_dim": input_dim, "state_dim": state_dim,
        "input": to_list(u), "input_shape": list(u.shape),
        "a_log": to_list(a_log), "a_log_shape": list(a_log.shape),
        "b": to_list(b), "b_shape": list(b.shape),
        "c": to_list(c), "c_shape": list(c.shape),
        "d": to_list(d), "d_shape": list(d.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


def gen_mamba_block():
    """Mamba selective scan with Conv1D + gating."""
    set_seed()
    batch, seq_len = 1, 8
    input_dim, hidden_dim = 4, 8
    state_size, conv_kernel = 4, 4

    x = torch.randn(batch, seq_len, input_dim)

    # In projection: x -> [x_branch, z_branch]
    w_in = torch.randn(input_dim, 2 * hidden_dim) * 0.1
    xz = x @ w_in
    x_branch, z_branch = xz.chunk(2, dim=-1)

    # Conv1D (causal, depthwise)
    # Simplified: just use 1D conv with left padding
    x_conv = x_branch.transpose(1, 2)  # [batch, hidden, seq]
    conv = nn.Conv1d(hidden_dim, hidden_dim, conv_kernel,
                     padding=conv_kernel - 1, groups=hidden_dim)
    x_conv_out = conv(x_conv)[:, :, :seq_len]  # causal: trim right
    x_branch2 = F.silu(x_conv_out.transpose(1, 2))

    # Delta, B, C projection
    w_dt = torch.randn(hidden_dim, state_size + state_size + 1) * 0.1
    dtbc = x_branch2 @ w_dt[:, :state_size + state_size + 1]
    dt_raw = dtbc[:, :, :1]
    B = dtbc[:, :, 1:1 + state_size]
    C = dtbc[:, :, 1 + state_size:]

    dt = F.softplus(dt_raw)

    # A parameter (log-parameterized diagonal)
    A_log = torch.randn(hidden_dim, state_size) * 0.1
    A = -torch.exp(A_log)

    # Selective scan
    outputs = []
    state = torch.zeros(batch, hidden_dim, state_size)
    for t in range(seq_len):
        u_t = x_branch2[:, t, :]
        dt_t = dt[:, t, :]
        B_t = B[:, t, :]
        C_t = C[:, t, :]

        dA = torch.exp(dt_t * A.unsqueeze(0))
        dB = dt_t * B_t.unsqueeze(1).expand_as(state)

        state = dA * state + dB * u_t.unsqueeze(-1)
        y_t = (state * C_t.unsqueeze(1).expand_as(state)).sum(dim=-1)
        outputs.append(y_t)

    y_scan = torch.stack(outputs, dim=1)

    # Gate: y * SiLU(z)
    y_gated = y_scan * F.silu(z_branch)

    # Out projection
    w_out = torch.randn(hidden_dim, input_dim) * 0.1
    y = y_gated @ w_out

    save_case("ssm_mamba", {
        "layer": "mamba_block",
        "description": "Simplified Mamba block for parity testing",
        "input_dim": input_dim, "hidden_dim": hidden_dim,
        "state_size": state_size, "conv_kernel": conv_kernel,
        "input": to_list(x), "input_shape": list(x.shape),
        "w_in": to_list(w_in), "w_in_shape": list(w_in.shape),
        "conv_weight": to_list(conv.weight), "conv_weight_shape": list(conv.weight.shape),
        "conv_bias": to_list(conv.bias), "conv_bias_shape": list(conv.bias.shape),
        "w_dt": to_list(w_dt), "w_dt_shape": list(w_dt.shape),
        "A_log": to_list(A_log), "A_log_shape": list(A_log.shape),
        "w_out": to_list(w_out), "w_out_shape": list(w_out.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-3,
    })


# ---------------------------------------------------------------------------
# Other ops
# ---------------------------------------------------------------------------

def gen_reduce_sum():
    set_seed()
    x = torch.randn(2, 3, 4)

    # Reduce over axis 1, keepdims
    y_keep = x.sum(dim=1, keepdim=True)
    # Reduce over axis 1, no keepdims
    y_nokeep = x.sum(dim=1, keepdim=False)

    save_case("op_reduce_sum", {
        "layer": "reduce_sum",
        "input": to_list(x), "input_shape": list(x.shape),
        "axes": [1], "keep_dims": True,
        "expected_output_keepdims": to_list(y_keep),
        "output_shape_keepdims": list(y_keep.shape),
        "expected_output_nokeepdims": to_list(y_nokeep),
        "output_shape_nokeepdims": list(y_nokeep.shape),
        "tolerance": 1e-6,
    })


def gen_transpose():
    set_seed()
    x = torch.randn(2, 3, 4)
    # Permute axes [0, 2, 1]
    y = x.permute(0, 2, 1)
    save_case("op_transpose", {
        "layer": "transpose",
        "axes": [0, 2, 1],
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y.contiguous()), "output_shape": list(y.shape),
        "tolerance": 1e-7,
    })


def gen_gather():
    """Gather/Embedding lookup."""
    set_seed()
    vocab, dim = 10, 4
    table = torch.randn(vocab, dim)
    indices = torch.tensor([2, 5, 0, 9])
    y = table[indices]
    save_case("op_gather", {
        "layer": "gather",
        "input": to_list(table), "input_shape": list(table.shape),
        "indices": indices.tolist(),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-7,
    })


def gen_dropout():
    """Dropout: verify scaling factor. Inference mode only (no randomness)."""
    set_seed()
    x = torch.randn(2, 8)
    # In eval mode, dropout is identity
    drop = nn.Dropout(p=0.5)
    drop.eval()
    y = drop(x)
    save_case("op_dropout", {
        "layer": "dropout",
        "rate": 0.5,
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output_eval": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-7,
    })


# ---------------------------------------------------------------------------
# Optimizer step verification
# ---------------------------------------------------------------------------

def gen_adamw_step():
    """Verify one AdamW step matches PyTorch."""
    set_seed()
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    param = torch.randn(4, 4)
    grad = torch.randn(4, 4)

    p = param.clone().requires_grad_(False)
    p_pt = nn.Parameter(param.clone())
    # Simulate one step manually
    opt = torch.optim.AdamW([p_pt], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)
    p_pt.grad = grad.clone()
    opt.step()

    save_case("optimizer_adamw", {
        "layer": "adamw",
        "lr": lr, "beta1": beta1, "beta2": beta2, "epsilon": eps, "weight_decay": wd,
        "param_before": to_list(param), "param_shape": list(param.shape),
        "grad": to_list(grad),
        "expected_param_after": to_list(p_pt.data),
        "tolerance": 1e-6,
    })


def gen_sgd_step():
    set_seed()
    lr = 0.01
    param = torch.randn(4, 4)
    grad = torch.randn(4, 4)

    p_pt = nn.Parameter(param.clone())
    opt = torch.optim.SGD([p_pt], lr=lr)
    p_pt.grad = grad.clone()
    opt.step()

    save_case("optimizer_sgd", {
        "layer": "sgd",
        "lr": lr,
        "param_before": to_list(param), "param_shape": list(param.shape),
        "grad": to_list(grad),
        "expected_param_after": to_list(p_pt.data),
        "tolerance": 1e-6,
    })


# ---------------------------------------------------------------------------
# Composite: Transformer Block
# ---------------------------------------------------------------------------

def gen_transformer_block():
    """TransformerBlock = RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual."""
    set_seed()
    batch, seq_len = 1, 4
    d_model, n_heads = 16, 2
    d_ff = 32
    eps = 1e-6
    d_k = d_model // n_heads

    x = torch.randn(batch, seq_len, d_model)

    # RMSNorm params
    attn_norm_gain = torch.ones(d_model)
    ffn_norm_gain = torch.ones(d_model)

    # Attention Q/K/V/O weights [in, out] (Zerfoo convention)
    wq = torch.randn(d_model, d_model) * 0.02
    wk = torch.randn(d_model, d_model) * 0.02
    wv = torch.randn(d_model, d_model) * 0.02
    wo = torch.randn(d_model, d_model) * 0.02

    # FFN weights [in, out]
    w1 = torch.randn(d_model, d_ff) * 0.02  # gate
    w3 = torch.randn(d_model, d_ff) * 0.02  # up
    w2 = torch.randn(d_ff, d_model) * 0.02  # down

    def rmsnorm(x, gain):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        return gain * (x / rms)

    # Step 1: RMSNorm
    normed = rmsnorm(x, attn_norm_gain)

    # Step 2: Self-attention
    q = normed @ wq
    k = normed @ wk
    v = normed @ wv

    # Reshape for multi-head: [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
    q = q.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
    k = k.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
    v = v.view(batch, seq_len, n_heads, d_k).transpose(1, 2)

    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    attn_out = attn_out @ wo

    # Step 3: Residual
    h = x + attn_out

    # Step 4: RMSNorm
    normed2 = rmsnorm(h, ffn_norm_gain)

    # Step 5: FFN (SwiGLU)
    gate = normed2 @ w1
    up = normed2 @ w3
    mid = F.silu(gate) * up
    ffn_out = mid @ w2

    # Step 6: Residual
    y = h + ffn_out

    save_case("composite_transformer_block", {
        "layer": "transformer_block",
        "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff, "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "attn_norm_gain": to_list(attn_norm_gain),
        "ffn_norm_gain": to_list(ffn_norm_gain),
        "wq": to_list(wq), "wk": to_list(wk),
        "wv": to_list(wv), "wo": to_list(wo),
        "w1": to_list(w1), "w2": to_list(w2), "w3": to_list(w3),
        "weight_shape_qkvo": list(wq.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    print("Generating golden test data...")

    generators = [
        # Activations
        ("ReLU", gen_relu),
        ("GELU", gen_gelu),
        ("Sigmoid", gen_sigmoid),
        ("Tanh", gen_tanh),
        ("SiLU", gen_silu),
        ("LeakyReLU", gen_leaky_relu),
        ("Softmax", gen_softmax),
        ("Erf", gen_erf),
        ("SwiGLU", gen_swiglu),
        # Normalization
        ("LayerNorm", gen_layer_norm),
        ("RMSNorm", gen_rms_norm),
        ("BatchNorm", gen_batch_norm),
        # Core
        ("Linear", gen_linear),
        ("Dense", gen_dense),
        ("Conv1D", gen_conv1d),
        ("Conv2D", gen_conv2d),
        ("MatMul", gen_matmul),
        ("FFN", gen_ffn),
        # Attention
        ("SDPA", gen_scaled_dot_product_attention),
        ("MultiHeadAttention", gen_multi_head_attention),
        # Embeddings
        ("TokenEmbedding", gen_token_embedding),
        ("RotaryEmbedding", gen_rotary_embedding),
        # Loss
        ("MSE", gen_mse_loss),
        ("BCE", gen_bce_loss),
        ("CrossEntropy", gen_cross_entropy_loss),
        # SSM/Recurrent
        ("SimpleRNN", gen_simple_rnn),
        ("S4", gen_s4),
        ("MambaBlock", gen_mamba_block),
        # Ops
        ("ReduceSum", gen_reduce_sum),
        ("Transpose", gen_transpose),
        ("Gather", gen_gather),
        ("Dropout", gen_dropout),
        # Optimizers
        ("AdamW", gen_adamw_step),
        ("SGD", gen_sgd_step),
        # Composite
        ("TransformerBlock", gen_transformer_block),
    ]

    passed, failed = 0, 0
    for name, gen_fn in generators:
        try:
            gen_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            failed += 1

    print(f"\nDone: {passed} generated, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
