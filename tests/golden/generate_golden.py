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
# EMA optimizer
# ---------------------------------------------------------------------------

def gen_ema():
    """Verify one EMA shadow update matches PyTorch formula.

    EMA shadow update: shadow = decay * shadow + (1 - decay) * param
    On the first call to Step, EMA initializes shadow as a copy of param
    and does NOT update (just copies). So we simulate a second call where
    the inner optimizer has already modified param, and the shadow update
    actually fires.
    """
    set_seed()
    decay = 0.999

    # After inner optimizer step, param has some new value.
    # shadow was initialized as copy of param_before (first Step).
    param_before = torch.randn(4, 4)
    shadow_before = param_before.clone()

    # Simulate inner optimizer modifying param (e.g., AdamW step)
    grad = torch.randn(4, 4)
    lr = 1e-3
    param_after_inner = param_before - lr * grad  # simplified SGD-like step

    # EMA update: shadow = decay * shadow + (1-decay) * param_after_inner
    expected_shadow_after = decay * shadow_before + (1 - decay) * param_after_inner

    save_case("optimizer_ema", {
        "layer": "ema",
        "decay": decay,
        "lr": lr,
        "param_before": to_list(param_before),
        "param_shape": list(param_before.shape),
        "grad": to_list(grad),
        "shadow_before": to_list(shadow_before),
        "param_after_inner": to_list(param_after_inner),
        "expected_shadow_after": to_list(expected_shadow_after),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# SWA optimizer
# ---------------------------------------------------------------------------

def gen_swa():
    """Verify SWA running average update.

    SWA formula: avg = avg + (param - avg) / (n + 1)
    First call (n=0): avg is initialized as copy of param (no update).
    Second call (n=0 still, but avg exists): avg = avg + (param - avg) / 1
    We simulate two UpdateAverage calls.
    """
    set_seed()
    # First epoch params
    param0 = torch.randn(4, 4)
    # After first UpdateAverage: avg = param0, n_averaged = 1

    # Second epoch params (simulate training changing weights)
    param1 = torch.randn(4, 4)
    # After second UpdateAverage (n_averaged was 1):
    # avg = avg + (param1 - avg) / (1 + 1) = (param0 + param1) / 2
    expected_avg = param0 + (param1 - param0) / 2.0

    save_case("optimizer_swa", {
        "layer": "swa",
        "param0": to_list(param0),
        "param1": to_list(param1),
        "param_shape": list(param0.shape),
        "expected_avg_after": to_list(expected_avg),
        "tolerance": 1e-5,
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
# FastGelu (T86.1.1)
# ---------------------------------------------------------------------------

def gen_fast_gelu():
    set_seed()
    x = torch.randn(2, 8)
    # FastGelu uses the same tanh approximation as GELU
    y = F.gelu(x, approximate="tanh")
    save_case("activation_fast_gelu", {
        "layer": "fast_gelu",
        "input": to_list(x), "input_shape": list(x.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# SimplifiedLayerNorm (T86.1.2) - same as RMSNorm (no mean subtraction)
# ---------------------------------------------------------------------------

def gen_simplified_layer_norm():
    set_seed()
    batch, dim = 2, 8
    x = torch.randn(batch, dim)
    gain = torch.randn(dim)
    eps = 1e-6

    # SimplifiedLayerNorm = RMSNorm: y = gain * x / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = gain * (x / rms)

    save_case("norm_simplified_layer_norm", {
        "layer": "simplified_layer_norm",
        "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "gain": to_list(gain), "gain_shape": list(gain.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# SkipSimplifiedLayerNorm (T86.1.3) - RMSNorm + residual
# ---------------------------------------------------------------------------

def gen_skip_simplified_layer_norm():
    set_seed()
    batch, dim = 2, 8
    x = torch.randn(batch, dim)
    gain = torch.randn(dim)
    eps = 1e-6

    # SimplifiedLayerNorm then add residual: y = x + gain * x / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = gain * (x / rms)
    y = x + normed

    save_case("norm_skip_simplified_layer_norm", {
        "layer": "skip_simplified_layer_norm",
        "epsilon": eps,
        "input": to_list(x), "input_shape": list(x.shape),
        "gain": to_list(gain), "gain_shape": list(gain.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# LMHead (T86.1.7) - linear projection hidden -> vocab
# ---------------------------------------------------------------------------

def gen_lm_head():
    set_seed()
    batch, seq_len, hidden_dim, vocab_size = 1, 4, 8, 16
    x = torch.randn(batch, seq_len, hidden_dim)
    # Zerfoo Linear stores weights as [in, out], LMHead uses Linear internally
    w = torch.randn(hidden_dim, vocab_size)

    # LMHead: reshape [B, S, H] -> [B*S, H], matmul, reshape back
    x_flat = x.view(batch * seq_len, hidden_dim)
    y_flat = x_flat @ w
    y = y_flat.view(batch, seq_len, vocab_size)

    save_case("core_lm_head", {
        "layer": "lm_head",
        "hidden_dim": hidden_dim, "vocab_size": vocab_size,
        "input": to_list(x), "input_shape": list(x.shape),
        "weight": to_list(w), "weight_shape": list(w.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# PatchEmbed (T86.1.12)
# ---------------------------------------------------------------------------

def gen_patch_embed():
    set_seed()
    batch, seq_len = 2, 16
    patch_size, embed_dim = 4, 8
    x = torch.randn(batch, seq_len)
    # Projection: [patch_size, embed_dim]
    proj = torch.randn(patch_size, embed_dim)

    num_patches = seq_len // patch_size
    # Reshape [batch, seq_len] -> [batch*num_patches, patch_size]
    x_reshaped = x.view(batch * num_patches, patch_size)
    # Project: [batch*num_patches, patch_size] @ [patch_size, embed_dim]
    y_flat = x_reshaped @ proj
    y = y_flat.view(batch, num_patches, embed_dim)

    save_case("timeseries_patch_embed", {
        "layer": "patch_embed",
        "patch_size": patch_size, "embed_dim": embed_dim,
        "input": to_list(x), "input_shape": list(x.shape),
        "proj": to_list(proj), "proj_shape": list(proj.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# TSMixerBlock (T86.1.14) - channel-independent mode (no feature mixing)
# ---------------------------------------------------------------------------

def gen_tsmixer_block():
    set_seed()
    batch, num_patches, d_model = 1, 4, 8

    x = torch.randn(batch, num_patches, d_model)

    # Time-mixing MLP weights (Linear: [in, out] = [num_patches, num_patches])
    time_mlp1_w = torch.randn(num_patches, num_patches) * 0.1
    time_mlp2_w = torch.randn(num_patches, num_patches) * 0.1
    # LayerNorm params
    time_ln_gamma = torch.ones(d_model)
    time_ln_beta = torch.zeros(d_model)

    # Step 1: LayerNorm
    ln = nn.LayerNorm(d_model, eps=1e-5)
    ln.weight.data.copy_(time_ln_gamma)
    ln.bias.data.copy_(time_ln_beta)
    normed = ln(x)

    # Step 2: Transpose [B, P, D] -> [B, D, P]
    transposed = normed.transpose(1, 2)

    # Step 3: MLP(GELU): timeMLP2(gelu(timeMLP1(x)))
    h = transposed @ time_mlp1_w
    h = F.gelu(h, approximate="tanh")
    h = h @ time_mlp2_w

    # Step 4: Transpose back [B, D, P] -> [B, P, D]
    h = h.transpose(1, 2)

    # Step 5: Residual add
    y = h + x

    save_case("timeseries_tsmixer_block", {
        "layer": "tsmixer_block",
        "num_patches": num_patches, "d_model": d_model,
        "channel_mixing": False,
        "input": to_list(x), "input_shape": list(x.shape),
        "time_mlp1_w": to_list(time_mlp1_w), "time_mlp1_w_shape": list(time_mlp1_w.shape),
        "time_mlp2_w": to_list(time_mlp2_w), "time_mlp2_w_shape": list(time_mlp2_w.shape),
        "time_ln_gamma": to_list(time_ln_gamma), "time_ln_gamma_shape": list(time_ln_gamma.shape),
        "time_ln_beta": to_list(time_ln_beta), "time_ln_beta_shape": list(time_ln_beta.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# SSMLayer (T86.1.17)
# ---------------------------------------------------------------------------

def gen_ssm_layer():
    set_seed()
    batch, seq_len = 1, 8
    d_state, d_input, d_output = 4, 3, 2

    u = torch.randn(batch, seq_len, d_input)

    # Parameters matching Zerfoo SSMLayer
    A_log = torch.randn(d_state) * 0.5  # [d_state]
    B = torch.randn(d_state, d_input) * 0.1  # [d_state, d_input]
    C = torch.randn(d_output, d_state) * 0.1  # [d_output, d_state]
    D_mat = torch.randn(d_output, d_input) * 0.01  # [d_output, d_input]
    log_dt = torch.tensor([math.log(0.01)])  # [1]

    # Discretize
    dt = torch.exp(log_dt)  # scalar
    A_diag = -torch.exp(A_log)  # [d_state], negative eigenvalues
    A_bar = torch.exp(A_diag * dt.item())  # [d_state]

    # B_bar = (A_bar - 1) / A_diag * B  (ZOH discretization)
    # Safe: for small A_diag, B_bar -> dt * B
    scale = (A_bar - 1) / A_diag
    # For near-zero A_diag, use dt
    scale = torch.where(torch.abs(A_diag) < 1e-12, dt.expand_as(A_diag), scale)
    B_bar = scale.unsqueeze(1) * B  # [d_state, d_input]

    # Sequential scan
    outputs = []
    state = torch.zeros(batch, d_state, 1)  # [batch, d_state, 1]
    A_bar_col = A_bar.unsqueeze(1)  # [d_state, 1]

    for t in range(seq_len):
        u_t = u[:, t, :].unsqueeze(-1)  # [batch, d_input, 1]
        Bu = B_bar @ u_t  # [d_state, d_input] @ [batch, d_input, 1] = [batch, d_state, 1]
        state = A_bar_col * state + Bu  # [batch, d_state, 1]
        Cx = C @ state  # [d_output, d_state] @ [batch, d_state, 1] = [batch, d_output, 1]
        Du = D_mat @ u_t  # [d_output, d_input] @ [batch, d_input, 1] = [batch, d_output, 1]
        y_t = Cx + Du  # [batch, d_output, 1]
        outputs.append(y_t.squeeze(-1))  # [batch, d_output]

    y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_output]

    save_case("timeseries_ssm_layer", {
        "layer": "ssm_layer",
        "d_state": d_state, "d_input": d_input, "d_output": d_output,
        "input": to_list(u), "input_shape": list(u.shape),
        "A_log": to_list(A_log), "A_log_shape": list(A_log.shape),
        "B": to_list(B), "B_shape": list(B.shape),
        "C": to_list(C), "C_shape": list(C.shape),
        "D": to_list(D_mat), "D_shape": list(D_mat.shape),
        "log_dt": to_list(log_dt), "log_dt_shape": list(log_dt.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# Core arithmetic ops (T86.1.21)
# ---------------------------------------------------------------------------

def gen_arithmetic_ops():
    set_seed()
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)

    save_case("op_add", {
        "layer": "add",
        "input_a": to_list(a), "input_b": to_list(b),
        "input_shape": list(a.shape),
        "expected_output": to_list(a + b),
        "tolerance": 1e-6,
    })
    save_case("op_sub", {
        "layer": "sub",
        "input_a": to_list(a), "input_b": to_list(b),
        "input_shape": list(a.shape),
        "expected_output": to_list(a - b),
        "tolerance": 1e-6,
    })
    save_case("op_mul", {
        "layer": "mul",
        "input_a": to_list(a), "input_b": to_list(b),
        "input_shape": list(a.shape),
        "expected_output": to_list(a * b),
        "tolerance": 1e-6,
    })
    save_case("op_div", {
        "layer": "div",
        "input_a": to_list(a), "input_b": to_list(b),
        "input_shape": list(a.shape),
        "expected_output": to_list(a / b),
        "tolerance": 1e-5,
    })

    # Pow: use abs(a) to avoid NaN for fractional exponents
    a_pos = torch.abs(a) + 0.1
    exp = torch.tensor([[2.0]])  # broadcast scalar exponent
    save_case("op_pow", {
        "layer": "pow",
        "input_a": to_list(a_pos), "input_b": to_list(exp.expand_as(a_pos)),
        "input_shape": list(a_pos.shape),
        "expected_output": to_list(torch.pow(a_pos, exp.expand_as(a_pos))),
        "tolerance": 1e-5,
    })

    save_case("op_sqrt", {
        "layer": "sqrt",
        "input": to_list(a_pos),
        "input_shape": list(a_pos.shape),
        "expected_output": to_list(torch.sqrt(a_pos)),
        "tolerance": 1e-6,
    })

    save_case("op_sin", {
        "layer": "sin",
        "input": to_list(a),
        "input_shape": list(a.shape),
        "expected_output": to_list(torch.sin(a)),
        "tolerance": 1e-6,
    })

    save_case("op_cos", {
        "layer": "cos",
        "input": to_list(a),
        "input_shape": list(a.shape),
        "expected_output": to_list(torch.cos(a)),
        "tolerance": 1e-6,
    })


# ---------------------------------------------------------------------------
# Core shape ops (T86.1.22)
# ---------------------------------------------------------------------------

def gen_shape_ops():
    set_seed()
    x = torch.randn(2, 3, 4)

    # Reshape
    y_reshape = x.reshape(6, 4)
    save_case("op_reshape", {
        "layer": "reshape",
        "input": to_list(x), "input_shape": list(x.shape),
        "target_shape": list(y_reshape.shape),
        "expected_output": to_list(y_reshape),
        "tolerance": 1e-7,
    })

    # Concat along axis 1
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 4)
    y_concat = torch.cat([a, b], dim=1)
    save_case("op_concat", {
        "layer": "concat",
        "input_a": to_list(a), "input_a_shape": list(a.shape),
        "input_b": to_list(b), "input_b_shape": list(b.shape),
        "axis": 1,
        "expected_output": to_list(y_concat), "output_shape": list(y_concat.shape),
        "tolerance": 1e-7,
    })


# ---------------------------------------------------------------------------
# Timeseries: DLinear
# ---------------------------------------------------------------------------

def gen_dlinear():
    """Generate golden data for DLinear forward pass.

    DLinear decomposes input into trend (moving average) and seasonal (residual),
    then applies separate linear projections per channel.
    """
    set_seed()
    input_len = 8
    output_len = 4
    channels = 2
    kernel_size = 3

    # Input: [channels][input_len]
    x = torch.randn(channels, input_len).float()

    # Weights for trend and seasonal linear projections per channel.
    # Weight shape per channel: [output_len, input_len], bias: [output_len]
    trend_w = torch.randn(channels, output_len, input_len).float() * 0.1
    trend_b = torch.zeros(channels, output_len).float()
    seasonal_w = torch.randn(channels, output_len, input_len).float() * 0.1
    seasonal_b = torch.zeros(channels, output_len).float()

    # Moving average decomposition (edge-padded, matching Zerfoo's implementation).
    half = kernel_size // 2
    trend = torch.zeros_like(x)
    for c in range(channels):
        for i in range(input_len):
            s = 0.0
            count = 0
            for j in range(i - half, i + half + 1):
                idx = max(0, min(j, input_len - 1))
                s += x[c, idx].item()
                count += 1
            trend[c, i] = s / count

    seasonal = x - trend

    # Forward: output[c] = trend_w[c] @ trend[c] + trend_b[c] + seasonal_w[c] @ seasonal[c] + seasonal_b[c]
    output = torch.zeros(channels, output_len)
    for c in range(channels):
        trend_out = trend_w[c] @ trend[c] + trend_b[c]
        seasonal_out = seasonal_w[c] @ seasonal[c] + seasonal_b[c]
        output[c] = trend_out + seasonal_out

    save_case("timeseries_dlinear", {
        "layer": "dlinear",
        "input_len": input_len,
        "output_len": output_len,
        "channels": channels,
        "kernel_size": kernel_size,
        "input": to_list(x), "input_shape": list(x.shape),
        "trend_w": to_list(trend_w), "trend_w_shape": list(trend_w.shape),
        "trend_b": to_list(trend_b), "trend_b_shape": list(trend_b.shape),
        "seasonal_w": to_list(seasonal_w), "seasonal_w_shape": list(seasonal_w.shape),
        "seasonal_b": to_list(seasonal_b), "seasonal_b_shape": list(seasonal_b.shape),
        "expected_trend": to_list(trend), "trend_shape": list(trend.shape),
        "expected_seasonal": to_list(seasonal), "seasonal_shape": list(seasonal.shape),
        "expected_output": to_list(output), "output_shape": list(output.shape),
        "tolerance": 1e-5,
    })


# ---------------------------------------------------------------------------
# PatchTST (Patch Time-Series Transformer) golden-file forward parity
# ---------------------------------------------------------------------------

def gen_patchtst():
    """Generate golden data for PatchTST forward pass.

    Tiny PatchTST: InputLength=16, PatchLength=4, Stride=4, DModel=8,
    NHeads=2, NLayers=1, OutputDim=4, batch=2, channels=1 (univariate).

    Architecture:
      1. Patch embedding: linear [patchLen -> dModel] (x @ W + b)
      2. Positional embedding: additive learned [numPatches, dModel]
      3. Pre-norm transformer encoder (LayerNorm -> MHA -> residual ->
         LayerNorm -> FFN(GELU tanh approx) -> residual)
      4. Head: flatten [numPatches*dModel] -> Linear -> [outputDim]

    Weight order in flat_params (matches Go's flatParams()):
      patchEmbW, patchEmbB, posEmb,
      per-layer: qW, qB, kW, kB, vW, vB, oW, oB,
                 ffn1W, ffn1B, ffn2W, ffn2B,
                 norm1, bias1, norm2, bias2,
      headW, headB
    """
    set_seed()

    batch = 2
    input_len = 16
    patch_len = 4
    stride = 4
    d_model = 8
    n_heads = 2
    n_layers = 1
    output_dim = 4
    num_patches = (input_len - patch_len) // stride + 1  # 4
    head_dim = d_model // n_heads  # 4
    ffn_dim = d_model * 4  # 32

    # Input: [batch, input_len] random float32.
    x = torch.randn(batch, input_len)

    # --- Build all weights with fixed seed ---
    # Patch embedding: W [patch_len, d_model], b [d_model]
    patch_emb_w = torch.randn(patch_len, d_model) * 0.1
    patch_emb_b = torch.randn(d_model) * 0.1

    # Positional embedding: [num_patches, d_model]
    pos_emb = torch.randn(num_patches, d_model) * 0.02

    # Encoder layer weights (1 layer)
    layers_weights = []
    for _ in range(n_layers):
        lw = {}
        # Q/K/V/O projections: W [d_model, d_model], b [d_model]
        lw["q_w"] = torch.randn(d_model, d_model) * 0.1
        lw["q_b"] = torch.randn(d_model) * 0.1
        lw["k_w"] = torch.randn(d_model, d_model) * 0.1
        lw["k_b"] = torch.randn(d_model) * 0.1
        lw["v_w"] = torch.randn(d_model, d_model) * 0.1
        lw["v_b"] = torch.randn(d_model) * 0.1
        lw["o_w"] = torch.randn(d_model, d_model) * 0.1
        lw["o_b"] = torch.randn(d_model) * 0.1
        # FFN: ffn1 [d_model, ffn_dim], ffn2 [ffn_dim, d_model]
        lw["ffn1_w"] = torch.randn(d_model, ffn_dim) * 0.1
        lw["ffn1_b"] = torch.randn(ffn_dim) * 0.1
        lw["ffn2_w"] = torch.randn(ffn_dim, d_model) * 0.1
        lw["ffn2_b"] = torch.randn(d_model) * 0.1
        # LayerNorm: scale (gamma), bias (beta) [d_model]
        lw["norm1"] = torch.ones(d_model)
        lw["bias1"] = torch.zeros(d_model)
        lw["norm2"] = torch.ones(d_model)
        lw["bias2"] = torch.zeros(d_model)
        layers_weights.append(lw)

    # Head: W [num_patches*d_model, output_dim], b [output_dim]
    head_w = torch.randn(num_patches * d_model, output_dim) * 0.1
    head_b = torch.randn(output_dim) * 0.1

    # --- Forward pass ---
    # Process each batch element (univariate, channels=1).
    outputs = []
    for bi in range(batch):
        row = x[bi]  # [input_len]

        # Extract patches: [num_patches, patch_len]
        patches = []
        for p in range(num_patches):
            start = p * stride
            patches.append(row[start:start + patch_len])
        patches = torch.stack(patches)  # [num_patches, patch_len]

        # Patch embedding: patches @ W + b -> [num_patches, d_model]
        embedded = patches @ patch_emb_w + patch_emb_b

        # Add positional embedding.
        embedded = embedded + pos_emb

        # Transformer encoder.
        h = embedded  # [num_patches, d_model]
        for li in range(n_layers):
            lw = layers_weights[li]

            # Pre-norm 1: LayerNorm.
            normed = F.layer_norm(h, [d_model], weight=lw["norm1"], bias=lw["bias1"])

            # MHA: Q, K, V projections.
            q = normed @ lw["q_w"] + lw["q_b"]  # [num_patches, d_model]
            k = normed @ lw["k_w"] + lw["k_b"]
            v = normed @ lw["v_w"] + lw["v_b"]

            # Split into heads: [num_patches, n_heads, head_dim] -> [n_heads, num_patches, head_dim]
            q = q.view(num_patches, n_heads, head_dim).transpose(0, 1)
            k = k.view(num_patches, n_heads, head_dim).transpose(0, 1)
            v = v.view(num_patches, n_heads, head_dim).transpose(0, 1)

            # Scaled dot-product attention (no causal mask).
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [n_heads, num_patches, num_patches]
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v)  # [n_heads, num_patches, head_dim]

            # Concat heads: [num_patches, d_model]
            attn_out = attn_out.transpose(0, 1).contiguous().view(num_patches, d_model)

            # Output projection.
            attn_out = attn_out @ lw["o_w"] + lw["o_b"]

            # Residual.
            h = h + attn_out

            # Pre-norm 2: LayerNorm.
            normed = F.layer_norm(h, [d_model], weight=lw["norm2"], bias=lw["bias2"])

            # FFN with GELU tanh approximation.
            ffn_h = normed @ lw["ffn1_w"] + lw["ffn1_b"]  # [num_patches, ffn_dim]
            ffn_h = F.gelu(ffn_h, approximate="tanh")
            ffn_out = ffn_h @ lw["ffn2_w"] + lw["ffn2_b"]  # [num_patches, d_model]

            # Residual.
            h = h + ffn_out

        # Flatten: [num_patches * d_model]
        flat = h.reshape(-1)

        # Head: linear [num_patches*d_model -> output_dim]
        out = flat @ head_w + head_b  # [output_dim]
        outputs.append(out)

    output = torch.stack(outputs)  # [batch, output_dim]

    # --- Build flat_params in Go's flatParams() order ---
    flat_params = []
    flat_params.extend(to_list(patch_emb_w))
    flat_params.extend(to_list(patch_emb_b))
    flat_params.extend(to_list(pos_emb))
    for li in range(n_layers):
        lw = layers_weights[li]
        flat_params.extend(to_list(lw["q_w"]))
        flat_params.extend(to_list(lw["q_b"]))
        flat_params.extend(to_list(lw["k_w"]))
        flat_params.extend(to_list(lw["k_b"]))
        flat_params.extend(to_list(lw["v_w"]))
        flat_params.extend(to_list(lw["v_b"]))
        flat_params.extend(to_list(lw["o_w"]))
        flat_params.extend(to_list(lw["o_b"]))
        flat_params.extend(to_list(lw["ffn1_w"]))
        flat_params.extend(to_list(lw["ffn1_b"]))
        flat_params.extend(to_list(lw["ffn2_w"]))
        flat_params.extend(to_list(lw["ffn2_b"]))
        flat_params.extend(to_list(lw["norm1"]))
        flat_params.extend(to_list(lw["bias1"]))
        flat_params.extend(to_list(lw["norm2"]))
        flat_params.extend(to_list(lw["bias2"]))
    flat_params.extend(to_list(head_w))
    flat_params.extend(to_list(head_b))

    save_case("model_patchtst", {
        "model": "patchtst",
        "batch": batch,
        "input_len": input_len,
        "patch_len": patch_len,
        "stride": stride,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "output_dim": output_dim,
        "num_patches": num_patches,
        "input": to_list(x), "input_shape": list(x.shape),
        "flat_params": flat_params,
        "param_count": len(flat_params),
        "expected_output": to_list(output), "output_shape": list(output.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# FreTS (Frequency-enhanced Time Series) golden-file forward parity
# ---------------------------------------------------------------------------

def gen_frets():
    """Generate golden data for FreTS forward pass.

    FreTS architecture:
      1. Manual DFT per channel -> select top-K frequency components
      2. Channel mixing MLP (2-layer with ReLU, residual) on real/imag separately
      3. Temporal mixing MLP (2-layer with ReLU, residual) on real/imag separately
      4. Inverse DFT -> linear projection per channel

    Uses manual DFT/IDFT to match Go implementation exactly (not np.fft).
    """
    np.random.seed(SEED)

    channels = 3
    input_len = 12
    output_len = 4
    top_k = 4
    hidden_size = 8
    n_freqs = input_len // 2 + 1  # 7

    # Generate deterministic input: [channels][input_len]
    x = np.random.randn(channels, input_len)

    # Generate deterministic weights (small scale for numerical stability).
    chan_w1 = np.random.randn(channels * hidden_size) * 0.1
    chan_b1 = np.random.randn(hidden_size) * 0.1
    chan_w2 = np.random.randn(hidden_size * channels) * 0.1
    chan_b2 = np.random.randn(channels) * 0.1

    temp_w1 = np.random.randn(top_k * hidden_size) * 0.1
    temp_b1 = np.random.randn(hidden_size) * 0.1
    temp_w2 = np.random.randn(hidden_size * top_k) * 0.1
    temp_b2 = np.random.randn(top_k) * 0.1

    out_w = np.random.randn(channels * output_len * input_len) * 0.1
    out_b = np.random.randn(channels * output_len) * 0.1

    # --- Manual DFT (matches Go's dft function exactly) ---
    def manual_dft(signal):
        """DFT of real signal -> complex coefficients [n/2+1]."""
        n = len(signal)
        nf = n // 2 + 1
        out = np.zeros(nf, dtype=np.complex128)
        for k in range(nf):
            s = 0.0 + 0.0j
            for t in range(n):
                angle = -2.0 * math.pi * k * t / n
                s += signal[t] * complex(math.cos(angle), math.sin(angle))
            out[k] = s
        return out

    # --- Manual IDFT (matches Go's idft function exactly) ---
    def manual_idft(coeffs, n):
        """IDFT from positive-frequency coefficients back to real signal of length n."""
        out = np.zeros(n)
        nf = len(coeffs)
        for t in range(n):
            s = 0.0 + 0.0j
            for k in range(nf):
                angle = 2.0 * math.pi * k * t / n
                c = complex(math.cos(angle), math.sin(angle))
                if k == 0 or (n % 2 == 0 and k == n // 2):
                    s += coeffs[k] * c
                else:
                    s += coeffs[k] * c + np.conj(coeffs[k]) * np.conj(c)
            out[t] = s.real / n
        return out

    # --- Top-K indices by magnitude (matches Go's topKIndices) ---
    def top_k_indices(coeffs, k):
        mags = np.abs(coeffs)
        # argsort descending, take first k, then sort ascending
        idx = np.argsort(-mags)[:k]
        return np.sort(idx)

    # --- mat_vec: vec[1,rows] @ mat[rows,cols] -> [cols] ---
    def mat_vec(vec, mat_flat, rows, cols):
        mat = mat_flat.reshape(rows, cols)
        return vec @ mat

    # Step 1: DFT per channel, select top-K frequencies.
    all_coeffs = []
    top_indices = []
    freq_real = np.zeros((channels, top_k))
    freq_imag = np.zeros((channels, top_k))

    for c in range(channels):
        coeffs = manual_dft(x[c])
        all_coeffs.append(coeffs)
        idx = top_k_indices(coeffs, top_k)
        top_indices.append(idx)
        for i, fi in enumerate(idx):
            freq_real[c, i] = coeffs[fi].real
            freq_imag[c, i] = coeffs[fi].imag

    # Step 2: Channel mixing MLP (per frequency bin k).
    for k in range(top_k):
        real_in = freq_real[:, k].copy()  # [channels]
        imag_in = freq_imag[:, k].copy()  # [channels]

        # Layer 1: input @ chanW1[channels, hidden] + chanB1, then ReLU
        raw_real = mat_vec(real_in, chan_w1, channels, hidden_size)
        raw_imag = mat_vec(imag_in, chan_w1, channels, hidden_size)
        h_real = np.maximum(raw_real + chan_b1, 0.0)
        h_imag = np.maximum(raw_imag + chan_b1, 0.0)

        # Layer 2: h @ chanW2[hidden, channels] + chanB2
        out_real = mat_vec(h_real, chan_w2, hidden_size, channels)
        out_imag = mat_vec(h_imag, chan_w2, hidden_size, channels)

        # Residual connection
        freq_real[:, k] = real_in + out_real + chan_b2
        freq_imag[:, k] = imag_in + out_imag + chan_b2

    # Step 3: Temporal mixing MLP (per channel).
    for c in range(channels):
        temp_real_in = freq_real[c].copy()  # [top_k]
        temp_imag_in = freq_imag[c].copy()  # [top_k]

        # Layer 1: input @ tempW1[topK, hidden] + tempB1, then ReLU
        raw_real = mat_vec(temp_real_in, temp_w1, top_k, hidden_size)
        raw_imag = mat_vec(temp_imag_in, temp_w1, top_k, hidden_size)
        h_real = np.maximum(raw_real + temp_b1, 0.0)
        h_imag = np.maximum(raw_imag + temp_b1, 0.0)

        # Layer 2: h @ tempW2[hidden, topK] + tempB2
        out_real = mat_vec(h_real, temp_w2, hidden_size, top_k)
        out_imag = mat_vec(h_imag, temp_w2, hidden_size, top_k)

        # Residual connection
        freq_real[c] += out_real + temp_b2
        freq_imag[c] += out_imag + temp_b2

    # Step 4: Inverse DFT per channel.
    reconstructed = np.zeros((channels, input_len))
    for c in range(channels):
        mixed = np.zeros(len(all_coeffs[c]), dtype=np.complex128)
        for i, fi in enumerate(top_indices[c]):
            mixed[fi] = complex(freq_real[c, i], freq_imag[c, i])
        reconstructed[c] = manual_idft(mixed, input_len)

    # Step 5: Linear projection per channel.
    # outW is [channels * outputLen * inputLen], stored as:
    #   for channel c: outW[c*outputLen*inputLen + o*inputLen + i]
    # Go transposes: outWt[i*outputLen+o] = outW[wOff+o*inputLen+i]
    # Then: reconstructed @ outWt[inputLen, outputLen] + outB
    output = np.zeros((channels, output_len))
    for c in range(channels):
        w_off = c * output_len * input_len
        b_off = c * output_len
        # Build transposed weight matrix [inputLen, outputLen]
        out_wt = np.zeros((input_len, output_len))
        for o in range(output_len):
            for i in range(input_len):
                out_wt[i, o] = out_w[w_off + o * input_len + i]
        proj = reconstructed[c] @ out_wt
        output[c] = proj + out_b[b_off:b_off + output_len]

    # Flatten output to [channels * outputLen] (same order as Go PredictWindowed).
    expected_output = output.flatten().tolist()

    save_case("model_frets", {
        "layer": "frets",
        "channels": channels,
        "input_len": input_len,
        "output_len": output_len,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "input": x.flatten().tolist(),
        "chan_w1": chan_w1.tolist(),
        "chan_b1": chan_b1.tolist(),
        "chan_w2": chan_w2.tolist(),
        "chan_b2": chan_b2.tolist(),
        "temp_w1": temp_w1.tolist(),
        "temp_b1": temp_b1.tolist(),
        "temp_w2": temp_w2.tolist(),
        "temp_b2": temp_b2.tolist(),
        "out_w": out_w.tolist(),
        "out_b": out_b.tolist(),
        "expected_output": expected_output,
        "tolerance": 1e-9,
    })


# ---------------------------------------------------------------------------
# GQA (Grouped Query Attention) -- no RoPE, no KV cache
# ---------------------------------------------------------------------------

def gen_gqa():
    """GQA with 4 query heads, 2 KV heads, d_model=16, no RoPE.

    Zerfoo GQA Dense layers store weights as [in, out] and compute x @ W + b.
    The golden file provides weights in that layout.
    """
    set_seed()
    batch, seq_len = 1, 4
    d_model = 16
    n_q_heads, n_kv_heads = 4, 2
    head_dim = d_model // n_q_heads  # 4

    x = torch.randn(batch, seq_len, d_model)

    # Q/K/V/O Dense weights [in, out] and biases [out] (Zerfoo layout)
    wq_w = torch.randn(d_model, d_model) * 0.02          # [16, 16]
    wq_b = torch.randn(d_model) * 0.02                    # [16]
    wk_w = torch.randn(d_model, n_kv_heads * head_dim) * 0.02  # [16, 8]
    wk_b = torch.randn(n_kv_heads * head_dim) * 0.02      # [8]
    wv_w = torch.randn(d_model, n_kv_heads * head_dim) * 0.02  # [16, 8]
    wv_b = torch.randn(n_kv_heads * head_dim) * 0.02      # [8]
    wo_w = torch.randn(d_model, d_model) * 0.02           # [16, 16]
    wo_b = torch.randn(d_model) * 0.02                    # [16]

    # Project: Q = x @ wq_w + wq_b, etc.
    q = x @ wq_w + wq_b                                    # [1, 4, 16]
    k = x @ wk_w + wk_b                                    # [1, 4, 8]
    v = x @ wv_w + wv_b                                    # [1, 4, 8]

    # Reshape into heads
    q = q.view(batch, seq_len, n_q_heads, head_dim).transpose(1, 2)   # [1, 4, 4, 4]
    k = k.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)  # [1, 2, 4, 4]
    v = v.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)  # [1, 2, 4, 4]

    # Repeat KV heads to match query heads (GQA expansion)
    repeat_factor = n_q_heads // n_kv_heads  # 2
    k = k.repeat_interleave(repeat_factor, dim=1)  # [1, 4, 4, 4]
    v = v.repeat_interleave(repeat_factor, dim=1)  # [1, 4, 4, 4]

    # Scaled dot-product attention (causal)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn, v)  # [1, 4, 4, 4]

    # Concat heads and output projection
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    y = attn_out @ wo_w + wo_b

    save_case("attention_gqa", {
        "layer": "grouped_query_attention",
        "d_model": d_model, "n_q_heads": n_q_heads, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "input": to_list(x), "input_shape": list(x.shape),
        "wq_w": to_list(wq_w), "wq_w_shape": list(wq_w.shape),
        "wq_b": to_list(wq_b), "wq_b_shape": list(wq_b.shape),
        "wk_w": to_list(wk_w), "wk_w_shape": list(wk_w.shape),
        "wk_b": to_list(wk_b), "wk_b_shape": list(wk_b.shape),
        "wv_w": to_list(wv_w), "wv_w_shape": list(wv_w.shape),
        "wv_b": to_list(wv_b), "wv_b_shape": list(wv_b.shape),
        "wo_w": to_list(wo_w), "wo_w_shape": list(wo_w.shape),
        "wo_b": to_list(wo_b), "wo_b_shape": list(wo_b.shape),
        "expected_output": to_list(y), "output_shape": list(y.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# MoE (Mixture of Experts)
# ---------------------------------------------------------------------------

def gen_moe():
    """MoE with 4 experts (simple linear), top-2 routing, softmax gating."""
    set_seed()
    seq_len = 4
    model_dim = 8
    n_experts = 4
    top_k = 2

    x = torch.randn(seq_len, model_dim)
    gate_w = torch.randn(n_experts, model_dim) * 0.1  # [4, 8]

    # Expert weights: each is a simple linear [model_dim, model_dim]
    expert_weights = [torch.randn(model_dim, model_dim) * 0.1 for _ in range(n_experts)]

    # Routing: logits = x @ gate_w^T, probs = softmax(logits)
    logits = x @ gate_w.T  # [4, 4]
    probs = F.softmax(logits, dim=-1)

    # Top-k selection per token
    topk_vals, topk_idxs = torch.topk(probs, top_k, dim=-1)
    # Normalize top-k weights
    topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

    # Dispatch: for each token, compute weighted sum of expert outputs
    output = torch.zeros(seq_len, model_dim)
    for t in range(seq_len):
        for k in range(top_k):
            expert_idx = topk_idxs[t, k].item()
            weight = topk_weights[t, k].item()
            expert_out = x[t:t+1] @ expert_weights[expert_idx]  # [1, model_dim]
            output[t] += weight * expert_out.squeeze(0)

    save_case("core_moe", {
        "layer": "mixture_of_experts",
        "model_dim": model_dim, "n_experts": n_experts, "top_k": top_k,
        "input": to_list(x), "input_shape": list(x.shape),
        "gate_weight": to_list(gate_w), "gate_weight_shape": list(gate_w.shape),
        "expert_weights": [to_list(w) for w in expert_weights],
        "expert_weight_shape": list(expert_weights[0].shape),
        "expected_output": to_list(output), "output_shape": list(output.shape),
        "tolerance": 1e-4,
    })


# ---------------------------------------------------------------------------
# CfC (Closed-form Continuous-time) golden-file forward parity
# ---------------------------------------------------------------------------

def gen_cfc():
    """Generate golden data for CfC forward pass.

    CfC uses liquid time-constant neurons with closed-form ODE solutions:
      tau = sigmoid(x @ Wtau_x + h @ Wtau_h + Btau)
      f = exp(-1.0 / max(tau, 1e-6))
      preact = tanh(x @ Wx + h @ Wh + Bh)
      h_new = f * h + (1 - f) * preact
    Output projection: h @ outW + outB
    """
    np.random.seed(SEED)

    input_size = 3
    hidden_size = 8
    output_size = 2
    num_layers = 1
    output_len = 4
    seq_len = 5

    # Input: [channels][seq_len] — channels == input_size for CfC.
    x_channels = np.random.randn(input_size, seq_len).astype(np.float64)

    # Layer weights (NumPy, deterministic).
    in_size = input_size
    wh = np.random.randn(hidden_size, hidden_size).astype(np.float64) * 0.1
    wx = np.random.randn(in_size, hidden_size).astype(np.float64) * 0.1
    bh = np.zeros(hidden_size, dtype=np.float64)
    wtau = np.random.randn(in_size + hidden_size, hidden_size).astype(np.float64) * 0.1
    btau = np.zeros(hidden_size, dtype=np.float64)

    # Output projection.
    out_dim = output_size * output_len
    out_w = np.random.randn(hidden_size, out_dim).astype(np.float64) * 0.1
    out_b = np.zeros(out_dim, dtype=np.float64)

    # Transpose input: [channels][seq_len] -> [seq_len][channels] (matches Go transposeWindow).
    x_seq = x_channels.T  # [seq_len, input_size]

    # Run CfC recurrence.
    h = np.zeros(hidden_size, dtype=np.float64)
    for t in range(seq_len):
        xt = x_seq[t]  # [input_size]

        # tau = sigmoid(xt @ Wtau[:in_size] + h @ Wtau[in_size:] + Btau)
        tau_x = xt @ wtau[:in_size]        # [hidden_size]
        tau_h = h @ wtau[in_size:]          # [hidden_size]
        tau = 1.0 / (1.0 + np.exp(-(btau + tau_x + tau_h)))

        # preact = tanh(xt @ Wx + h @ Wh + Bh)
        pre_x = xt @ wx        # [hidden_size]
        pre_h = h @ wh          # [hidden_size]
        preact = np.tanh(bh + pre_x + pre_h)

        # f = exp(-1.0 / max(tau, 1e-6))
        tau_clamped = np.maximum(tau, 1e-6)
        f = np.exp(-1.0 / tau_clamped)

        # h_new = f * h + (1 - f) * preact
        h = f * h + (1.0 - f) * preact

    # Output projection: h @ outW + outB
    output = h @ out_w + out_b  # [out_dim]

    # Save in cfcWeights JSON format compatible with Go's loadWeights.
    save_case("model_cfc", {
        "layer": "cfc",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "num_layers": num_layers,
        "output_len": output_len,
        "seq_len": seq_len,
        "input": x_channels.flatten().tolist(),
        "input_shape": list(x_channels.shape),
        # Layer weights (2D as nested lists for JSON).
        "wh": wh.tolist(),
        "wx": wx.tolist(),
        "bh": bh.tolist(),
        "wtau": wtau.tolist(),
        "btau": btau.tolist(),
        # Output projection.
        "out_w": out_w.tolist(),
        "out_b": out_b.tolist(),
        "expected_output": output.tolist(),
        "output_shape": [out_dim],
        "tolerance": 1e-7,
    })


# ---------------------------------------------------------------------------
# TimeMixer golden-file forward parity
# ---------------------------------------------------------------------------

def gen_timemixer():
    """Generate golden data for TimeMixer forward pass.

    TimeMixer (ICLR 2024) decomposes input into trend/seasonal at multiple
    scales using learnable moving averages, mixes across scales with 2-layer
    MLPs, projects via per-scale linear heads, and combines with softmax-gated
    mixing weights.

    Config: InputLen=12, OutputLen=4, NumFeatures=3, NumScales=2,
            HiddenSize=8, NumLayers=1.
    """
    np.random.seed(42)

    input_len = 12
    output_len = 4
    num_features = 3
    num_scales = 2
    hidden_size = 8
    num_layers = 1

    # Input: [num_features][input_len]
    x = np.random.randn(num_features, input_len)

    # --- MA weights per scale (softmax-normalized) ---
    # Scale s has kernel size 2^(s+1).
    ma_weights = []
    for s in range(num_scales):
        ks = 1 << (s + 1)
        raw = np.random.randn(ks) * 0.1
        # softmax normalize
        e = np.exp(raw - np.max(raw))
        ma_weights.append((e / e.sum()).tolist())

    # --- MLP weights (per layer: seasonal then trend) ---
    # Each MLP: w1 [hidden_size, num_scales], b1 [hidden_size],
    #           w2 [num_scales, hidden_size], b2 [num_scales].
    seasonal_mlps = []
    trend_mlps = []
    for _ in range(num_layers):
        seasonal_mlps.append({
            "w1": np.random.randn(hidden_size, num_scales).tolist(),
            "b1": np.random.randn(hidden_size).tolist(),
            "w2": np.random.randn(num_scales, hidden_size).tolist(),
            "b2": np.random.randn(num_scales).tolist(),
        })
        trend_mlps.append({
            "w1": np.random.randn(hidden_size, num_scales).tolist(),
            "b1": np.random.randn(hidden_size).tolist(),
            "w2": np.random.randn(num_scales, hidden_size).tolist(),
            "b2": np.random.randn(num_scales).tolist(),
        })

    # --- Projection heads per scale: [input_len][output_len] ---
    trend_heads = []
    seasonal_heads = []
    for _ in range(num_scales):
        trend_heads.append(np.random.randn(input_len, output_len).tolist())
        seasonal_heads.append(np.random.randn(input_len, output_len).tolist())

    # --- Mix weights (pre-softmax, initialized to 0) ---
    mix_weights = [0.0] * num_scales

    # ---- Forward pass in NumPy (mirrors Go implementation) ----

    # 1. Decompose: weighted moving average per scale.
    def weighted_moving_avg(series, kernel):
        n = len(series)
        k = len(kernel)
        out = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(k):
                idx = max(0, i - j)
                s += kernel[j] * series[idx]
            out[i] = s
        return out

    scales_trend = []    # [num_scales][num_features][input_len]
    scales_seasonal = []
    for s in range(num_scales):
        kernel = np.array(ma_weights[s])
        st = []
        ss = []
        for f in range(num_features):
            tr = weighted_moving_avg(x[f], kernel)
            st.append(tr)
            ss.append(x[f] - tr)
        scales_trend.append(st)
        scales_seasonal.append(ss)

    # 2. Past decomposable mixing (bottom-up).
    for l in range(num_layers):
        s_mlp = seasonal_mlps[l]
        t_mlp = trend_mlps[l]

        def mlp_forward(inp, w1, b1, w2, b2):
            """Two-layer MLP with ReLU: out = W2 @ ReLU(W1 @ x + b1) + b2."""
            w1 = np.array(w1)
            b1 = np.array(b1)
            w2 = np.array(w2)
            b2 = np.array(b2)
            hidden = np.maximum(0, w1 @ inp + b1)  # ReLU
            return w2 @ hidden + b2

        new_seasonal = [[np.zeros(input_len) for _ in range(num_features)]
                        for _ in range(num_scales)]
        new_trend = [[np.zeros(input_len) for _ in range(num_features)]
                     for _ in range(num_scales)]

        # Mix seasonal across scales.
        for f in range(num_features):
            for t in range(input_len):
                scale_vec = np.array([scales_seasonal[s][f][t]
                                      for s in range(num_scales)])
                mixed = mlp_forward(scale_vec, s_mlp["w1"], s_mlp["b1"],
                                    s_mlp["w2"], s_mlp["b2"])
                for s in range(num_scales):
                    new_seasonal[s][f][t] = mixed[s]

        # Mix trend across scales.
        for f in range(num_features):
            for t in range(input_len):
                scale_vec = np.array([scales_trend[s][f][t]
                                      for s in range(num_scales)])
                mixed = mlp_forward(scale_vec, t_mlp["w1"], t_mlp["b1"],
                                    t_mlp["w2"], t_mlp["b2"])
                for s in range(num_scales):
                    new_trend[s][f][t] = mixed[s]

        # Bottom-up residual: coarse -> fine.
        for s in range(num_scales - 2, -1, -1):
            for f in range(num_features):
                new_seasonal[s][f] = new_seasonal[s][f] + new_seasonal[s + 1][f]
                new_trend[s][f] = new_trend[s][f] + new_trend[s + 1][f]

        scales_seasonal = new_seasonal
        scales_trend = new_trend

    # 3. Softmax mixing weights.
    mw = np.array(mix_weights)
    e = np.exp(mw - np.max(mw))
    sm_weights = e / e.sum()

    # 4. Project each scale and combine.
    forecast = np.zeros((num_features, output_len))
    for f in range(num_features):
        for s in range(num_scales):
            th = np.array(trend_heads[s])   # [input_len, output_len]
            sh = np.array(seasonal_heads[s])
            trend_proj = np.array(scales_trend[s][f]) @ th
            season_proj = np.array(scales_seasonal[s][f]) @ sh
            forecast[f] += sm_weights[s] * (trend_proj + season_proj)

    # --- Build flat_params in same order as Go's FlatParams ---
    # Order: ma_weights (per scale), then per layer: seasonal MLP (w1, b1, w2, b2),
    #        trend MLP (w1, b1, w2, b2).
    flat_params = []
    for s in range(num_scales):
        flat_params.extend(ma_weights[s])
    for l in range(num_layers):
        flat_params.extend(np.array(seasonal_mlps[l]["w1"]).flatten().tolist())
        flat_params.extend(seasonal_mlps[l]["b1"])
        flat_params.extend(np.array(seasonal_mlps[l]["w2"]).flatten().tolist())
        flat_params.extend(seasonal_mlps[l]["b2"])
        flat_params.extend(np.array(trend_mlps[l]["w1"]).flatten().tolist())
        flat_params.extend(trend_mlps[l]["b1"])
        flat_params.extend(np.array(trend_mlps[l]["w2"]).flatten().tolist())
        flat_params.extend(trend_mlps[l]["b2"])

    save_case("model_timemixer", {
        "layer": "timemixer",
        "input_len": input_len,
        "output_len": output_len,
        "num_features": num_features,
        "num_scales": num_scales,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input": x.flatten().tolist(),
        "input_shape": list(x.shape),
        "flat_params": flat_params,
        "trend_heads": [np.array(h).flatten().tolist() for h in trend_heads],
        "seasonal_heads": [np.array(h).flatten().tolist() for h in seasonal_heads],
        "mix_weights": mix_weights,
        "expected_output": forecast.flatten().tolist(),
        "output_shape": list(forecast.shape),
        "tolerance": 1e-7,
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
        ("EMA", gen_ema),
        ("SWA", gen_swa),
        # Composite
        ("TransformerBlock", gen_transformer_block),
        # New layers (E86)
        ("FastGelu", gen_fast_gelu),
        ("SimplifiedLayerNorm", gen_simplified_layer_norm),
        ("SkipSimplifiedLayerNorm", gen_skip_simplified_layer_norm),
        ("LMHead", gen_lm_head),
        ("PatchEmbed", gen_patch_embed),
        ("TSMixerBlock", gen_tsmixer_block),
        ("SSMLayer", gen_ssm_layer),
        ("ArithmeticOps", gen_arithmetic_ops),
        ("ShapeOps", gen_shape_ops),
        # Timeseries models (E86.4)
        ("DLinear", gen_dlinear),
        ("PatchTST", gen_patchtst),
        ("FreTS", gen_frets),
        # Complex layers (GQA, MoE)
        ("GQA", gen_gqa),
        ("MoE", gen_moe),
        # Timeseries models (E88)
        ("CfC", gen_cfc),
        ("TimeMixer", gen_timemixer),
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
