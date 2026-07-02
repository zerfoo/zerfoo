#!/usr/bin/env python3
"""Generate golden JSON files for N-BEATS and ITransformer parity tests.

N-BEATS: InputLength=12, OutputLength=6, 2 stacks (Trend + Seasonality),
         1 block per stack, HiddenDim=16, NHarmonics=4.

ITransformer: Channels=3, InputLen=8, OutputLen=4, DModel=8, DFF=16,
              NHeads=2, NLayers=1. Uses float64.
"""

import json
import math
import numpy as np

np.random.seed(42)


# ===========================================================================
# N-BEATS golden file generation
# ===========================================================================

def polynomial_basis(degree, length):
    """Polynomial basis [degree, length]. Row i = (t/T)^i."""
    T = max(length - 1, 1)
    basis = np.zeros((degree, length), dtype=np.float32)
    for i in range(degree):
        for t in range(length):
            basis[i, t] = (t / T) ** i
    return basis


def fourier_basis(n_harmonics, length):
    """Fourier basis [2*n_harmonics, length]."""
    T = float(length)
    basis = np.zeros((2 * n_harmonics, length), dtype=np.float32)
    for k in range(n_harmonics):
        freq = 2.0 * math.pi * (k + 1) / T
        for t in range(length):
            basis[k, t] = math.cos(freq * t)
            basis[n_harmonics + k, t] = math.sin(freq * t)
    return basis


def linear_forward(x, w, b):
    """y = x @ W^T + b. w is [out, in], x is [batch, in]."""
    return x @ w.T + b


def relu(x):
    return np.maximum(x, 0)


def gen_nbeats():
    input_len = 12
    output_len = 6
    hidden_dim = 16
    n_harmonics = 4
    batch = 1

    # Trend: theta_dim = 3 (polynomial degree 2 -> 3 coefficients)
    # Seasonality: theta_dim = 2 * n_harmonics = 8
    trend_theta_dim = 3
    season_theta_dim = 2 * n_harmonics

    # Generate random input
    x_input = np.random.randn(batch, input_len).astype(np.float32)

    # Generate weights for each block.
    # Each block has: 4 FC layers + thetaB + thetaF
    # FC layers: [input_len->hidden, hidden->hidden, hidden->hidden, hidden->hidden]
    # For trend block:
    #   thetaB: [hidden_dim, trend_theta_dim], thetaF: [hidden_dim, trend_theta_dim]
    # For seasonality block:
    #   thetaB: [hidden_dim, season_theta_dim], thetaF: [hidden_dim, season_theta_dim]

    def make_block_weights(fc_in_dim, theta_dim):
        """Generate FC + theta weights. Returns dict and flat params list."""
        params = []
        layers = []

        dims = [fc_in_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim]
        for i in range(4):
            w = np.random.randn(dims[i+1], dims[i]).astype(np.float32) * 0.1
            b = np.random.randn(dims[i+1]).astype(np.float32) * 0.01
            layers.append((w, b))
            params.extend(w.flatten().tolist())
            params.extend(b.flatten().tolist())

        # thetaB
        tb_w = np.random.randn(theta_dim, hidden_dim).astype(np.float32) * 0.1
        tb_b = np.random.randn(theta_dim).astype(np.float32) * 0.01
        params.extend(tb_w.flatten().tolist())
        params.extend(tb_b.flatten().tolist())

        # thetaF
        tf_w = np.random.randn(theta_dim, hidden_dim).astype(np.float32) * 0.1
        tf_b = np.random.randn(theta_dim).astype(np.float32) * 0.01
        params.extend(tf_w.flatten().tolist())
        params.extend(tf_b.flatten().tolist())

        return layers, tb_w, tb_b, tf_w, tf_b, params

    # Trend block
    trend_fc, trend_tb_w, trend_tb_b, trend_tf_w, trend_tf_b, trend_params = \
        make_block_weights(input_len, trend_theta_dim)
    # Seasonality block
    season_fc, season_tb_w, season_tb_b, season_tf_w, season_tf_b, season_params = \
        make_block_weights(input_len, season_theta_dim)

    all_params = trend_params + season_params

    # Compute forward pass
    # Basis matrices
    trend_backcast_basis = polynomial_basis(trend_theta_dim, input_len)
    trend_forecast_basis = polynomial_basis(trend_theta_dim, output_len)
    season_backcast_basis = fourier_basis(n_harmonics, input_len)
    season_forecast_basis = fourier_basis(n_harmonics, output_len)

    def block_forward(x, fc_layers, tb_w, tb_b, tf_w, tf_b, bc_basis, fc_basis):
        h = x
        for w, b in fc_layers:
            h = linear_forward(h, w, b)
            h = relu(h)
        theta_b = linear_forward(h, tb_w, tb_b)
        theta_f = linear_forward(h, tf_w, tf_b)
        backcast = theta_b @ bc_basis  # [batch, theta_dim] @ [theta_dim, input_len]
        forecast = theta_f @ fc_basis  # [batch, theta_dim] @ [theta_dim, output_len]
        return backcast, forecast

    residual = x_input.copy()
    total_forecast = np.zeros((batch, output_len), dtype=np.float32)

    # Trend block
    bc, fc = block_forward(residual, trend_fc, trend_tb_w, trend_tb_b,
                           trend_tf_w, trend_tf_b,
                           trend_backcast_basis, trend_forecast_basis)
    residual = residual - bc
    total_forecast = total_forecast + fc

    # Seasonality block
    bc, fc = block_forward(residual, season_fc, season_tb_w, season_tb_b,
                           season_tf_w, season_tf_b,
                           season_backcast_basis, season_forecast_basis)
    residual = residual - bc
    total_forecast = total_forecast + fc

    golden = {
        "model": "nbeats",
        "input_length": input_len,
        "output_length": output_len,
        "hidden_dim": hidden_dim,
        "n_harmonics": n_harmonics,
        "n_blocks_per_stack": 1,
        "batch": batch,
        "input": x_input.flatten().tolist(),
        "params": all_params,
        "expected_output": total_forecast.flatten().tolist(),
        "tolerance": 1e-5
    }

    with open("layers/model_nbeats.json", "w") as f:
        json.dump(golden, f, indent=2)
    print(f"N-BEATS: {len(all_params)} params, output shape {total_forecast.shape}")


# ===========================================================================
# ITransformer golden file generation
# ===========================================================================

def layer_norm_1d(x, scale, bias, eps=1e-5):
    """Layer norm over last dimension. x: [n]."""
    mean = np.mean(x)
    var = np.mean((x - mean) ** 2)
    std = np.sqrt(var + eps)
    return scale * (x - mean) / std + bias


def linear_forward_vec(x, w, b):
    """y = x @ W + b. w: [in, out], x: [in] -> y: [out]."""
    return x @ w + b


def gelu(x):
    """Approximate GELU matching Go implementation."""
    inner = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1 + np.tanh(inner))


def scaled_dot_product_attention(Q, K, V, n_heads):
    """Multi-head attention. Q,K,V: [seq, dModel]."""
    seq, d_model = Q.shape
    d_head = d_model // n_heads

    # Reshape to [n_heads, seq, d_head]
    Q_h = Q.reshape(seq, n_heads, d_head).transpose(1, 0, 2)
    K_h = K.reshape(seq, n_heads, d_head).transpose(1, 0, 2)
    V_h = V.reshape(seq, n_heads, d_head).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(d_head)
    # [n_heads, seq, seq]
    scores = Q_h @ K_h.transpose(0, 2, 1) * scale

    # Softmax per head
    for h in range(n_heads):
        for i in range(seq):
            row = scores[h, i]
            row = row - np.max(row)
            exp_row = np.exp(row)
            scores[h, i] = exp_row / np.sum(exp_row)

    # [n_heads, seq, d_head]
    attn_out = scores @ V_h

    # Concat heads: [seq, d_model]
    return attn_out.transpose(1, 0, 2).reshape(seq, d_model)


def gen_itransformer():
    channels = 3
    input_len = 8
    output_len = 4
    d_model = 8
    d_ff = 16
    n_heads = 2
    n_layers = 1

    # Generate random input: [channels, input_len]
    x_input = np.random.randn(channels, input_len)

    # Xavier initialization helper
    def xavier(rows, cols):
        scale = np.sqrt(2.0 / (rows + cols))
        return np.random.randn(rows, cols) * scale

    # Embedding: [input_len, d_model]
    embed_w = xavier(input_len, d_model)
    embed_b = np.zeros(d_model)

    # Encoder layer weights
    qW = xavier(d_model, d_model)
    kW = xavier(d_model, d_model)
    vW = xavier(d_model, d_model)
    oW = xavier(d_model, d_model)
    qB = np.zeros(d_model)
    kB = np.zeros(d_model)
    vB = np.zeros(d_model)
    oB = np.zeros(d_model)
    ln1_scale = np.ones(d_model)
    ln1_bias = np.zeros(d_model)
    fc1W = xavier(d_model, d_ff)
    fc1B = np.zeros(d_ff)
    fc2W = xavier(d_ff, d_model)
    fc2B = np.zeros(d_model)
    ln2_scale = np.ones(d_model)
    ln2_bias = np.zeros(d_model)

    # Output projection: [d_model, output_len]
    proj_w = xavier(d_model, output_len)
    proj_b = np.zeros(output_len)

    # Collect flat params in the same order as Go's flatParams()
    flat_params = []
    # embed_w: [input_len][d_model]
    flat_params.extend(embed_w.flatten().tolist())
    flat_params.extend(embed_b.tolist())
    # Layer: qW, kW, vW, oW (each [d_model][d_model])
    for w in [qW, kW, vW, oW]:
        flat_params.extend(w.flatten().tolist())
    for b in [qB, kB, vB, oB]:
        flat_params.extend(b.tolist())
    # LN1
    flat_params.extend(ln1_scale.tolist())
    flat_params.extend(ln1_bias.tolist())
    # FFN
    flat_params.extend(fc1W.flatten().tolist())
    flat_params.extend(fc1B.tolist())
    flat_params.extend(fc2W.flatten().tolist())
    flat_params.extend(fc2B.tolist())
    # LN2
    flat_params.extend(ln2_scale.tolist())
    flat_params.extend(ln2_bias.tolist())
    # Output projection
    flat_params.extend(proj_w.flatten().tolist())
    flat_params.extend(proj_b.tolist())

    # Forward pass
    # Step 1: Variate embedding
    tokens = np.zeros((channels, d_model))
    for c in range(channels):
        tokens[c] = linear_forward_vec(x_input[c], embed_w, embed_b)

    # Step 2: Encoder layer
    # MHA
    Q = np.zeros((channels, d_model))
    K = np.zeros((channels, d_model))
    V = np.zeros((channels, d_model))
    for c in range(channels):
        Q[c] = linear_forward_vec(tokens[c], qW, qB)
        K[c] = linear_forward_vec(tokens[c], kW, kB)
        V[c] = linear_forward_vec(tokens[c], vW, vB)

    attn_concat = scaled_dot_product_attention(Q, K, V, n_heads)

    # Output projection of attention
    attn_out = np.zeros((channels, d_model))
    for c in range(channels):
        attn_out[c] = linear_forward_vec(attn_concat[c], oW, oB)

    # Residual + LN1
    for c in range(channels):
        tokens[c] = tokens[c] + attn_out[c]
        tokens[c] = layer_norm_1d(tokens[c], ln1_scale, ln1_bias)

    # FFN per variate
    for c in range(channels):
        ffn_out = linear_forward_vec(tokens[c], fc1W, fc1B)
        ffn_out = gelu(ffn_out)
        ffn_out = linear_forward_vec(ffn_out, fc2W, fc2B)
        tokens[c] = tokens[c] + ffn_out
        tokens[c] = layer_norm_1d(tokens[c], ln2_scale, ln2_bias)

    # Step 3: Output projection
    output = np.zeros((channels, output_len))
    for c in range(channels):
        output[c] = linear_forward_vec(tokens[c], proj_w, proj_b)

    golden = {
        "model": "itransformer",
        "channels": channels,
        "input_len": input_len,
        "output_len": output_len,
        "d_model": d_model,
        "d_ff": d_ff,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "input": x_input.tolist(),
        "params": flat_params,
        "expected_output": output.tolist(),
        "tolerance": 1e-9
    }

    with open("layers/model_itransformer.json", "w") as f:
        json.dump(golden, f, indent=2)
    print(f"ITransformer: {len(flat_params)} params, output shape {output.shape}")


if __name__ == "__main__":
    gen_nbeats()
    gen_itransformer()
    print("Done.")
