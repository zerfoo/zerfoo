#!/usr/bin/env python3
"""
BERT parity reference: constructs the same tiny 2-layer BERT model as
inference/arch_bert_parity_test.go with identical deterministic weights,
runs the same input, and prints per-layer outputs for manual comparison.

Usage:
    pip install torch numpy
    python bert_parity_reference.py
"""

import numpy as np
import torch
import torch.nn as nn


def make_rng_tensor(rng: np.random.RandomState, shape: tuple) -> torch.Tensor:
    """Generate a tensor with values in [-0.1, 0.1] matching Go's math/rand."""
    data = rng.random_sample(np.prod(shape)).astype(np.float32) * 0.2 - 0.1
    return torch.from_numpy(data.reshape(shape))


def main():
    # Match Go's rand.NewSource(42).
    # NOTE: Go's math/rand and NumPy use different algorithms, so the exact
    # values will differ. This script uses the same STRUCTURE and scale to
    # enable manual comparison of output ranges and shapes. For exact parity,
    # export the Go tensors to a file and load them here.
    rng = np.random.RandomState(42)

    hidden = 32
    num_heads = 2
    inter = 64
    max_seq = 8
    vocab = 100
    num_labels = 3
    num_layers = 2
    eps = 1e-12

    # Build weights.
    token_embd = make_rng_tensor(rng, (vocab, hidden))
    pos_embd = make_rng_tensor(rng, (max_seq, hidden))
    type_embd = make_rng_tensor(rng, (2, hidden))
    emb_norm_w = make_rng_tensor(rng, (hidden,))
    emb_norm_b = make_rng_tensor(rng, (hidden,))

    layers = []
    for i in range(num_layers):
        layer = {}
        for proj in ["q", "k", "v"]:
            layer[f"attn_{proj}_w"] = make_rng_tensor(rng, (hidden, hidden))
            layer[f"attn_{proj}_b"] = make_rng_tensor(rng, (hidden,))
        layer["attn_o_w"] = make_rng_tensor(rng, (hidden, hidden))
        layer["attn_o_b"] = make_rng_tensor(rng, (hidden,))
        layer["attn_norm_w"] = make_rng_tensor(rng, (hidden,))
        layer["attn_norm_b"] = make_rng_tensor(rng, (hidden,))
        layer["ffn_up_w"] = make_rng_tensor(rng, (inter, hidden))
        layer["ffn_up_b"] = make_rng_tensor(rng, (inter,))
        layer["ffn_down_w"] = make_rng_tensor(rng, (hidden, inter))
        layer["ffn_down_b"] = make_rng_tensor(rng, (hidden,))
        layer["ffn_norm_w"] = make_rng_tensor(rng, (hidden,))
        layer["ffn_norm_b"] = make_rng_tensor(rng, (hidden,))
        layers.append(layer)

    cls_w = make_rng_tensor(rng, (num_labels, hidden))
    cls_b = make_rng_tensor(rng, (num_labels,))

    # Input: [CLS]=1, tokens 42, 73, [SEP]=2 (within vocab_size=100).
    input_ids = torch.tensor([[1, 42, 73, 2]], dtype=torch.long)
    seq_len = input_ids.shape[1]

    # Step 1: Embedding = token + position + type, then LayerNorm.
    tok = token_embd[input_ids[0]]  # [seq, hidden]
    pos = pos_embd[:seq_len]        # [seq, hidden]
    typ = type_embd[torch.zeros(seq_len, dtype=torch.long)]  # [seq, hidden]

    emb = tok + pos + typ  # [seq, hidden]
    emb = emb.unsqueeze(0)  # [1, seq, hidden]

    # Manual LayerNorm.
    mean = emb.mean(dim=-1, keepdim=True)
    var_ = emb.var(dim=-1, keepdim=True, unbiased=False)
    emb_normed = (emb - mean) / torch.sqrt(var_ + eps)
    emb_normed = emb_normed * emb_norm_w + emb_norm_b

    print(f"Embedding output shape: {emb_normed.shape}")
    print(f"Embedding output[0, 0, :8]: {emb_normed[0, 0, :8].tolist()}")

    hidden_state = emb_normed

    for li, layer in enumerate(layers):
        # --- Self-attention (bidirectional) ---
        q = hidden_state @ layer["attn_q_w"].T + layer["attn_q_b"]  # [1, seq, hidden]
        k = hidden_state @ layer["attn_k_w"].T + layer["attn_k_b"]
        v = hidden_state @ layer["attn_v_w"].T + layer["attn_v_b"]

        head_dim = hidden // num_heads
        # Reshape to [1, num_heads, seq, head_dim]
        q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention (no causal mask).
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        print(f"\nLayer {li} attention weights sum per row (should be ~1.0):")
        print(f"  head 0, row 0: {attn_weights[0, 0, 0].sum().item():.6f}")
        print(f"  head 1, row 0: {attn_weights[0, 1, 0].sum().item():.6f}")

        attn_out = torch.matmul(attn_weights, v)  # [1, num_heads, seq, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, hidden)

        # Output projection + bias.
        attn_out = attn_out @ layer["attn_o_w"].T + layer["attn_o_b"]

        # Post-attention residual + LayerNorm.
        residual = hidden_state + attn_out
        mean = residual.mean(dim=-1, keepdim=True)
        var_ = residual.var(dim=-1, keepdim=True, unbiased=False)
        normed = (residual - mean) / torch.sqrt(var_ + eps)
        normed = normed * layer["attn_norm_w"] + layer["attn_norm_b"]

        print(f"Layer {li} post-attn norm shape: {normed.shape}")
        print(f"Layer {li} post-attn norm[0, 0, :8]: {normed[0, 0, :8].tolist()}")

        # --- FFN: up -> GELU -> down ---
        ffn_up = normed @ layer["ffn_up_w"].T + layer["ffn_up_b"]
        ffn_act = nn.functional.gelu(ffn_up)
        ffn_down = ffn_act @ layer["ffn_down_w"].T + layer["ffn_down_b"]

        # Post-FFN residual + LayerNorm.
        residual2 = normed + ffn_down
        mean = residual2.mean(dim=-1, keepdim=True)
        var_ = residual2.var(dim=-1, keepdim=True, unbiased=False)
        normed2 = (residual2 - mean) / torch.sqrt(var_ + eps)
        hidden_state = normed2 * layer["ffn_norm_w"] + layer["ffn_norm_b"]

        print(f"Layer {li} post-FFN norm shape: {hidden_state.shape}")
        print(f"Layer {li} post-FFN norm[0, 0, :8]: {hidden_state[0, 0, :8].tolist()}")

    # --- Classification head: pool CLS token, then linear ---
    cls_hidden = hidden_state[:, 0, :]  # [1, hidden]
    logits = cls_hidden @ cls_w.T + cls_b  # [1, num_labels]

    print(f"\nFinal logits shape: {logits.shape}")
    print(f"Final logits: {logits[0].tolist()}")
    print(f"Logit spread: {(logits.max() - logits.min()).item():.6f}")


if __name__ == "__main__":
    main()
