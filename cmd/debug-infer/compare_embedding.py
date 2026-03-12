#!/usr/bin/env python3
"""Compare embedding lookup and first-layer computation against Zerfoo.

Usage: python3 compare_embedding.py /path/to/model.gguf
"""
import sys
import struct
import math
import numpy as np

def read_gguf_f16(raw_bytes):
    """Convert 2-byte little-endian fp16 to float32."""
    bits = struct.unpack('<H', raw_bytes)[0]
    return np.float16(np.frombuffer(struct.pack('<H', bits), dtype=np.float16)[0]).astype(np.float32)

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/ndungu/models/gemma3-gguf/model.gguf"

    from gguf import GGUFReader
    reader = GGUFReader(model_path)

    # Build tensor dict
    tensors = {}
    for t in reader.tensors:
        name = str(t.name)
        data = t.data
        shape = tuple(t.shape)
        tensors[name] = (shape, data)

    # Get embedding weight
    emb_shape, emb_data = tensors['token_embd.weight']
    print(f"Embedding shape (GGUF ne order): {emb_shape}")
    # GGUF ne order: ne[0]=hidden_dim, ne[1]=vocab_size
    # So data is stored as vocab_size rows of hidden_dim columns
    hidden_dim = emb_shape[0]
    vocab_size = emb_shape[1]
    print(f"hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    # Reshape to [vocab_size, hidden_dim]
    emb = np.array(emb_data, dtype=np.float32).reshape(vocab_size, hidden_dim)

    # BOS token = 2 for Gemma
    bos_embedding = emb[2].copy()
    print(f"\nBOS (token 2) embedding first 20 values:")
    print(bos_embedding[:20].tolist())

    # Gemma scales embeddings by sqrt(hidden_dim)
    embed_scale = math.sqrt(hidden_dim)
    scaled = bos_embedding * embed_scale
    print(f"\nScaled by sqrt({hidden_dim})={embed_scale:.4f}, first 20:")
    print(scaled[:20].tolist())

    # RMS norm of scaled embedding
    rms = np.sqrt(np.mean(scaled ** 2))
    print(f"\nRMS of scaled embedding: {rms:.6f}")

    # Get input_layernorm.weight for layer 0
    ln_shape, ln_data = tensors['blk.0.attn_norm.weight']
    ln_weight = np.array(ln_data, dtype=np.float32)
    print(f"\nLayer 0 input_layernorm.weight first 20: {ln_weight[:20].tolist()}")

    # Apply RMSNorm
    eps = 1e-6
    rms_val = np.sqrt(np.mean(scaled ** 2) + eps)
    normed = (scaled / rms_val) * ln_weight
    print(f"\nAfter RMSNorm first 20: {normed[:20].tolist()}")

    # Q projection
    q_shape, q_data = tensors['blk.0.attn_q.weight']
    print(f"\nQ weight shape (GGUF ne): {q_shape}")
    # ne[0]=hidden_dim=1152, ne[1]=num_heads*head_dim=1024
    # data stored as 1024 rows of 1152 columns
    q_rows = q_shape[1]  # 1024
    q_cols = q_shape[0]  # 1152
    q_weight = np.array(q_data, dtype=np.float32).reshape(q_rows, q_cols)
    print(f"Q weight reshaped: [{q_rows}, {q_cols}]")

    # Q = normed @ W_q.T (standard MatMul)
    q_out = normed @ q_weight.T
    print(f"Q projection output shape: {q_out.shape}")
    print(f"Q projection first 20: {q_out[:20].tolist()}")

    # K projection
    k_shape, k_data = tensors['blk.0.attn_k.weight']
    k_rows = k_shape[1]
    k_cols = k_shape[0]
    k_weight = np.array(k_data, dtype=np.float32).reshape(k_rows, k_cols)
    k_out = normed @ k_weight.T
    print(f"\nK projection output shape: {k_out.shape}, first 20: {k_out[:20].tolist()}")

    # V projection
    v_shape, v_data = tensors['blk.0.attn_v.weight']
    v_rows = v_shape[1]
    v_cols = v_shape[0]
    v_weight = np.array(v_data, dtype=np.float32).reshape(v_rows, v_cols)
    v_out = normed @ v_weight.T
    print(f"V projection output shape: {v_out.shape}, first 20: {v_out[:20].tolist()}")

    print("\n=== Summary ===")
    print(f"BOS embedding L2 norm: {np.linalg.norm(bos_embedding):.6f}")
    print(f"Scaled embedding L2 norm: {np.linalg.norm(scaled):.6f}")
    print(f"Normed L2 norm: {np.linalg.norm(normed):.6f}")
    print(f"Q output L2 norm: {np.linalg.norm(q_out):.6f}")

if __name__ == '__main__':
    main()
