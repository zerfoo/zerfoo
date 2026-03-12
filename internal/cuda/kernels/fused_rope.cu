// fused_rope.cu -- CUDA kernel for fused Rotary Position Embeddings (RoPE).
// Applies rotary position embeddings in a single pass, eliminating the need
// for separate Split, Mul, Sub, Add, Concat operations.
//
// Input shape:  [batch, seq_len, head_dim]
// cos/sin shape: [seq_len, half_dim]  (row stride may differ from half_dim)
// rotaryDim:    number of dimensions receiving rotation (<= head_dim)

#include <cuda_runtime.h>

// One thread per output element. 1D grid over batch * seq_len * head_dim.
__global__ void kernel_fused_rope(const float* __restrict__ input,
                                   const float* __restrict__ cos_angles,
                                   const float* __restrict__ sin_angles,
                                   float* __restrict__ output,
                                   int batch, int seq_len, int head_dim,
                                   int half_rotary, int cos_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * head_dim;
    if (idx >= total) return;

    // Decompose flat index into (b, s, d).
    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int s = tmp % seq_len;
    // b = tmp / seq_len  (unused, input is contiguous)

    if (d < half_rotary) {
        // First half of rotary dimensions:
        // out[b,s,d] = in[b,s,d] * cos[s,d] - in[b,s,d+halfRotary] * sin[s,d]
        float x0 = __ldg(&input[idx]);
        float x1 = __ldg(&input[idx + half_rotary]);
        float c  = __ldg(&cos_angles[s * cos_stride + d]);
        float sn = __ldg(&sin_angles[s * cos_stride + d]);
        output[idx] = x0 * c - x1 * sn;
    } else if (d < half_rotary * 2) {
        // Second half of rotary dimensions:
        // out[b,s,d] = in[b,s,d] * cos[s,d-halfRotary] + in[b,s,d-halfRotary] * sin[s,d-halfRotary]
        int d2 = d - half_rotary;
        float x1 = __ldg(&input[idx]);
        float x0 = __ldg(&input[idx - half_rotary]);
        float c  = __ldg(&cos_angles[s * cos_stride + d2]);
        float sn = __ldg(&sin_angles[s * cos_stride + d2]);
        output[idx] = x1 * c + x0 * sn;
    } else {
        // Pass-through for non-rotary dimensions.
        output[idx] = __ldg(&input[idx]);
    }
}

// ---------- Launcher function (extern "C" for CGO) ----------

extern "C" {

cudaError_t fused_rope_f32(const float* input, const float* cos_angles,
                            const float* sin_angles, float* output,
                            int batch, int seq_len, int head_dim,
                            int half_rotary, int cos_stride,
                            cudaStream_t stream) {
    int total = batch * seq_len * head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_fused_rope<<<grid, block, 0, stream>>>(
        input, cos_angles, sin_angles, output,
        batch, seq_len, head_dim, half_rotary, cos_stride);
    return cudaGetLastError();
}

} // extern "C"
