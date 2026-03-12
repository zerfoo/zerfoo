// fused_qk_norm_rope.cu -- Fused per-head RMSNorm + RoPE for Q and K heads.
//
// Replaces 4 kernel launches (Q_norm + K_norm + Q_RoPE + K_RoPE) with 1.
// Input: contiguous [totalHeads, headDim] buffer containing Q heads followed
// by K heads. Heads 0..numQHeads-1 use weightQ, the rest use weightK.
//
// Each block processes one head: RMSNorm then RoPE in shared memory.

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

// Per-head: RMSNorm(x, weight, eps) then RoPE(x, cos, sin).
// Block per head, blockDim.x threads cooperate on headDim elements.
__global__ void kernel_fused_qk_norm_rope(
    const float* __restrict__ input,      // [totalHeads, headDim]
    const float* __restrict__ weightQ,    // [headDim]
    const float* __restrict__ weightK,    // [headDim]
    const float* __restrict__ cosAngles,  // [halfRotary]
    const float* __restrict__ sinAngles,  // [halfRotary]
    float*       __restrict__ output,     // [totalHeads, headDim]
    float eps, int headDim, int numQHeads, int halfRotary)
{
    int head = blockIdx.x;
    const float* x = input + head * headDim;
    float* o = output + head * headDim;
    const float* w = (head < numQHeads) ? weightQ : weightK;

    extern __shared__ float smem[];
    // smem layout: [headDim floats for normalized x] [blockDim.x floats for reduction]
    float* x_norm = smem;
    float* reduce = smem + headDim;

    // Phase 1: Compute sum of squares for RMSNorm.
    float local_sq = 0.0f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float v = x[d];
        local_sq += v * v;
    }
    reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            reduce[threadIdx.x] += reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(reduce[0] / (float)headDim + eps);

    // Phase 2: Normalize and store in shared memory.
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        x_norm[d] = x[d] * scale * w[d];
    }
    __syncthreads();

    // Phase 3: Apply RoPE. First halfRotary dimensions are rotated,
    // next halfRotary are the paired dimensions, rest are passthrough.
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        if (d < halfRotary) {
            float c = cosAngles[d];
            float s_val = sinAngles[d];
            float x0 = x_norm[d];
            float x1 = x_norm[d + halfRotary];
            o[d] = x0 * c - x1 * s_val;
            o[d + halfRotary] = x0 * s_val + x1 * c;
        } else if (d >= 2 * halfRotary) {
            // Passthrough dimensions (beyond rotary range).
            o[d] = x_norm[d];
        }
        // d in [halfRotary, 2*halfRotary) is written by the d < halfRotary branch.
    }
}

static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

extern "C" {

// fused_qk_norm_rope_f32 applies per-head RMSNorm + RoPE to combined Q+K data.
//
// input:      [totalHeads, headDim] device pointer (Q heads then K heads).
// weightQ:    [headDim] RMSNorm weight for Q heads.
// weightK:    [headDim] RMSNorm weight for K heads.
// cosAngles:  [halfRotary] cosine angles for RoPE.
// sinAngles:  [halfRotary] sine angles for RoPE.
// output:     [totalHeads, headDim] device pointer.
// eps_bits:   RMSNorm epsilon as uint32 bit pattern.
// totalHeads: numQHeads + numKVHeads.
// headDim:    dimension per head.
// numQHeads:  boundary index (heads 0..numQHeads-1 use weightQ).
// halfRotary: half of the rotary dimension.
cudaError_t fused_qk_norm_rope_f32(
    const float* input, const float* weightQ, const float* weightK,
    const float* cosAngles, const float* sinAngles, float* output,
    unsigned int eps_bits,
    int totalHeads, int headDim, int numQHeads, int halfRotary,
    cudaStream_t stream)
{
    float eps = bits_to_float(eps_bits);
    // Block size: power of 2 up to min(headDim, 256).
    int block = 1;
    while (block < headDim && block < 256) block <<= 1;
    // Shared memory: headDim (normalized x) + block (reduction buffer).
    size_t smem = (headDim + block) * sizeof(float);
    kernel_fused_qk_norm_rope<<<totalHeads, block, smem, stream>>>(
        input, weightQ, weightK, cosAngles, sinAngles, output,
        eps, headDim, numQHeads, halfRotary);
    return cudaGetLastError();
}

} // extern "C"
