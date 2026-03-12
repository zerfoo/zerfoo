// fused_norm_add.cu -- Fused RMSNorm + elementwise Add in one kernel launch.
//
// Computes: output = rmsnorm(input, weight, eps) + residual
// Replaces 2 kernel launches (RMSNorm + Add) with 1.
// Input: [rows, D], weight: [D], residual: [rows, D], output: [rows, D].

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

__global__ void kernel_fused_norm_add(
    const float* __restrict__ input,    // [rows, D]
    const float* __restrict__ weight,   // [D]
    const float* __restrict__ residual, // [rows, D]
    float*       __restrict__ output,   // [rows, D]
    float eps, int D)
{
    int row = blockIdx.x;
    const float* x = input + row * D;
    const float* r = residual + row * D;
    float* o = output + row * D;

    extern __shared__ float smem[];

    // Phase 1: Compute sum of squares for RMSNorm.
    float local_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = x[d];
        local_sq += v * v;
    }
    smem[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scale = rsqrtf(smem[0] / (float)D + eps);

    // Phase 2: Normalize, multiply by weight, add residual.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        o[d] = x[d] * scale * weight[d] + r[d];
    }
}

static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

extern "C" {

// fused_norm_add_f32 applies RMSNorm then adds residual in one kernel launch.
//
// input:    [rows, D] device pointer (data to normalize).
// weight:   [D] RMSNorm weight.
// residual: [rows, D] device pointer (added after normalization).
// output:   [rows, D] device pointer.
// eps_bits: RMSNorm epsilon as uint32 bit pattern.
// rows:     number of rows.
// D:        row dimension.
// stream:   CUDA stream (pass NULL for default stream).
cudaError_t fused_norm_add_f32(
    const float* input, const float* weight, const float* residual,
    float* output,
    unsigned int eps_bits, int rows, int D,
    cudaStream_t stream)
{
    float eps = bits_to_float(eps_bits);
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_fused_norm_add<<<rows, block, smem, stream>>>(
        input, weight, residual, output, eps, D);
    return cudaGetLastError();
}

} // extern "C"
