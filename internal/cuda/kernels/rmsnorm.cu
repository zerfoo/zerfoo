// rmsnorm.cu -- CUDA kernel for fused RMSNorm.
// Computes: output[row] = input[row] * rsqrt(mean(input[row]^2) + eps) * weight
// Single-pass per row using shared-memory reduction.

#include <cuda_runtime.h>
#include <math.h>

// Each block handles one row of length D.
// Uses shared memory for parallel sum-of-squares reduction.
__global__ void kernel_rmsnorm(const float* __restrict__ input,
                                const float* __restrict__ weight,
                                float* __restrict__ output,
                                float* __restrict__ scales,
                                float eps, int D) {
    int row = blockIdx.x;
    const float* x = input + row * D;
    float* y = output + row * D;

    extern __shared__ float sdata[];

    // Phase 1: Compute partial sum of squares.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = x[i];
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();

    // Parallel reduction for sum of squares.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute scale = rsqrt(mean_sq + eps).
    float scale = rsqrtf(sdata[0] / (float)D + eps);

    // Store per-row scale for backward pass.
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }

    // Phase 2: Normalize and scale by weight.
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        y[i] = x[i] * scale * weight[i];
    }
}

// ---------- Launcher function (extern "C" for CGO) ----------

extern "C" {

cudaError_t launch_rmsnorm(const float* input, const float* weight,
                            float* output, float* scales, float eps,
                            int rows, int D, cudaStream_t stream) {
    // Block size: next power of 2 up to min(D, 256).
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_rmsnorm<<<rows, block, smem, stream>>>(input, weight, output, scales, eps, D);
    return cudaGetLastError();
}

} // extern "C"
