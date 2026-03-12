// fused_add_rmsnorm.cu -- CUDA kernel for fused Add + RMSNorm.
// Combines residual addition and RMS normalization into a single kernel,
// saving one kernel launch per transformer layer normalization.
//
// Operation:
//   residual[i] = input[i] + residual[i]           (in-place update)
//   output[i]   = residual[i] / rms * weight[i]    (RMSNorm)
//   where rms = sqrt(mean(residual^2) + eps)
//
// Each block handles one row of length D.
// Uses shared memory for parallel sum-of-squares reduction.

#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_fused_add_rmsnorm(const float* __restrict__ input,
                                          const float* __restrict__ residual,
                                          const float* __restrict__ weight,
                                          float* __restrict__ normed_out,
                                          float* __restrict__ sum_out,
                                          float eps, int D) {
    int row = blockIdx.x;
    const float* inp = input + row * D;
    const float* res = residual + row * D;
    float* nout = normed_out + row * D;
    float* sout = sum_out + row * D;

    extern __shared__ float sdata[];

    // Phase 1: Compute sum = input + residual, write to sum_out, accumulate sum of squares.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = inp[i] + res[i];
        sout[i] = v;
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

    // Phase 2: Normalize sum and scale by weight.
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        nout[i] = sout[i] * scale * weight[i];
    }
}

// bits_to_float reinterprets a uint32 bit pattern as float32.
// Used because the purego/ccall calling convention passes all arguments
// through integer registers.
static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------

extern "C" {

cudaError_t fused_add_rmsnorm_f32(const float* input, const float* residual,
                                    const float* weight, float* normed_out,
                                    float* sum_out,
                                    unsigned int eps_bits,
                                    int rows, int D, cudaStream_t stream) {
    float eps = bits_to_float(eps_bits);
    // Block size: next power of 2 up to min(D, 256).
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_fused_add_rmsnorm<<<rows, block, smem, stream>>>(input, residual, weight, normed_out, sum_out, eps, D);
    return cudaGetLastError();
}

} // extern "C"
