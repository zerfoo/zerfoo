// fused_swiglu.cu -- CUDA kernel for fused SwiGLU activation.
// Computes output[i] = w1[i] * sigmoid(w1[i]) * w3[i] in a single pass,
// eliminating the Concat + Split + sigmoid + Mul + Mul chain.
//
// w1, w3 shape: [n] (flattened, same length)
// output shape: [n]

#include <cuda_runtime.h>

__global__ void kernel_fused_swiglu(const float* __restrict__ w1,
                                     const float* __restrict__ w3,
                                     float* __restrict__ output,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = __ldg(&w1[idx]);
    float gate = 1.0f / (1.0f + expf(-x));  // sigmoid(x)
    output[idx] = x * gate * __ldg(&w3[idx]);  // silu(x) * w3
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------

extern "C" {

cudaError_t fused_swiglu_f32(const float* w1, const float* w3, float* output,
                              int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_fused_swiglu<<<grid, block, 0, stream>>>(w1, w3, output, n);
    return cudaGetLastError();
}

} // extern "C"
