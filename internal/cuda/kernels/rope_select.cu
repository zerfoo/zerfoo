// rope_select.cu -- CUDA kernel for GPU-indexed RoPE table selection.
// Reads counter[0] to compute offset and copies the correct cos/sin slice
// from the precomputed table.

#include <cuda_runtime.h>

__global__ void kernel_rope_select(const float* __restrict__ cos_table,
                                    const float* __restrict__ sin_table,
                                    float* __restrict__ cos_out,
                                    float* __restrict__ sin_out,
                                    const int* __restrict__ counter,
                                    int halfRotary) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= halfRotary) return;
    int pos = counter[0];
    int offset = pos * halfRotary + idx;
    cos_out[idx] = cos_table[offset];
    sin_out[idx] = sin_table[offset];
}

extern "C" {

cudaError_t launch_rope_select(const float* cos_table, const float* sin_table,
                                float* cos_out, float* sin_out,
                                const int* counter, int halfRotary,
                                cudaStream_t stream) {
    int block = 256;
    int grid = (halfRotary + block - 1) / block;
    kernel_rope_select<<<grid, block, 0, stream>>>(
        cos_table, sin_table, cos_out, sin_out, counter, halfRotary);
    return cudaGetLastError();
}

} // extern "C"
