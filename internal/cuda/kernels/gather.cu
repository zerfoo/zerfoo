// gather.cu -- CUDA kernel for embedding table gather (lookup).
// Each thread block handles one index, copying D elements from the table.
// Indices are int64 (Go int on 64-bit platforms) to avoid CPU-side conversion.

#include <cuda_runtime.h>

// kernel_gather: output[i, :] = table[indices[i], :]
// table: [V, D], indices: [N] int64, output: [N, D]
__global__ void kernel_gather(const float* __restrict__ table,
                               const long long* __restrict__ indices,
                               float* __restrict__ output,
                               int N, int D, int V) {
    int row = blockIdx.x;
    if (row >= N) return;

    int idx = (int)indices[row];
    // Clamp index to valid range.
    if (idx < 0) idx = 0;
    if (idx >= V) idx = V - 1;

    const float* src = table + idx * D;
    float* dst = output + row * D;

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        dst[col] = src[col];
    }
}

// ---------- Launcher function (extern "C" for CGO) ----------

extern "C" {

cudaError_t launch_gather(const float* table, const long long* indices,
                           float* output, int N, int D, int V,
                           cudaStream_t stream) {
    int block = 256;
    if (D < block) block = D;
    // Round up to next power of 2 for efficiency.
    int b = 1;
    while (b < block) b <<= 1;
    if (b > 256) b = 256;

    kernel_gather<<<N, b, 0, stream>>>(table, indices, output, N, D, V);
    return cudaGetLastError();
}

} // extern "C"
