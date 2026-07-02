// gather.cu -- CUDA kernel for embedding table gather (lookup).
// Each thread block handles one index, copying D elements from the table.
// Supports both int32 and int64 indices.

#include <cuda_runtime.h>

// ---------- Templated gather kernel ----------

// kernel_gather_t: output[i, :] = table[indices[i], :]
// table: [V, D], indices: [N], output: [N, D]
template <typename IndexT>
__global__ void kernel_gather_t(const float* __restrict__ table,
                                const IndexT* __restrict__ indices,
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

// ---------- Helper: compute block size ----------

static inline int gather_block_size(int D) {
    int block = 256;
    if (D < block) block = D;
    int b = 1;
    while (b < block) b <<= 1;
    if (b > 256) b = 256;
    return b;
}

// ---------- Launcher functions (extern "C" for purego / CGo) ----------

extern "C" {

// launch_gather: int64 (long long) indices.
cudaError_t launch_gather(const float* table, const long long* indices,
                           float* output, int N, int D, int V,
                           cudaStream_t stream) {
    int b = gather_block_size(D);
    kernel_gather_t<long long><<<N, b, 0, stream>>>(table, indices, output, N, D, V);
    return cudaGetLastError();
}

// launch_gather_i32: int32 indices.
cudaError_t launch_gather_i32(const float* table, const int* indices,
                               float* output, int N, int D, int V,
                               cudaStream_t stream) {
    int b = gather_block_size(D);
    kernel_gather_t<int><<<N, b, 0, stream>>>(table, indices, output, N, D, V);
    return cudaGetLastError();
}

} // extern "C"
