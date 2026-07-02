// transpose.cu -- CUDA kernels for tensor transpose operations.
// Supports 2D, 3D, and 4D tensors with arbitrary permutations.
// Uses shared-memory tiling for coalesced memory access.

#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// ---------- 2D transpose with shared-memory tiling ----------
// Transposes a rows x cols matrix into cols x rows.
// Uses 32x32 shared-memory tiles with +1 padding to avoid bank conflicts.

__global__ void kernel_transpose_2d(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIdx = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIdx = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load tile from input (coalesced reads).
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((yIdx + j) < rows && xIdx < cols) {
            tile[threadIdx.y + j][threadIdx.x] = input[(yIdx + j) * cols + xIdx];
        }
    }
    __syncthreads();

    // Write tile to output (coalesced writes, transposed indices).
    int outX = blockIdx.y * TILE_DIM + threadIdx.x;
    int outY = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((outY + j) < cols && outX < rows) {
            output[(outY + j) * rows + outX] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ---------- General N-D transpose ----------
// Permutes dimensions of an N-D tensor using stride-based indexing.
// Each thread handles one element: computes source flat index from output flat index.
// Output strides are precomputed on the host to avoid O(ndim^2) work per thread.

__global__ void kernel_transpose_nd(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const int* __restrict__ in_strides,
                                     const int* __restrict__ out_strides,
                                     const int* __restrict__ perm,
                                     int ndim, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose output flat index into output coordinates using
    // precomputed output strides, then map to input via perm.
    int remaining = idx;
    int src_idx = 0;
    for (int d = 0; d < ndim; d++) {
        int coord = remaining / out_strides[d];
        remaining = remaining % out_strides[d];
        src_idx += coord * in_strides[perm[d]];
    }

    output[idx] = input[src_idx];
}

// ---------- Launcher functions (extern "C" for CGO) ----------

extern "C" {

cudaError_t launch_transpose_2d(const float* input, float* output,
                                 int rows, int cols, cudaStream_t stream) {
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM,
              (rows + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    kernel_transpose_2d<<<grid, block, 0, stream>>>(input, output, rows, cols);
    return cudaGetLastError();
}

cudaError_t launch_transpose_nd(const float* input, float* output,
                                 const int* in_strides, const int* out_strides,
                                 const int* perm, int ndim, int total,
                                 cudaStream_t stream) {
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_transpose_nd<<<grid, block, 0, stream>>>(input, output, in_strides,
                                                     out_strides, perm, ndim, total);
    return cudaGetLastError();
}

} // extern "C"
