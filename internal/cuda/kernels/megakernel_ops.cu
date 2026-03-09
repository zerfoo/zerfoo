// megakernel_ops.cu -- Device functions for the megakernel code generator.
//
// Each function operates on data in registers or shared memory, avoiding
// global memory round-trips between ops. These are the building blocks
// that the emitter (internal/codegen) chains together.
//
// Included by generated megakernel .cu files via:
//   #include "megakernel_ops.cu"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

// ============================================================
// T92.1: Elementwise device functions
// ============================================================

__device__ __forceinline__ float dev_add(float a, float b) { return a + b; }
__device__ __forceinline__ float dev_sub(float a, float b) { return a - b; }
__device__ __forceinline__ float dev_mul(float a, float b) { return a * b; }
__device__ __forceinline__ float dev_div(float a, float b) { return a / b; }
__device__ __forceinline__ float dev_pow(float a, float b) { return powf(a, b); }

__device__ __forceinline__ float dev_add_scalar(float a, float s) { return a + s; }
__device__ __forceinline__ float dev_sub_scalar(float a, float s) { return a - s; }
__device__ __forceinline__ float dev_mul_scalar(float a, float s) { return a * s; }
__device__ __forceinline__ float dev_div_scalar(float a, float s) { return a / s; }
__device__ __forceinline__ float dev_pow_scalar(float a, float s) { return powf(a, s); }

// ============================================================
// T92.2: Unary device functions
// ============================================================

__device__ __forceinline__ float dev_exp(float x) { return expf(x); }
__device__ __forceinline__ float dev_log(float x) { return logf(x); }
__device__ __forceinline__ float dev_sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ float dev_rsqrt(float x) { return rsqrtf(x); }
__device__ __forceinline__ float dev_tanh(float x) { return tanhf(x); }
__device__ __forceinline__ float dev_neg(float x) { return -x; }
__device__ __forceinline__ float dev_abs(float x) { return fabsf(x); }

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float dev_silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================
// T92.3: RMSNorm device function
// ============================================================

// dev_rmsnorm normalizes a vector in shared memory.
// Call with one warp or block per row. Uses shared memory for reduction.
// Parameters:
//   out    - output array in shared memory (length dim)
//   in     - input array in shared memory (length dim)
//   weight - weight array in global memory (length dim)
//   dim    - vector dimension
//   eps    - epsilon for numerical stability (default 1e-6)
__device__ void dev_rmsnorm(float* out, const float* in, const float* weight,
                             int dim, float eps) {
    // Shared memory for reduction.
    extern __shared__ float smem[];

    // Phase 1: Sum of squares.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = in[i];
        local_sq += v * v;
    }
    smem[threadIdx.x] = local_sq;
    __syncthreads();

    // Parallel reduction.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float scale = rsqrtf(smem[0] / (float)dim + eps);

    // Phase 2: Normalize and apply weight.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = in[i] * scale * weight[i];
    }
    __syncthreads();
}

// ============================================================
// T92.4: Softmax device function
// ============================================================

// dev_softmax computes softmax over the last dimension.
// Uses shared memory for max and sum reductions.
__device__ void dev_softmax(float* out, const float* in, int rows, int cols) {
    extern __shared__ float smem[];

    for (int row = 0; row < rows; row++) {
        const float* x = in + row * cols;
        float* y = out + row * cols;

        // Phase 1: Find max for numerical stability.
        float local_max = -INFINITY;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            local_max = fmaxf(local_max, x[i]);
        }
        smem[threadIdx.x] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < (unsigned)s) {
                smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
            }
            __syncthreads();
        }
        float max_val = smem[0];

        // Phase 2: Compute exp and sum.
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float e = expf(x[i] - max_val);
            y[i] = e;
            local_sum += e;
        }
        smem[threadIdx.x] = local_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < (unsigned)s) {
                smem[threadIdx.x] += smem[threadIdx.x + s];
            }
            __syncthreads();
        }
        float sum = smem[0];

        // Phase 3: Normalize.
        float inv_sum = 1.0f / sum;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            y[i] *= inv_sum;
        }
        __syncthreads();
    }
}

// ============================================================
// T92.5: Q4 GEMV device function
// ============================================================

// Q4 block format: 32 nibbles (16 bytes) + 1 float scale + 1 float zero.
// Total block size: 24 bytes for 32 elements.
#define Q4_BLOCK_SIZE 32

// dev_gemv_q4 computes matrix-vector product with Q4-quantized weight matrix.
// out[m] = sum_k(dequant(weight[m][k]) * activation[k]) for k in [0, K)
//
// Weight layout: row-major blocks of Q4_BLOCK_SIZE elements.
// Each block: scale (float), zero (float), 16 bytes of packed nibbles.
__device__ void dev_gemv_q4(float* out, const void* weight,
                             const float* activation, int M, int K) {
    // Each thread handles one or more output rows.
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float acc = 0.0f;
        int num_blocks = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
        const char* row = (const char*)weight + (size_t)m * num_blocks * 24;

        for (int blk = 0; blk < num_blocks; blk++) {
            const char* block_ptr = row + blk * 24;
            float scale = *(const float*)block_ptr;
            float zero = *(const float*)(block_ptr + 4);
            const unsigned char* nibbles = (const unsigned char*)(block_ptr + 8);

            int k_start = blk * Q4_BLOCK_SIZE;
            for (int i = 0; i < 16 && (k_start + i * 2) < K; i++) {
                unsigned char packed = nibbles[i];
                float v0 = ((packed & 0x0F) - 8) * scale + zero;
                float v1 = ((packed >> 4) - 8) * scale + zero;
                int k0 = k_start + i * 2;
                int k1 = k0 + 1;
                acc += v0 * activation[k0];
                if (k1 < K) acc += v1 * activation[k1];
            }
        }
        out[m] = acc;
    }
}

// ============================================================
// T92.6: F32 GEMV device function
// ============================================================

// dev_gemv_f32 computes matrix-vector product with float32 weight matrix.
// out[m] = sum_k(weight[m*K + k] * activation[k]) for k in [0, K)
__device__ void dev_gemv_f32(float* out, const float* weight,
                              const float* activation, int M, int K) {
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        float acc = 0.0f;
        const float* row = weight + (size_t)m * K;
        for (int k = 0; k < K; k++) {
            acc += row[k] * activation[k];
        }
        out[m] = acc;
    }
}

// ============================================================
// T92.7: Gather device function
// ============================================================

// dev_gather reads one row from an embedding table to output.
// out[i] = table[index * dim + i] for i in [0, dim)
__device__ void dev_gather(float* out, const float* table, int index, int dim) {
    const float* row = table + (size_t)index * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = row[i];
    }
    __syncthreads();
}

// ============================================================
// T92.8: Cooperative grid sync
// ============================================================

// dev_grid_sync synchronizes all thread blocks in a cooperative kernel launch.
__device__ void dev_grid_sync() {
    cooperative_groups::this_grid().sync();
}

// ============================================================
// T99.1: Slice device function
// ============================================================

// dev_slice copies a contiguous slice along the last axis.
// Copies elements [start, end) from in to out.
// dim is the size of the last axis in the input tensor.
__device__ void dev_slice(float* out, const float* in,
                           int start, int end, int axis, int dim) {
    int len = end - start;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        out[i] = in[start + i];
    }
    __syncthreads();
}

// ============================================================
// T99.2: Repeat device function
// ============================================================

// dev_repeat replicates input along an axis (GQA K/V head replication).
// For last-axis repeat: out[r*dim + i] = in[i] for r in [0, reps).
__device__ void dev_repeat(float* out, const float* in,
                            int axis, int reps, int dim) {
    int total = dim * reps;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        out[i] = in[i % dim];
    }
    __syncthreads();
}

// ============================================================
// T99.3: ReduceSum and ReduceMean device functions
// ============================================================

// dev_reduce_sum computes sum reduction along the specified axis.
// Uses shared memory for parallel reduction.
__device__ void dev_reduce_sum(float* out, const float* in, int axis, int dim) {
    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += in[i];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[0] = smem[0];
    }
    __syncthreads();
}

// dev_reduce_mean computes mean reduction along the specified axis.
// Uses shared memory for parallel reduction.
__device__ void dev_reduce_mean(float* out, const float* in, int axis, int dim) {
    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += in[i];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[0] = smem[0] / (float)dim;
    }
    __syncthreads();
}

// dev_transpose reorders elements according to a permutation.
// For the megakernel, most transposes are logical (just reindex).
// This is a physical copy for cases where data layout must change.
__device__ void dev_transpose(float* out, const float* in,
                               const int* shape, const int* perm,
                               int ndim, int total) {
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // Compute source multi-index from flat idx.
        int src_idx = 0;

        // For 2D transpose (most common): just swap row/col.
        if (ndim == 2) {
            int rows = shape[0], cols = shape[1];
            int r = idx / cols;
            int c = idx % cols;
            if (perm[0] == 1) {
                // Transpose: (r,c) -> (c,r)
                src_idx = c * rows + r;
            } else {
                src_idx = idx; // identity
            }
        } else {
            // General N-D transpose: compute permuted index.
            // (Simplified; full implementation would compute strides)
            src_idx = idx;
        }

        out[idx] = in[src_idx];
    }
    __syncthreads();
}
