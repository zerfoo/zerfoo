// argmax.cu -- CUDA kernel to find the index of the maximum element in a float array.
// Uses a two-stage parallel reduction: intra-block shared memory reduction,
// then a single-block second pass over block-level results.

#include <cuda_runtime.h>
#include <float.h>

// Stage 1: Each block reduces a chunk of the input to one (value, index) pair.
__global__ void kernel_argmax_stage1(const float* __restrict__ input,
                                      float* __restrict__ blockVals,
                                      int* __restrict__ blockIdxs,
                                      int n) {
    extern __shared__ char smem[];
    float* svals = (float*)smem;
    int* sidxs = (int*)(svals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize with -FLT_MAX so unused threads lose.
    float val = -FLT_MAX;
    int idx = 0;
    if (gid < n) {
        val = input[gid];
        idx = gid;
    }

    svals[tid] = val;
    sidxs[tid] = idx;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (svals[tid + s] > svals[tid]) {
                svals[tid] = svals[tid + s];
                sidxs[tid] = sidxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockVals[blockIdx.x] = svals[0];
        blockIdxs[blockIdx.x] = sidxs[0];
    }
}

// Stage 2: Single block reduces the block-level results to one final answer.
__global__ void kernel_argmax_stage2(const float* __restrict__ blockVals,
                                      const int* __restrict__ blockIdxs,
                                      int* __restrict__ result,
                                      int numBlocks) {
    extern __shared__ char smem[];
    float* svals = (float*)smem;
    int* sidxs = (int*)(svals + blockDim.x);

    int tid = threadIdx.x;

    float val = -FLT_MAX;
    int idx = 0;
    if (tid < numBlocks) {
        val = blockVals[tid];
        idx = blockIdxs[tid];
    }

    svals[tid] = val;
    sidxs[tid] = idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (svals[tid + s] > svals[tid]) {
                svals[tid] = svals[tid + s];
                sidxs[tid] = sidxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = sidxs[0];
    }
}

// ---------- Launcher function (extern "C" for dlsym) ----------

extern "C" {

// launch_argmax finds the index of the maximum element in input[0..n-1].
// result is a device pointer to a single int.
// scratch is a device pointer to temporary storage of at least
// 2 * ceil(n/256) * sizeof(float) bytes (for blockVals + blockIdxs).
cudaError_t launch_argmax(const float* input, int* result,
                           void* scratch, int n,
                           cudaStream_t stream) {
    const int BLOCK = 256;
    int numBlocks = (n + BLOCK - 1) / BLOCK;

    // scratch layout: [numBlocks floats] [numBlocks ints]
    float* blockVals = (float*)scratch;
    int* blockIdxs = (int*)((char*)scratch + numBlocks * sizeof(float));

    int smemSize = BLOCK * (sizeof(float) + sizeof(int));

    // Stage 1: per-block reduction.
    kernel_argmax_stage1<<<numBlocks, BLOCK, smemSize, stream>>>(
        input, blockVals, blockIdxs, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // Stage 2: reduce block results.
    // Use next power-of-2 >= numBlocks, capped at 1024.
    int stage2Block = 1;
    while (stage2Block < numBlocks) stage2Block <<= 1;
    if (stage2Block > 1024) stage2Block = 1024;

    int smem2 = stage2Block * (sizeof(float) + sizeof(int));
    kernel_argmax_stage2<<<1, stage2Block, smem2, stream>>>(
        blockVals, blockIdxs, result, numBlocks);
    return cudaGetLastError();
}

} // extern "C"
