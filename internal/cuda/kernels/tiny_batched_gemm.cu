// tiny_batched_gemm.cu -- custom small-matrix strided-batched GEMM for f32.
//
// Motivation (ADR 075 lever L3): cuBLAS SgemmStridedBatched routes tiny
// matrices (e.g. CrossAsset attention's 12x12 Q@K^T and 12x64 weights@V over
// batch = B*heads = 1024) through a GEMV + split-K reduction fan-out
// (gemvNSP_kernel + splitKreduce_kernel, the T11.0c fingerprint): 2-3 internal
// kernels per logical GEMM, none of which tile efficiently for m,n,k <= ~64.
// This kernel computes one strided-batched GEMM in ONE launch with one CUDA
// thread-block per batch element, staging the A[m,k] and B[k,n] tiles in shared
// memory and writing the full C[m,n] tile cooperatively. It is GENERAL: any
// small batched matmul (m,n,k all small, batch large) benefits.
//
// Semantics match SgemmStridedBatched with alpha=1, beta=0, row-major operands:
//   C_b[i,j] = sum_l A_b[i,l] * B_b[l,j],   A_b = A + b*strideA (elements), etc.
// Accumulation is in f32, matching cuBLAS Sgemm (the reference path this
// replaces) so the result is bit-comparable within f32 GEMM tolerance.
//
// Dispatch guard (host side, gpu_engine.go): used only when m,n,k are all
// <= TINY_GEMM_MAX_DIM and batch > 1; otherwise the cuBLAS path stays.

#include <cuda_runtime.h>

// Max supported dimension per side. The shared-memory tiles are sized for this
// bound: A (m*k) + B (k*n) f32 floats. At 64 that is 64*64*2*4 = 32 KiB, within
// the 48 KiB default dynamic-smem budget on sm_121 (GB10). The host launcher
// also enforces this; the kernel guards defensively.
#define TINY_GEMM_MAX_DIM 64

// kernel_tiny_batched_gemm: one block per batch element. Threads cooperatively
// load A_b (m x k) and B_b (k x n) into shared memory, then each thread owns a
// strided subset of the m*n output cells and computes its dot products.
//
// blockDim.x threads span the m*n output cells in a grid-stride loop within the
// block, so the kernel works for any (m,n) <= TINY_GEMM_MAX_DIM regardless of
// the chosen block size.
__global__ void kernel_tiny_batched_gemm(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int m, int n, int k,
                                         long long strideA,
                                         long long strideB,
                                         long long strideC) {
    int b = blockIdx.x;

    const float* Ab = A + (long long)b * strideA;
    const float* Bb = B + (long long)b * strideB;
    float* Cb = C + (long long)b * strideC;

    // Shared tiles for this batch element's A and B.
    __shared__ float sA[TINY_GEMM_MAX_DIM * TINY_GEMM_MAX_DIM];
    __shared__ float sB[TINY_GEMM_MAX_DIM * TINY_GEMM_MAX_DIM];

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Cooperative load of A (m*k) and B (k*n) into shared memory.
    int aElems = m * k;
    for (int idx = tid; idx < aElems; idx += nthreads) {
        sA[idx] = Ab[idx];
    }
    int bElems = k * n;
    for (int idx = tid; idx < bElems; idx += nthreads) {
        sB[idx] = Bb[idx];
    }
    __syncthreads();

    // Each thread computes a strided subset of the m*n output cells.
    int cElems = m * n;
    for (int idx = tid; idx < cElems; idx += nthreads) {
        int i = idx / n;   // row in [0,m)
        int j = idx % n;   // col in [0,n)
        float acc = 0.0f;
        const float* aRow = &sA[i * k];
        // sB is row-major [k, n]; element (l, j) is sB[l*n + j].
        #pragma unroll 4
        for (int l = 0; l < k; ++l) {
            acc += aRow[l] * sB[l * n + j];
        }
        Cb[i * n + j] = acc;
    }
}

extern "C" {

// tiny_batched_gemm_f32 launches one strided-batched GEMM. All arguments cross
// as integers / pointers -- there are NO floating-point scalars, so the purego
// integer-register ABI (see fused_adamw.cu ABI note) is satisfied trivially.
// alpha is fixed at 1 and beta at 0 to match the attention MatMul this serves.
//
// Returns cudaErrorInvalidValue if any dimension exceeds TINY_GEMM_MAX_DIM so
// the host can fall back to cuBLAS rather than silently truncating.
cudaError_t tiny_batched_gemm_f32(const float* A, const float* B, float* C,
                                  int m, int n, int k,
                                  long long strideA, long long strideB,
                                  long long strideC,
                                  int batch, cudaStream_t stream) {
    if (m <= 0 || n <= 0 || k <= 0 || batch <= 0) {
        return cudaErrorInvalidValue;
    }
    if (m > TINY_GEMM_MAX_DIM || n > TINY_GEMM_MAX_DIM || k > TINY_GEMM_MAX_DIM) {
        return cudaErrorInvalidValue;
    }

    // Block size: cap at 256, but no fewer than the output tile so small tiles
    // still parallelize. Round up to a warp multiple for occupancy.
    int cells = m * n;
    int block = cells < 256 ? cells : 256;
    block = ((block + 31) / 32) * 32;
    if (block < 32) block = 32;
    if (block > 1024) block = 1024;

    dim3 grid(batch);
    dim3 threads(block);
    kernel_tiny_batched_gemm<<<grid, threads, 0, stream>>>(
        A, B, C, m, n, k, strideA, strideB, strideC);
    return cudaGetLastError();
}

} // extern "C"
