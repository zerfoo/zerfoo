// offset_memcpy.cu -- CUDA kernel that copies dim floats from src to dst
// at an offset determined by a GPU-resident counter.
// Used for GPU-driven KV cache append without CPU readback.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// kernel_offset_memcpy: dst[counter[0] * dim + idx] = src[idx]
// Single-thread read of counter, then dim threads do the copy.
__global__ void kernel_offset_memcpy(float* __restrict__ dst,
                                      const float* __restrict__ src,
                                      const int* __restrict__ counter,
                                      int dim, int maxSeqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    int pos = counter[0];
    if (pos >= maxSeqLen) return;  // bounds safety
    dst[pos * dim + idx] = src[idx];
}

extern "C" {

cudaError_t launch_offset_memcpy(float* dst, const float* src,
                                  const int* counter, int dim,
                                  int maxSeqLen, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    kernel_offset_memcpy<<<grid, block, 0, stream>>>(
        dst, src, counter, dim, maxSeqLen);
    return cudaGetLastError();
}

} // extern "C"

// FP16 variant: converts F32 src to FP16 during offset copy
__global__ void kernel_offset_memcpy_fp16(__half* __restrict__ dst,
                                          const float* __restrict__ src,
                                          const int* __restrict__ counter,
                                          int dim, int maxSeqLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    int pos = counter[0];
    if (pos >= maxSeqLen) return;  // bounds safety
    dst[pos * dim + idx] = __float2half(src[idx]);
}

extern "C" {

cudaError_t launch_offset_memcpy_fp16(void* dst, const float* src,
                                       const int* counter, int dim,
                                       int maxSeqLen, cudaStream_t stream) {
    int block = 256;
    int grid = (dim + block - 1) / block;
    kernel_offset_memcpy_fp16<<<grid, block, 0, stream>>>(
        (__half*)dst, src, counter, dim, maxSeqLen);
    return cudaGetLastError();
}

} // extern "C"
