// scaled_softmax.cu -- CUDA kernel for fused scaled softmax.
// Computes output = softmax(input * scale) in a single pass,
// eliminating the MulScalar + Softmax kernel pair (2 launches -> 1).
//
// input shape:  [outer, axisSize, inner] (logical view)
// output shape: [outer, axisSize, inner]

#include <math.h>
#include <cuda_runtime.h>

// Each block handles one (outer, inner) stripe along the softmax axis.
// Uses shared memory for inter-warp reductions and warp shuffle for the
// final intra-warp steps to reduce shared memory traffic.
__global__ void kernel_scaled_softmax(const float* input, float* output,
                                       int outer, int inner, int axisSize,
                                       float scale) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    // Phase 1: Find max along axis for numerical stability (after scaling)
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        float val = input[base + k * step] * scale;
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // Inter-warp max reduction via shared memory.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Final warp max reduction using __shfl_down_sync.
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, val, offset);
            if (other > val) val = other;
        }
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float max_val = sdata[0];
    __syncthreads();

    // Phase 2: Compute exp(x * scale - max) and accumulate sum
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        float ex = expf(input[idx] * scale - max_val);
        output[idx] = ex;
        local_sum += ex;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Inter-warp sum reduction via shared memory.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Final warp sum reduction using __shfl_down_sync.
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();
    float sum_val = sdata[0];
    __syncthreads();

    // Phase 3: Normalize
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        output[idx] /= sum_val;
    }
}

// ---------- Launcher function (extern "C" for CGO / purego) ----------

// bits_to_float reinterprets a uint32 bit pattern as float32.
// Used because the purego/ccall calling convention passes all arguments
// through integer registers. Float parameters sent as IEEE 754 bits
// must be reinterpreted before use.
static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

extern "C" {

cudaError_t scaled_softmax_f32(const float* input, float* output,
                                int outer, int inner, int axisSize,
                                unsigned int scale_bits,
                                cudaStream_t stream) {
    float scale = bits_to_float(scale_bits);
    // Block size: next power of 2 up to min(axisSize, 256)
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    kernel_scaled_softmax<<<numStripes, block, smem, stream>>>(
        input, output, outer, inner, axisSize, scale);
    return cudaGetLastError();
}

} // extern "C"
