// elementwise_fp16.cu -- FP16 CUDA kernels for elementwise tensor operations.
// Uses __half/__half2 types from cuda_fp16.h for 2-wide SIMD processing.
// Reductions (rmsnorm, scaled_softmax) accumulate in FP32 for precision.

#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------- Binary elementwise (no broadcasting, __half2 SIMD) ----------

__global__ void kernel_add_fp16(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* c2 = reinterpret_cast<__half2*>(c);
        c2[idx] = __hadd2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hadd(a[idx2], b[idx2]);
    }
}

__global__ void kernel_sub_fp16(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* c2 = reinterpret_cast<__half2*>(c);
        c2[idx] = __hsub2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hsub(a[idx2], b[idx2]);
    }
}

__global__ void kernel_mul_fp16(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* c2 = reinterpret_cast<__half2*>(c);
        c2[idx] = __hmul2(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hmul(a[idx2], b[idx2]);
    }
}

__global__ void kernel_div_fp16(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* c2 = reinterpret_cast<__half2*>(c);
        c2[idx] = __h2div(a2[idx], b2[idx]);
    } else if (idx2 < n) {
        c[idx2] = __hdiv(a[idx2], b[idx2]);
    }
}

// ---------- RMSNorm FP16 ----------
// input/output/weight are __half, but accumulation uses FP32 for precision.
// output[row] = input[row] * rsqrt(mean(input[row]^2) + eps) * weight

__global__ void kernel_rmsnorm_fp16(const __half* __restrict__ input,
                                     const __half* __restrict__ weight,
                                     __half* __restrict__ output,
                                     float eps, int D) {
    int row = blockIdx.x;
    const __half* x = input + row * D;
    __half* y = output + row * D;

    extern __shared__ float sdata[];

    // Phase 1: Compute partial sum of squares in FP32.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(x[i]);
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();

    // Parallel reduction: shared memory for inter-warp, then warp shuffle.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float val = sdata[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) sdata[0] = val;
    }
    __syncthreads();

    // Compute scale = rsqrt(mean_sq + eps) in FP32.
    float scale = rsqrtf(sdata[0] / (float)D + eps);

    // Phase 2: Normalize and scale by weight.
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(x[i]) * scale * __half2float(weight[i]);
        y[i] = __float2half(v);
    }
}

// ---------- Scaled Softmax FP16 ----------
// input/output are __half, reductions in FP32 for precision.
// output = softmax(input * scale)

__global__ void kernel_scaled_softmax_fp16(const __half* input, __half* output,
                                            int outer, int inner, int axisSize,
                                            float scale) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    // Phase 1: Find max along axis (FP32 accumulation)
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        float val = __half2float(input[base + k * step]) * scale;
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
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

    // Phase 2: Compute exp(x * scale - max) and sum (FP32 accumulation)
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        float ex = expf(__half2float(input[idx]) * scale - max_val);
        output[idx] = __float2half(ex);
        local_sum += ex;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
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
        float v = __half2float(output[idx]) / sum_val;
        output[idx] = __float2half(v);
    }
}

// ---------- Launcher helpers ----------

static inline void grid_config_fp16(int n, int* grid, int* block) {
    *block = 256;
    // Each thread handles 2 elements for binary ops
    *grid = ((n + 1) / 2 + *block - 1) / *block;
}

static inline float bits_to_float_fp16(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Launcher functions (extern "C") ----------

extern "C" {

cudaError_t launch_add_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_fp16(n, &grid, &block);
    kernel_add_fp16<<<grid, block, 0, stream>>>(
        (const __half*)a, (const __half*)b, (__half*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_sub_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_fp16(n, &grid, &block);
    kernel_sub_fp16<<<grid, block, 0, stream>>>(
        (const __half*)a, (const __half*)b, (__half*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_fp16(n, &grid, &block);
    kernel_mul_fp16<<<grid, block, 0, stream>>>(
        (const __half*)a, (const __half*)b, (__half*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_div_fp16(const void* a, const void* b, void* c, int n, cudaStream_t stream) {
    int grid, block;
    grid_config_fp16(n, &grid, &block);
    kernel_div_fp16<<<grid, block, 0, stream>>>(
        (const __half*)a, (const __half*)b, (__half*)c, n);
    return cudaGetLastError();
}

cudaError_t launch_rmsnorm_fp16(const void* input, const void* weight,
                                 void* output, unsigned int eps_bits,
                                 int rows, int D, cudaStream_t stream) {
    float eps = bits_to_float_fp16(eps_bits);
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_rmsnorm_fp16<<<rows, block, smem, stream>>>(
        (const __half*)input, (const __half*)weight, (__half*)output, eps, D);
    return cudaGetLastError();
}

cudaError_t launch_scaled_softmax_fp16(const void* input, void* output,
                                        int outer, int inner, int axisSize,
                                        unsigned int scale_bits,
                                        cudaStream_t stream) {
    float scale = bits_to_float_fp16(scale_bits);
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    kernel_scaled_softmax_fp16<<<numStripes, block, smem, stream>>>(
        (const __half*)input, (__half*)output, outer, inner, axisSize, scale);
    return cudaGetLastError();
}

} // extern "C"

// ---------- F32 <-> FP16 conversion kernels ----------

__global__ void kernel_f32_to_fp16(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void kernel_fp16_to_f32(const __half* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

extern "C" {

cudaError_t launch_f32_to_fp16(const void* src, void* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_f32_to_fp16<<<grid, block, 0, stream>>>((const float*)src, (__half*)dst, n);
    return cudaGetLastError();
}

cudaError_t launch_fp16_to_f32(const void* src, void* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_fp16_to_f32<<<grid, block, 0, stream>>>((const __half*)src, (float*)dst, n);
    return cudaGetLastError();
}

} // extern "C"
