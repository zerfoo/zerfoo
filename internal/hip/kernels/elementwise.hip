// elementwise.hip -- HIP kernels for elementwise tensor operations.
// Compiled by hipcc into libhipkernels.a, linked via CGO.
// This is a direct port of the CUDA elementwise.cu kernel source.

#include <math.h>
#include <hip/hip_runtime.h>

// ---------- Binary elementwise (no broadcasting) ----------

__global__ void kernel_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void kernel_sub(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void kernel_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void kernel_div(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / b[idx];
}

__global__ void kernel_pow(const float* base, const float* exp, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = powf(base[idx], exp[idx]);
}

// ---------- Scalar operations ----------

__global__ void kernel_add_scalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + scalar;
}

__global__ void kernel_mul_scalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * scalar;
}

__global__ void kernel_div_scalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] / scalar;
}

// ---------- Unary math ops ----------

__global__ void kernel_exp(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = expf(a[idx]);
}

__global__ void kernel_log(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = logf(a[idx]);
}

__global__ void kernel_sqrt(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = sqrtf(a[idx]);
}

__global__ void kernel_rsqrt(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = rsqrtf(a[idx]);
}

__global__ void kernel_tanh(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = tanhf(a[idx]);
}

// tanh_prime: (1 - tanh(a)^2) * upstream
__global__ void kernel_tanh_prime(const float* a, const float* upstream, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(a[idx]);
        c[idx] = (1.0f - t * t) * upstream[idx];
    }
}

// ---------- Fill ----------

__global__ void kernel_fill(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

// ---------- Softmax (shared-memory reduction) ----------

__global__ void kernel_softmax(const float* input, float* output,
                                int outer, int inner, int axisSize) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        float val = input[base + k * step];
        if (val > local_max) local_max = val;
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        float ex = expf(input[idx] - max_val);
        output[idx] = ex;
        local_sum += ex;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];
    __syncthreads();

    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        output[idx] /= sum_val;
    }
}

// ---------- Sum reduction along axis ----------

__global__ void kernel_sum_axis(const float* input, float* output,
                                int outer, int inner, int axisSize) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        local_sum += input[base + k * step];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[stripe] = sdata[0];
    }
}

// ---------- Launcher functions (extern "C" for CGO) ----------

static inline void grid_config(int n, int* grid, int* block) {
    *block = 256;
    *grid = (n + *block - 1) / *block;
}

extern "C" {

hipError_t launch_add(const float* a, const float* b, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_add, dim3(grid), dim3(block), 0, stream, a, b, c, n);
    return hipGetLastError();
}

hipError_t launch_sub(const float* a, const float* b, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_sub, dim3(grid), dim3(block), 0, stream, a, b, c, n);
    return hipGetLastError();
}

hipError_t launch_mul(const float* a, const float* b, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_mul, dim3(grid), dim3(block), 0, stream, a, b, c, n);
    return hipGetLastError();
}

hipError_t launch_div(const float* a, const float* b, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_div, dim3(grid), dim3(block), 0, stream, a, b, c, n);
    return hipGetLastError();
}

hipError_t launch_pow(const float* base, const float* exp, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_pow, dim3(grid), dim3(block), 0, stream, base, exp, c, n);
    return hipGetLastError();
}

hipError_t launch_add_scalar(const float* a, float scalar, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_add_scalar, dim3(grid), dim3(block), 0, stream, a, scalar, c, n);
    return hipGetLastError();
}

hipError_t launch_mul_scalar(const float* a, float scalar, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_mul_scalar, dim3(grid), dim3(block), 0, stream, a, scalar, c, n);
    return hipGetLastError();
}

hipError_t launch_div_scalar(const float* a, float scalar, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_div_scalar, dim3(grid), dim3(block), 0, stream, a, scalar, c, n);
    return hipGetLastError();
}

hipError_t launch_exp(const float* a, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_exp, dim3(grid), dim3(block), 0, stream, a, c, n);
    return hipGetLastError();
}

hipError_t launch_log(const float* a, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_log, dim3(grid), dim3(block), 0, stream, a, c, n);
    return hipGetLastError();
}

hipError_t launch_sqrt(const float* a, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_sqrt, dim3(grid), dim3(block), 0, stream, a, c, n);
    return hipGetLastError();
}

hipError_t launch_rsqrt(const float* a, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_rsqrt, dim3(grid), dim3(block), 0, stream, a, c, n);
    return hipGetLastError();
}

hipError_t launch_tanh(const float* a, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_tanh, dim3(grid), dim3(block), 0, stream, a, c, n);
    return hipGetLastError();
}

hipError_t launch_tanh_prime(const float* a, const float* upstream, float* c, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_tanh_prime, dim3(grid), dim3(block), 0, stream, a, upstream, c, n);
    return hipGetLastError();
}

hipError_t launch_fill(float* data, float value, int n, hipStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    hipLaunchKernelGGL(kernel_fill, dim3(grid), dim3(block), 0, stream, data, value, n);
    return hipGetLastError();
}

hipError_t launch_sum_axis(const float* input, float* output,
                           int outer, int inner, int axisSize, hipStream_t stream) {
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    hipLaunchKernelGGL(kernel_sum_axis, dim3(numStripes), dim3(block), smem, stream,
                       input, output, outer, inner, axisSize);
    return hipGetLastError();
}

hipError_t launch_softmax(const float* input, float* output,
                          int outer, int inner, int axisSize, hipStream_t stream) {
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    hipLaunchKernelGGL(kernel_softmax, dim3(numStripes), dim3(block), smem, stream,
                       input, output, outer, inner, axisSize);
    return hipGetLastError();
}

} // extern "C"
