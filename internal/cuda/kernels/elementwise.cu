// elementwise.cu -- CUDA kernels for elementwise tensor operations.
// Compiled by nvcc into libkernels.a, linked via CGO.

#include <math.h>
#include <cuda_runtime.h>

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

// ---------- Broadcast binary elementwise ----------
// Supports 2D broadcasting via row/col strides.
// For [M,D] op [1,D]: stride_b_row=0, stride_b_col=1
// For [M,D] op [M,1]: stride_b_row=1, stride_b_col=0
// For [M,D] op [M,D]: stride_b_row=D, stride_b_col=1

__global__ void kernel_add_broadcast(const float* a, const float* b, float* c,
                                      int stride_a_row, int stride_a_col,
                                      int stride_b_row, int stride_b_col,
                                      int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D;
    int col = idx % D;
    c[idx] = a[row * stride_a_row + col * stride_a_col]
           + b[row * stride_b_row + col * stride_b_col];
}

__global__ void kernel_sub_broadcast(const float* a, const float* b, float* c,
                                      int stride_a_row, int stride_a_col,
                                      int stride_b_row, int stride_b_col,
                                      int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D;
    int col = idx % D;
    c[idx] = a[row * stride_a_row + col * stride_a_col]
           - b[row * stride_b_row + col * stride_b_col];
}

__global__ void kernel_mul_broadcast(const float* a, const float* b, float* c,
                                      int stride_a_row, int stride_a_col,
                                      int stride_b_row, int stride_b_col,
                                      int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D;
    int col = idx % D;
    c[idx] = a[row * stride_a_row + col * stride_a_col]
           * b[row * stride_b_row + col * stride_b_col];
}

__global__ void kernel_div_broadcast(const float* a, const float* b, float* c,
                                      int stride_a_row, int stride_a_col,
                                      int stride_b_row, int stride_b_col,
                                      int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D;
    int col = idx % D;
    c[idx] = a[row * stride_a_row + col * stride_a_col]
           / b[row * stride_b_row + col * stride_b_col];
}

// ---------- 4D broadcast binary elementwise ----------
// Supports broadcasting up to 4 dimensions using per-dimension strides.
// For a dimension where a tensor has size 1 (broadcast), set stride to 0.
// Output shape is [d0, d1, d2, d3], total = d0*d1*d2*d3 threads.

__global__ void kernel_add_broadcast4d(const float* a, const float* b, float* c,
                                        int d0, int d1, int d2, int d3,
                                        int sa0, int sa1, int sa2, int sa3,
                                        int sb0, int sb1, int sb2, int sb3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;
    int i3 = idx % d3;
    int tmp = idx / d3;
    int i2 = tmp % d2;
    tmp = tmp / d2;
    int i1 = tmp % d1;
    int i0 = tmp / d1;
    c[idx] = a[i0*sa0 + i1*sa1 + i2*sa2 + i3*sa3]
           + b[i0*sb0 + i1*sb1 + i2*sb2 + i3*sb3];
}

__global__ void kernel_sub_broadcast4d(const float* a, const float* b, float* c,
                                        int d0, int d1, int d2, int d3,
                                        int sa0, int sa1, int sa2, int sa3,
                                        int sb0, int sb1, int sb2, int sb3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;
    int i3 = idx % d3;
    int tmp = idx / d3;
    int i2 = tmp % d2;
    tmp = tmp / d2;
    int i1 = tmp % d1;
    int i0 = tmp / d1;
    c[idx] = a[i0*sa0 + i1*sa1 + i2*sa2 + i3*sa3]
           - b[i0*sb0 + i1*sb1 + i2*sb2 + i3*sb3];
}

__global__ void kernel_mul_broadcast4d(const float* a, const float* b, float* c,
                                        int d0, int d1, int d2, int d3,
                                        int sa0, int sa1, int sa2, int sa3,
                                        int sb0, int sb1, int sb2, int sb3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;
    int i3 = idx % d3;
    int tmp = idx / d3;
    int i2 = tmp % d2;
    tmp = tmp / d2;
    int i1 = tmp % d1;
    int i0 = tmp / d1;
    c[idx] = a[i0*sa0 + i1*sa1 + i2*sa2 + i3*sa3]
           * b[i0*sb0 + i1*sb1 + i2*sb2 + i3*sb3];
}

__global__ void kernel_div_broadcast4d(const float* a, const float* b, float* c,
                                        int d0, int d1, int d2, int d3,
                                        int sa0, int sa1, int sa2, int sa3,
                                        int sb0, int sb1, int sb2, int sb3) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = d0 * d1 * d2 * d3;
    if (idx >= total) return;
    int i3 = idx % d3;
    int tmp = idx / d3;
    int i2 = tmp % d2;
    tmp = tmp / d2;
    int i1 = tmp % d1;
    int i0 = tmp / d1;
    c[idx] = a[i0*sa0 + i1*sa1 + i2*sa2 + i3*sa3]
           / b[i0*sb0 + i1*sb1 + i2*sb2 + i3*sb3];
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

__global__ void kernel_sub_scalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - scalar;
}

__global__ void kernel_pow_scalar(const float* a, float scalar, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Fast path for x^2 (used by ONNX RMSNorm): avoids powf() which
        // returns NaN for negative bases (powf computes exp(s*log(x))).
        if (scalar == 2.0f) {
            c[idx] = a[idx] * a[idx];
        } else {
            c[idx] = powf(fabsf(a[idx]), scalar);
        }
    }
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

__global__ void kernel_sin(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = sinf(a[idx]);
}

__global__ void kernel_cos(const float* a, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = cosf(a[idx]);
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

// Each block handles one (outer, inner) stripe along the softmax axis.
// Uses shared memory for parallel max and sum reductions.
__global__ void kernel_softmax(const float* input, float* output,
                                int outer, int inner, int axisSize) {
    int stripe = blockIdx.x;
    int o = stripe / inner;
    int in_ = stripe % inner;
    int base = o * axisSize * inner + in_;
    int step = inner;

    extern __shared__ float sdata[];

    // Phase 1: Find max along axis for numerical stability
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

    // Phase 2: Compute exp(x - max) and accumulate sum
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

    // Phase 3: Normalize
    for (int k = threadIdx.x; k < axisSize; k += blockDim.x) {
        int idx = base + k * step;
        output[idx] /= sum_val;
    }
}

// ---------- Sum reduction along axis ----------

// Each block handles one (outer, inner) stripe, reducing axisSize elements
// into a single output value.
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
// All launchers accept a cudaStream_t for async execution.
// Pass NULL (0) for the default stream.

static inline void grid_config(int n, int* grid, int* block) {
    *block = 256;
    *grid = (n + *block - 1) / *block;
}

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

cudaError_t launch_add_broadcast(const float* a, const float* b, float* c,
                                  int stride_a_row, int stride_a_col,
                                  int stride_b_row, int stride_b_col,
                                  int M, int D, cudaStream_t stream) {
    int n = M * D;
    int grid, block; grid_config(n, &grid, &block);
    kernel_add_broadcast<<<grid, block, 0, stream>>>(a, b, c,
        stride_a_row, stride_a_col, stride_b_row, stride_b_col, M, D);
    return cudaGetLastError();
}

cudaError_t launch_sub_broadcast(const float* a, const float* b, float* c,
                                  int stride_a_row, int stride_a_col,
                                  int stride_b_row, int stride_b_col,
                                  int M, int D, cudaStream_t stream) {
    int n = M * D;
    int grid, block; grid_config(n, &grid, &block);
    kernel_sub_broadcast<<<grid, block, 0, stream>>>(a, b, c,
        stride_a_row, stride_a_col, stride_b_row, stride_b_col, M, D);
    return cudaGetLastError();
}

cudaError_t launch_mul_broadcast(const float* a, const float* b, float* c,
                                  int stride_a_row, int stride_a_col,
                                  int stride_b_row, int stride_b_col,
                                  int M, int D, cudaStream_t stream) {
    int n = M * D;
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul_broadcast<<<grid, block, 0, stream>>>(a, b, c,
        stride_a_row, stride_a_col, stride_b_row, stride_b_col, M, D);
    return cudaGetLastError();
}

cudaError_t launch_div_broadcast(const float* a, const float* b, float* c,
                                  int stride_a_row, int stride_a_col,
                                  int stride_b_row, int stride_b_col,
                                  int M, int D, cudaStream_t stream) {
    int n = M * D;
    int grid, block; grid_config(n, &grid, &block);
    kernel_div_broadcast<<<grid, block, 0, stream>>>(a, b, c,
        stride_a_row, stride_a_col, stride_b_row, stride_b_col, M, D);
    return cudaGetLastError();
}

cudaError_t launch_add_broadcast4d(const float* a, const float* b, float* c,
                                    int d0, int d1, int d2, int d3,
                                    int sa0, int sa1, int sa2, int sa3,
                                    int sb0, int sb1, int sb2, int sb3,
                                    cudaStream_t stream) {
    int n = d0 * d1 * d2 * d3;
    int grid, block; grid_config(n, &grid, &block);
    kernel_add_broadcast4d<<<grid, block, 0, stream>>>(a, b, c, d0, d1, d2, d3,
        sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
    return cudaGetLastError();
}

cudaError_t launch_sub_broadcast4d(const float* a, const float* b, float* c,
                                    int d0, int d1, int d2, int d3,
                                    int sa0, int sa1, int sa2, int sa3,
                                    int sb0, int sb1, int sb2, int sb3,
                                    cudaStream_t stream) {
    int n = d0 * d1 * d2 * d3;
    int grid, block; grid_config(n, &grid, &block);
    kernel_sub_broadcast4d<<<grid, block, 0, stream>>>(a, b, c, d0, d1, d2, d3,
        sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
    return cudaGetLastError();
}

cudaError_t launch_mul_broadcast4d(const float* a, const float* b, float* c,
                                    int d0, int d1, int d2, int d3,
                                    int sa0, int sa1, int sa2, int sa3,
                                    int sb0, int sb1, int sb2, int sb3,
                                    cudaStream_t stream) {
    int n = d0 * d1 * d2 * d3;
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul_broadcast4d<<<grid, block, 0, stream>>>(a, b, c, d0, d1, d2, d3,
        sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
    return cudaGetLastError();
}

cudaError_t launch_div_broadcast4d(const float* a, const float* b, float* c,
                                    int d0, int d1, int d2, int d3,
                                    int sa0, int sa1, int sa2, int sa3,
                                    int sb0, int sb1, int sb2, int sb3,
                                    cudaStream_t stream) {
    int n = d0 * d1 * d2 * d3;
    int grid, block; grid_config(n, &grid, &block);
    kernel_div_broadcast4d<<<grid, block, 0, stream>>>(a, b, c, d0, d1, d2, d3,
        sa0, sa1, sa2, sa3, sb0, sb1, sb2, sb3);
    return cudaGetLastError();
}

cudaError_t launch_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_add<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sub(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_sub<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_div(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_div<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_pow(const float* base, const float* exp, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_pow<<<grid, block, 0, stream>>>(base, exp, c, n);
    return cudaGetLastError();
}

// Scalar launchers accept float as uint32 bits (IEEE 754) because the
// purego ccall convention passes all arguments through integer registers.

cudaError_t launch_add_scalar(const float* a, unsigned int scalar_bits, float* c, int n, cudaStream_t stream) {
    float scalar = bits_to_float(scalar_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_add_scalar<<<grid, block, 0, stream>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul_scalar(const float* a, unsigned int scalar_bits, float* c, int n, cudaStream_t stream) {
    float scalar = bits_to_float(scalar_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul_scalar<<<grid, block, 0, stream>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_div_scalar(const float* a, unsigned int scalar_bits, float* c, int n, cudaStream_t stream) {
    float scalar = bits_to_float(scalar_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_div_scalar<<<grid, block, 0, stream>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sub_scalar(const float* a, unsigned int scalar_bits, float* c, int n, cudaStream_t stream) {
    float scalar = bits_to_float(scalar_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_sub_scalar<<<grid, block, 0, stream>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_pow_scalar(const float* a, unsigned int scalar_bits, float* c, int n, cudaStream_t stream) {
    float scalar = bits_to_float(scalar_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_pow_scalar<<<grid, block, 0, stream>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_exp(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_exp<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_log(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_log<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sqrt(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_sqrt<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_rsqrt(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_rsqrt<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sin(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_sin<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_cos(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_cos<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh(const float* a, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_tanh<<<grid, block, 0, stream>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_prime(const float* a, const float* upstream, float* c, int n, cudaStream_t stream) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_tanh_prime<<<grid, block, 0, stream>>>(a, upstream, c, n);
    return cudaGetLastError();
}

cudaError_t launch_fill(float* data, unsigned int value_bits, int n, cudaStream_t stream) {
    float value = bits_to_float(value_bits);
    int grid, block; grid_config(n, &grid, &block);
    kernel_fill<<<grid, block, 0, stream>>>(data, value, n);
    return cudaGetLastError();
}

cudaError_t launch_sum_axis(const float* input, float* output,
                            int outer, int inner, int axisSize, cudaStream_t stream) {
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    kernel_sum_axis<<<numStripes, block, smem, stream>>>(input, output, outer, inner, axisSize);
    return cudaGetLastError();
}

cudaError_t launch_softmax(const float* input, float* output,
                           int outer, int inner, int axisSize, cudaStream_t stream) {
    // Block size: next power of 2 up to min(axisSize, 256)
    int block = 1;
    while (block < axisSize && block < 256) block <<= 1;
    int numStripes = outer * inner;
    size_t smem = block * sizeof(float);
    kernel_softmax<<<numStripes, block, smem, stream>>>(input, output, outer, inner, axisSize);
    return cudaGetLastError();
}

// ---------- Repeat along axis ----------
// Repeats a tensor along a given axis. The tensor is viewed as
// [outerSize, axisDim, innerSize] and the output is
// [outerSize, axisDim * reps, innerSize].
// Each thread copies one element from src to the correct position in dst.
__global__ void kernel_repeat(const float* src, float* dst,
                              int outerSize, int axisDim, int innerSize, int reps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * axisDim * reps * innerSize;
    if (idx >= total) return;

    // Decompose flat index in output to (outer, repAxis, inner)
    int inner = idx % innerSize;
    int temp = idx / innerSize;
    int repAxis = temp % (axisDim * reps);
    int outer = temp / (axisDim * reps);

    // Map to source: strip out the repetition
    int srcAxis = repAxis % axisDim;
    int srcIdx = (outer * axisDim + srcAxis) * innerSize + inner;
    dst[idx] = src[srcIdx];
}

cudaError_t launch_repeat(const float* src, float* dst,
                          int outerSize, int axisDim, int innerSize, int reps,
                          cudaStream_t stream) {
    int total = outerSize * axisDim * reps * innerSize;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_repeat<<<blocks, threads, 0, stream>>>(src, dst, outerSize, axisDim, innerSize, reps);
    return cudaGetLastError();
}

} // extern "C"
