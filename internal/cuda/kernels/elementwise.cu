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

// ---------- Launcher functions (extern "C" for CGO) ----------

static inline void grid_config(int n, int* grid, int* block) {
    *block = 256;
    *grid = (n + *block - 1) / *block;
}

extern "C" {

cudaError_t launch_add(const float* a, const float* b, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_add<<<grid, block>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sub(const float* a, const float* b, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_sub<<<grid, block>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul(const float* a, const float* b, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul<<<grid, block>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_div(const float* a, const float* b, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_div<<<grid, block>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t launch_pow(const float* base, const float* exp, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_pow<<<grid, block>>>(base, exp, c, n);
    return cudaGetLastError();
}

cudaError_t launch_add_scalar(const float* a, float scalar, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_add_scalar<<<grid, block>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_mul_scalar(const float* a, float scalar, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_mul_scalar<<<grid, block>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_div_scalar(const float* a, float scalar, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_div_scalar<<<grid, block>>>(a, scalar, c, n);
    return cudaGetLastError();
}

cudaError_t launch_exp(const float* a, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_exp<<<grid, block>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_log(const float* a, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_log<<<grid, block>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_sqrt(const float* a, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_sqrt<<<grid, block>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_rsqrt(const float* a, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_rsqrt<<<grid, block>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh(const float* a, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_tanh<<<grid, block>>>(a, c, n);
    return cudaGetLastError();
}

cudaError_t launch_tanh_prime(const float* a, const float* upstream, float* c, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_tanh_prime<<<grid, block>>>(a, upstream, c, n);
    return cudaGetLastError();
}

cudaError_t launch_fill(float* data, float value, int n) {
    int grid, block; grid_config(n, &grid, &block);
    kernel_fill<<<grid, block>>>(data, value, n);
    return cudaGetLastError();
}

} // extern "C"
