// fp8_ops.cu -- FP8 E4M3 dequantize-on-load kernels.
// Reads FP8 E4M3 from global memory, converts to FP16 in registers,
// computes element-wise ops in FP16, writes FP16 output.
// Compiled by nvcc into libkernels.a.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string.h>

// ---------- FP8 E4M3 decode (manual bit manipulation) ----------
// E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa bits.
// All 256 bit patterns are finite (no infinity/NaN). Max value: 448.

__device__ __forceinline__ __half fp8e4m3_to_fp16(unsigned char b) {
    unsigned int sign = (b >> 7) & 1;
    int exp = (b >> 3) & 0x0F;
    unsigned int mant = b & 0x07;

    if (exp == 0 && mant == 0) {
        // +-0: return signed zero in FP16
        unsigned short bits = (unsigned short)(sign << 15);
        return __ushort_as_half(bits);
    }
    if (exp == 0) {
        // Subnormal E4M3: value = (-1)^sign * 2^(1-7) * (mant/8)
        // = (-1)^sign * mant * 2^(-9)
        // FP16 can represent this as a normal or subnormal.
        float val = (float)mant / 8.0f * (1.0f / 64.0f); // 2^(1-7) = 1/64
        if (sign) val = -val;
        return __float2half(val);
    }

    // Normal E4M3: unbiased_exp = exp - 7
    // Map to FP16: bias=15, so fp16_exp = exp - 7 + 15 = exp + 8
    int fp16_exp = exp + 8;
    // FP16: 1 sign, 5 exponent, 10 mantissa
    // E4M3 mantissa is 3 bits; shift left by 7 to fill 10-bit mantissa field.
    unsigned short fp16_mant = (unsigned short)(mant << 7);
    unsigned short bits = (unsigned short)((sign << 15) | (fp16_exp << 10) | fp16_mant);
    return __ushort_as_half(bits);
}

// ---------- bits_to_float ----------

static inline float bits_to_float(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ---------- Dequantize kernel: FP8 E4M3 -> FP16 ----------
// out[i] = fp8_to_fp16(in[i]) * scale

__global__ void kernel_dequant_fp8e4m3_to_fp16(
    const unsigned char* __restrict__ input,
    __half* __restrict__ output,
    __half scale,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half val = fp8e4m3_to_fp16(input[idx]);
        output[idx] = __hmul(val, scale);
    }
}

// ---------- FP8 Add: dequant both inputs, add in FP16 ----------
// out[i] = fp8_to_fp16(a[i]) * scale_a + fp8_to_fp16(b[i]) * scale_b

__global__ void kernel_fp8_add(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    __half* __restrict__ c,
    __half scale_a,
    __half scale_b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half va = __hmul(fp8e4m3_to_fp16(a[idx]), scale_a);
        __half vb = __hmul(fp8e4m3_to_fp16(b[idx]), scale_b);
        c[idx] = __hadd(va, vb);
    }
}

// ---------- FP8 Mul: dequant both inputs, multiply in FP16 ----------
// out[i] = fp8_to_fp16(a[i]) * scale_a * fp8_to_fp16(b[i]) * scale_b

__global__ void kernel_fp8_mul(
    const unsigned char* __restrict__ a,
    const unsigned char* __restrict__ b,
    __half* __restrict__ c,
    __half scale_a,
    __half scale_b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half va = __hmul(fp8e4m3_to_fp16(a[idx]), scale_a);
        __half vb = __hmul(fp8e4m3_to_fp16(b[idx]), scale_b);
        c[idx] = __hmul(va, vb);
    }
}

// ---------- FP8 RMSNorm: dequant input, compute in FP32 accum, output FP16 ----------
// Each block handles one row of length D.
// output[row,i] = (x[i] * rsqrt(mean(x^2) + eps)) * weight[i]
// where x[i] = fp8_to_fp16(input[row*D+i]) * scale

__global__ void kernel_fp8_rmsnorm(
    const unsigned char* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    __half scale,
    float eps,
    int D)
{
    int row = blockIdx.x;
    const unsigned char* x_fp8 = input + row * D;
    __half* y = output + row * D;

    extern __shared__ float sdata[];

    // Phase 1: Dequant + compute partial sum of squares in FP32.
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(__hmul(fp8e4m3_to_fp16(x_fp8[i]), scale));
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();

    // Parallel reduction.
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

    float rms_scale = rsqrtf(sdata[0] / (float)D + eps);

    // Phase 2: Normalize and scale by weight, output FP16.
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(__hmul(fp8e4m3_to_fp16(x_fp8[i]), scale));
        y[i] = __float2half(v * rms_scale * __half2float(weight[i]));
    }
}

// ---------- Grid configuration ----------

static inline void grid_config(int n, int* grid, int* block) {
    *block = 256;
    *grid = (n + *block - 1) / *block;
}

// ---------- Launcher functions (extern "C") ----------

extern "C" {

cudaError_t launch_dequant_fp8e4m3_to_fp16(
    const void* input, void* output,
    unsigned int scale_bits, int n, cudaStream_t stream)
{
    float scale_f = bits_to_float(scale_bits);
    __half scale_h = __float2half(scale_f);
    int grid, block;
    grid_config(n, &grid, &block);
    kernel_dequant_fp8e4m3_to_fp16<<<grid, block, 0, stream>>>(
        (const unsigned char*)input, (__half*)output, scale_h, n);
    return cudaGetLastError();
}

cudaError_t launch_fp8_add(
    const void* a, const void* b, void* c,
    unsigned int scale_a_bits, unsigned int scale_b_bits,
    int n, cudaStream_t stream)
{
    __half scale_a = __float2half(bits_to_float(scale_a_bits));
    __half scale_b = __float2half(bits_to_float(scale_b_bits));
    int grid, block;
    grid_config(n, &grid, &block);
    kernel_fp8_add<<<grid, block, 0, stream>>>(
        (const unsigned char*)a, (const unsigned char*)b, (__half*)c,
        scale_a, scale_b, n);
    return cudaGetLastError();
}

cudaError_t launch_fp8_mul(
    const void* a, const void* b, void* c,
    unsigned int scale_a_bits, unsigned int scale_b_bits,
    int n, cudaStream_t stream)
{
    __half scale_a = __float2half(bits_to_float(scale_a_bits));
    __half scale_b = __float2half(bits_to_float(scale_b_bits));
    int grid, block;
    grid_config(n, &grid, &block);
    kernel_fp8_mul<<<grid, block, 0, stream>>>(
        (const unsigned char*)a, (const unsigned char*)b, (__half*)c,
        scale_a, scale_b, n);
    return cudaGetLastError();
}

cudaError_t launch_fp8_rmsnorm(
    const void* input, const void* weight, void* output,
    unsigned int scale_bits, unsigned int eps_bits,
    int rows, int D, cudaStream_t stream)
{
    __half scale_h = __float2half(bits_to_float(scale_bits));
    float eps = bits_to_float(eps_bits);
    int block = 1;
    while (block < D && block < 256) block <<= 1;
    size_t smem = block * sizeof(float);
    kernel_fp8_rmsnorm<<<rows, block, smem, stream>>>(
        (const unsigned char*)input, (const __half*)weight, (__half*)output,
        scale_h, eps, D);
    return cudaGetLastError();
}

} // extern "C"
