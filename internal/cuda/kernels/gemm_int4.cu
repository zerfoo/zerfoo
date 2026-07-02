/* INT4 mixed-precision GEMM kernel (packed INT4 weights * FP32 activations -> FP32).
 *
 * Weights are stored as packed INT4 (two values per byte, low nibble first)
 * with per-group scale factors and zero points. The kernel dequantizes each
 * weight value on the fly: float_val = (int4_val - zero) * scale.
 *
 * Uses tiled computation to reduce global memory traffic. Each thread
 * accumulates a dot product for one output element, iterating over K.
 */

#include "gemm_int4.h"
#include <stdint.h>

#define TILE_N_I4 32
#define TILE_M_I4 32

/* Extract INT4 value at position k from packed byte array.
 * Two values per byte: even k -> low nibble, odd k -> high nibble.
 * Returns signed value in [-8, 7] range. */
__device__ __forceinline__ int extract_int4(const uint8_t* packed, int idx)
{
    uint8_t byte = packed[idx >> 1];
    int val = (idx & 1) ? (byte >> 4) : (byte & 0x0F);
    /* Sign extend from 4-bit: if bit 3 is set, value is negative. */
    if (val >= 8) val -= 16;
    return val;
}

__global__ void gemm_int4_kernel(
    const uint8_t* __restrict__ A,
    const float*   __restrict__ B,
    float*         __restrict__ C,
    const float*   __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    int M, int K, int N, int group_size)
{
    int row = blockIdx.y * TILE_M_I4 + threadIdx.y;
    int col = blockIdx.x * TILE_N_I4 + threadIdx.x;

    if (row >= M || col >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    float acc = 0.0f;

    /* Iterate over K, dequantizing INT4 weights per group. */
    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        float scale = scales[row * num_groups + group_idx];
        int zero = (int)zeros[row * num_groups + group_idx];

        int a_val = extract_int4(A + row * (K / 2), k);
        float dequant = (float)(a_val - zero) * scale;

        acc += dequant * B[k * N + col];
    }

    C[row * N + col] = acc;
}

extern "C" cudaError_t gemm_int4_f32(
    const void* A, const float* B, float* C,
    const float* scales, const void* zeros,
    int M, int K, int N, int group_size,
    cudaStream_t stream)
{
    if (K % 2 != 0) {
        return cudaErrorInvalidValue;
    }
    if (group_size <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(TILE_N_I4, TILE_M_I4);
    dim3 grid((N + TILE_N_I4 - 1) / TILE_N_I4, (M + TILE_M_I4 - 1) / TILE_M_I4);

    gemm_int4_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)A, B, C, scales, (const uint8_t*)zeros,
        M, K, N, group_size);

    return cudaGetLastError();
}

/* Right-multiply kernel: C[i,j] = sum_k B[i,k] * dequant(W[k,j])
 * W is [in_features, out_features/2] packed INT4 (row-major).
 * B is [batch, in_features] FP32 (row-major).
 * C is [batch, out_features] FP32 (row-major).
 * Quantization is along the out_features (column) dimension of W.
 */
__global__ void gemm_int4_rmul_kernel(
    const uint8_t* __restrict__ W,
    const float*   __restrict__ B,
    float*         __restrict__ C,
    const float*   __restrict__ scales,
    const uint8_t* __restrict__ zeros,
    int batch, int in_features, int out_features, int group_size)
{
    int row = blockIdx.y * TILE_M_I4 + threadIdx.y;  /* batch index */
    int col = blockIdx.x * TILE_N_I4 + threadIdx.x;  /* output feature */

    if (row >= batch || col >= out_features) return;

    int num_groups = (out_features + group_size - 1) / group_size;
    float acc = 0.0f;

    int col_group = col / group_size;

    for (int k = 0; k < in_features; k++) {
        /* W[k, col]: row k, column col in the unpacked weight matrix.
         * Packed at W + k * (out_features / 2), position col. */
        float scale = scales[k * num_groups + col_group];
        int zero = (int)zeros[k * num_groups + col_group];

        int w_val = extract_int4(W + k * (out_features / 2), col);
        float dequant = (float)(w_val - zero) * scale;

        acc += B[row * in_features + k] * dequant;
    }

    C[row * out_features + col] = acc;
}

extern "C" cudaError_t gemm_int4_f32_rmul(
    const void* W, const float* B, float* C,
    const float* scales, const void* zeros,
    int batch, int in_features, int out_features, int group_size,
    cudaStream_t stream)
{
    if (out_features % 2 != 0) {
        return cudaErrorInvalidValue;
    }
    if (group_size <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(TILE_N_I4, TILE_M_I4);
    dim3 grid((out_features + TILE_N_I4 - 1) / TILE_N_I4,
              (batch + TILE_M_I4 - 1) / TILE_M_I4);

    gemm_int4_rmul_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)W, B, C, scales, (const uint8_t*)zeros,
        batch, in_features, out_features, group_size);

    return cudaGetLastError();
}
