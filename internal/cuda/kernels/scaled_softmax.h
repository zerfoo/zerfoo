/* Fused scaled softmax kernel interface.
 *
 * Computes output = softmax(input * scale) in one kernel launch,
 * replacing the MulScalar + Softmax chain.
 */
#ifndef SCALED_SOFTMAX_H
#define SCALED_SOFTMAX_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* scaled_softmax_f32 applies scaled softmax in one kernel launch.
 *
 * input:      device pointer to float array.
 * output:     device pointer to float array (same size as input).
 * outer:      product of dimensions before the softmax axis.
 * inner:      product of dimensions after the softmax axis.
 * axisSize:   size of the softmax axis.
 * scale_bits: scale factor as IEEE 754 uint32 bits.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t scaled_softmax_f32(
    const float* input, float* output,
    int outer, int inner, int axisSize,
    unsigned int scale_bits,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* SCALED_SOFTMAX_H */
