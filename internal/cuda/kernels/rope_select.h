/* GPU-indexed RoPE table selection kernel interface.
 *
 * Reads counter[0] to compute offset and copies halfRotary cos/sin values
 * from the precomputed table.
 */
#ifndef ROPE_SELECT_H
#define ROPE_SELECT_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* launch_rope_select copies halfRotary cos/sin values from the precomputed
 * table at position counter[0].
 *
 * cos_table:  device pointer to precomputed cos table [positions * halfRotary].
 * sin_table:  device pointer to precomputed sin table [positions * halfRotary].
 * cos_out:    device pointer to [halfRotary] float output for cos.
 * sin_out:    device pointer to [halfRotary] float output for sin.
 * counter:    device pointer to single int32 position counter.
 * halfRotary: number of elements per position slice.
 * stream:     CUDA stream (pass NULL for default stream).
 */
cudaError_t launch_rope_select(
    const float* cos_table, const float* sin_table,
    float* cos_out, float* sin_out,
    const int* counter, int halfRotary,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* ROPE_SELECT_H */
