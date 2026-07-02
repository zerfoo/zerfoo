// counter.cu -- CUDA kernels for GPU-resident counter operations.
// Used to track decode position on GPU without CPU involvement,
// enabling CUDA graph capture.

#include <cuda_runtime.h>

// Single-thread kernel that atomically increments a GPU-resident int32.
__global__ void kernel_increment_counter(int* counter, int delta) {
    atomicAdd(counter, delta);
}

// Single-thread kernel that resets a GPU-resident int32 to a given value.
__global__ void kernel_reset_counter(int* counter, int value) {
    *counter = value;
}

// ---------- Launcher functions (extern "C" for dlsym) ----------

extern "C" {

cudaError_t launch_increment_counter(int* counter, int delta, cudaStream_t stream) {
    kernel_increment_counter<<<1, 1, 0, stream>>>(counter, delta);
    return cudaGetLastError();
}

cudaError_t launch_reset_counter(int* counter, int value, cudaStream_t stream) {
    kernel_reset_counter<<<1, 1, 0, stream>>>(counter, value);
    return cudaGetLastError();
}

} // extern "C"
