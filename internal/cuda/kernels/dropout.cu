// dropout.cu -- GB10 (sm_121) inverted-dropout with a deterministic Philox mask.
//
// The mask is drawn from a counter-based Philox4x32-10 generator keyed by
// (seed, element offset), bit-identical to the CPU reference in
// compute/philox.go. Because the draw is a pure function of (seed, offset, p),
// the same seed produces the same mask on CPU and GPU -- which is what makes
// ztensor's CPU-GPU parity gate pass -- and backward recomputes the mask from
// (seed, p) rather than reading a cached buffer (capture-safe, no save pinned
// across an arena reset; ztensor ADR 006).
//
// Inverted-dropout semantics (match torch.nn.functional.dropout): training mode
// out[i] = keep(i) ? in[i] / (1 - p) : 0 with keep iff uniform >= p; eval mode
// (training == 0) or p == 0 is exact identity. Forward and backward share one
// kernel because dropout is linear in its input given the mask, so the host
// passes the upstream gradient as `in` for the backward call.
//
// The Philox constants and round structure mirror compute/philox.go exactly.

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

// host_uint_as_float reinterprets a 32-bit pattern as a float on the HOST
// (the device intrinsic __uint_as_float is not callable from host code).
static inline float host_uint_as_float(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u
#define PHILOX_ROUNDS 10

// Single Philox round: same word layout as the Go reference.
__device__ __forceinline__ void philox_round(uint32_t c[4], const uint32_t k[2]) {
    uint32_t hi0 = __umulhi(PHILOX_M0, c[0]);
    uint32_t lo0 = PHILOX_M0 * c[0];
    uint32_t hi1 = __umulhi(PHILOX_M1, c[2]);
    uint32_t lo1 = PHILOX_M1 * c[2];
    uint32_t n0 = hi1 ^ c[1] ^ k[0];
    uint32_t n1 = lo1;
    uint32_t n2 = hi0 ^ c[3] ^ k[1];
    uint32_t n3 = lo0;
    c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
}

// philox_uniform returns a uniform float in [0,1) for element `offset` under
// `seed`, consuming only the first output lane (matches the Go reference).
__device__ __forceinline__ float philox_uniform(uint64_t seed, uint64_t offset) {
    uint32_t k[2] = { (uint32_t)seed, (uint32_t)(seed >> 32) };
    uint32_t c[4] = { (uint32_t)offset, (uint32_t)(offset >> 32), 0u, 0u };
#pragma unroll
    for (int i = 0; i < PHILOX_ROUNDS; ++i) {
        philox_round(c, k);
        k[0] += PHILOX_W0;
        k[1] += PHILOX_W1;
    }
    // Map a 32-bit word to [0,1) by dividing by 2^32; same keep/drop boundary
    // (uniform >= p) the Go reference uses.
    return (float)c[0] * (1.0f / 4294967296.0f);
}

__global__ void kernel_dropout(const float* __restrict__ in,
                               float* __restrict__ out,
                               int n, float p, uint64_t seed, float invKeep) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float u = philox_uniform(seed, (uint64_t)gid);
    out[gid] = (u >= p) ? (in[gid] * invKeep) : 0.0f;
}

__global__ void kernel_identity_copy(const float* __restrict__ in,
                                     float* __restrict__ out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    out[gid] = in[gid];
}

extern "C" {

// dropout_f32 applies inverted dropout to in[0..n-1], writing out[0..n-1].
// training != 0 and p > 0 => masked-and-scaled; otherwise exact identity copy.
//
// p and invKeep (= 1/(1-p)) are passed as their 32-bit IEEE-754 BIT PATTERNS in
// integer parameters, not as `float`. The purego/dlopen launch path (no CGO)
// loads every argument into integer registers only (the AAPCS64 trampoline
// never populates the V float registers), so a `float` parameter would read
// garbage and shift every following argument. Passing the bits as uint32 and
// reinterpreting here with __uint_as_float keeps the ABI integer-only and
// identical between the CGO and purego paths. invKeep is computed host-side so
// the scale matches the CPU path bit-for-bit.
cudaError_t dropout_f32(const float* in, float* out, int n,
                        uint32_t pBits, uint64_t seed, int training, uint32_t invKeepBits,
                        cudaStream_t stream) {
    const int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid < 1) grid = 1;
    float p = host_uint_as_float(pBits);
    float invKeep = host_uint_as_float(invKeepBits);
    if (training == 0 || p == 0.0f) {
        kernel_identity_copy<<<grid, BLOCK, 0, stream>>>(in, out, n);
        return cudaGetLastError();
    }
    kernel_dropout<<<grid, BLOCK, 0, stream>>>(in, out, n, p, seed, invKeep);
    return cudaGetLastError();
}

} // extern "C"
