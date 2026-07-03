package timeseries

import (
	"math/rand/v2"
	"sync"
)

// weightInitRand is the package-level generator behind every model's weight
// initialization in this package (Xavier/He linear-layer init in
// newLinearLayer/newLinearXavier, and PatchTST's positional-embedding init).
//
// math/rand/v2's top-level convenience functions (rand.NormFloat64,
// rand.Float64, ...) are deliberately NOT seedable: v2 dropped the global
// Seed function entirely, so each process draws from a distinct,
// non-reproducible source by design (see the math/rand/v2 package docs).
// Calling classic math/rand's rand.Seed(), or passing a fixed seed to a
// caller-owned generator elsewhere, has NO effect on code that calls the
// math/rand/v2 package-level functions directly -- which every model
// constructor in this package did until this file existed.
//
// This is a real nondeterminism source found while proving out
// ZTENSOR_DETERMINISTIC=1 (zerfoo plan-gpu-training-hardening.md T4.1):
// that flag covers GPU kernel reduction order and cuBLAS GEMM configuration,
// but weight initialization happens on the CPU host, before any GPU kernel
// runs, and is an orthogonal source of run-to-run divergence the flag does
// not and cannot address. Two ZTENSOR_DETERMINISTIC=1 training runs cannot
// be bitwise-identical if their initial weights differ.
//
// Default behavior is unchanged from before this file existed: an
// auto-seeded (process-random) generator, so any caller that never calls
// SeedWeightInit keeps the same "different weights every run" behavior it
// always had.
var (
	weightInitMu   sync.Mutex
	weightInitRand = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
)

// SeedWeightInit reseeds the package-level weight-initialization generator
// used by every model in this package. Call before constructing any model
// for reproducible weights. Reproducing a training run bitwise end-to-end
// also requires ZTENSOR_DETERMINISTIC=1 (GPU kernel/cuBLAS determinism) and
// a fixed data-generation seed -- see cmd/bench_train's -seed flag.
func SeedWeightInit(seed1, seed2 uint64) {
	weightInitMu.Lock()
	defer weightInitMu.Unlock()
	weightInitRand = rand.New(rand.NewPCG(seed1, seed2))
}

// weightInitNormFloat64 draws from the package-level weight-init generator.
// Guarded by weightInitMu: math/rand/v2's *rand.Rand is documented as not
// safe for concurrent use, unlike the (unseedable) package-level functions
// it replaces here.
func weightInitNormFloat64() float64 {
	weightInitMu.Lock()
	defer weightInitMu.Unlock()
	return weightInitRand.NormFloat64()
}
