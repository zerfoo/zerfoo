// Package timeseries — weight hash debug helper.
//
// Dev-only diagnostic for the GPU training convergence regression
// investigation (see docs/plans/gpu-training-convergence-regression.md,
// Epic E1). Enabled only when ZERFOO_DEBUG_WEIGHT_HASH=1.
//
// Walks a slice of parameter tensors, snapshots each tensor's data via
// .Data(), and logs an FNV-1a 64-bit hex hash per tensor. Intended to be
// called before and after an optimizer step to detect parameter drift
// between host-side and device-side copies.
package timeseries

import (
	"encoding/binary"
	"hash/fnv"
	"log"
	"math"
	"os"

	"github.com/zerfoo/ztensor/tensor"
)

// weightHashDebugEnabled reports whether the weight-hash debug helper
// should emit output. Gated on ZERFOO_DEBUG_WEIGHT_HASH=1.
func weightHashDebugEnabled() bool {
	return os.Getenv("ZERFOO_DEBUG_WEIGHT_HASH") == "1"
}

// hashFloat32Slice computes an FNV-1a 64-bit hash over the bit pattern of a
// float32 slice. NaN values are canonicalised so two runs that produce
// distinct NaN payloads still compare equal.
func hashFloat32Slice(data []float32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range data {
		bits := math.Float32bits(v)
		if math.IsNaN(float64(v)) {
			bits = 0x7fc00000 // canonical quiet NaN
		}
		binary.LittleEndian.PutUint32(buf[:], bits)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

// HashParamTensors hashes each tensor in tensors using FNV-1a over its
// float32 data (obtained via .Data() so GPU-resident tensors are
// snapshotted back to the host) and logs one line per tensor.
//
// Output format (per line):
//
//	[weight-hash] tag=<tag> idx=<i> shape=<[...]> n=<len> hash=0x<hex>
//
// The helper is a no-op unless ZERFOO_DEBUG_WEIGHT_HASH=1 is set in the
// environment, so it is safe to leave installed on hot paths.
func HashParamTensors(tag string, tensors []*tensor.TensorNumeric[float32]) {
	if !weightHashDebugEnabled() {
		return
	}
	for i, t := range tensors {
		if t == nil {
			log.Printf("[weight-hash] tag=%s idx=%d <nil>", tag, i)
			continue
		}
		data := t.Data()
		h := hashFloat32Slice(data)
		log.Printf("[weight-hash] tag=%s idx=%d shape=%v n=%d hash=0x%016x",
			tag, i, t.Shape(), len(data), h)
	}
}
