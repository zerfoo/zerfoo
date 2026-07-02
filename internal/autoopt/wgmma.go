package autoopt

import "fmt"

// WGMMADataType identifies the input data type for warp-group MMA.
type WGMMADataType int

const (
	WGMMAFP16 WGMMADataType = iota
	WGMMABF16
	WGMMAFP8E4M3
	WGMMAFP8E5M2
	WGMMAINT8
)

// String returns the data type name.
func (dt WGMMADataType) String() string {
	switch dt {
	case WGMMAFP16:
		return "fp16"
	case WGMMABF16:
		return "bf16"
	case WGMMAFP8E4M3:
		return "fp8_e4m3"
	case WGMMAFP8E5M2:
		return "fp8_e5m2"
	case WGMMAINT8:
		return "int8"
	default:
		return "unknown"
	}
}

// WGMMAConfig describes the configuration for a warp-group matrix multiply-accumulate
// operation on SM90+ hardware. WGMMA allows a warp group (4 warps = 128 threads)
// to collectively execute large MMA operations using tensor cores.
type WGMMAConfig struct {
	// M is the tile height (output rows per warp group).
	M int

	// N is the tile width (output columns per warp group).
	N int

	// K is the tile depth (reduction dimension per MMA step).
	K int

	// InputType is the data type for the A and B input matrices.
	InputType WGMMADataType

	// AccumulatorFP32 uses FP32 accumulators when true, FP16 when false.
	AccumulatorFP32 bool

	// TransposeB transposes the B matrix (column-major layout).
	TransposeB bool
}

// Validate checks that the WGMMA configuration is valid for SM90 hardware.
func (c *WGMMAConfig) Validate() error {
	// SM90 wgmma supports M=64 as the warp-group M dimension.
	if c.M != 64 {
		return fmt.Errorf("wgmma: M must be 64 (warp-group size), got %d", c.M)
	}

	validN := map[int]bool{8: true, 16: true, 24: true, 32: true, 48: true, 64: true, 80: true, 96: true, 112: true, 128: true, 144: true, 160: true, 176: true, 192: true, 208: true, 224: true, 240: true, 256: true}
	if !validN[c.N] {
		return fmt.Errorf("wgmma: N=%d is not a supported tile width (must be multiple of 8 in [8, 256])", c.N)
	}

	validK := validKForType(c.InputType)
	if !validK[c.K] {
		return fmt.Errorf("wgmma: K=%d is not valid for input type %s", c.K, c.InputType)
	}

	return nil
}

// validKForType returns valid K dimensions for each WGMMA input type.
func validKForType(dt WGMMADataType) map[int]bool {
	switch dt {
	case WGMMAFP16, WGMMABF16:
		// FP16/BF16: K=16 per MMA step
		return map[int]bool{16: true}
	case WGMMAFP8E4M3, WGMMAFP8E5M2, WGMMAINT8:
		// FP8/INT8: K=32 per MMA step
		return map[int]bool{32: true}
	default:
		return nil
	}
}

// OutputElements returns the number of output elements per warp-group MMA step.
func (c *WGMMAConfig) OutputElements() int {
	return c.M * c.N
}

// SelectWGMMATile chooses optimal WGMMA tile dimensions for a given GEMM problem size.
// Returns a WGMMAConfig with the best tile shape for the given M, N, K dimensions
// and data type.
func SelectWGMMATile(m, n, k int, dt WGMMADataType) *WGMMAConfig {
	// WGMMA M is always 64 (warp-group size).
	cfg := &WGMMAConfig{
		M:               64,
		InputType:       dt,
		AccumulatorFP32: true, // FP32 accumulators for numerical stability
	}

	// Select N tile: largest valid N that divides the problem N,
	// capped at 256.
	cfg.N = selectTileN(n)

	// Select K based on data type.
	switch dt {
	case WGMMAFP16, WGMMABF16:
		cfg.K = 16
	case WGMMAFP8E4M3, WGMMAFP8E5M2, WGMMAINT8:
		cfg.K = 32
	}

	return cfg
}

// selectTileN picks the largest N tile dimension that is a valid WGMMA size.
func selectTileN(n int) int {
	// Valid N values: multiples of 8 from 8 to 256.
	// Pick the largest that does not exceed the problem N, capped at 256.
	maxN := n
	if maxN > 256 {
		maxN = 256
	}

	// Round down to nearest multiple of 8.
	tileN := (maxN / 8) * 8
	if tileN < 8 {
		tileN = 8
	}
	return tileN
}

// EstimateWGMMAIterations returns the number of warp-group MMA iterations
// needed to cover a GEMM of the given dimensions with the selected tile.
func EstimateWGMMAIterations(m, n, k int, cfg *WGMMAConfig) int {
	if cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0 {
		return 0
	}
	mIter := (m + cfg.M - 1) / cfg.M
	nIter := (n + cfg.N - 1) / cfg.N
	kIter := (k + cfg.K - 1) / cfg.K
	return mIter * nIter * kIter
}
