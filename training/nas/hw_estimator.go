package nas

import "math"

// HWProfile describes the hardware capabilities of a target device.
type HWProfile struct {
	// Name is a human-readable identifier for the device.
	Name string
	// FLOPSThroughput is the peak FP32 throughput in GFLOPS.
	FLOPSThroughput float64
	// MemBandwidthGBs is the memory bandwidth in GB/s.
	MemBandwidthGBs float64
}

// DGXSpark returns the hardware profile for the NVIDIA DGX Spark (GB10 GPU).
func DGXSpark() HWProfile {
	return HWProfile{
		Name:            "DGX Spark GB10",
		FLOPSThroughput: 1000, // ~1 TFLOPS FP32
		MemBandwidthGBs: 200,  // ~200 GB/s
	}
}

// OpCost defines the per-instance cost of an operation type in terms of
// compute (FLOPs) and memory transfers (bytes read + written).
type OpCost struct {
	FLOPs    float64 // floating-point operations per instance
	MemBytes float64 // bytes of memory traffic per instance
}

// DefaultOpCosts returns cost estimates for each operation type. The values
// model relative cost assuming a spatial dimension of 32x32 with 64 channels.
func DefaultOpCosts() map[OpType]OpCost {
	spatial := 32.0 * 32.0
	channels := 64.0

	return map[OpType]OpCost{
		OpConv3x3: {
			FLOPs:    2 * 9 * channels * channels * spatial, // 2*K^2*C_in*C_out*H*W
			MemBytes: 4 * (channels*spatial + 9*channels*channels + channels*spatial),
		},
		OpConv5x5: {
			FLOPs:    2 * 25 * channels * channels * spatial,
			MemBytes: 4 * (channels*spatial + 25*channels*channels + channels*spatial),
		},
		OpSepConv3x3: {
			// Depthwise + pointwise
			FLOPs:    2*9*channels*spatial + 2*channels*channels*spatial,
			MemBytes: 4 * (channels*spatial + 9*channels + channels*channels + 2*channels*spatial),
		},
		OpSepConv5x5: {
			FLOPs:    2*25*channels*spatial + 2*channels*channels*spatial,
			MemBytes: 4 * (channels*spatial + 25*channels + channels*channels + 2*channels*spatial),
		},
		OpAvgPool3x3: {
			FLOPs:    9 * channels * spatial,
			MemBytes: 4 * 2 * channels * spatial, // read + write
		},
		OpMaxPool3x3: {
			FLOPs:    9 * channels * spatial,
			MemBytes: 4 * 2 * channels * spatial,
		},
		OpSkipConnect: {
			FLOPs:    0,
			MemBytes: 4 * channels * spatial, // just a copy
		},
		OpZero: {
			FLOPs:    0,
			MemBytes: 0,
		},
	}
}

// LatencyEstimator predicts inference latency for cell architectures using a
// linear cost model calibrated against measured hardware benchmarks.
type LatencyEstimator struct {
	hw      HWProfile
	opCosts map[OpType]OpCost
	// Fitted coefficients: latency = alpha * compute_time + beta * mem_time + bias
	alpha float64
	beta  float64
	bias  float64
}

// NewLatencyEstimator creates an estimator for the given hardware profile using
// default operation costs.
func NewLatencyEstimator(hw HWProfile) *LatencyEstimator {
	return &LatencyEstimator{
		hw:      hw,
		opCosts: DefaultOpCosts(),
		alpha:   1.0,
		beta:    1.0,
		bias:    0.0,
	}
}

// cellFeatures computes the raw compute time and memory time for a cell.
func (e *LatencyEstimator) cellFeatures(c Cell) (computeTime, memTime float64) {
	var totalFLOPs, totalMemBytes float64
	for _, edge := range c.Edges {
		cost, ok := e.opCosts[edge.Op]
		if !ok {
			continue
		}
		totalFLOPs += cost.FLOPs
		totalMemBytes += cost.MemBytes
	}
	// Convert FLOPs to time: FLOPs / (GFLOPS * 1e9) = seconds
	if e.hw.FLOPSThroughput > 0 {
		computeTime = totalFLOPs / (e.hw.FLOPSThroughput * 1e9)
	}
	// Convert memory bytes to time: bytes / (GB/s * 1e9) = seconds
	if e.hw.MemBandwidthGBs > 0 {
		memTime = totalMemBytes / (e.hw.MemBandwidthGBs * 1e9)
	}
	return computeTime, memTime
}

// Estimate returns the predicted inference latency in seconds for a cell.
func (e *LatencyEstimator) Estimate(c Cell) float64 {
	ct, mt := e.cellFeatures(c)
	return e.alpha*ct + e.beta*mt + e.bias
}

// CalibrationPoint pairs a cell architecture with its measured latency.
type CalibrationPoint struct {
	Cell    Cell
	Latency float64 // measured latency in seconds
}

// Calibrate fits the linear model coefficients (alpha, beta, bias) using
// ordinary least squares on the provided calibration data.
func (e *LatencyEstimator) Calibrate(data []CalibrationPoint) {
	n := float64(len(data))
	if n < 2 {
		return
	}

	// Extract features and targets.
	xs := make([][2]float64, len(data))
	ys := make([]float64, len(data))
	for i, dp := range data {
		ct, mt := e.cellFeatures(dp.Cell)
		xs[i] = [2]float64{ct, mt}
		ys[i] = dp.Latency
	}

	// OLS for y = alpha*x0 + beta*x1 + bias using normal equations.
	// [X^T X] [alpha, beta, bias]^T = X^T y
	// where X is [x0, x1, 1] augmented matrix.
	var xtx [3][3]float64
	var xty [3]float64
	for i := range data {
		row := [3]float64{xs[i][0], xs[i][1], 1.0}
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				xtx[r][c] += row[r] * row[c]
			}
			xty[r] += row[r] * ys[i]
		}
	}

	// Solve 3x3 system via Gaussian elimination with partial pivoting.
	aug := [3][4]float64{
		{xtx[0][0], xtx[0][1], xtx[0][2], xty[0]},
		{xtx[1][0], xtx[1][1], xtx[1][2], xty[1]},
		{xtx[2][0], xtx[2][1], xtx[2][2], xty[2]},
	}

	for col := 0; col < 3; col++ {
		// Partial pivoting.
		maxRow := col
		for row := col + 1; row < 3; row++ {
			if math.Abs(aug[row][col]) > math.Abs(aug[maxRow][col]) {
				maxRow = row
			}
		}
		aug[col], aug[maxRow] = aug[maxRow], aug[col]

		pivot := aug[col][col]
		if math.Abs(pivot) < 1e-30 {
			return // singular, keep defaults
		}
		for j := col; j < 4; j++ {
			aug[col][j] /= pivot
		}
		for row := 0; row < 3; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := col; j < 4; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	e.alpha = aug[0][3]
	e.beta = aug[1][3]
	e.bias = aug[2][3]
	_ = n
}

// RSquared computes the coefficient of determination (R^2) of the estimator
// on the given data points.
func (e *LatencyEstimator) RSquared(data []CalibrationPoint) float64 {
	if len(data) == 0 {
		return 0
	}
	var sumY, sumY2, ssRes float64
	for _, dp := range data {
		sumY += dp.Latency
		sumY2 += dp.Latency * dp.Latency
		predicted := e.Estimate(dp.Cell)
		residual := dp.Latency - predicted
		ssRes += residual * residual
	}
	n := float64(len(data))
	mean := sumY / n
	ssTot := sumY2 - n*mean*mean
	if ssTot < 1e-30 {
		return 1.0 // no variance in targets
	}
	return 1.0 - ssRes/ssTot
}

// LatencyEstimate predicts inference latency for a cell architecture using
// the calibrated model. This is an alias for Estimate for API convenience.
func (e *LatencyEstimator) LatencyEstimate(c Cell) float64 {
	return e.Estimate(c)
}
