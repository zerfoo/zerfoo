package nas

import (
	"math"
	"math/rand/v2"
	"testing"
)

// generateCalibrationData creates 20 synthetic architectures with "measured"
// latencies that follow a known linear model plus small noise, simulating
// DGX Spark benchmarks.
func generateCalibrationData(t *testing.T) []CalibrationPoint {
	t.Helper()

	hw := DGXSpark()
	est := NewLatencyEstimator(hw)

	// Ground-truth coefficients for the synthetic benchmark data.
	trueAlpha := 1.2
	trueBeta := 0.8
	trueBias := 5e-6 // small fixed overhead in seconds

	rng := rand.New(rand.NewPCG(42, 0))
	ss := NewSearchSpace(4) // 4 nodes → 6 edges per cell

	points := make([]CalibrationPoint, 20)
	for i := range points {
		cell := ss.Sample(rng)
		ct, mt := est.cellFeatures(cell)
		// Simulated measured latency with small noise (< 2% of signal).
		noise := 1.0 + (rng.Float64()-0.5)*0.02
		latency := (trueAlpha*ct + trueBeta*mt + trueBias) * noise
		points[i] = CalibrationPoint{Cell: cell, Latency: latency}
	}
	return points
}

func TestLatencyEstimator(t *testing.T) {
	data := generateCalibrationData(t)

	// Split: first 15 for training, last 5 for held-out test.
	train := data[:15]
	test := data[15:]

	hw := DGXSpark()
	est := NewLatencyEstimator(hw)
	est.Calibrate(train)

	// R^2 on training data should be very high.
	trainR2 := est.RSquared(train)
	t.Logf("Train R^2 = %.4f", trainR2)
	if trainR2 < 0.95 {
		t.Errorf("train R^2 = %.4f, want >= 0.95", trainR2)
	}

	// R^2 on held-out data must exceed 0.85 (acceptance criterion).
	testR2 := est.RSquared(test)
	t.Logf("Test R^2 = %.4f (held-out)", testR2)
	if testR2 < 0.85 {
		t.Errorf("held-out R^2 = %.4f, want >= 0.85", testR2)
	}
}

func TestLatencyEstimatorUncalibrated(t *testing.T) {
	hw := DGXSpark()
	est := NewLatencyEstimator(hw)

	ss := NewSearchSpace(4)
	rng := rand.New(rand.NewPCG(99, 0))
	cell := ss.Sample(rng)

	lat := est.Estimate(cell)
	if lat < 0 {
		t.Errorf("uncalibrated estimate = %v, want >= 0", lat)
	}
	if lat == 0 {
		// Should have some non-zero ops.
		t.Errorf("uncalibrated estimate = 0, expected non-zero for non-trivial cell")
	}
}

func TestLatencyEstimateAlias(t *testing.T) {
	hw := DGXSpark()
	est := NewLatencyEstimator(hw)

	ss := NewSearchSpace(3)
	rng := rand.New(rand.NewPCG(7, 0))
	cell := ss.Sample(rng)

	if est.Estimate(cell) != est.LatencyEstimate(cell) {
		t.Error("LatencyEstimate and Estimate should return the same value")
	}
}

func TestDGXSparkProfile(t *testing.T) {
	hw := DGXSpark()
	if hw.FLOPSThroughput != 1000 {
		t.Errorf("FLOPSThroughput = %v, want 1000", hw.FLOPSThroughput)
	}
	if hw.MemBandwidthGBs != 200 {
		t.Errorf("MemBandwidthGBs = %v, want 200", hw.MemBandwidthGBs)
	}
}

func TestDefaultOpCosts(t *testing.T) {
	costs := DefaultOpCosts()
	allOps := AllOps()
	for _, op := range allOps {
		if _, ok := costs[op]; !ok {
			t.Errorf("missing cost for op %q", op)
		}
	}

	// Zero op should have zero cost.
	z := costs[OpZero]
	if z.FLOPs != 0 || z.MemBytes != 0 {
		t.Errorf("zero op cost = %+v, want zero", z)
	}

	// Conv5x5 should cost more than Conv3x3.
	if costs[OpConv5x5].FLOPs <= costs[OpConv3x3].FLOPs {
		t.Error("conv5x5 should have more FLOPs than conv3x3")
	}
}

func TestCalibrateMinimalData(t *testing.T) {
	hw := DGXSpark()
	est := NewLatencyEstimator(hw)

	// With only 1 data point, Calibrate should not panic.
	est.Calibrate([]CalibrationPoint{{
		Cell:    Cell{NumNodes: 2, Edges: []Edge{{From: 0, To: 1, Op: OpConv3x3}}},
		Latency: 1e-5,
	}})

	// Original defaults should be preserved.
	if est.alpha != 1.0 || est.beta != 1.0 || est.bias != 0.0 {
		t.Error("calibrate with 1 point should keep defaults")
	}
}

func TestRSquaredPerfectFit(t *testing.T) {
	hw := DGXSpark()
	est := NewLatencyEstimator(hw)

	// With default coefficients (alpha=1, beta=1, bias=0), predictions
	// exactly match compute+mem time. Create data that matches.
	ss := NewSearchSpace(3)
	rng := rand.New(rand.NewPCG(123, 0))

	var data []CalibrationPoint
	for range 10 {
		cell := ss.Sample(rng)
		lat := est.Estimate(cell)
		data = append(data, CalibrationPoint{Cell: cell, Latency: lat})
	}

	r2 := est.RSquared(data)
	if math.Abs(r2-1.0) > 1e-10 {
		t.Errorf("R^2 = %.6f, want 1.0 for perfect fit", r2)
	}
}
