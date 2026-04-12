package crossasset

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func defaultConfig() Config {
	return Config{
		NSources:          4,
		FeaturesPerSource: 8,
		DModel:            16,
		NHeads:            4,
		NLayers:           2,
		DropoutRate:        0.0,
		LearningRate:      0.001,
	}
}

func TestCrossAsset_Forward(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float32, cfg.NSources)
	for i := range features {
		features[i] = make([]float32, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float32(i*cfg.FeaturesPerSource+j) * 0.1
		}
	}

	output, err := m.Forward(features)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Check output shape.
	if len(output) != cfg.NSources {
		t.Fatalf("expected %d sources in output, got %d", cfg.NSources, len(output))
	}
	for i, o := range output {
		if len(o) != cfg.DModel {
			t.Fatalf("source %d: expected %d dims, got %d", i, cfg.DModel, len(o))
		}
	}

	// Verify outputs are non-zero.
	for i, o := range output {
		allZero := true
		for _, v := range o {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("source %d: output is all zeros", i)
		}
	}

	// Verify outputs are finite.
	for i, o := range output {
		for j, v := range o {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("source %d, dim %d: non-finite value %v", i, j, v)
			}
		}
	}
}

// TestCrossAsset_ForwardEngineParity verifies that Forward() produces the same
// output when using a separate CPU engine as when using the default engine,
// and that outputs are deterministic across two calls with the same engine.
func TestCrossAsset_ForwardEngineParity(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float32, cfg.NSources)
	for i := range features {
		features[i] = make([]float32, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float32(i*cfg.FeaturesPerSource+j) * 0.1
		}
	}

	// Run forward with default (package-level) CPU engine.
	out1, err := m.Forward(features)
	if err != nil {
		t.Fatalf("Forward (default engine): %v", err)
	}

	// Run forward again — should be deterministic.
	out2, err := m.Forward(features)
	if err != nil {
		t.Fatalf("Forward (second call): %v", err)
	}

	// Switch to a fresh CPU engine and run forward.
	freshEngine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m.SetEngine(freshEngine)
	out3, err := m.Forward(features)
	if err != nil {
		t.Fatalf("Forward (fresh engine): %v", err)
	}

	// Restore default engine.
	m.SetEngine(cpuEngine)

	const tol = 1e-4
	for s := 0; s < cfg.NSources; s++ {
		for d := 0; d < cfg.DModel; d++ {
			// Determinism check.
			if diff := math.Abs(float64(out1[s][d] - out2[s][d])); diff > tol {
				t.Errorf("determinism: source %d dim %d: diff=%.6f", s, d, diff)
			}
			// Engine parity check.
			if diff := math.Abs(float64(out1[s][d] - out3[s][d])); diff > tol {
				t.Errorf("engine parity: source %d dim %d: default=%.6f fresh=%.6f diff=%.6f",
					s, d, out1[s][d], out3[s][d], diff)
			}
		}
	}
}

func TestCrossAsset_Forward_InputValidation(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	t.Run("wrong number of sources", func(t *testing.T) {
		features := make([][]float32, cfg.NSources+1)
		for i := range features {
			features[i] = make([]float32, cfg.FeaturesPerSource)
		}
		_, err := m.Forward(features)
		if err == nil {
			t.Fatal("expected error for wrong number of sources")
		}
	})

	t.Run("wrong features per source", func(t *testing.T) {
		features := make([][]float32, cfg.NSources)
		for i := range features {
			features[i] = make([]float32, cfg.FeaturesPerSource+1)
		}
		_, err := m.Forward(features)
		if err == nil {
			t.Fatal("expected error for wrong features per source")
		}
	})
}

func TestCrossAsset_AttentionWeights(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float32, cfg.NSources)
	for i := range features {
		features[i] = make([]float32, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float32(i*cfg.FeaturesPerSource+j) * 0.1
		}
	}

	attn, err := m.AttentionWeights(features)
	if err != nil {
		t.Fatalf("AttentionWeights: %v", err)
	}

	// Check shape.
	if len(attn) != cfg.NSources {
		t.Fatalf("expected %d rows, got %d", cfg.NSources, len(attn))
	}
	for i, row := range attn {
		if len(row) != cfg.NSources {
			t.Fatalf("row %d: expected %d cols, got %d", i, cfg.NSources, len(row))
		}
	}

	// Verify weights sum to 1 across attended sources (j dimension).
	for i, row := range attn {
		sum := float64(0)
		for _, w := range row {
			sum += float64(w)
		}
		if math.Abs(sum-1.0) > 1e-4 {
			t.Errorf("row %d: attention weights sum to %f, expected 1.0", i, sum)
		}
	}

	// Verify all weights are non-negative.
	for i, row := range attn {
		for j, w := range row {
			if w < 0 {
				t.Errorf("attn[%d][%d] = %f, expected non-negative", i, j, w)
			}
		}
	}

	// Verify weights are finite.
	for i, row := range attn {
		for j, w := range row {
			if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) {
				t.Errorf("attn[%d][%d] = %v, expected finite", i, j, w)
			}
		}
	}
}

func TestCrossAsset_Predict(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float32, cfg.NSources)
	for i := range features {
		features[i] = make([]float32, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float32(i+j) * 0.1
		}
	}

	dirs, confs, err := m.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if len(dirs) != cfg.NSources {
		t.Fatalf("expected %d directions, got %d", cfg.NSources, len(dirs))
	}
	if len(confs) != cfg.NSources {
		t.Fatalf("expected %d confidences, got %d", cfg.NSources, len(confs))
	}

	for i, d := range dirs {
		if d < 0 || d > 2 {
			t.Errorf("source %d: direction %d out of range [0, 2]", i, d)
		}
	}
	for i, c := range confs {
		if c < 0 || c > 1 {
			t.Errorf("source %d: confidence %f out of range [0, 1]", i, c)
		}
	}
}

func TestCrossAsset_Train(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	nSamples := 20
	data := make([][][]float32, nSamples)
	labels := make([][]int, nSamples)
	for i := 0; i < nSamples; i++ {
		data[i] = make([][]float32, cfg.NSources)
		labels[i] = make([]int, cfg.NSources)
		for s := 0; s < cfg.NSources; s++ {
			data[i][s] = make([]float32, cfg.FeaturesPerSource)
			for f := 0; f < cfg.FeaturesPerSource; f++ {
				data[i][s][f] = float32(i+s+f) * 0.01
			}
			labels[i][s] = i % 3
		}
	}

	tc := TrainConfig{
		Epochs:       5,
		BatchSize:    10,
		LearningRate: 0.01,
	}

	err := m.Train(data, labels, tc)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Verify model can still predict after training.
	dirs, confs, err := m.Predict(data[0])
	if err != nil {
		t.Fatalf("Predict after train: %v", err)
	}
	if len(dirs) != cfg.NSources {
		t.Fatalf("expected %d directions, got %d", cfg.NSources, len(dirs))
	}
	if len(confs) != cfg.NSources {
		t.Fatalf("expected %d confidences, got %d", cfg.NSources, len(confs))
	}
}

func TestCrossAsset_Train_Validation(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	t.Run("no data", func(t *testing.T) {
		err := m.Train(nil, nil, TrainConfig{Epochs: 1})
		if err == nil {
			t.Fatal("expected error for no data")
		}
	})

	t.Run("mismatched lengths", func(t *testing.T) {
		data := make([][][]float32, 2)
		labels := make([][]int, 3)
		err := m.Train(data, labels, TrainConfig{Epochs: 1})
		if err == nil {
			t.Fatal("expected error for mismatched lengths")
		}
	})

	t.Run("zero epochs", func(t *testing.T) {
		data := make([][][]float32, 1)
		labels := make([][]int, 1)
		err := m.Train(data, labels, TrainConfig{Epochs: 0})
		if err == nil {
			t.Fatal("expected error for zero epochs")
		}
	})
}

func TestCrossAsset_DifferentInputsProduceDifferentOutputs(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	feat1 := make([][]float32, cfg.NSources)
	feat2 := make([][]float32, cfg.NSources)
	for i := 0; i < cfg.NSources; i++ {
		feat1[i] = make([]float32, cfg.FeaturesPerSource)
		feat2[i] = make([]float32, cfg.FeaturesPerSource)
		for j := 0; j < cfg.FeaturesPerSource; j++ {
			feat1[i][j] = float32(j) * 0.1
			feat2[i][j] = float32(j) * 0.5
		}
	}

	out1, err := m.Forward(feat1)
	if err != nil {
		t.Fatalf("Forward feat1: %v", err)
	}
	out2, err := m.Forward(feat2)
	if err != nil {
		t.Fatalf("Forward feat2: %v", err)
	}

	same := true
	for s := 0; s < cfg.NSources; s++ {
		for d := 0; d < cfg.DModel; d++ {
			if out1[s][d] != out2[s][d] {
				same = false
				break
			}
		}
	}
	if same {
		t.Error("different inputs produced identical outputs")
	}
}

// TestCrossAsset_GradientParity verifies that the node-based backward pass
// produces gradients within tolerance of numerical (finite-difference) gradients
// for a single layer's weight parameters.
func TestCrossAsset_GradientParity(t *testing.T) {
	cfg := Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
	m := NewModel(cfg)

	features := make([][]float32, cfg.NSources)
	for i := range features {
		features[i] = make([]float32, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float32(i*cfg.FeaturesPerSource+j+1) * 0.05
		}
	}
	label := []int{0, 1, 2}

	// Compute loss and analytic gradients via one backward step.
	ns := cfg.NSources
	dm := cfg.DModel

	// Forward: input projection via engine.
	x := make([][]float32, ns)
	for s := range ns {
		x[s] = make([]float32, dm)
		matVecMulEngine(m.engine, x[s], m.inputW[s], features[s], cfg.FeaturesPerSource, dm)
		vecAddEngine(m.engine, x[s], m.inputB[s])
	}

	// Forward through layer with cache.
	xOut, cache := m.forwardLayerCached(m.engine, x, m.layers[0])

	// Head forward + cross-entropy loss.
	lossAndGrad := func(xOut [][]float32, label []int) (float64, [][]float32) {
		totalLoss := 0.0
		dx := make([][]float32, ns)
		for s := range ns {
			logits := make([]float32, 3)
			matVecMulEngine(m.engine, logits, m.headW, xOut[s], dm, 3)
			vecAddEngine(m.engine, logits, m.headB)
			probs := softmaxEngine(m.engine, logits)
			totalLoss -= math.Log(float64(probs[label[s]]) + 1e-12)
			dLogits := make([]float32, 3)
			copy(dLogits, probs)
			dLogits[label[s]] -= 1.0
			for j := range dLogits {
				dLogits[j] /= float32(ns)
			}
			dx[s] = make([]float32, dm)
			for d := range dm {
				for c := range 3 {
					dx[s][d] += dLogits[c] * m.headW[d*3+c]
				}
			}
		}
		return totalLoss / float64(ns), dx
	}

	_, dxHead := lossAndGrad(xOut, label)

	// Backward through layer.
	dl := zeroLayer(dm)
	_ = m.backwardLayer(m.engine, dxHead, cache, &m.layers[0], &dl)

	// Helper: compute loss for a perturbed weight.
	computeLoss := func() float64 {
		xFwd := make([][]float32, ns)
		for s := range ns {
			xFwd[s] = make([]float32, dm)
			matVecMulEngine(m.engine, xFwd[s], m.inputW[s], features[s], cfg.FeaturesPerSource, dm)
			vecAddEngine(m.engine, xFwd[s], m.inputB[s])
		}
		ctx := context.Background()
		xT := slicesToTensor(xFwd, ns, dm)
		for _, l := range m.layers {
			var err error
			xT, err = m.forwardLayerEngine(ctx, m.engine, xT, l)
			if err != nil {
				t.Fatalf("forwardLayerEngine: %v", err)
			}
		}
		xFwd = tensorToSlices(xT, ns, dm)
		loss, _ := lossAndGrad(xFwd, label)
		return loss
	}

	// Check a subset of qW gradients via finite differences.
	const h = 1e-3 // larger step for float32 precision
	const tol = 0.5 // relaxed tolerance for float32 finite differences
	nCheck := 10
	if nCheck > len(m.layers[0].qW) {
		nCheck = len(m.layers[0].qW)
	}

	mismatches := 0
	for i := 0; i < nCheck; i++ {
		orig := m.layers[0].qW[i]

		m.layers[0].qW[i] = orig + h
		lPlus := computeLoss()
		m.layers[0].qW[i] = orig - h
		lMinus := computeLoss()
		m.layers[0].qW[i] = orig

		numerical := (lPlus - lMinus) / (2 * h)
		analytic := float64(dl.qW[i])

		diff := math.Abs(numerical - analytic)
		scale := math.Max(math.Abs(numerical), math.Abs(analytic))
		if scale > 1e-6 {
			diff /= scale // relative error
		}
		if diff > tol {
			mismatches++
			if mismatches <= 3 {
				t.Errorf("qW[%d]: numerical=%.8f, analytic=%.8f, rel_diff=%.6f", i, numerical, analytic, diff)
			}
		}
	}

	// Also check ffnW2 gradients.
	nCheckFFN := 10
	if nCheckFFN > len(m.layers[0].ffnW2) {
		nCheckFFN = len(m.layers[0].ffnW2)
	}
	for i := 0; i < nCheckFFN; i++ {
		orig := m.layers[0].ffnW2[i]

		m.layers[0].ffnW2[i] = orig + h
		lPlus := computeLoss()
		m.layers[0].ffnW2[i] = orig - h
		lMinus := computeLoss()
		m.layers[0].ffnW2[i] = orig

		numerical := (lPlus - lMinus) / (2 * h)
		analytic := float64(dl.ffnW2[i])

		diff := math.Abs(numerical - analytic)
		scale := math.Max(math.Abs(numerical), math.Abs(analytic))
		if scale > 1e-6 {
			diff /= scale
		}
		if diff > tol {
			mismatches++
			if mismatches <= 6 {
				t.Errorf("ffnW2[%d]: numerical=%.8f, analytic=%.8f, rel_diff=%.6f", i, numerical, analytic, diff)
			}
		}
	}

	if mismatches > 0 {
		t.Logf("Total gradient mismatches: %d / %d", mismatches, nCheck+nCheckFFN)
	}
}
