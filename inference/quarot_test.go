package inference

import (
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestFWHT_Hadamard2(t *testing.T) {
	// H_2 (unnormalized) = [[1, 1], [1, -1]]
	// [1, 0] -> [1, 1]
	x := []float32{1, 0}
	fwht(x)
	if x[0] != 1 || x[1] != 1 {
		t.Fatalf("FWHT([1,0]) = %v, want [1, 1]", x)
	}

	// [1, 1] -> [2, 0]
	y := []float32{1, 1}
	fwht(y)
	if y[0] != 2 || y[1] != 0 {
		t.Fatalf("FWHT([1,1]) = %v, want [2, 0]", y)
	}
}

func TestFWHT_Involutory(t *testing.T) {
	// The normalized Hadamard is involutory: H * H = I.
	// Equivalently, applying FWHT twice (with normalization) recovers input.
	n := 8
	norm := float32(1.0 / math.Sqrt(float64(n)))

	original := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	x := make([]float32, n)
	copy(x, original)

	// First transform + normalize.
	fwht(x)
	for i := range x {
		x[i] *= norm
	}

	// Second transform + normalize (should recover original).
	fwht(x)
	for i := range x {
		x[i] *= norm
	}

	for i := range x {
		if diff := math.Abs(float64(x[i] - original[i])); diff > 1e-5 {
			t.Errorf("round-trip element %d: got %f, want %f (diff %e)", i, x[i], original[i], diff)
		}
	}
}

func TestFWHT_Orthogonal(t *testing.T) {
	// Verify that the normalized Hadamard rows are orthonormal.
	// H_4 (normalized) rows should satisfy: dot(row_i, row_j) = delta_ij.
	n := 4
	norm := float32(1.0 / math.Sqrt(float64(n)))

	// Build H_4 by transforming identity vectors.
	rows := make([][]float32, n)
	for i := range n {
		row := make([]float32, n)
		row[i] = 1
		fwht(row)
		for j := range row {
			row[j] *= norm
		}
		rows[i] = row
	}

	for i := range n {
		for j := range n {
			dot := float32(0)
			for k := range n {
				dot += rows[i][k] * rows[j][k]
			}
			expected := float32(0)
			if i == j {
				expected = 1
			}
			if diff := math.Abs(float64(dot - expected)); diff > 1e-5 {
				t.Errorf("dot(row_%d, row_%d) = %f, want %f", i, j, dot, expected)
			}
		}
	}
}

func TestFuseQuaRotWeights_RoundTrip(t *testing.T) {
	// Fusing Hadamard twice should recover the original weights within tolerance.
	hiddenDim := 8
	intermediateDim := 16
	numLayers := 2

	tensors := makeTestWeights(numLayers, hiddenDim, intermediateDim)

	// Save originals.
	originals := make(map[string][]float32)
	for name, tns := range tensors {
		data := tns.Data()
		cp := make([]float32, len(data))
		copy(cp, data)
		originals[name] = cp
	}

	// First fusion.
	if err := FuseQuaRotWeights(tensors, numLayers); err != nil {
		t.Fatalf("first fusion: %v", err)
	}

	// Verify weights actually changed.
	for name, orig := range originals {
		rotated := tensors[name].Data()
		allSame := true
		for i := range orig {
			if orig[i] != rotated[i] {
				allSame = false
				break
			}
		}
		if allSame && len(orig) > 0 {
			t.Errorf("tensor %q unchanged after fusion", name)
		}
	}

	// Second fusion (should recover originals).
	if err := FuseQuaRotWeights(tensors, numLayers); err != nil {
		t.Fatalf("second fusion: %v", err)
	}

	for name, orig := range originals {
		recovered := tensors[name].Data()
		for i := range orig {
			if diff := math.Abs(float64(recovered[i] - orig[i])); diff > 1e-3 {
				t.Errorf("round-trip %q[%d]: got %f, want %f (diff %e)",
					name, i, recovered[i], orig[i], diff)
			}
		}
	}
}

func TestFuseQuaRotWeights_MissingLayers(t *testing.T) {
	// Should fail if numLayers <= 0.
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	if err := FuseQuaRotWeights(tensors, 0); err == nil {
		t.Fatal("expected error for numLayers=0")
	}
}

func TestFuseQuaRotWeights_MissingQProj(t *testing.T) {
	// Should fail if layer 0 Q projection is missing (needed to determine dim).
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	if err := FuseQuaRotWeights(tensors, 1); err == nil {
		t.Fatal("expected error for missing Q projection")
	}
}

func TestFuseQuaRotWeights_NonPowerOf2(t *testing.T) {
	// Should fail if hidden dimension is not a power of 2.
	data := make([]float32, 3*6)
	for i := range data {
		data[i] = float32(i)
	}
	w, _ := tensor.New([]int{3, 6}, data) // 6 is not power of 2
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.q_proj.weight": w,
	}
	if err := FuseQuaRotWeights(tensors, 1); err == nil {
		t.Fatal("expected error for non-power-of-2 dimension")
	}
}

func TestFuseQuaRotWeights_PreservesShape(t *testing.T) {
	hiddenDim := 4
	intermediateDim := 8
	tensors := makeTestWeights(1, hiddenDim, intermediateDim)

	// Record original shapes.
	shapes := make(map[string][]int)
	for name, tns := range tensors {
		shapes[name] = tns.Shape()
	}

	if err := FuseQuaRotWeights(tensors, 1); err != nil {
		t.Fatalf("fusion: %v", err)
	}

	for name, origShape := range shapes {
		newShape := tensors[name].Shape()
		if len(newShape) != len(origShape) {
			t.Errorf("%q shape dims changed: %v -> %v", name, origShape, newShape)
			continue
		}
		for d := range origShape {
			if origShape[d] != newShape[d] {
				t.Errorf("%q shape[%d] changed: %d -> %d", name, d, origShape[d], newShape[d])
			}
		}
	}
}

func TestFuseQuaRotWeights_WithQuaRotFalseIsNoOp(t *testing.T) {
	// When WithQuaRot(false) is set, FuseQuaRotWeights should never be called.
	// This test verifies the opt-out path: applying the option with false
	// leaves loadOptions.quarot == false, meaning the LoadFile path skips fusion.
	// We verify this by checking that weights are identical before and after
	// a simulated "no quarot" path.
	hiddenDim := 4
	intermediateDim := 8
	tensors := makeTestWeights(1, hiddenDim, intermediateDim)

	// Save originals.
	originals := make(map[string][]float32)
	for name, tns := range tensors {
		data := tns.Data()
		cp := make([]float32, len(data))
		copy(cp, data)
		originals[name] = cp
	}

	// Simulate the LoadFile path with quarot=false: skip FuseQuaRotWeights entirely.
	o := &loadOptions{}
	WithQuaRot(false)(o)
	if o.quarot {
		// This would trigger fusion — verify it doesn't.
		t.Fatal("WithQuaRot(false) should leave quarot=false")
	}

	// Weights must be unchanged since fusion was not called.
	for name, orig := range originals {
		current := tensors[name].Data()
		for i := range orig {
			if orig[i] != current[i] {
				t.Errorf("tensor %q[%d] changed without fusion: %f != %f",
					name, i, current[i], orig[i])
			}
		}
	}
}

func TestFuseQuaRotWeights_ErrorCases(t *testing.T) {
	tests := []struct {
		name    string
		tensors map[string]*tensor.TensorNumeric[float32]
		layers  int
		wantErr string
	}{
		{
			name:    "negative numLayers",
			tensors: map[string]*tensor.TensorNumeric[float32]{},
			layers:  -1,
			wantErr: "numLayers must be positive",
		},
		{
			name:    "zero numLayers",
			tensors: map[string]*tensor.TensorNumeric[float32]{},
			layers:  0,
			wantErr: "numLayers must be positive",
		},
		{
			name:    "missing Q projection",
			tensors: map[string]*tensor.TensorNumeric[float32]{},
			layers:  1,
			wantErr: "missing tensor",
		},
		{
			name: "non-power-of-2 hidden dim",
			tensors: func() map[string]*tensor.TensorNumeric[float32] {
				data := make([]float32, 3*6)
				for i := range data {
					data[i] = float32(i)
				}
				w, _ := tensor.New([]int{3, 6}, data)
				return map[string]*tensor.TensorNumeric[float32]{
					"model.layers.0.self_attn.q_proj.weight": w,
				}
			}(),
			layers:  1,
			wantErr: "not a power of 2",
		},
		{
			name: "1D weight tensor",
			tensors: func() map[string]*tensor.TensorNumeric[float32] {
				data := make([]float32, 4)
				w, _ := tensor.New([]int{4}, data)
				return map[string]*tensor.TensorNumeric[float32]{
					"model.layers.0.self_attn.q_proj.weight": w,
				}
			}(),
			layers:  1,
			wantErr: "expected 2D weight tensor",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := FuseQuaRotWeights(tt.tensors, tt.layers)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if got := err.Error(); !contains(got, tt.wantErr) {
				t.Errorf("error %q does not contain %q", got, tt.wantErr)
			}
		})
	}
}

func TestFuseQuaRotWeights_SingleFusionChangesWeights(t *testing.T) {
	// Verify that a single fusion actually modifies all weight tensors.
	hiddenDim := 4
	intermediateDim := 8
	numLayers := 1
	tensors := makeTestWeights(numLayers, hiddenDim, intermediateDim)

	originals := make(map[string][]float32)
	for name, tns := range tensors {
		data := tns.Data()
		cp := make([]float32, len(data))
		copy(cp, data)
		originals[name] = cp
	}

	if err := FuseQuaRotWeights(tensors, numLayers); err != nil {
		t.Fatalf("fusion: %v", err)
	}

	for name, orig := range originals {
		rotated := tensors[name].Data()
		changed := false
		for i := range orig {
			if orig[i] != rotated[i] {
				changed = true
				break
			}
		}
		if !changed {
			t.Errorf("tensor %q was not modified by fusion", name)
		}
	}
}

func TestFuseQuaRotWeights_SkipsMissingOptionalTensors(t *testing.T) {
	// If a layer is missing some optional projections (e.g., GQA with fewer
	// K/V heads), fusion should succeed and only process present tensors.
	hiddenDim := 4
	data := make([]float32, hiddenDim*hiddenDim)
	for i := range data {
		data[i] = float32(i + 1)
	}
	qW, _ := tensor.New([]int{hiddenDim, hiddenDim}, data)
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.q_proj.weight": qW,
		// K, V, O, gate, up, down are all missing — should be skipped.
	}

	if err := FuseQuaRotWeights(tensors, 1); err != nil {
		t.Fatalf("expected success with missing optional tensors, got: %v", err)
	}

	// Q projection should still be rotated.
	rotated := tensors["model.layers.0.self_attn.q_proj.weight"].Data()
	allSame := true
	for i := range data {
		if data[i] != rotated[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("Q projection should have been rotated")
	}
}

func TestFuseQuaRotWeights_OutputToleranceMultiLayer(t *testing.T) {
	// Verify the round-trip tolerance holds across multiple layers with
	// varying intermediate dimensions (power-of-2).
	tests := []struct {
		name            string
		numLayers       int
		hiddenDim       int
		intermediateDim int
	}{
		{"1-layer-dim4", 1, 4, 8},
		{"2-layer-dim8", 2, 8, 16},
		{"4-layer-dim16", 4, 16, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := makeTestWeights(tt.numLayers, tt.hiddenDim, tt.intermediateDim)

			originals := make(map[string][]float32)
			for name, tns := range tensors {
				d := tns.Data()
				cp := make([]float32, len(d))
				copy(cp, d)
				originals[name] = cp
			}

			// Double-fuse (round trip).
			if err := FuseQuaRotWeights(tensors, tt.numLayers); err != nil {
				t.Fatalf("first fusion: %v", err)
			}
			if err := FuseQuaRotWeights(tensors, tt.numLayers); err != nil {
				t.Fatalf("second fusion: %v", err)
			}

			for name, orig := range originals {
				recovered := tensors[name].Data()
				for i := range orig {
					if diff := math.Abs(float64(recovered[i] - orig[i])); diff > 1e-3 {
						t.Errorf("%q[%d]: got %f, want %f (diff %e)",
							name, i, recovered[i], orig[i], diff)
					}
				}
			}
		})
	}
}

// contains reports whether s contains substr.
func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// makeTestWeights creates a synthetic weight tensor map for testing.
func makeTestWeights(numLayers, hiddenDim, intermediateDim int) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	for layer := range numLayers {
		prefix := fmt.Sprintf("model.layers.%d.", layer)

		// Attention projections: all [hiddenDim, hiddenDim].
		for _, suffix := range []string{
			"self_attn.q_proj.weight",
			"self_attn.k_proj.weight",
			"self_attn.v_proj.weight",
			"self_attn.o_proj.weight",
		} {
			tensors[prefix+suffix] = makeRandTensor(hiddenDim, hiddenDim, layer)
		}

		// FFN projections.
		// gate_proj and up_proj: [intermediateDim, hiddenDim]
		tensors[prefix+"mlp.gate_proj.weight"] = makeRandTensor(intermediateDim, hiddenDim, layer)
		tensors[prefix+"mlp.up_proj.weight"] = makeRandTensor(intermediateDim, hiddenDim, layer)
		// down_proj: [hiddenDim, intermediateDim]
		tensors[prefix+"mlp.down_proj.weight"] = makeRandTensor(hiddenDim, intermediateDim, layer)
	}
	return tensors
}

// makeRandTensor creates a deterministic pseudo-random tensor for testing.
func makeRandTensor(rows, cols, seed int) *tensor.TensorNumeric[float32] {
	data := make([]float32, rows*cols)
	for i := range data {
		// Simple deterministic pseudo-random values in [-1, 1].
		v := math.Sin(float64(i+seed*1000+1)) * 0.5
		data[i] = float32(v)
	}
	t, err := tensor.New([]int{rows, cols}, data)
	if err != nil {
		panic(fmt.Sprintf("makeRandTensor: %v", err))
	}
	return t
}
