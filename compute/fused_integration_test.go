package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestFusedPipeline_RMSNorm_RoPE_SiLUGate exercises the full fused pipeline
// (RMSNorm -> RoPE -> SiLUGate) and compares the output against the unfused
// reference path. This is an end-to-end integration test for the fused kernel
// chain used in transformer decode. Tests skip without a GPU.
func TestFusedPipeline_RMSNorm_RoPE_SiLUGate(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	cases := []struct {
		name      string
		batch     int
		seqLen    int
		headDim   int
		ffnDim    int
		epsilon   float32
		tolerance float64
	}{
		{"small_1x4x8_ffn16", 1, 4, 8, 16, 1e-6, 1e-5},
		{"medium_2x8x32_ffn64", 2, 8, 32, 64, 1e-5, 1e-5},
		{"large_1x64x128_ffn256", 1, 64, 128, 256, 1e-6, 1e-5},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			halfDim := tc.headDim / 2

			// Build inputs.
			input, _ := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.headDim}, nil)
			if err := engine.RandomUniform(ctx, input, -1, 1); err != nil {
				t.Fatal(err)
			}
			weight, _ := tensor.New[float32]([]int{tc.headDim}, nil)
			if err := engine.RandomUniform(ctx, weight, 0.5, 1.5); err != nil {
				t.Fatal(err)
			}

			cosData := make([]float32, tc.seqLen*halfDim)
			sinData := make([]float32, tc.seqLen*halfDim)
			for s := range tc.seqLen {
				for i := range halfDim {
					angle := float64(s) / math.Pow(10000.0, float64(2*i)/float64(tc.headDim))
					cosData[s*halfDim+i] = float32(math.Cos(angle))
					sinData[s*halfDim+i] = float32(math.Sin(angle))
				}
			}
			cosAngles, _ := tensor.New([]int{tc.seqLen, halfDim}, cosData)
			sinAngles, _ := tensor.New([]int{tc.seqLen, halfDim}, sinData)

			gate, _ := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.ffnDim}, nil)
			up, _ := tensor.New[float32]([]int{tc.batch, tc.seqLen, tc.ffnDim}, nil)
			if err := engine.RandomUniform(ctx, gate, -2, 2); err != nil {
				t.Fatal(err)
			}
			if err := engine.RandomUniform(ctx, up, -2, 2); err != nil {
				t.Fatal(err)
			}

			// --- Fused pipeline ---
			fusedNorm, _, err := FusedRMSNorm(input, weight, tc.epsilon)
			if err != nil {
				t.Fatalf("FusedRMSNorm: %v", err)
			}
			fusedRope, err := FusedRoPE(fusedNorm, cosAngles, sinAngles, tc.headDim)
			if err != nil {
				t.Fatalf("FusedRoPE: %v", err)
			}
			fusedSilu, err := FusedSiLUGate(gate, up)
			if err != nil {
				t.Fatalf("FusedSiLUGate: %v", err)
			}

			// --- Unfused pipeline ---
			// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
			squared, _ := engine.Mul(ctx, input, input, nil)
			lastDim := len(input.Shape()) - 1
			meanSq, _ := engine.ReduceMean(ctx, squared, lastDim, true)
			meanSqEps, _ := engine.AddScalar(ctx, meanSq, tc.epsilon, nil)
			rsqrt, _ := engine.Rsqrt(ctx, meanSqEps, nil)
			normalized, _ := engine.Mul(ctx, input, rsqrt, nil)
			unfusedNorm, _ := engine.Mul(ctx, normalized, weight, nil)

			// RoPE
			xRot0, _ := unfusedNorm.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{0, halfDim})
			xRot1, _ := unfusedNorm.Slice([2]int{0, tc.batch}, [2]int{0, tc.seqLen}, [2]int{halfDim, tc.headDim})
			t1, _ := engine.Mul(ctx, xRot0, cosAngles)
			t2, _ := engine.Mul(ctx, xRot1, sinAngles)
			rotX0, _ := engine.Sub(ctx, t1, t2)
			m1, _ := engine.Mul(ctx, xRot1, cosAngles)
			m2, _ := engine.Mul(ctx, xRot0, sinAngles)
			rotX1, _ := engine.Add(ctx, m1, m2)
			unfusedRope, _ := engine.Concat(ctx, []*tensor.TensorNumeric[float32]{rotX0, rotX1}, 2)

			// SiLUGate: silu(gate) * up
			sigGate, _ := engine.UnaryOp(ctx, gate, func(v float32) float32 {
				return float32(1.0 / (1.0 + math.Exp(-float64(v))))
			})
			siluGate, _ := engine.Mul(ctx, gate, sigGate, nil)
			unfusedSilu, _ := engine.Mul(ctx, siluGate, up, nil)

			// --- Compare each stage ---
			stages := []struct {
				name   string
				fused  *tensor.TensorNumeric[float32]
				unfused *tensor.TensorNumeric[float32]
			}{
				{"RMSNorm", fusedNorm, unfusedNorm},
				{"RoPE", fusedRope, unfusedRope},
				{"SiLUGate", fusedSilu, unfusedSilu},
			}

			for _, stage := range stages {
				fusedData := stage.fused.Data()
				unfusedData := stage.unfused.Data()
				if len(fusedData) != len(unfusedData) {
					t.Fatalf("%s: size mismatch %d vs %d", stage.name, len(fusedData), len(unfusedData))
				}

				maxDiff := float64(0)
				for i := range fusedData {
					diff := math.Abs(float64(fusedData[i] - unfusedData[i]))
					if diff > maxDiff {
						maxDiff = diff
					}
				}

				t.Logf("%s: max diff = %e", stage.name, maxDiff)
				if maxDiff > tc.tolerance {
					t.Errorf("%s: max diff %e exceeds tolerance %e", stage.name, maxDiff, tc.tolerance)
				}
			}
		})
	}
}

// TestFusedPipeline_Composability verifies that fused kernels can be
// composed in different orders without accumulating unexpected error.
func TestFusedPipeline_Composability(t *testing.T) {
	engine := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	batch, seqLen, dim := 1, 16, 64
	halfDim := dim / 2

	input, _ := tensor.New[float32]([]int{batch, seqLen, dim}, nil)
	if err := engine.RandomUniform(ctx, input, -1, 1); err != nil {
		t.Fatal(err)
	}
	weight, _ := tensor.New[float32]([]int{dim}, nil)
	if err := engine.RandomUniform(ctx, weight, 0.5, 1.5); err != nil {
		t.Fatal(err)
	}

	cosData := make([]float32, seqLen*halfDim)
	sinData := make([]float32, seqLen*halfDim)
	for s := range seqLen {
		for i := range halfDim {
			angle := float64(s) / math.Pow(10000.0, float64(2*i)/float64(dim))
			cosData[s*halfDim+i] = float32(math.Cos(angle))
			sinData[s*halfDim+i] = float32(math.Sin(angle))
		}
	}
	cosAngles, _ := tensor.New([]int{seqLen, halfDim}, cosData)
	sinAngles, _ := tensor.New([]int{seqLen, halfDim}, sinData)

	// Run fused pipeline twice on same input -- must be deterministic.
	norm1, _, _ := FusedRMSNorm(input, weight, 1e-6)
	rope1, _ := FusedRoPE(norm1, cosAngles, sinAngles, dim)

	norm2, _, _ := FusedRMSNorm(input, weight, 1e-6)
	rope2, _ := FusedRoPE(norm2, cosAngles, sinAngles, dim)

	data1 := rope1.Data()
	data2 := rope2.Data()

	for i := range data1 {
		if data1[i] != data2[i] {
			t.Fatalf("non-deterministic at index %d: %f vs %f", i, data1[i], data2[i])
		}
	}

	// Compare with unfused to confirm error stays bounded after composition.
	squared, _ := engine.Mul(ctx, input, input, nil)
	meanSq, _ := engine.ReduceMean(ctx, squared, 2, true)
	meanSqEps, _ := engine.AddScalar(ctx, meanSq, float32(1e-6), nil)
	rsqrt, _ := engine.Rsqrt(ctx, meanSqEps, nil)
	normalized, _ := engine.Mul(ctx, input, rsqrt, nil)
	unfusedNorm, _ := engine.Mul(ctx, normalized, weight, nil)

	xRot0, _ := unfusedNorm.Slice([2]int{0, batch}, [2]int{0, seqLen}, [2]int{0, halfDim})
	xRot1, _ := unfusedNorm.Slice([2]int{0, batch}, [2]int{0, seqLen}, [2]int{halfDim, dim})
	t1, _ := engine.Mul(ctx, xRot0, cosAngles)
	t2, _ := engine.Mul(ctx, xRot1, sinAngles)
	rotX0, _ := engine.Sub(ctx, t1, t2)
	m1, _ := engine.Mul(ctx, xRot1, cosAngles)
	m2, _ := engine.Mul(ctx, xRot0, sinAngles)
	rotX1, _ := engine.Add(ctx, m1, m2)
	unfusedRope, _ := engine.Concat(ctx, []*tensor.TensorNumeric[float32]{rotX0, rotX1}, 2)

	unfusedData := unfusedRope.Data()
	maxDiff := float64(0)
	for i := range data1 {
		diff := math.Abs(float64(data1[i] - unfusedData[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("composed fused vs unfused max diff: %e", maxDiff)
	if maxDiff > 1e-5 {
		t.Errorf("composed max diff %e exceeds 1e-5", maxDiff)
	}
}
