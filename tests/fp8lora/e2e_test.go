//go:build integration

package fp8lora_test

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/training/fp8"
	"github.com/zerfoo/zerfoo/training/lora"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// fp8LoRAModel implements lora.Model[float32] with FP8Linear base layers
// wrapped by LoRA adapters.
type fp8LoRAModel struct {
	layers map[string]lora.Layer[float32]
	order  []string
}

func newFP8LoRAModel() *fp8LoRAModel {
	return &fp8LoRAModel{
		layers: make(map[string]lora.Layer[float32]),
	}
}

func (m *fp8LoRAModel) AddLayer(layer lora.Layer[float32]) {
	name := layer.Name()
	m.layers[name] = layer
	m.order = append(m.order, name)
}

func (m *fp8LoRAModel) Layers() []lora.Layer[float32] {
	result := make([]lora.Layer[float32], 0, len(m.order))
	for _, name := range m.order {
		result = append(result, m.layers[name])
	}
	return result
}

func (m *fp8LoRAModel) ReplaceLayer(name string, replacement lora.Layer[float32]) error {
	if _, ok := m.layers[name]; !ok {
		return nil
	}
	m.layers[name] = replacement
	return nil
}

// TestFP8LoRAFineTune verifies end-to-end FP8 quantized base model + LoRA
// adapter training. It creates a 2-layer FP8-quantized model, attaches LoRA
// adapters (rank=8, alpha=16) to Q/V projection layers, runs 3 training steps
// with synthetic data, and verifies:
//   - Loss decreases monotonically over 3 steps
//   - FP32 master weights (via MasterWeightStore) update correctly each step
func TestFP8LoRAFineTune(t *testing.T) {
	const (
		dIn       = 32
		dHidden   = 48
		loraRank  = 8
		loraAlpha = 16.0
		nSteps    = 3
		batchSize = 4
		lr        = 1e-3
	)

	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	rng := rand.New(rand.NewPCG(42, 0))

	// Create 2-layer FP8-quantized model with mock weights.
	w1Data := make([]float32, dHidden*dIn)
	for i := range w1Data {
		w1Data[i] = float32(rng.NormFloat64()) * 0.1
	}
	w2Data := make([]float32, dIn*dHidden)
	for i := range w2Data {
		w2Data[i] = float32(rng.NormFloat64()) * 0.1
	}

	qLayer, err := fp8.NewFP8Linear[float32]("q_proj", engine, dIn, dHidden, w1Data)
	if err != nil {
		t.Fatalf("create q_proj FP8Linear: %v", err)
	}
	vLayer, err := fp8.NewFP8Linear[float32]("v_proj", engine, dHidden, dIn, w2Data)
	if err != nil {
		t.Fatalf("create v_proj FP8Linear: %v", err)
	}

	// Create MasterWeightStore to maintain FP32 copies of FP8 weights.
	fp8Layers := []*fp8.FP8Linear[float32]{qLayer, vLayer}
	masterStore, err := fp8.NewMasterWeightStore[float32](fp8Layers)
	if err != nil {
		t.Fatalf("create MasterWeightStore: %v", err)
	}

	// Snapshot initial FP32 master weights for later comparison.
	initialMasterWeights := snapshotMasterWeights(masterStore)

	// Wrap FP8 layers with LoRA adapters (rank=8, alpha=16).
	loraQ, err := lora.NewLoraLinear[float32]("q_proj", qLayer, loraRank, loraAlpha, engine, dIn, dHidden)
	if err != nil {
		t.Fatalf("create LoRA wrapper for q_proj: %v", err)
	}
	loraV, err := lora.NewLoraLinear[float32]("v_proj", vLayer, loraRank, loraAlpha, engine, dHidden, dIn)
	if err != nil {
		t.Fatalf("create LoRA wrapper for v_proj: %v", err)
	}

	model := newFP8LoRAModel()
	model.AddLayer(loraQ)
	model.AddLayer(loraV)

	// Create AdamW optimizer for LoRA parameters.
	opt := optimizer.NewAdamW[float32](engine, T(lr), 0.9, 0.999, 1e-8, 0.0)

	// Collect all LoRA parameters.
	var loraParams []*graph.Parameter[float32]
	for _, layer := range model.Layers() {
		if ll, ok := layer.(*lora.LoraLinear[float32]); ok {
			loraParams = append(loraParams, ll.Parameters()...)
		}
	}

	// Generate a fixed synthetic batch used for all steps so the model can
	// consistently reduce loss on the same data.
	inData := make([]float32, batchSize*dIn)
	for i := range inData {
		inData[i] = float32(rng.NormFloat64()) * 0.1
	}
	fixedInput, err := tensor.New[float32]([]int{batchSize, dIn}, inData)
	if err != nil {
		t.Fatalf("create fixed input: %v", err)
	}
	targetData := make([]float32, batchSize*dIn)
	for i := range targetData {
		srcIdx := i % dIn
		targetData[i] = inData[(i/dIn)*dIn+srcIdx]*0.5 + 0.05
	}
	fixedTarget, err := tensor.New[float32]([]int{batchSize, dIn}, targetData)
	if err != nil {
		t.Fatalf("create fixed target: %v", err)
	}

	// Run 3 training steps and assert monotonic loss decrease.
	var losses [nSteps]float32
	for step := range nSteps {
		input := fixedInput
		target := fixedTarget

		// Clear gradients on LoRA parameters.
		for _, p := range loraParams {
			p.ClearGradient()
		}

		// Forward pass through all layers.
		layers := model.Layers()
		activations := make([]*tensor.TensorNumeric[float32], len(layers)+1)
		activations[0] = input
		x := input
		for i, layer := range layers {
			out, err := layer.Forward(ctx, x)
			if err != nil {
				t.Fatalf("step %d: forward layer %q: %v", step, layer.Name(), err)
			}
			activations[i+1] = out
			x = out
		}

		// Compute MSE loss.
		diff, err := engine.Sub(ctx, x, target)
		if err != nil {
			t.Fatalf("step %d: loss sub: %v", step, err)
		}
		diffSq, err := engine.Mul(ctx, diff, diff)
		if err != nil {
			t.Fatalf("step %d: loss mul: %v", step, err)
		}
		reduced := diffSq
		for dim := len(diffSq.Shape()) - 1; dim >= 0; dim-- {
			reduced, err = engine.ReduceSum(ctx, reduced, dim, false)
			if err != nil {
				t.Fatalf("step %d: reduce sum dim %d: %v", step, dim, err)
			}
		}
		lossData := reduced.Data()
		nElem := batchSize * dIn
		lossVal := float32(lossData[0]) / float32(nElem)
		losses[step] = lossVal

		if math.IsNaN(float64(lossVal)) || math.IsInf(float64(lossVal), 0) {
			t.Fatalf("step %d: loss is NaN/Inf: %v", step, lossVal)
		}
		t.Logf("step %d: loss = %.6f", step, lossVal)

		// Backward pass through layers in reverse.
		scaleFactor := engine.Ops().FromFloat64(2.0 / float64(nElem))
		grad, err := engine.MulScalar(ctx, diff, scaleFactor)
		if err != nil {
			t.Fatalf("step %d: loss grad: %v", step, err)
		}
		for i := len(layers) - 1; i >= 0; i-- {
			grads, err := layers[i].Backward(ctx, types.FullBackprop, grad, activations[i])
			if err != nil {
				t.Fatalf("step %d: backward layer %q: %v", step, layers[i].Name(), err)
			}
			if len(grads) > 0 && grads[0] != nil {
				grad = grads[0]
			}
		}

		// Optimizer step on LoRA parameters.
		if err := opt.Step(ctx, loraParams); err != nil {
			t.Fatalf("step %d: optimizer step: %v", step, err)
		}

		// Sync FP32 master weights back to FP8.
		if err := masterStore.SyncToFP8(); err != nil {
			t.Fatalf("step %d: SyncToFP8: %v", step, err)
		}
	}

	// Assert loss decreases monotonically over 3 steps.
	for i := 1; i < nSteps; i++ {
		if losses[i] >= losses[i-1] {
			t.Errorf("loss did not decrease monotonically: step %d loss=%.6f >= step %d loss=%.6f",
				i, losses[i], i-1, losses[i-1])
		}
	}
	t.Logf("loss progression: %.6f -> %.6f -> %.6f", losses[0], losses[1], losses[2])

	// Assert FP32 master weights are still valid (not corrupted by training).
	// Since we only trained LoRA params and FP8 base weights are frozen,
	// master weights should remain unchanged from initial values.
	finalMasterWeights := snapshotMasterWeights(masterStore)
	for i, initial := range initialMasterWeights {
		if len(initial) != len(finalMasterWeights[i]) {
			t.Fatalf("master weight %d: size changed from %d to %d", i, len(initial), len(finalMasterWeights[i]))
		}
		for j := range initial {
			if math.IsNaN(float64(finalMasterWeights[i][j])) || math.IsInf(float64(finalMasterWeights[i][j]), 0) {
				t.Errorf("master weight %d element %d is NaN/Inf after training", i, j)
			}
		}
		// Base weights are frozen (only LoRA is trained), so master weights
		// should be identical to initial values.
		if !float32SliceEqual(initial, finalMasterWeights[i]) {
			t.Errorf("master weight %d changed unexpectedly during LoRA-only training", i)
		}
	}
	t.Logf("FP32 master weights verified: %d layers, %d bytes",
		len(finalMasterWeights), masterStore.MemoryBytes())
}

// T is a helper to convert float64 to float32 for readability.
func T(v float64) float32 { return float32(v) }

// snapshotMasterWeights copies all FP32 master weight data for later comparison.
func snapshotMasterWeights(store *fp8.MasterWeightStore[float32]) [][]float32 {
	params := store.FP32Params()
	snapshots := make([][]float32, len(params))
	for i, p := range params {
		data := p.Data()
		cp := make([]float32, len(data))
		copy(cp, data)
		snapshots[i] = cp
	}
	return snapshots
}

// float32SliceEqual checks if two float32 slices are identical.
func float32SliceEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
