package training_test

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/training/lora"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// loraTestLinear is a minimal Linear layer with NF4-quantized base weights
// for LoRA integration testing. OpType() returns "Linear" so InjectLoRA
// can target it.
type loraTestLinear struct {
	weights   *graph.Parameter[float32]
	engine    compute.Engine[float32]
	layerName string
	dIn, dOut int
}

func newLoraTestLinear(name string, engine compute.Engine[float32], dIn, dOut int, data []float32) (*loraTestLinear, error) {
	nf4 := tensor.NewNF4Storage(data, []int{dIn, dOut})
	wTensor, err := tensor.NewWithStorage[float32]([]int{dIn, dOut}, nf4)
	if err != nil {
		return nil, err
	}
	param, err := graph.NewParameter[float32](name+"_weights", wTensor, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return &loraTestLinear{
		weights:   param,
		engine:    engine,
		layerName: name,
		dIn:       dIn,
		dOut:      dOut,
	}, nil
}

func (l *loraTestLinear) OpType() string                          { return "Linear" }
func (l *loraTestLinear) Attributes() map[string]any              { return nil }
func (l *loraTestLinear) OutputShape() []int                      { return []int{-1, l.dOut} }
func (l *loraTestLinear) Parameters() []*graph.Parameter[float32] { return []*graph.Parameter[float32]{l.weights} }
func (l *loraTestLinear) Name() string                            { return l.layerName }

func (l *loraTestLinear) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return l.engine.MatMul(ctx, inputs[0], l.weights.Value)
}

func (l *loraTestLinear) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	wT, err := l.engine.Transpose(ctx, l.weights.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dx, err := l.engine.MatMul(ctx, outputGradient, wT)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[float32]{dx}, nil
}

// loraTestModel implements lora.Model[float32] for integration testing.
type loraTestModel struct {
	layers map[string]lora.Layer[float32]
	order  []string
}

func newLoraTestModel() *loraTestModel {
	return &loraTestModel{
		layers: make(map[string]lora.Layer[float32]),
	}
}

func (m *loraTestModel) AddLayer(layer lora.Layer[float32]) {
	name := layer.Name()
	m.layers[name] = layer
	m.order = append(m.order, name)
}

func (m *loraTestModel) Layers() []lora.Layer[float32] {
	result := make([]lora.Layer[float32], 0, len(m.order))
	for _, name := range m.order {
		result = append(result, m.layers[name])
	}
	return result
}

func (m *loraTestModel) ReplaceLayer(name string, replacement lora.Layer[float32]) error {
	if _, ok := m.layers[name]; !ok {
		return nil
	}
	m.layers[name] = replacement
	return nil
}

// instructionPair represents a synthetic instruction fine-tuning example.
type instructionPair struct {
	input  *tensor.TensorNumeric[float32]
	target *tensor.TensorNumeric[float32]
}

// generateInstructionPairs creates n synthetic instruction pairs.
// Each pair maps a random input embedding to a deterministic target embedding
// based on a simple pattern (scaled + shifted), simulating the kind of
// input/output mapping a LoRA adapter would learn.
func generateInstructionPairs(rng *rand.Rand, n, dIn, dOut int) []instructionPair {
	pairs := make([]instructionPair, n)
	for i := range n {
		inData := make([]float32, dIn)
		for j := range inData {
			inData[j] = float32(rng.NormFloat64()) * 0.1
		}
		inTensor, _ := tensor.New[float32]([]int{1, dIn}, inData)

		outData := make([]float32, dOut)
		for j := range outData {
			srcIdx := j % dIn
			outData[j] = inData[srcIdx]*0.5 + 0.1*float32(j%3)
		}
		outTensor, _ := tensor.New[float32]([]int{1, dOut}, outData)

		pairs[i] = instructionPair{input: inTensor, target: outTensor}
	}
	return pairs
}

// TestLoRAFinetune verifies end-to-end LoRA fine-tuning with QLoRATrainer
// and AdamW8bit on 1000 synthetic instruction pairs.
//
// Acceptance criteria:
//   - Fine-tune on 1000 synthetic instruction pairs
//   - Eval loss decreases from baseline after training
//   - Adapter size < 50MB when saved
func TestLoRAFinetune(t *testing.T) {
	const (
		dModel    = 32
		dHidden   = 64
		dOut      = 32
		loraRank  = 4
		loraAlpha = 8.0
		nPairs    = 1000
		nEpochs   = 3
		batchSize = 4
		evalSize  = 100
	)

	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	rng := rand.New(rand.NewPCG(42, 0))

	// Build a 3-layer synthetic model with NF4-quantized base weights.
	w1Data := make([]float32, dModel*dHidden)
	for i := range w1Data {
		w1Data[i] = float32(rng.NormFloat64()) * 0.05
	}
	w2Data := make([]float32, dHidden*dHidden)
	for i := range w2Data {
		w2Data[i] = float32(rng.NormFloat64()) * 0.05
	}
	w3Data := make([]float32, dHidden*dOut)
	for i := range w3Data {
		w3Data[i] = float32(rng.NormFloat64()) * 0.05
	}

	model := newLoraTestModel()

	layer1, err := newLoraTestLinear("q_proj", engine, dModel, dHidden, w1Data)
	if err != nil {
		t.Fatalf("create layer1: %v", err)
	}
	layer2, err := newLoraTestLinear("v_proj", engine, dHidden, dHidden, w2Data)
	if err != nil {
		t.Fatalf("create layer2: %v", err)
	}
	layer3, err := newLoraTestLinear("o_proj", engine, dHidden, dOut, w3Data)
	if err != nil {
		t.Fatalf("create layer3: %v", err)
	}
	model.AddLayer(layer1)
	model.AddLayer(layer2)
	model.AddLayer(layer3)

	// Generate 1000 synthetic instruction pairs.
	allPairs := generateInstructionPairs(rng, nPairs, dModel, dOut)
	trainPairs := allPairs[:nPairs-evalSize]
	evalPairs := allPairs[nPairs-evalSize:]

	// Create AdamW8bit optimizer (4x memory savings over FP32 AdamW).
	opt := optimizer.NewAdamW8bit[float32](engine, 1e-4, 0.9, 0.999, 1e-8, 0.0)

	// Create QLoRA trainer — injects LoRA adapters into all three layers.
	trainer, err := lora.NewQLoRATrainer[float32](
		model, loraRank, loraAlpha,
		[]string{"q_proj", "v_proj", "o_proj"},
		engine, opt,
	)
	if err != nil {
		t.Fatalf("NewQLoRATrainer: %v", err)
	}

	// Verify LoRA injection.
	for _, layer := range model.Layers() {
		if layer.OpType() != "LoraLinear" {
			t.Fatalf("expected layer %q to be LoraLinear after injection, got %q",
				layer.Name(), layer.OpType())
		}
	}

	// Compute baseline eval loss before training.
	baselineLoss := loraEvalLoss(t, ctx, model, evalPairs)
	t.Logf("baseline eval loss: %.6f", baselineLoss)

	// Training loop: iterate over instruction pairs in mini-batches.
	var stepCount int
	for epoch := range nEpochs {
		// Shuffle training data each epoch.
		shuffled := make([]instructionPair, len(trainPairs))
		copy(shuffled, trainPairs)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		var epochLossSum float64
		var epochSteps int

		for i := 0; i < len(shuffled); i += batchSize {
			end := i + batchSize
			if end > len(shuffled) {
				end = len(shuffled)
			}
			batch := shuffled[i:end]

			batchIn, batchTarget := concatBatch(t, batch)

			loss, err := trainer.Step(ctx, batchIn, batchTarget)
			if err != nil {
				t.Fatalf("epoch %d step %d: Step failed: %v", epoch, stepCount, err)
			}
			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				t.Fatalf("epoch %d step %d: loss is NaN/Inf: %v", epoch, stepCount, loss)
			}

			epochLossSum += float64(loss)
			epochSteps++
			stepCount++
		}

		avgEpochLoss := epochLossSum / float64(epochSteps)
		t.Logf("epoch %d: avg train loss = %.6f (%d steps)", epoch, avgEpochLoss, epochSteps)
	}

	// Compute final eval loss after training.
	finalEvalLoss := loraEvalLoss(t, ctx, model, evalPairs)
	t.Logf("final eval loss: %.6f (baseline: %.6f)", finalEvalLoss, baselineLoss)

	// Assert: eval loss decreased from baseline.
	if finalEvalLoss >= baselineLoss {
		t.Errorf("eval loss did not decrease: baseline=%.6f final=%.6f", baselineLoss, finalEvalLoss)
	}

	// Assert: adapter size < 50MB.
	params := trainer.TrainableParams()
	var totalElements int
	for _, p := range params {
		shape := p.Value.Shape()
		elems := 1
		for _, d := range shape {
			elems *= d
		}
		totalElements += elems
	}
	adapterBytes := totalElements * int(unsafe.Sizeof(float32(0)))
	adapterMB := float64(adapterBytes) / (1024 * 1024)
	t.Logf("adapter size: %d parameters, %.2f MB", totalElements, adapterMB)

	const maxAdapterMB = 50.0
	if adapterMB >= maxAdapterMB {
		t.Errorf("adapter size %.2f MB exceeds limit of %.0f MB", adapterMB, maxAdapterMB)
	}

	reduction := (1 - finalEvalLoss/baselineLoss) * 100
	t.Logf("loss reduction: %.1f%% (baseline=%.6f -> final=%.6f)", reduction, baselineLoss, finalEvalLoss)
}

// loraEvalLoss computes the average MSE loss over eval pairs by running
// forward through all model layers without updating weights.
func loraEvalLoss(t *testing.T, ctx context.Context, model *loraTestModel, pairs []instructionPair) float32 {
	t.Helper()
	var totalLoss float64
	for _, pair := range pairs {
		x := pair.input
		for _, layer := range model.Layers() {
			out, err := layer.Forward(ctx, x)
			if err != nil {
				t.Fatalf("eval forward failed: %v", err)
			}
			x = out
		}
		outData := x.Data()
		targetData := pair.target.Data()
		var mse float64
		for i := range outData {
			diff := float64(outData[i] - targetData[i])
			mse += diff * diff
		}
		mse /= float64(len(outData))
		totalLoss += mse
	}
	return float32(totalLoss / float64(len(pairs)))
}

// concatBatch concatenates multiple instruction pairs into a single batch.
func concatBatch(t *testing.T, pairs []instructionPair) (*tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
	t.Helper()
	n := len(pairs)
	if n == 0 {
		t.Fatal("empty batch")
	}
	dIn := len(pairs[0].input.Data())
	dOut := len(pairs[0].target.Data())

	inData := make([]float32, n*dIn)
	outData := make([]float32, n*dOut)
	for i, p := range pairs {
		copy(inData[i*dIn:], p.input.Data())
		copy(outData[i*dOut:], p.target.Data())
	}

	inTensor, err := tensor.New[float32]([]int{n, dIn}, inData)
	if err != nil {
		t.Fatalf("concat input: %v", err)
	}
	outTensor, err := tensor.New[float32]([]int{n, dOut}, outData)
	if err != nil {
		t.Fatalf("concat target: %v", err)
	}
	return inTensor, outTensor
}
