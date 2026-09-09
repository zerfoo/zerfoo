//go:build r01audit

package tabular

import (
	"bytes"
	"context"
	"math"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	writer "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// This is a format feasibility spike, not a public deployment loader. R07
// must implement the manifest, validation, atomic writes and legacy migration.
func TestR01GGUFClassifierSpike(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	model, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{2}}, engine, ops)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"layer0.weights": model.layers[0].weights, "layer0.biases": model.layers[0].biases,
		"head.weights": model.head.weights, "head.biases": model.head.biases,
	}
	values := map[string][]float32{
		"layer0.weights": {1, 0, 0, 1}, "layer0.biases": {0.25, 0.5},
		"head.weights": {1, 0, -1, 0, 1, -1}, "head.biases": {0, 0, 0},
	}
	gw := writer.NewWriter()
	gw.AddMetadataString("general.architecture", "zerfoo.tabular.mlp.v1")
	for _, name := range []string{"layer0.weights", "layer0.biases", "head.weights", "head.biases"} {
		tensors[name].SetData(values[name])
		gw.AddTensorF32(name, tensors[name].Shape(), values[name])
	}
	var buf bytes.Buffer
	if err := gw.Write(&buf); err != nil {
		t.Fatal(err)
	}
	reader := bytes.NewReader(buf.Bytes())
	parsed, err := gguf.Parse(reader)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := gguf.LoadTensors(parsed, reader)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded) != 4 {
		t.Fatalf("tensor count %d", len(loaded))
	}
	for name, before := range tensors {
		after, ok := loaded[name]
		if !ok || !reflect.DeepEqual(before.Shape(), after.Shape()) || !reflect.DeepEqual(before.Data(), after.Data()) {
			t.Fatalf("tensor round-trip mismatch: %s", name)
		}
	}
	restored := &Model{config: model.config, engine: engine, ops: ops, layers: []mlpLayer{{weights: loaded["layer0.weights"], biases: loaded["layer0.biases"]}}, head: mlpLayer{weights: loaded["head.weights"], biases: loaded["head.biases"]}}
	for _, input := range [][]float64{{1, 2}, {-1, 1}, {2, -1}} {
		h0, h1 := math.Max(0, input[0]+0.25), math.Max(0, input[1]+0.5)
		logits := []float64{h0, h1, -h0 - h1}
		best := 0
		for i := 1; i < 3; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		sum := 0.0
		for _, v := range logits {
			sum += math.Exp(v - logits[best])
		}
		want := 1 / sum
		for _, m := range []*Model{model, restored} {
			label, confidence, err := m.Predict(input)
			if err != nil {
				t.Fatal(err)
			}
			delta := math.Abs(confidence - want)
			if int(label) != best || math.IsNaN(confidence) || delta > 1e-6+1e-5*math.Abs(want) {
				t.Fatalf("prediction got=(%d,%g), want=(%d,%g)", label, confidence, best, want)
			}
			t.Logf("input=%v class=%d confidence=%g abs_error=%g", input, label, confidence, delta)
		}
	}
	t.Logf("GGUF bytes=%d tensors=%d; public tabular GGUF loader remains missing", buf.Len(), len(loaded))
}

func TestR01ReLUGradientFiniteDifference(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)
	ctx := context.Background()
	model, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{2}}, engine, ops)
	if err != nil {
		t.Fatal(err)
	}
	model.layers[0].weights.SetData([]float32{0.2, -0.3, 0.4, 0.1})
	model.layers[0].biases.SetData([]float32{1, 1})
	model.head.weights.SetData([]float32{0.3, -0.2, 0.1, -0.1, 0.4, 0.2})
	model.head.biases.SetData([]float32{0.1, -0.1, 0.2})
	input, err := tensor.New[float32]([]int{2, 2}, []float32{1, 0.5, -0.5, 1})
	if err != nil {
		t.Fatal(err)
	}
	labels := []int{0, 2}
	logits, acts, pre, err := forwardPass(ctx, model, input)
	if err != nil {
		t.Fatal(err)
	}
	_, softmax, err := crossEntropyLoss(ctx, engine, logits, labels, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	params, err := buildParams(model)
	if err != nil {
		t.Fatal(err)
	}
	if err := backwardPass(ctx, model, engine, ops, params, acts, pre, input, softmax, labels, 2, 3); err != nil {
		t.Fatal(err)
	}
	// Scalar float64 reference computes its own forward/mean cross entropy.
	reference := func() float64 {
		loss := 0.0
		for row := 0; row < 2; row++ {
			h := [2]float64{}
			for j := 0; j < 2; j++ {
				h[j] = float64(model.layers[0].biases.Data()[j])
				for i := 0; i < 2; i++ {
					h[j] += float64(input.Data()[row*2+i]) * float64(model.layers[0].weights.Data()[i*2+j])
				}
				h[j] = math.Max(0, h[j])
			}
			z := [3]float64{}
			for j := 0; j < 3; j++ {
				z[j] = float64(model.head.biases.Data()[j])
				for i := 0; i < 2; i++ {
					z[j] += h[i] * float64(model.head.weights.Data()[i*3+j])
				}
			}
			mx := math.Max(z[0], math.Max(z[1], z[2]))
			sum := 0.0
			for _, v := range z {
				sum += math.Exp(v - mx)
			}
			loss += mx + math.Log(sum) - z[labels[row]]
		}
		return loss / 2
	}
	maxErr := 0.0
	count := 0
	for _, p := range params {
		for i, got := range p.Gradient.Data() {
			data := p.Value.Data()
			old := data[i]
			hi, lo := old+0.001, old-0.001
			data[i] = hi
			p.Value.SetData(data)
			plus := reference()
			data[i] = lo
			p.Value.SetData(data)
			minus := reference()
			data[i] = old
			p.Value.SetData(data)
			want := (plus - minus) / float64(hi-lo)
			delta := math.Abs(float64(got) - want)
			if math.IsNaN(float64(got)) || delta > 2e-5+2e-4*math.Abs(want) {
				t.Fatalf("%s[%d] got=%g want=%g error=%g", p.Name, i, got, want, delta)
			}
			maxErr = math.Max(maxErr, delta)
			count++
		}
	}
	t.Logf("independent scalar central differences: parameters=%d max_abs_error=%g", count, maxErr)
}
