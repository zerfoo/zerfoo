package tabular

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestClassifierContracts(t *testing.T) {
	t.Run("float32", func(t *testing.T) { checkClassifierContracts(t, compute.NewCPUEngine(numeric.Float32Ops{})) })
	t.Run("float64", func(t *testing.T) { checkClassifierContracts(t, compute.NewCPUEngine(numeric.Float64Ops{})) })
}

func checkClassifierContracts[T tensor.Float](t *testing.T, engine compute.Engine[T]) {
	t.Helper()
	for _, classes := range []int{2, 3, 5} {
		for _, hidden := range [][]int{nil, {2}} {
			t.Run(fmt.Sprintf("classes=%d/hidden=%v", classes, hidden), func(t *testing.T) {
				labels := []string{"α", "accepted", "class five", "🚀", "unknown"}[:classes]
				model, err := NewClassifier(ClassifierConfig{InputDim: 2, ClassCount: classes, Labels: labels, HiddenDims: hidden, Seed: 42}, engine)
				if err != nil {
					t.Fatal(err)
				}
				// Fixed weights give logits [1.25,2.5,0,...] for input [1,2],
				// independently of whether an identity ReLU hidden layer is present.
				for _, p := range model.params {
					p.Value.SetData(make([]T, len(p.Value.Data())))
				}
				if len(hidden) > 0 {
					model.params[0].Value.SetData([]T{1, 0, 0, 1})
				}
				head := model.params[len(model.params)-2].Value
				weights := make([]T, 2*classes)
				weights[0] = 1
				weights[classes+1] = 1
				head.SetData(weights)
				biases := make([]T, classes)
				biases[0] = 0.25
				biases[1] = 0.5
				model.params[len(model.params)-1].Value.SetData(biases)
				got, err := model.Predict(context.Background(), []float64{1, 2})
				if err != nil {
					t.Fatal(err)
				}
				if got.ClassID != 1 || got.Label != labels[1] || len(got.Probabilities) != classes {
					t.Fatalf("bad prediction: %+v", got)
				}
				logits := make([]float64, classes)
				logits[0] = 1.25
				logits[1] = 2.5
				sum := 0.0
				for _, v := range logits {
					sum += math.Exp(v)
				}
				for i, p := range got.Probabilities {
					want := math.Exp(logits[i]) / sum
					if math.Abs(p-want) > 1e-6+1e-5*math.Abs(want) {
						t.Fatalf("probability %d: %g want %g", i, p, want)
					}
				}
				batch, err := model.PredictBatch(context.Background(), [][]float64{{1, 2}, {1, 2}})
				if err != nil {
					t.Fatal(err)
				}
				if len(batch) != 2 || !reflect.DeepEqual(batch[0], got) || !reflect.DeepEqual(batch[1], got) {
					t.Fatalf("batch mismatch: %+v", batch)
				}
				objective, err := model.Loss(context.Background(), [][]float64{{1, 2}, {1, 2}}, []int{1, 0})
				if err != nil {
					t.Fatal(err)
				}
				wantLoss := math.Log(sum) - (2.5+1.25)/2
				if math.Abs(objective-wantLoss) > 1e-6+1e-5*wantLoss {
					t.Fatalf("loss %g want %g", objective, wantLoss)
				}
			})
		}
	}
}

func TestClassifierValidationAndOwnership(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	valid := ClassifierConfig{InputDim: 2, ClassCount: 2, Labels: []string{"no", "yes"}, Seed: 17}
	for name, change := range map[string]func(*ClassifierConfig){
		"zero inputs":  func(c *ClassifierConfig) { c.InputDim = 0 },
		"one class":    func(c *ClassifierConfig) { c.ClassCount = 1 },
		"label count":  func(c *ClassifierConfig) { c.Labels = []string{"only"} },
		"duplicate":    func(c *ClassifierConfig) { c.Labels = []string{"same", "same"} },
		"empty label":  func(c *ClassifierConfig) { c.Labels = []string{"ok", " "} },
		"invalid UTF8": func(c *ClassifierConfig) { c.Labels = []string{"ok", string([]byte{255})} },
		"zero hidden":  func(c *ClassifierConfig) { c.HiddenDims = []int{0} },
		"overflow":     func(c *ClassifierConfig) { c.InputDim = math.MaxInt },
	} {
		t.Run(name, func(t *testing.T) {
			c := valid
			change(&c)
			if _, err := NewClassifier(c, engine); err == nil {
				t.Fatal("invalid config accepted")
			}
		})
	}
	if _, err := NewClassifier[float32](valid, nil); err == nil {
		t.Fatal("nil engine accepted")
	}
	model, err := NewClassifier(valid, engine)
	if err != nil {
		t.Fatal(err)
	}
	same, err := NewClassifier(valid, engine)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range model.params {
		if !reflect.DeepEqual(p.Value.Data(), same.params[i].Value.Data()) {
			t.Fatal("seed not reproducible")
		}
	}
	valid.Labels[0] = "changed"
	config := model.Config()
	config.Labels[1] = "mutated"
	if !reflect.DeepEqual(model.Config().Labels, []string{"no", "yes"}) {
		t.Fatal("caller mutated model labels")
	}
	for name, rows := range map[string][][]float64{
		"empty": nil, "short": {{1}}, "ragged": {{1, 2}, {3}}, "nan": {{math.NaN(), 0}}, "inf": {{0, math.Inf(1)}}, "f32 overflow": {{math.MaxFloat64, 0}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := model.PredictBatch(context.Background(), rows); err == nil {
				t.Fatal("invalid data accepted")
			}
		})
	}
	for _, targets := range [][]int{nil, {-1}, {2}, {0, 1}} {
		if _, err := model.Loss(context.Background(), [][]float64{{1, 2}}, targets); err == nil {
			t.Fatalf("invalid targets accepted: %v", targets)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := model.Predict(ctx, []float64{1, 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

func TestClassifierExtremeLogits(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	model, err := NewClassifier(ClassifierConfig{InputDim: 1, ClassCount: 3, Labels: []string{"a", "b", "c"}}, engine)
	if err != nil {
		t.Fatal(err)
	}
	model.params[0].Value.SetData([]float32{-1000, 0, 1000})
	model.params[1].Value.SetData([]float32{0, 0, 0})
	p, err := model.Predict(context.Background(), []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	if p.ClassID != 2 || p.Probabilities[2] != 1 {
		t.Fatalf("unstable softmax: %+v", p)
	}
	objective, err := model.Loss(context.Background(), [][]float64{{1}}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(objective-2000) > 0.001 {
		t.Fatalf("unstable loss: %g", objective)
	}
}
