package tabular

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func classifierFixture(t *testing.T) *Dataset {
	t.Helper()
	raw, err := os.ReadFile("testdata/model_creation/iris.csv")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := os.ReadFile("testdata/model_creation/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Splits map[string][]int `json:"splits"`
	}
	if err := json.Unmarshal(manifest, &fixture); err != nil {
		t.Fatal(err)
	}
	data, err := InspectCSV(context.Background(), strings.NewReader(string(raw)), DatasetOptions{Target: "species", Split: "stratified", Assignments: fixture.Splits})
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestClassifierTrainingIris(t *testing.T) {
	data := classifierFixture(t)
	manifest := data.Manifest()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	for _, hidden := range [][]int{nil, {16}} {
		for _, seed := range []uint64{17, 42, 91} {
			config := ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: manifest.Labels, HiddenDims: hidden, Seed: seed}
			result, err := FitClassifier(context.Background(), data, config, FitOptions{Epochs: 200, BatchSize: 15, LearningRate: 0.01, WeightDecay: 0.0001, Seed: seed, MaxSteps: 1200}, engine, nil)
			if err != nil {
				t.Fatal(err)
			}
			if result.Status != "succeeded" || result.Steps != 1200 {
				t.Fatalf("wrong training status/steps: %+v", result)
			}
			rows, labels, err := data.Partition("test")
			if err != nil {
				t.Fatal(err)
			}
			predictions, err := result.Model.PredictBatch(context.Background(), rows)
			if err != nil {
				t.Fatal(err)
			}
			correct := 0
			confusion := [3][3]int{}
			ce := 0.0
			for i, p := range predictions {
				confusion[labels[i]][p.ClassID]++
				if labels[i] == p.ClassID {
					correct++
				}
				ce -= math.Log(math.Max(1e-7, p.Probabilities[labels[i]]))
			}
			macro := 0.0
			for c := 0; c < 3; c++ {
				truth, predicted := 0, 0
				for j := 0; j < 3; j++ {
					truth += confusion[c][j]
					predicted += confusion[j][c]
				}
				macro += 2 * float64(confusion[c][c]) / float64(truth+predicted) / 3
			}
			ce /= float64(len(rows))
			t.Logf("hidden=%v seed=%d steps=%d correct=%d/30 macro_f1=%g mean_ce=%g", hidden, seed, result.Steps, correct, macro, ce)
			if correct < 26 || macro < 0.85 || ce > 0.6 {
				t.Fatalf("frozen convergence threshold failed: correct=%d macro=%g CE=%g", correct, macro, ce)
			}
		}
	}
}

func TestClassifierTrainingCancellationAndLimits(t *testing.T) {
	data := classifierFixture(t)
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	config := ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: data.Manifest().Labels, Seed: 42}
	options := FitOptions{Epochs: 2, BatchSize: 15, LearningRate: 0.01, Seed: 42, MaxSteps: 1}
	limited, err := FitClassifier(context.Background(), data, config, options, engine, nil)
	if !errors.Is(err, ErrTrainingLimit) || limited.Status != "budget_exhausted" || limited.Steps != 1 {
		t.Fatalf("wrong limit result: %+v %v", limited, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	options.MaxSteps = 0
	canceled, err := FitClassifier(ctx, data, config, options, engine, func(TrainingProgress) error { cancel(); return nil })
	if !errors.Is(err, context.Canceled) || canceled.Status != "canceled" || canceled.Steps != 1 {
		t.Fatalf("wrong cancellation result: %+v %v", canceled, err)
	}
	first, err := FitClassifier(context.Background(), data, config, options, engine, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := FitClassifier(context.Background(), data, config, options, engine, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range first.Model.params {
		if !reflect.DeepEqual(p.Value.Data(), second.Model.params[i].Value.Data()) {
			t.Fatal("seed replay mismatch")
		}
	}
	initial, err := NewClassifier(config, engine)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(initial.params[0].Value.Data(), first.Model.params[0].Value.Data()) {
		t.Fatal("weights did not change")
	}
}

func TestClassifierTrainingUsesLabels(t *testing.T) {
	original := classifierFixture(t)
	raw, err := os.ReadFile("testdata/model_creation/iris.csv")
	if err != nil {
		t.Fatal(err)
	}
	swapped := strings.NewReplacer("Iris-setosa", "Iris-versicolor", "Iris-versicolor", "Iris-setosa").Replace(string(raw))
	options := original.Manifest().Options
	options.Assignments = original.Manifest().Assignments
	changed, err := InspectCSV(context.Background(), strings.NewReader(swapped), options)
	if err != nil {
		t.Fatal(err)
	}
	config := ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: original.Manifest().Labels, Seed: 42}
	fit := FitOptions{Epochs: 200, BatchSize: 15, LearningRate: 0.01, WeightDecay: 0.0001, Seed: 42}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	first, err := FitClassifier(context.Background(), original, config, fit, engine, nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := FitClassifier(context.Background(), changed, config, fit, engine, nil)
	if err != nil {
		t.Fatal(err)
	}
	rows, labels, err := original.Partition("validation")
	if err != nil {
		t.Fatal(err)
	}
	before, err := first.Model.PredictBatch(context.Background(), rows)
	if err != nil {
		t.Fatal(err)
	}
	after, err := second.Model.PredictBatch(context.Background(), rows)
	if err != nil {
		t.Fatal(err)
	}
	differing := 0
	for i := range rows {
		if labels[i] < 2 && before[i].ClassID != after[i].ClassID {
			differing++
		}
	}
	if differing < 18 {
		t.Fatalf("swapping labels changed only %d of the 20 affected predictions", differing)
	}
	if first.DatasetHash == second.DatasetHash {
		t.Fatal("label changes retained run identity")
	}
}
