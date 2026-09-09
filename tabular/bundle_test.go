package tabular

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestClassifierBundleFreshProcess(t *testing.T) {
	if path := os.Getenv("ZERFOO_TEST_BUNDLE_PATH"); path != "" {
		bundle, err := LoadClassifierBundle(context.Background(), path, os.Getenv("ZERFOO_TEST_BUNDLE_ID"), compute.NewCPUEngine(numeric.Float32Ops{}))
		if err != nil {
			t.Fatal(err)
		}
		prediction, err := bundle.PredictRaw(context.Background(), []float64{5.1, 3.5, 1.4, 0.2})
		if err != nil {
			t.Fatal(err)
		}
		raw, err := json.Marshal(prediction)
		if err != nil {
			t.Fatal(err)
		}
		if string(raw) != os.Getenv("ZERFOO_TEST_BUNDLE_PREDICTION") {
			t.Fatalf("fresh process changed prediction: %s", raw)
		}
		return
	}
	data := classifierFixture(t)
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	model, err := NewClassifier(ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: data.Manifest().Labels, HiddenDims: []int{16}, Seed: 42}, engine)
	if err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "model")
	id, err := SaveClassifierBundle(context.Background(), output, model, data, "")
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := LoadClassifierBundle(context.Background(), output, id, engine)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range model.params {
		if !reflect.DeepEqual(p.Value.Data(), bundle.Model.params[i].Value.Data()) {
			t.Fatal("weights changed")
		}
	}
	originalRow, err := data.Manifest().Preprocessing.Transform([]float64{5.1, 3.5, 1.4, 0.2})
	if err != nil {
		t.Fatal(err)
	}
	expected, err := model.Predict(context.Background(), originalRow)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(expected)
	if err != nil {
		t.Fatal(err)
	}
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	command := exec.CommandContext(context.Background(), binary, "-test.run=^TestClassifierBundleFreshProcess$", "-test.v")
	command.Env = append(os.Environ(), "ZERFOO_TEST_BUNDLE_PATH="+output, "ZERFOO_TEST_BUNDLE_ID="+id, "ZERFOO_TEST_BUNDLE_PREDICTION="+string(raw))
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("fresh-process load: %v\n%s", err, output)
	}
}

func TestClassifierBundleRejectsCorruption(t *testing.T) {
	data := classifierFixture(t)
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	model, err := NewClassifier(ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: data.Manifest().Labels}, engine)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"weights", "manifest_hash", "version", "shape", "preprocessing", "unknown_field", "wrong_identity"} {
		t.Run(kind, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bundle")
			id, err := SaveClassifierBundle(context.Background(), path, model, data, "")
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "weights":
				if err := os.WriteFile(filepath.Join(path, "weights.gguf"), []byte("GGUF"), 0600); err != nil {
					t.Fatal(err)
				}
			case "manifest_hash":
				if err := os.WriteFile(filepath.Join(path, "bundle.sha256"), []byte("wrong"), 0600); err != nil {
					t.Fatal(err)
				}
			case "wrong_identity":
				id = "wrong"
			default:
				raw, err := os.ReadFile(filepath.Join(path, "manifest.json"))
				if err != nil {
					t.Fatal(err)
				}
				var manifest map[string]any
				if err := json.Unmarshal(raw, &manifest); err != nil {
					t.Fatal(err)
				}
				switch kind {
				case "version":
					manifest["version"] = 99
				case "shape":
					manifest["config"].(map[string]any)["input_dim"] = 5
				case "preprocessing":
					manifest["preprocessing"].(map[string]any)["scale"] = []float64{0, 1, 1, 1}
				case "unknown_field":
					manifest["unexpected"] = true
				}
				raw, err = json.Marshal(manifest)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(path, "manifest.json"), raw, 0600); err != nil {
					t.Fatal(err)
				}
				id = contentHash(raw)
				if err := os.WriteFile(filepath.Join(path, "bundle.sha256"), []byte(id), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := LoadClassifierBundle(context.Background(), path, id, engine); err == nil {
				t.Fatal("corrupt bundle accepted")
			}
		})
	}
}

func TestClassifierLegacyMigration(t *testing.T) {
	var csv strings.Builder
	csv.WriteString("a,b,label\n")
	for i := 0; i < 30; i++ {
		csv.WriteString(strings.Join([]string{string(rune('1' + i%8)), string(rune('1' + i/8)), string(rune('0' + i%3))}, ",") + "\n")
	}
	data, err := InspectCSV(context.Background(), strings.NewReader(csv.String()), DatasetOptions{Target: "label", Split: "stratified", Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	legacy, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{4}}, engine, numeric.Float32Ops{})
	if err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(t.TempDir(), "legacy.ztab")
	if err := Save(legacy, source); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "migrated")
	id, err := MigrateLegacyClassifier(context.Background(), source, output, data, []string{"0", "1", "2"}, engine)
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := LoadClassifierBundle(context.Background(), output, id, engine)
	if err != nil {
		t.Fatal(err)
	}
	// Legacy models consume preprocessed numeric inputs. Compare at that exact boundary.
	for _, row := range [][]float64{{1, 2}, {2, 1}, {0, 0}} {
		direction, confidence, err := legacy.Predict(row)
		if err != nil {
			t.Fatal(err)
		}
		got, err := bundle.Model.Predict(context.Background(), row)
		if err != nil {
			t.Fatal(err)
		}
		if int(direction) != got.ClassID || confidence != got.Probabilities[got.ClassID] {
			t.Fatalf("migration changed legacy prediction: %v vs %+v", direction, got)
		}
	}
}
