package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestTrainTabularPublicCommand(t *testing.T) {
	csvPath := "../../tabular/testdata/model_creation/iris.csv"
	raw, err := os.ReadFile(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	fixtureJSON, err := os.ReadFile("../../tabular/testdata/model_creation/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Splits map[string][]int `json:"splits"`
	}
	if err := json.Unmarshal(fixtureJSON, &fixture); err != nil {
		t.Fatal(err)
	}
	dataset, err := tabular.InspectCSV(context.Background(), bytes.NewReader(raw), tabular.DatasetOptions{Target: "species", Split: "stratified", Assignments: fixture.Splits})
	if err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(t.TempDir(), "dataset.json")
	manifestJSON, err := json.Marshal(dataset.Manifest())
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, manifestJSON, 0600); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "trained.bundle")
	var log bytes.Buffer
	command := NewTrainCommand(&log)
	err = command.Run(context.Background(), []string{"tabular", "--data", csvPath, "--target", "species", "--output", output, "--dataset-manifest", manifestPath, "--recipe", "mlp"})
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(&log)
	progress := 0
	var result tabularTrainEvent
	for {
		var event tabularTrainEvent
		err := decoder.Decode(&event)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("invalid machine output: %v", err)
		}
		if event.Version != 1 {
			t.Fatal("event version")
		}
		if event.Type == "progress" {
			progress++
		}
		if event.Type == "result" {
			result = event
		}
	}
	if progress != 1200 || result.Status != "succeeded" || result.BundleID == "" {
		t.Fatalf("wrong result: %+v progress=%d", result, progress)
	}
	bundle, err := tabular.LoadClassifierBundle(context.Background(), output, result.BundleID, compute.NewCPUEngine(numeric.Float32Ops{}))
	if err != nil {
		t.Fatal(err)
	}
	rows, labels, err := dataset.Partition("test")
	if err != nil {
		t.Fatal(err)
	}
	predictions, err := bundle.Model.PredictBatch(context.Background(), rows)
	if err != nil {
		t.Fatal(err)
	}
	correct := 0
	for i, p := range predictions {
		if p.ClassID == labels[i] {
			correct++
		}
	}
	if correct < 26 {
		t.Fatalf("CLI artifact fails frozen accuracy: %d/30", correct)
	}
	t.Logf("public train command: steps=%d held-out correct=%d/30 bundle=%s", progress, correct, result.BundleID)
}

func TestTrainTabularErrorSchema(t *testing.T) {
	for _, args := range [][]string{{"tabular"}, {"tabular", "--data", "missing.csv", "--target", "label", "--output", "unused"}, {"tabular", "--recipe", "unsupported"}} {
		var out bytes.Buffer
		err := NewTrainCommand(&out).Run(context.Background(), args)
		if err == nil {
			t.Fatal("invalid command succeeded")
		}
		var event tabularTrainEvent
		if err := json.Unmarshal(out.Bytes(), &event); err != nil {
			t.Fatal(err)
		}
		if event.Version != 1 || event.Type != "error" || event.Status != "failed" || strings.TrimSpace(event.Error) == "" {
			t.Fatalf("bad error event: %+v", event)
		}
	}
}
