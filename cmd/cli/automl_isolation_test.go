package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/zerfoo/training/automl"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTabularWorkerRejectsLeakingDuplicateFixture(t *testing.T) {
	path := filepath.Join(t.TempDir(), "balanced.csv")
	if err := os.WriteFile(path, []byte("x,label\n0,0\n0,0\n0,1\n0,1\n0,2\n0,2\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := newTabularWorker(path, "accuracy"); err == nil {
		t.Fatal("accepted data that cannot form duplicate-isolated held-out splits")
	}
}

func TestTabularWorkerRejectsUnknownMetric(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data.csv")
	if err := os.WriteFile(path, []byte("x,label\n0,0\n1,1\n2,0\n3,1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := newTabularWorker(path, "unsupported"); err == nil {
		t.Fatal("unknown metric accepted")
	}
}

func TestTabularWorkerScoresValidationAndRetainsWinner(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data.csv")
	var csv strings.Builder
	csv.WriteString("x,y,label\n")
	for i := 0; i < 60; i++ {
		fmt.Fprintf(&csv, "%.4f,%.4f,%d\n", float64(i%3)+float64(i)*0.001, float64(i)/60, i%3)
	}
	if err := os.WriteFile(path, []byte(csv.String()), 0600); err != nil {
		t.Fatal(err)
	}
	worker, err := newTabularWorker(path, "loss")
	if err != nil {
		t.Fatal(err)
	}
	config := automl.Config{Params: map[string]float64{"lr": 0.01, "batch_size": 15}}
	metric, err := worker.RunTrial(config)
	if err != nil {
		t.Fatal(err)
	}
	if worker.lastEvaluation.Samples != 12 {
		t.Fatalf("scored %d rows instead of validation's 12", worker.lastEvaluation.Samples)
	}
	if metric.Score != -worker.lastEvaluation.CrossEntropy {
		t.Fatal("loss utility direction mismatch")
	}
	output := filepath.Join(t.TempDir(), "best.json")
	var log bytes.Buffer
	if err := NewAutoMLCommand(&log).Run(context.Background(), []string{"--model", "tabular", "--dataset", path, "--trials", "2", "--metric", "loss", "--output", output}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	var best bestConfigOutput
	if err := json.Unmarshal(raw, &best); err != nil {
		t.Fatal(err)
	}
	if best.Score < 0 || best.ArtifactSHA256 == "" {
		t.Fatalf("wrong public metric/artifact: %+v", best)
	}
	bundle, err := tabular.LoadClassifierBundle(context.Background(), best.ArtifactPath, best.ArtifactSHA256, compute.NewCPUEngine(numeric.Float32Ops{}))
	if err != nil {
		t.Fatal(err)
	}
	rows, labels, err := worker.dataset.Partition("validation")
	if err != nil {
		t.Fatal(err)
	}
	// Both default to seed 42, so the manifest assignments are identical.
	evaluated, err := bundle.Model.Evaluate(context.Background(), rows, labels)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(evaluated.CrossEntropy-best.Score) > 1e-12 {
		t.Fatalf("saved winner score %g differs from %g", evaluated.CrossEntropy, best.Score)
	}
}
