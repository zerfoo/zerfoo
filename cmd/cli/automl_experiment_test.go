package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func experimentFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "data.csv")
	var data strings.Builder
	data.WriteString("x,label\n")
	for i := 0; i < 100; i++ {
		fmt.Fprintf(&data, "%d,%d\n", i, i%5/4)
	}
	if err := os.WriteFile(path, []byte(data.String()), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}
func readExperiment(t *testing.T, dir string) experimentRecord {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "experiment.json"))
	if err != nil {
		t.Fatal(err)
	}
	var record experimentRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		t.Fatal(err)
	}
	return record
}
func TestExperimentSelectionDirection(t *testing.T) {
	low, high := 0.2, 0.8
	a := &experimentTrial{Status: "succeeded", Score: &low}
	b := &experimentTrial{Status: "succeeded", Score: &high}
	for _, metric := range []string{"accuracy", "macro_f1", "loss", "cross_entropy"} {
		t.Run(metric, func(t *testing.T) {
			want := metric == "loss" || metric == "cross_entropy"
			if betterExperimentTrial(a, b, metric) != want {
				t.Fatal("wrong selection direction")
			}
			if betterExperimentTrial(a, a, metric) {
				t.Fatal("tie displaced earlier baseline")
			}
			failed := &experimentTrial{Status: "failed", Score: &high}
			if betterExperimentTrial(failed, a, metric) {
				t.Fatal("failed candidate won")
			}
		})
	}
}
func TestExperimentPersistsBaselineWinnerAndExposure(t *testing.T) {
	data := experimentFixture(t)
	output := filepath.Join(t.TempDir(), "best.json")
	args := []string{"--model", "tabular", "--dataset", data, "--trials", "2", "--metric", "accuracy", "--output", output, "--patience", "0"}
	var log bytes.Buffer
	cmd := NewAutoMLCommand(&log)
	if err := cmd.Run(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	record := readExperiment(t, output+".run")
	if record.Status != "succeeded" || record.TestExposures != 1 || record.Test == nil {
		t.Fatalf("missing final evaluation: %+v", record)
	}
	if record.Winner.Recipe != "majority" {
		t.Fatalf("expected honest baseline win, got %s", record.Winner.Recipe)
	}
	worker, err := newTabularWorker(data, "accuracy")
	if err != nil {
		t.Fatal(err)
	}
	rows, labels, err := worker.dataset.Partition("validation")
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 5; i++ {
		raw, err := os.ReadFile(filepath.Join(output+".run", fmt.Sprintf("trial-%04d.json", i)))
		if err != nil {
			t.Fatal(err)
		}
		var trial experimentTrial
		if err := json.Unmarshal(raw, &trial); err != nil {
			t.Fatal(err)
		}
		if trial.Status != "succeeded" || trial.ElapsedNS <= 0 || trial.AllocatedBytes == 0 {
			t.Fatalf("incomplete trial: %+v", trial)
		}
		bundle, err := tabular.LoadClassifierBundle(context.Background(), trial.ArtifactPath, trial.ArtifactSHA256, compute.NewCPUEngine(numeric.Float32Ops{}))
		if err != nil {
			t.Fatal(err)
		}
		report, err := bundle.Model.Evaluate(context.Background(), rows, labels)
		if err != nil {
			t.Fatal(err)
		}
		if report.Accuracy != *trial.Score {
			t.Fatal("artifact does not reproduce score")
		}
	}
	if err := cmd.Run(context.Background(), args); err == nil {
		t.Fatal("reused experiment allowed")
	}
	next := filepath.Join(t.TempDir(), "next.json")
	args[9] = next
	args = append(args, "--prior-experiment", output+".run")
	if err := cmd.Run(context.Background(), args); err != nil {
		t.Fatal(err)
	}
	replay := readExperiment(t, next+".run")
	if replay.PriorTestExposures != 1 || replay.TestExposures != 1 {
		t.Fatalf("lost exposure lineage: %+v", replay)
	}
}

type cancelExperimentWriter struct{ cancel context.CancelFunc }

func (w cancelExperimentWriter) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte(`"type":"experiment"`)) {
		w.cancel()
	}
	return len(p), nil
}
func TestExperimentCancellationRetainsFailure(t *testing.T) {
	output := filepath.Join(t.TempDir(), "best.json")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cmd := NewAutoMLCommand(cancelExperimentWriter{cancel: cancel})
	err := cmd.Run(ctx, []string{"--model", "tabular", "--dataset", experimentFixture(t), "--trials", "1", "--output", output})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
	record := readExperiment(t, output+".run")
	if record.Status != "canceled" || record.TestExposures != 0 {
		t.Fatalf("bad cancellation: %+v", record)
	}
	raw, err := os.ReadFile(filepath.Join(output+".run", "trial-0001.json"))
	if err != nil {
		t.Fatal(err)
	}
	var trial experimentTrial
	if err := json.Unmarshal(raw, &trial); err != nil {
		t.Fatal(err)
	}
	if trial.Status != "canceled" || trial.Error == "" || trial.Score != nil {
		t.Fatalf("failure not retained: %+v", trial)
	}
}
