package automl

import (
	"math"
	"testing"
)

// syntheticData generates a simple linearly-separable dataset for testing.
// Class 0: features mostly positive. Class 1: features mostly negative.
// Class 2 (Flat): features near zero.
func syntheticData(n, features int) ([][]float64, []int) {
	data := make([][]float64, n)
	labels := make([]int, n)
	for i := range n {
		row := make([]float64, features)
		label := i % 3
		for j := range features {
			switch label {
			case 0:
				row[j] = 0.5 + float64(j%3)*0.1
			case 1:
				row[j] = -0.5 - float64(j%3)*0.1
			case 2:
				row[j] = float64(j%2)*0.05 - 0.025
			}
		}
		data[i] = row
		labels[i] = label
	}
	return data, labels
}

func TestAutoML_SearchSpace(t *testing.T) {
	archs := AllArchitectures()

	want := map[ArchKind]bool{
		ArchMLP:           true,
		ArchFTTransformer: true,
		ArchTabNet:        true,
		ArchSAINT:         true,
		ArchTabResNet:     true,
		ArchTFT:           true,
		ArchNBEATS:        true,
		ArchPatchTST:      true,
	}

	if len(archs) != len(want) {
		t.Errorf("AllArchitectures() returned %d archs, want %d", len(archs), len(want))
	}

	for _, arch := range archs {
		if !want[arch] {
			t.Errorf("unexpected architecture in search space: %q", arch)
		}
		delete(want, arch)
	}

	for arch := range want {
		t.Errorf("missing architecture from search space: %q", arch)
	}
}

func TestAutoML_ArchHParams(t *testing.T) {
	tests := []struct {
		arch      ArchKind
		wantParam string
	}{
		{ArchMLP, "hidden_dim"},
		{ArchFTTransformer, "d_token"},
		{ArchTabNet, "n_steps"},
		{ArchSAINT, "d_model"},
		{ArchTabResNet, "num_blocks"},
		{ArchTFT, "n_lstm_layers"},
		{ArchNBEATS, "stack_width"},
		{ArchPatchTST, "patch_len"},
	}

	for _, tt := range tests {
		t.Run(string(tt.arch), func(t *testing.T) {
			params := archHParams(tt.arch)
			if len(params) == 0 {
				t.Errorf("archHParams(%q) returned no params", tt.arch)
			}
			found := false
			for _, p := range params {
				if p.Name == tt.wantParam {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("archHParams(%q) missing param %q", tt.arch, tt.wantParam)
			}
		})
	}
}

func TestAutoML_FindsBestArchitecture(t *testing.T) {
	data, labels := syntheticData(60, 4)

	cfg := AutoMLConfig{
		MaxTrials:       8,
		TrialsPerArch:   1,
		Seed:            42,
		TrainEpochs:     3,
		ValidationSplit: 0.2,
	}

	best, report, err := AutoML(data, labels, cfg)
	if err != nil {
		t.Fatalf("AutoML() error: %v", err)
	}

	if best == nil {
		t.Fatal("AutoML() returned nil BestModel")
	}
	if report == nil {
		t.Fatal("AutoML() returned nil SearchReport")
	}

	if best.Architecture == "" {
		t.Error("BestModel.Architecture is empty")
	}

	if best.Predictor == nil {
		t.Error("BestModel.Predictor is nil")
	}

	if len(report.Trials) == 0 {
		t.Error("SearchReport has no trials")
	}

	if report.BestScore < 0 || report.BestScore > 1 {
		t.Errorf("BestScore = %v, want in [0, 1]", report.BestScore)
	}

	if math.IsInf(report.BestScore, 0) || math.IsNaN(report.BestScore) {
		t.Errorf("BestScore is invalid: %v", report.BestScore)
	}

	// Predictor should produce a valid result.
	dir, conf, err := best.Predict(data[0])
	if err != nil {
		t.Errorf("BestModel.Predict() error: %v", err)
	}
	if conf < 0 || conf > 1 {
		t.Errorf("BestModel.Predict() confidence = %v, want in [0, 1]", conf)
	}
	if int(dir) < 0 || int(dir) > 2 {
		t.Errorf("BestModel.Predict() direction = %v, want in [0, 2]", dir)
	}
}

func TestAutoML_AllArchitecturesSearched(t *testing.T) {
	data, labels := syntheticData(80, 4)

	cfg := AutoMLConfig{
		Architectures:   AllArchitectures(),
		TrialsPerArch:   1,
		Seed:            7,
		TrainEpochs:     2,
		ValidationSplit: 0.2,
	}

	_, report, err := AutoML(data, labels, cfg)
	if err != nil {
		t.Fatalf("AutoML() error: %v", err)
	}

	searched := make(map[ArchKind]bool)
	for _, trial := range report.Trials {
		searched[trial.Architecture] = true
	}

	for _, arch := range AllArchitectures() {
		if !searched[arch] {
			t.Errorf("architecture %q was not searched", arch)
		}
	}
}

func TestAutoML_ReportContainsAllTrials(t *testing.T) {
	data, labels := syntheticData(60, 4)

	trialsPerArch := 2
	archs := AllArchitectures()

	cfg := AutoMLConfig{
		Architectures:   archs,
		TrialsPerArch:   trialsPerArch,
		Seed:            1,
		TrainEpochs:     2,
		ValidationSplit: 0.2,
	}

	_, report, err := AutoML(data, labels, cfg)
	if err != nil {
		t.Fatalf("AutoML() error: %v", err)
	}

	wantTrials := len(archs) * trialsPerArch
	if len(report.Trials) != wantTrials {
		t.Errorf("SearchReport.Trials length = %d, want %d", len(report.Trials), wantTrials)
	}
}

func TestAutoML_CustomArchSubset(t *testing.T) {
	data, labels := syntheticData(60, 4)

	subset := []ArchKind{ArchMLP, ArchTabNet, ArchTFT}

	cfg := AutoMLConfig{
		Architectures:   subset,
		TrialsPerArch:   1,
		Seed:            99,
		TrainEpochs:     2,
		ValidationSplit: 0.2,
	}

	best, report, err := AutoML(data, labels, cfg)
	if err != nil {
		t.Fatalf("AutoML() error: %v", err)
	}

	if best == nil || report == nil {
		t.Fatal("AutoML() returned nil")
	}

	searched := make(map[ArchKind]bool)
	for _, trial := range report.Trials {
		searched[trial.Architecture] = true
	}

	for _, arch := range subset {
		if !searched[arch] {
			t.Errorf("expected architecture %q to be searched", arch)
		}
	}

	// Architectures outside the subset should not appear.
	for _, trial := range report.Trials {
		found := false
		for _, arch := range subset {
			if trial.Architecture == arch {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unexpected architecture in report: %q", trial.Architecture)
		}
	}
}

func TestAutoML_BestModelMatchesReport(t *testing.T) {
	data, labels := syntheticData(60, 4)

	cfg := AutoMLConfig{
		Architectures:   AllArchitectures(),
		TrialsPerArch:   1,
		Seed:            5,
		TrainEpochs:     2,
		ValidationSplit: 0.2,
	}

	best, report, err := AutoML(data, labels, cfg)
	if err != nil {
		t.Fatalf("AutoML() error: %v", err)
	}

	if best.Architecture != report.BestArch {
		t.Errorf("BestModel.Architecture = %q, report.BestArch = %q, want match",
			best.Architecture, report.BestArch)
	}
}

func TestAutoML_ClampHelper(t *testing.T) {
	tests := []struct {
		v, min, max, want float64
	}{
		{5.0, 1.0, 10.0, 5.0},
		{0.5, 1.0, 10.0, 1.0},
		{15.0, 1.0, 10.0, 10.0},
		{1.0, 1.0, 1.0, 1.0},
	}

	for _, tt := range tests {
		got := clamp(tt.v, tt.min, tt.max)
		if got != tt.want {
			t.Errorf("clamp(%v, %v, %v) = %v, want %v", tt.v, tt.min, tt.max, got, tt.want)
		}
	}
}
