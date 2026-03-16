package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFlags(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
		check   func(*testing.T, *PredictConfig)
	}{
		{
			name:    "missing_data",
			args:    []string{"-model", "m.zmf", "-output", "out.csv"},
			wantErr: "data path is required",
		},
		{
			name:    "missing_model",
			args:    []string{"-data", "d.csv", "-output", "out.csv"},
			wantErr: "model path is required",
		},
		{
			name:    "missing_output",
			args:    []string{"-data", "d.csv", "-model", "m.zmf"},
			wantErr: "output path is required",
		},
		{
			name: "valid_defaults",
			args: []string{"-data", "d.csv", "-model", "m.zmf", "-output", "/tmp/nonexistent_predict_test.csv"},
			check: func(t *testing.T, c *PredictConfig) {
				t.Helper()
				if c.DataPath != "d.csv" {
					t.Errorf("DataPath = %q, want d.csv", c.DataPath)
				}
				if c.BatchSize != 10000 {
					t.Errorf("BatchSize = %d, want 10000", c.BatchSize)
				}
				if c.OutputFormat != "csv" {
					t.Errorf("OutputFormat = %q, want csv", c.OutputFormat)
				}
				if c.IDColumn != "id" {
					t.Errorf("IDColumn = %q, want id", c.IDColumn)
				}
			},
		},
		{
			name: "custom_options",
			args: []string{
				"-data", "d.csv", "-model", "m.zmf", "-output", "/tmp/nonexistent_predict_test.json",
				"-batch-size", "500", "-format", "json", "-include-probs",
				"-features", "a, b, c", "-id-col", "pk", "-group-col", "era",
				"-verbose", "-overwrite",
			},
			check: func(t *testing.T, c *PredictConfig) {
				t.Helper()
				if c.BatchSize != 500 {
					t.Errorf("BatchSize = %d, want 500", c.BatchSize)
				}
				if c.OutputFormat != "json" {
					t.Errorf("OutputFormat = %q, want json", c.OutputFormat)
				}
				if !c.IncludeProbs {
					t.Error("IncludeProbs should be true")
				}
				if len(c.FeatureColumns) != 3 || c.FeatureColumns[1] != "b" {
					t.Errorf("FeatureColumns = %v, want [a b c]", c.FeatureColumns)
				}
				if c.IDColumn != "pk" {
					t.Errorf("IDColumn = %q, want pk", c.IDColumn)
				}
				if c.GroupColumn != "era" {
					t.Errorf("GroupColumn = %q, want era", c.GroupColumn)
				}
				if !c.Verbose {
					t.Error("Verbose should be true")
				}
				if !c.Overwrite {
					t.Error("Overwrite should be true")
				}
			},
		},
		{
			name:    "invalid_flag",
			args:    []string{"-unknown-flag"},
			wantErr: "flag provided but not defined",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config, err := parseFlags(tt.args)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tt.wantErr)
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.check != nil {
				tt.check(t, config)
			}
		})
	}
}

func TestParseFlags_OutputExists(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "existing.csv")
	if err := os.WriteFile(outPath, []byte("data"), 0o600); err != nil {
		t.Fatal(err)
	}

	// Without overwrite, should fail.
	_, err := parseFlags([]string{"-data", "d.csv", "-model", "m.zmf", "-output", outPath})
	if err == nil || !strings.Contains(err.Error(), "output file exists") {
		t.Errorf("expected 'output file exists' error, got: %v", err)
	}

	// With overwrite, should succeed.
	config, err := parseFlags([]string{"-data", "d.csv", "-model", "m.zmf", "-output", outPath, "-overwrite"})
	if err != nil {
		t.Fatalf("unexpected error with -overwrite: %v", err)
	}
	if config.OutputPath != outPath {
		t.Errorf("OutputPath = %q, want %q", config.OutputPath, outPath)
	}
}

func TestRun_CSVOutput(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "predictions.csv")

	var buf bytes.Buffer
	err := run([]string{
		"-data", "input.csv", "-model", "model.zmf",
		"-output", outPath, "-format", "csv",
	}, &buf)
	if err != nil {
		t.Fatalf("run() error: %v", err)
	}

	if !strings.Contains(buf.String(), "Prediction completed successfully") {
		t.Errorf("stdout = %q, want 'Prediction completed successfully'", buf.String())
	}

	// Check output file exists and has content.
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	if !strings.Contains(string(data), "id,prediction") {
		t.Error("CSV output should contain header 'id,prediction'")
	}

	// Check metadata file was created.
	metaPath := filepath.Join(dir, "predictions_metadata.json")
	metaData, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("failed to read metadata: %v", err)
	}
	var result PredictionResult
	if err := json.Unmarshal(metaData, &result); err != nil {
		t.Fatalf("failed to unmarshal metadata: %v", err)
	}
	if !result.Success {
		t.Error("metadata should show success=true")
	}
}

func TestRun_JSONOutput(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "predictions.json")

	var buf bytes.Buffer
	err := run([]string{
		"-data", "input.csv", "-model", "model.zmf",
		"-output", outPath, "-format", "json",
	}, &buf)
	if err != nil {
		t.Fatalf("run() error: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}

	var rows []map[string]any
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}
	if len(rows) == 0 {
		t.Error("JSON output should have at least one row")
	}
}

func TestRun_UnsupportedFormat(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "predictions.parquet")

	var buf bytes.Buffer
	err := run([]string{
		"-data", "input.csv", "-model", "model.zmf",
		"-output", outPath, "-format", "parquet",
	}, &buf)
	if err == nil || !strings.Contains(err.Error(), "unsupported output format") {
		t.Errorf("expected 'unsupported output format' error, got: %v", err)
	}
}

func TestRun_VerboseMode(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "predictions.csv")

	var buf bytes.Buffer
	err := run([]string{
		"-data", "input.csv", "-model", "model.zmf",
		"-output", outPath, "-verbose",
	}, &buf)
	if err != nil {
		t.Fatalf("run() error: %v", err)
	}

	if !strings.Contains(buf.String(), "Starting prediction with config") {
		t.Errorf("verbose output should contain config info, got: %q", buf.String())
	}
}

func TestRun_WithGroupAndProbs(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "predictions.csv")

	var buf bytes.Buffer
	err := run([]string{
		"-data", "input.csv", "-model", "model.zmf",
		"-output", outPath, "-group-col", "era", "-include-probs",
	}, &buf)
	if err != nil {
		t.Fatalf("run() error: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read output: %v", err)
	}
	header := strings.Split(strings.SplitN(string(data), "\n", 2)[0], ",")
	if len(header) != 4 {
		t.Errorf("expected 4 CSV columns (id, prediction, era, prediction_prob), got %d: %v", len(header), header)
	}
}

func TestRun_MissingArgs(t *testing.T) {
	var buf bytes.Buffer
	err := run([]string{}, &buf)
	if err == nil {
		t.Error("expected error for missing args")
	}
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  *PredictConfig
		wantErr string
	}{
		{
			name:    "empty_data",
			config:  &PredictConfig{ModelPath: "m", OutputPath: "/tmp/nonexistent_validate_test"},
			wantErr: "data path is required",
		},
		{
			name:    "empty_model",
			config:  &PredictConfig{DataPath: "d", OutputPath: "/tmp/nonexistent_validate_test"},
			wantErr: "model path is required",
		},
		{
			name:    "empty_output",
			config:  &PredictConfig{DataPath: "d", ModelPath: "m"},
			wantErr: "output path is required",
		},
		{
			name:   "valid",
			config: &PredictConfig{DataPath: "d", ModelPath: "m", OutputPath: "/tmp/nonexistent_validate_test"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConfig(tt.config)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("validateConfig() error = %v, want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestSavePredictionResult(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "test_output.csv")
	config := &PredictConfig{OutputPath: outPath}
	result := &PredictionResult{Success: true, NumSamples: 100}

	savePredictionResult(config, result)

	metaPath := filepath.Join(dir, "test_output_metadata.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		t.Fatalf("metadata file not created: %v", err)
	}

	var loaded PredictionResult
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("failed to unmarshal metadata: %v", err)
	}
	if !loaded.Success {
		t.Error("expected success=true in metadata")
	}
	if loaded.NumSamples != 100 {
		t.Errorf("NumSamples = %d, want 100", loaded.NumSamples)
	}
}
