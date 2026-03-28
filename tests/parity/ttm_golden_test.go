package parity

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// GoldenCase represents a single TTM golden test case from granite-tsfm.
type GoldenCase struct {
	TestCase       int       `json:"test_case"`
	Description    string    `json:"description"`
	InputShape     []int     `json:"input_shape"`
	OutputShape    []int     `json:"output_shape"`
	Input          []float32 `json:"input"`
	ExpectedOutput []float32 `json:"expected_output"`
	Tolerance      float64   `json:"tolerance"`
}

func TestTTMGoldenFiles_Structure(t *testing.T) {
	goldenDir := filepath.Join("..", "..", "tests", "golden", "ttm")
	entries, err := os.ReadDir(goldenDir)
	if err != nil {
		t.Skipf("golden files not found: %v", err)
	}

	caseCount := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) != ".json" || e.Name() == "model_config.json" {
			continue
		}

		t.Run(e.Name(), func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(goldenDir, e.Name()))
			if err != nil {
				t.Fatalf("read golden file: %v", err)
			}

			var gc GoldenCase
			if err := json.Unmarshal(data, &gc); err != nil {
				t.Fatalf("parse golden file: %v", err)
			}

			// Verify input shape is [1, 32, 1] (batch=1, context=32, channels=1).
			if len(gc.InputShape) != 3 {
				t.Errorf("input shape has %d dims, want 3", len(gc.InputShape))
			}
			if gc.InputShape[0] != 1 || gc.InputShape[1] != 32 || gc.InputShape[2] != 1 {
				t.Errorf("input shape = %v, want [1, 32, 1]", gc.InputShape)
			}

			// Verify output shape is [1, 8, 1] (batch=1, forecast=8, channels=1).
			if len(gc.OutputShape) != 3 {
				t.Errorf("output shape has %d dims, want 3", len(gc.OutputShape))
			}
			if gc.OutputShape[0] != 1 || gc.OutputShape[1] != 8 || gc.OutputShape[2] != 1 {
				t.Errorf("output shape = %v, want [1, 8, 1]", gc.OutputShape)
			}

			// Verify input has correct number of elements.
			if len(gc.Input) != 32 {
				t.Errorf("input has %d elements, want 32", len(gc.Input))
			}

			// Verify output has correct number of elements.
			if len(gc.ExpectedOutput) != 8 {
				t.Errorf("expected_output has %d elements, want 8", len(gc.ExpectedOutput))
			}

			// Verify all expected outputs are finite.
			for i, v := range gc.ExpectedOutput {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Errorf("expected_output[%d] = %v, want finite", i, v)
				}
			}
		})
		caseCount++
	}

	if caseCount != 10 {
		t.Errorf("found %d golden cases, want 10", caseCount)
	}
}
