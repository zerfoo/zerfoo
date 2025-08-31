package parity_test

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/zerfoo/zerfoo/tests/helpers"
	"github.com/zerfoo/zerfoo/tests/internal/testutil"
)

func TestLogitParity(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo")
	}
	const n = 1000
	const maxNew = 1
	const tol32 = 1e-3
	const topK = 5
	prompts := testutil.LoadPrompts("tests/testdata/prompts.txt", n, 42)

	var agree float64
	var mre float64
	for _, p := range prompts {
		a, err := helpers.ImplZerfoo.Logits(p, maxNew)
		if err != nil {
			t.Fatalf("logits: %v", err)
		}
		b, err := helpers.ImplZerfoo.RefLogits(p, maxNew)
		if err != nil {
			t.Fatalf("ref logits: %v", err)
		}
		if len(a) != len(b) {
			t.Fatalf("shape mismatch %d vs %d", len(a), len(b))
		}
		mre += testutil.MeanRelativeError(a, b)
		agree += testutil.TopKAgreement(a, b, topK)
	}
	mre /= float64(n)
	agree /= float64(n)

	if mre > tol32 {
		t.Fatalf("mean relative error %.6f > tol %.6f", mre, tol32)
	}
	if agree < 0.99 {
		t.Fatalf("top-%d agreement %.4f < 0.99", topK, agree)
	}

	// Save parity results as specified in S1.3.2
	saveParityResults(t, mre, agree, tol32, topK, n)
}

func saveParityResults(t *testing.T, mre, agree, tol32 float64, topK, n int) {
	timestamp := time.Now().Format("2006-01-02")
	
	// Save JSON results
	results := map[string]interface{}{
		"timestamp":        timestamp,
		"test_name":        "logit_parity",
		"mean_rel_error":   mre,
		"top_k_agreement":  agree,
		"tolerance":        tol32,
		"top_k":            topK,
		"num_prompts":      n,
		"passed":           mre <= tol32 && agree >= 0.99,
	}
	
	jsonFile := fmt.Sprintf("artifacts/parity/%s.json", timestamp)
	jsonData, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		t.Logf("Failed to marshal JSON: %v", err)
		return
	}
	
	if err := os.WriteFile(jsonFile, jsonData, 0600); err != nil {
		t.Logf("Failed to write JSON file %s: %v", jsonFile, err)
	} else {
		t.Logf("Saved parity results to %s", jsonFile)
	}
	
	// Save CSV results
	csvFile := fmt.Sprintf("artifacts/parity/%s.csv", timestamp)
	file, err := os.OpenFile(csvFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600) // #nosec G304 -- controlled test artifact path
	if err != nil {
		t.Logf("Failed to create CSV file %s: %v", csvFile, err)
		return
	}
	defer func() { _ = file.Close() }()
	
	writer := csv.NewWriter(file)
	defer writer.Flush()
	
	// Write header
	if err := writer.Write([]string{"timestamp", "test_name", "mean_rel_error", "top_k_agreement", "tolerance", "top_k", "num_prompts", "passed"}); err != nil {
		t.Logf("Failed to write CSV header: %v", err)
		return
	}
	
	// Write data
	passed := "false"
	if mre <= tol32 && agree >= 0.99 {
		passed = "true"
	}
	
	record := []string{
		timestamp,
		"logit_parity",
		fmt.Sprintf("%.6f", mre),
		fmt.Sprintf("%.4f", agree),
		fmt.Sprintf("%.3f", tol32),
		fmt.Sprintf("%d", topK),
		fmt.Sprintf("%d", n),
		passed,
	}
	
	if err := writer.Write(record); err != nil {
		t.Logf("Failed to write CSV record: %v", err)
		return
	}
	
	t.Logf("Saved parity results to %s", csvFile)
}
