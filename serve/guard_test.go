package serve

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zerfoo/zerfoo/inference/guardian"
)

// mockGuardEvaluator implements GuardEvaluator for testing.
type mockGuardEvaluator struct {
	verdicts    []guardian.Verdict
	batchResult *guardian.BatchResult
	scanResult  *guardian.ScanResult
	err         error
}

func (m *mockGuardEvaluator) Evaluate(_ context.Context, req guardian.GuardianRequest) ([]guardian.Verdict, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.verdicts != nil {
		return m.verdicts, nil
	}
	// Default: return one safe verdict per risk.
	risks := req.Risks
	if len(risks) == 0 {
		risks = guardian.HarmRiskCategories()
	}
	out := make([]guardian.Verdict, len(risks))
	for i, r := range risks {
		out[i] = guardian.Verdict{
			Risk:       r,
			Unsafe:     false,
			Confidence: 0.1,
		}
	}
	return out, nil
}

func (m *mockGuardEvaluator) EvaluateBatch(_ context.Context, inputs []guardian.GuardianInput, risks []string) (*guardian.BatchResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.batchResult != nil {
		return m.batchResult, nil
	}
	// Default: return safe results for each input.
	result := &guardian.BatchResult{
		Results: make([]guardian.InputResult, len(inputs)),
	}
	for i := range inputs {
		verdicts := make([]guardian.Verdict, len(risks))
		for j, r := range risks {
			verdicts[j] = guardian.Verdict{
				Risk:       r,
				Unsafe:     false,
				Confidence: 0.1,
			}
		}
		result.Results[i] = guardian.InputResult{
			Index:    i,
			Verdicts: verdicts,
			Flagged:  false,
		}
	}
	return result, nil
}

func (m *mockGuardEvaluator) Scan(_ context.Context, _ guardian.GuardianInput) (*guardian.ScanResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	if m.scanResult != nil {
		return m.scanResult, nil
	}
	return &guardian.ScanResult{
		Flagged:  false,
		Verdicts: []guardian.Verdict{{Risk: "harm", Unsafe: false, Confidence: 0.1}},
	}, nil
}

func newGuardServer(t *testing.T, e GuardEvaluator) *httptest.Server {
	t.Helper()
	mdl := buildTestModel(t)
	srv := NewServer(mdl, WithGuardEvaluator(e))
	return httptest.NewServer(srv.Handler())
}

func TestGuardEndpoint(t *testing.T) {
	eval := &mockGuardEvaluator{
		verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: true, Confidence: 0.9},
			{Risk: "jailbreaking", Unsafe: false, Confidence: 0.3},
		},
	}
	ts := newGuardServer(t, eval)
	defer ts.Close()

	body := `{"model":"granite-guardian-3.3-8b","input":{"user":"How to hack a computer"},"risks":["harm","jailbreaking"]}`
	resp := doPost(t, ts.URL+"/v1/guard", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result GuardResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if result.Model != "granite-guardian-3.3-8b" {
		t.Errorf("model = %q, want %q", result.Model, "granite-guardian-3.3-8b")
	}
	if !result.Flagged {
		t.Error("flagged = false, want true")
	}
	if len(result.Verdicts) != 2 {
		t.Fatalf("verdicts length = %d, want 2", len(result.Verdicts))
	}
	if result.Verdicts[0].Risk != "harm" {
		t.Errorf("verdicts[0].risk = %q, want %q", result.Verdicts[0].Risk, "harm")
	}
	if !result.Verdicts[0].Unsafe {
		t.Error("verdicts[0].unsafe = false, want true")
	}
	if result.Verdicts[0].Confidence != 0.9 {
		t.Errorf("verdicts[0].confidence = %f, want 0.9", result.Verdicts[0].Confidence)
	}
	if result.Verdicts[1].Unsafe {
		t.Error("verdicts[1].unsafe = true, want false")
	}
}

func TestGuardBatchEndpoint(t *testing.T) {
	eval := &mockGuardEvaluator{
		batchResult: &guardian.BatchResult{
			Results: []guardian.InputResult{
				{
					Index:   0,
					Flagged: true,
					Verdicts: []guardian.Verdict{
						{Risk: "harm", Unsafe: true, Confidence: 0.95},
					},
				},
				{
					Index:   1,
					Flagged: false,
					Verdicts: []guardian.Verdict{
						{Risk: "harm", Unsafe: false, Confidence: 0.1},
					},
				},
			},
		},
	}
	ts := newGuardServer(t, eval)
	defer ts.Close()

	body := `{"model":"granite-guardian-3.3-8b","inputs":[{"user":"text1"},{"user":"text2"}],"risks":["harm"]}`
	resp := doPost(t, ts.URL+"/v1/guard/batch", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result GuardBatchResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if result.Model != "granite-guardian-3.3-8b" {
		t.Errorf("model = %q, want %q", result.Model, "granite-guardian-3.3-8b")
	}
	if len(result.Results) != 2 {
		t.Fatalf("results length = %d, want 2", len(result.Results))
	}
	if !result.Results[0].Flagged {
		t.Error("results[0].flagged = false, want true")
	}
	if result.Results[1].Flagged {
		t.Error("results[1].flagged = true, want false")
	}
}

func TestGuardScanEndpoint(t *testing.T) {
	eval := &mockGuardEvaluator{
		scanResult: &guardian.ScanResult{
			Flagged:     true,
			HighestRisk: "harm",
			Verdicts: []guardian.Verdict{
				{Risk: "harm", Unsafe: true, Confidence: 0.9},
				{Risk: "jailbreaking", Unsafe: false, Confidence: 0.2},
			},
		},
	}
	ts := newGuardServer(t, eval)
	defer ts.Close()

	body := `{"model":"granite-guardian-3.3-8b","input":{"user":"dangerous text"}}`
	resp := doPost(t, ts.URL+"/v1/guard/scan", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 200; body: %s", resp.StatusCode, data)
	}

	var result GuardScanResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode error: %v", err)
	}

	if result.Model != "granite-guardian-3.3-8b" {
		t.Errorf("model = %q, want %q", result.Model, "granite-guardian-3.3-8b")
	}
	if !result.Flagged {
		t.Error("flagged = false, want true")
	}
	if result.HighestRisk != "harm" {
		t.Errorf("highest_risk = %q, want %q", result.HighestRisk, "harm")
	}
	if len(result.Verdicts) != 2 {
		t.Fatalf("verdicts length = %d, want 2", len(result.Verdicts))
	}
}

func TestGuardMissingModel(t *testing.T) {
	ts := newGuardServer(t, &mockGuardEvaluator{})
	defer ts.Close()

	body := `{"input":{"user":"hello"},"risks":["harm"]}`
	resp := doPost(t, ts.URL+"/v1/guard", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body: %s", resp.StatusCode, data)
	}
}

func TestGuardInvalidRisk(t *testing.T) {
	ts := newGuardServer(t, &mockGuardEvaluator{})
	defer ts.Close()

	body := `{"model":"granite-guardian-3.3-8b","input":{"user":"hello"},"risks":["nonexistent_risk"]}`
	resp := doPost(t, ts.URL+"/v1/guard", "application/json", body)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		data, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d, want 400; body: %s", resp.StatusCode, data)
	}
}
