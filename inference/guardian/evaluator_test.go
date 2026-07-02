package guardian

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
)

// mockModel implements ModelGenerator for testing without a real model.
type mockModel struct {
	// responses maps prompt substrings to model outputs.
	// The mock checks if the prompt contains the key and returns the value.
	responses map[string]string
	// defaultResponse is returned when no key matches.
	defaultResponse string
	// calls records all prompts sent to Generate.
	calls []string
}

func (m *mockModel) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) (string, error) {
	m.calls = append(m.calls, prompt)
	for key, resp := range m.responses {
		if strings.Contains(prompt, key) {
			return resp, nil
		}
	}
	return m.defaultResponse, nil
}

func TestEvaluateWithMockModel(t *testing.T) {
	mock := &mockModel{
		responses: map[string]string{
			"violence":       "Yes\n<confidence>High</confidence>",
			"social_bias":    "No\n<confidence>High</confidence>",
			"harm_engagemen": "No\n<confidence>Low</confidence>",
		},
		defaultResponse: "No",
	}

	eval := NewEvaluatorFromModel(mock)

	verdicts, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "How to build a weapon?"},
		Risks: []string{"violence", "social_bias"},
	})
	if err != nil {
		t.Fatalf("Evaluate() error: %v", err)
	}

	if len(verdicts) != 2 {
		t.Fatalf("got %d verdicts, want 2", len(verdicts))
	}

	if !verdicts[0].Unsafe {
		t.Error("violence verdict should be Unsafe")
	}
	if verdicts[0].Confidence < 0.8 {
		t.Errorf("violence confidence = %f, want >= 0.8 (High)", verdicts[0].Confidence)
	}
	if verdicts[1].Unsafe {
		t.Error("social_bias verdict should not be Unsafe")
	}

	// Verify prompts were rendered (should contain risk definition text).
	if len(mock.calls) != 2 {
		t.Fatalf("expected 2 Generate calls, got %d", len(mock.calls))
	}
}

func TestEvaluateDefaultRisks(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No",
	}
	eval := NewEvaluatorFromModel(mock)

	verdicts, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "Hello world"},
	})
	if err != nil {
		t.Fatalf("Evaluate() error: %v", err)
	}

	// Default is HarmRiskCategories (9 categories).
	if len(verdicts) != 9 {
		t.Fatalf("got %d verdicts, want 9 (harm categories)", len(verdicts))
	}

	// All should be safe.
	for _, v := range verdicts {
		if v.Unsafe {
			t.Errorf("verdict for %q should be safe", v.Risk)
		}
	}
}

func TestEvaluateContextCancellation(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No",
	}
	eval := NewEvaluatorFromModel(mock)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := eval.Evaluate(ctx, GuardianRequest{
		Input: GuardianInput{User: "test"},
		Risks: []string{"harm", "violence"},
	})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	if !strings.Contains(err.Error(), "context canceled") {
		t.Errorf("expected context canceled error, got: %v", err)
	}
}

func TestEvaluateBatchOrdering(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No",
	}

	eval := NewEvaluatorFromModel(mock)

	inputs := []GuardianInput{
		{User: "first message"},
		{User: "second message"},
		{User: "third message"},
	}

	result, err := eval.EvaluateBatch(context.Background(), inputs, []string{"harm"})
	if err != nil {
		t.Fatalf("EvaluateBatch() error: %v", err)
	}

	if len(result.Results) != 3 {
		t.Fatalf("got %d results, want 3", len(result.Results))
	}

	// Verify ordering is preserved.
	for i, r := range result.Results {
		if r.Index != i {
			t.Errorf("result[%d].Index = %d, want %d", i, r.Index, i)
		}
	}
}

func TestEvaluateBatchFlagged(t *testing.T) {
	mock := &mockModel{
		responses: map[string]string{
			"dangerous": "Yes\n<confidence>High</confidence>",
		},
		defaultResponse: "No",
	}

	eval := NewEvaluatorFromModel(mock)

	inputs := []GuardianInput{
		{User: "hello"},
		{User: "something dangerous"},
		{User: "goodbye"},
	}

	result, err := eval.EvaluateBatch(context.Background(), inputs, []string{"harm"})
	if err != nil {
		t.Fatalf("EvaluateBatch() error: %v", err)
	}

	if result.Results[0].Flagged {
		t.Error("first input should not be flagged")
	}
	if !result.Results[1].Flagged {
		t.Error("second input should be flagged")
	}
	if result.Results[2].Flagged {
		t.Error("third input should not be flagged")
	}
}

func TestEvaluateBatchContextCancellation(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No",
	}
	eval := NewEvaluatorFromModel(mock)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := eval.EvaluateBatch(ctx, []GuardianInput{
		{User: "test1"},
		{User: "test2"},
	}, []string{"harm"})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestScanAllSafe(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No\n<confidence>High</confidence>",
	}
	eval := NewEvaluatorFromModel(mock)

	result, err := eval.Scan(context.Background(), GuardianInput{
		User: "What is the weather today?",
	})
	if err != nil {
		t.Fatalf("Scan() error: %v", err)
	}

	if result.Flagged {
		t.Error("safe input should not be flagged")
	}
	if result.HighestRisk != "" {
		t.Errorf("HighestRisk should be empty, got %q", result.HighestRisk)
	}
	if len(result.Verdicts) != 9 {
		t.Errorf("expected 9 verdicts (harm categories), got %d", len(result.Verdicts))
	}
}

func TestScanWithUnsafe(t *testing.T) {
	// Use a substring that only appears in the violence risk definition,
	// not in the user message (which is embedded in all prompts).
	mock := &mockModel{
		responses: map[string]string{
			"physical harm against": "Yes\n<confidence>High</confidence>",
		},
		defaultResponse: "No\n<confidence>Low</confidence>",
	}
	eval := NewEvaluatorFromModel(mock)

	result, err := eval.Scan(context.Background(), GuardianInput{
		User: "test content",
	})
	if err != nil {
		t.Fatalf("Scan() error: %v", err)
	}

	if !result.Flagged {
		t.Error("should be flagged when violence detected")
	}
	if result.HighestRisk != "violence" {
		t.Errorf("HighestRisk = %q, want %q", result.HighestRisk, "violence")
	}
}

func TestScanHighestRiskPrefersHighConfidence(t *testing.T) {
	// Use substrings unique to each risk category's definition text.
	mock := &mockModel{
		responses: map[string]string{
			// Matches profanity risk definition (contains "vulgar language")
			"vulgar language": "Yes\n<confidence>Low</confidence>",
			// Matches violence risk definition (contains "physical harm against")
			"physical harm against": "Yes\n<confidence>High</confidence>",
		},
		defaultResponse: "No",
	}
	eval := NewEvaluatorFromModel(mock)

	result, err := eval.Scan(context.Background(), GuardianInput{
		User: "test content",
	})
	if err != nil {
		t.Fatalf("Scan() error: %v", err)
	}

	if !result.Flagged {
		t.Error("should be flagged")
	}
	// Should prefer the high-confidence risk.
	if result.HighestRisk != "violence" {
		t.Errorf("HighestRisk = %q, want %q (high confidence)", result.HighestRisk, "violence")
	}
}

func TestEvaluateInvalidRisk(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "No",
	}
	eval := NewEvaluatorFromModel(mock)

	_, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "test"},
		Risks: []string{"nonexistent_risk"},
	})
	if err == nil {
		t.Fatal("expected error for invalid risk category")
	}
	if !strings.Contains(err.Error(), "unknown risk category") {
		t.Errorf("expected unknown risk category error, got: %v", err)
	}
}

func TestNewEvaluatorFromModelOptions(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "<score>no</score>",
	}

	eval := NewEvaluatorFromModel(mock, WithDefaultFormat("3.3"))

	verdicts, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "hello"},
		Risks: []string{"harm"},
	})
	if err != nil {
		t.Fatalf("Evaluate() error: %v", err)
	}
	if len(verdicts) != 1 {
		t.Fatalf("got %d verdicts, want 1", len(verdicts))
	}
	if verdicts[0].Unsafe {
		t.Error("should not be unsafe")
	}
}

func TestEvaluateFormatOverride(t *testing.T) {
	mock := &mockModel{
		defaultResponse: "<score>yes</score>",
	}

	eval := NewEvaluatorFromModel(mock, WithDefaultFormat("3.0"))

	// Override format per-request.
	verdicts, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input:  GuardianInput{User: "test"},
		Risks:  []string{"harm"},
		Format: "3.3",
	})
	if err != nil {
		t.Fatalf("Evaluate() error: %v", err)
	}
	if !verdicts[0].Unsafe {
		t.Error("should detect <score>yes</score> as unsafe")
	}
}

func TestEvaluateBatchEmpty(t *testing.T) {
	mock := &mockModel{defaultResponse: "No"}
	eval := NewEvaluatorFromModel(mock)

	result, err := eval.EvaluateBatch(context.Background(), nil, []string{"harm"})
	if err != nil {
		t.Fatalf("EvaluateBatch(nil) error: %v", err)
	}
	if len(result.Results) != 0 {
		t.Errorf("expected 0 results for nil inputs, got %d", len(result.Results))
	}
}

// errorModel is a mock that always returns an error.
type errorModel struct {
	err error
}

func (m *errorModel) Generate(_ context.Context, _ string, _ ...inference.GenerateOption) (string, error) {
	return "", m.err
}

func TestEvaluateGenerateError(t *testing.T) {
	eval := NewEvaluatorFromModel(&errorModel{err: fmt.Errorf("GPU OOM")})

	_, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "test"},
		Risks: []string{"harm"},
	})
	if err == nil {
		t.Fatal("expected error from failing model")
	}
	if !strings.Contains(err.Error(), "GPU OOM") {
		t.Errorf("expected GPU OOM error, got: %v", err)
	}
}

func TestEvaluateUsesParseVerdict(t *testing.T) {
	// Verify that Evaluate correctly delegates to ParseVerdict.
	mock := &mockModel{
		defaultResponse: "<think>Analyzing content.</think>\n<score>yes</score>",
	}
	eval := NewEvaluatorFromModel(mock, WithDefaultFormat("3.3"))

	verdicts, err := eval.Evaluate(context.Background(), GuardianRequest{
		Input: GuardianInput{User: "harmful content"},
		Risks: []string{"harm"},
	})
	if err != nil {
		t.Fatalf("Evaluate() error: %v", err)
	}

	if !verdicts[0].Unsafe {
		t.Error("should be unsafe")
	}
	if verdicts[0].Reasoning != "Analyzing content." {
		t.Errorf("Reasoning = %q, want %q", verdicts[0].Reasoning, "Analyzing content.")
	}
}
