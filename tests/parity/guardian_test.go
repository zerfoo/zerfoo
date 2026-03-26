package parity_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/inference/guardian"
)

// goldenCase represents a single golden test case loaded from testdata JSON.
type goldenCase struct {
	Name   string `json:"name"`
	Input  struct {
		User      string `json:"user"`
		Assistant string `json:"assistant,omitempty"`
		Context   string `json:"context,omitempty"`
	} `json:"input"`
	Risk   string `json:"risk"`
	Unsafe bool   `json:"unsafe"`
}

type goldenFile struct {
	Source string       `json:"source"`
	Cases  []goldenCase `json:"cases"`
}

func loadGoldenCases(t *testing.T) []goldenCase {
	t.Helper()
	data, err := os.ReadFile("testdata/guardian/golden.json")
	if err != nil {
		t.Fatalf("failed to read golden file: %v", err)
	}
	var gf goldenFile
	if err := json.Unmarshal(data, &gf); err != nil {
		t.Fatalf("failed to parse golden file: %v", err)
	}
	return gf.Cases
}

// TestGuardianTemplateRendering verifies that all 13 risk category templates
// render without error for the three prompt modes (user-only, user+assistant,
// RAG with context).
func TestGuardianTemplateRendering(t *testing.T) {
	categories := guardian.AllRiskCategories()
	if len(categories) != 13 {
		t.Fatalf("expected 13 risk categories, got %d", len(categories))
	}

	inputs := []struct {
		name  string
		input guardian.GuardianInput
	}{
		{
			name:  "user_only",
			input: guardian.GuardianInput{User: "Test message"},
		},
		{
			name: "user_assistant",
			input: guardian.GuardianInput{
				User:      "Test question",
				Assistant: "Test answer",
			},
		},
		{
			name: "rag",
			input: guardian.GuardianInput{
				Context:   "Test context document.",
				User:      "Test question",
				Assistant: "Test answer from context.",
			},
		},
	}

	formats := []string{"3.0", "3.2", "3.3"}

	for _, in := range inputs {
		for _, risk := range categories {
			for _, format := range formats {
				name := in.name + "/" + risk + "/" + format
				t.Run(name, func(t *testing.T) {
					think := format == "3.3" // enable thinking only for 3.3
					prompt, err := guardian.RenderTemplate(in.input, guardian.TemplateOptions{
						Risk:   risk,
						Format: format,
						Think:  think,
					})
					if err != nil {
						t.Fatalf("RenderTemplate failed: %v", err)
					}
					if prompt == "" {
						t.Fatal("RenderTemplate returned empty prompt")
					}
				})
			}
		}
	}
}

// TestGuardianVerdictParsing verifies ParseVerdict handles all three format
// variants (3.0 plain Yes/No, 3.2 with confidence tags, 3.3 with score tags
// and optional thinking).
func TestGuardianVerdictParsing(t *testing.T) {
	tests := []struct {
		name       string
		output     string
		risk       string
		logprobs   []float64
		wantUnsafe bool
		wantConf   float64 // expected confidence (approximate)
		wantReason string  // expected reasoning substring
	}{
		// 3.0 format: plain text
		{
			name:       "3.0_yes",
			output:     "Yes",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0, // no logprobs
		},
		{
			name:       "3.0_no",
			output:     "No",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "3.0_yes_with_logprobs",
			output:     "Yes",
			risk:       "harm",
			logprobs:   []float64{-0.1, -2.5},
			wantUnsafe: true,
		},
		{
			name:       "3.0_no_with_logprobs",
			output:     "No",
			risk:       "harm",
			logprobs:   []float64{-2.5, -0.1},
			wantUnsafe: false,
		},
		{
			name:       "3.0_yes_whitespace",
			output:     "  Yes\n",
			risk:       "violence",
			wantUnsafe: true,
		},
		{
			name:       "3.0_no_whitespace",
			output:     "\n  No  \n",
			risk:       "violence",
			wantUnsafe: false,
		},
		// 3.2 format: confidence tags
		{
			name:       "3.2_yes_high_confidence",
			output:     "Yes\n<confidence>High</confidence>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0.9,
		},
		{
			name:       "3.2_yes_low_confidence",
			output:     "Yes\n<confidence>Low</confidence>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   0.3,
		},
		{
			name:       "3.2_no_high_confidence",
			output:     "No\n<confidence>High</confidence>",
			risk:       "jailbreaking",
			wantUnsafe: false,
			wantConf:   0.9,
		},
		{
			name:       "3.2_no_low_confidence",
			output:     "No\n<confidence>low</confidence>",
			risk:       "social_bias",
			wantUnsafe: false,
			wantConf:   0.3,
		},
		// 3.3 format: score tags with optional thinking
		{
			name:       "3.3_yes",
			output:     "<score>yes</score>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   1.0,
		},
		{
			name:       "3.3_no",
			output:     "<score>No</score>",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   1.0,
		},
		{
			name:       "3.3_with_thinking",
			output:     "<think>The user is asking about explosives, which is dangerous.</think>\n<score>Yes</score>",
			risk:       "harm",
			wantUnsafe: true,
			wantConf:   1.0,
			wantReason: "explosives",
		},
		{
			name:       "3.3_safe_with_thinking",
			output:     "<think>This is a harmless geography question.</think>\n<score>no</score>",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   1.0,
			wantReason: "geography",
		},
		// Unrecognized output
		{
			name:       "unrecognized",
			output:     "I cannot determine this.",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
		{
			name:       "empty",
			output:     "",
			risk:       "harm",
			wantUnsafe: false,
			wantConf:   0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			v := guardian.ParseVerdict(tc.output, tc.risk, tc.logprobs)

			if v.Unsafe != tc.wantUnsafe {
				t.Errorf("Unsafe = %v, want %v", v.Unsafe, tc.wantUnsafe)
			}
			if v.Risk != tc.risk {
				t.Errorf("Risk = %q, want %q", v.Risk, tc.risk)
			}
			if tc.wantConf > 0 {
				const epsilon = 0.01
				if diff := v.Confidence - tc.wantConf; diff > epsilon || diff < -epsilon {
					t.Errorf("Confidence = %f, want %f", v.Confidence, tc.wantConf)
				}
			}
			if tc.wantReason != "" && !strings.Contains(v.Reasoning, tc.wantReason) {
				t.Errorf("Reasoning = %q, want substring %q", v.Reasoning, tc.wantReason)
			}
		})
	}
}

// TestGuardianPromptFormat verifies that rendered prompts contain the expected
// structural markers from the Granite Guardian specification.
func TestGuardianPromptFormat(t *testing.T) {
	prompt, err := guardian.RenderTemplate(
		guardian.GuardianInput{User: "Test message"},
		guardian.TemplateOptions{Risk: "harm", Format: "3.2"},
	)
	if err != nil {
		t.Fatalf("RenderTemplate: %v", err)
	}

	markers := []string{
		"<start_of_turn>",
		"<end_of_turn>",
		"<start_of_risk_definition>",
		"<end_of_risk_definition>",
		"User Message:",
		"Your answer must be either 'Yes' or 'No'",
	}
	for _, m := range markers {
		if !strings.Contains(prompt, m) {
			t.Errorf("prompt missing expected marker %q", m)
		}
	}

	// Verify RAG mode includes context.
	ragPrompt, err := guardian.RenderTemplate(
		guardian.GuardianInput{
			Context:   "Test context.",
			User:      "Test question",
			Assistant: "Test answer",
		},
		guardian.TemplateOptions{Risk: "groundedness", Format: "3.2"},
	)
	if err != nil {
		t.Fatalf("RenderTemplate (RAG): %v", err)
	}

	ragMarkers := []string{
		"Context:",
		"Assistant Response:",
		"grounded",
	}
	for _, m := range ragMarkers {
		if !strings.Contains(ragPrompt, m) {
			t.Errorf("RAG prompt missing expected marker %q", m)
		}
	}

	// Verify 3.3 thinking mode includes thinking instruction.
	thinkPrompt, err := guardian.RenderTemplate(
		guardian.GuardianInput{User: "Test message"},
		guardian.TemplateOptions{Risk: "harm", Format: "3.3", Think: true},
	)
	if err != nil {
		t.Fatalf("RenderTemplate (thinking): %v", err)
	}
	if !strings.Contains(thinkPrompt, "think step by step") {
		t.Error("thinking prompt missing 'think step by step' instruction")
	}
}

// mockGenerator is a test double that returns pre-configured responses.
type mockGenerator struct {
	responses map[string]string // prompt substring -> response
	fallback  string
}

func (m *mockGenerator) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) (string, error) {
	for substr, resp := range m.responses {
		if strings.Contains(prompt, substr) {
			return resp, nil
		}
	}
	return m.fallback, nil
}

// TestGuardianGoldenCases loads golden test data and, if GUARDIAN_GGUF is set,
// verifies the full inference pipeline produces matching verdicts. When the env
// var is not set, it tests template rendering and verdict parsing against
// expected outputs using a mock generator.
func TestGuardianGoldenCases(t *testing.T) {
	cases := loadGoldenCases(t)

	ggufPath := os.Getenv("GUARDIAN_GGUF")
	if ggufPath != "" {
		t.Logf("running full inference parity against %s", ggufPath)
		runGuardianInferenceParity(t, ggufPath, cases)
		return
	}

	t.Log("GUARDIAN_GGUF not set; testing template rendering + verdict parsing with mock generator")

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			input := guardian.GuardianInput{
				User:      tc.Input.User,
				Assistant: tc.Input.Assistant,
				Context:   tc.Input.Context,
			}

			// Build a per-case mock that returns the expected verdict.
			var response string
			if tc.Unsafe {
				response = "Yes\n<confidence>High</confidence>"
			} else {
				response = "No\n<confidence>High</confidence>"
			}
			gen := &mockGenerator{fallback: response}
			eval := guardian.NewEvaluatorFromModel(gen)

			verdicts, err := eval.Evaluate(context.Background(), guardian.GuardianRequest{
				Input: input,
				Risks: []string{tc.Risk},
			})
			if err != nil {
				t.Fatalf("Evaluate: %v", err)
			}
			if len(verdicts) != 1 {
				t.Fatalf("expected 1 verdict, got %d", len(verdicts))
			}
			v := verdicts[0]
			if v.Unsafe != tc.Unsafe {
				t.Errorf("verdict Unsafe = %v, want %v (Ollama golden)", v.Unsafe, tc.Unsafe)
			}
			if v.Risk != tc.Risk {
				t.Errorf("verdict Risk = %q, want %q", v.Risk, tc.Risk)
			}
		})
	}
}

// runGuardianInferenceParity runs the full Guardian pipeline against a real
// GGUF model and compares verdicts to golden data from Ollama.
func runGuardianInferenceParity(t *testing.T, ggufPath string, cases []goldenCase) {
	t.Helper()

	eval, err := guardian.NewEvaluator(ggufPath)
	if err != nil {
		t.Skipf("failed to load Guardian model at %s: %v", ggufPath, err)
	}

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			input := guardian.GuardianInput{
				User:      tc.Input.User,
				Assistant: tc.Input.Assistant,
				Context:   tc.Input.Context,
			}

			verdicts, err := eval.Evaluate(context.Background(), guardian.GuardianRequest{
				Input: input,
				Risks: []string{tc.Risk},
			})
			if err != nil {
				t.Fatalf("Evaluate: %v", err)
			}
			if len(verdicts) != 1 {
				t.Fatalf("expected 1 verdict, got %d", len(verdicts))
			}
			v := verdicts[0]
			if v.Unsafe != tc.Unsafe {
				t.Errorf("Zerfoo verdict Unsafe=%v but Ollama golden=%v for risk %q",
					v.Unsafe, tc.Unsafe, tc.Risk)
			}
		})
	}
}
