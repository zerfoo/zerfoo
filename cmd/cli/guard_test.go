package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/inference/guardian"
)

// fakeGenerator implements guardian.ModelGenerator for testing.
type fakeGenerator struct {
	responses map[string]string // risk keyword -> response
}

func (f *fakeGenerator) Generate(_ context.Context, prompt string, _ ...inference.GenerateOption) (string, error) {
	// Return "Yes" for harm-related prompts, "No" otherwise.
	for keyword, resp := range f.responses {
		if strings.Contains(prompt, keyword) {
			return resp, nil
		}
	}
	return "No\n<confidence>High</confidence>", nil
}

func newTestEvaluator(responses map[string]string) *guardian.Evaluator {
	gen := &fakeGenerator{responses: responses}
	return guardian.NewEvaluatorFromModel(gen)
}

func TestGuardCommand_Name(t *testing.T) {
	cmd := NewGuardCommand(nil)
	if cmd.Name() != "guard" {
		t.Errorf("expected name 'guard', got %q", cmd.Name())
	}
}

func TestGuardCommand_Description(t *testing.T) {
	cmd := NewGuardCommand(nil)
	if cmd.Description() == "" {
		t.Error("expected non-empty description")
	}
}

func TestGuardCommand_Interface(t *testing.T) {
	var _ Command = (*GuardCommand)(nil)
}

func TestGuardCommand_ParseArgs(t *testing.T) {
	cmd := NewGuardCommand(nil)

	tests := []struct {
		name     string
		args     []string
		wantErr  bool
		validate func(*testing.T, *guardConfig)
	}{
		{
			name: "basic input",
			args: []string{"--model", "test.gguf", "--input", "hello"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if cfg.model != "test.gguf" {
					t.Errorf("model = %q, want test.gguf", cfg.model)
				}
				if cfg.input != "hello" {
					t.Errorf("input = %q, want hello", cfg.input)
				}
			},
		},
		{
			name: "with risks",
			args: []string{"--model", "m", "--input", "x", "--risks", "harm,jailbreaking,profanity"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if len(cfg.risks) != 3 {
					t.Fatalf("risks len = %d, want 3", len(cfg.risks))
				}
				if cfg.risks[0] != "harm" || cfg.risks[1] != "jailbreaking" || cfg.risks[2] != "profanity" {
					t.Errorf("risks = %v, want [harm jailbreaking profanity]", cfg.risks)
				}
			},
		},
		{
			name: "scan flag",
			args: []string{"--model", "m", "--input", "x", "--scan"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if !cfg.scan {
					t.Error("expected scan to be true")
				}
			},
		},
		{
			name: "json flag",
			args: []string{"--model", "m", "--input", "x", "--json"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if !cfg.jsonOut {
					t.Error("expected jsonOut to be true")
				}
			},
		},
		{
			name: "response flag",
			args: []string{"--model", "m", "--input", "x", "--response", "some reply"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if cfg.response != "some reply" {
					t.Errorf("response = %q, want 'some reply'", cfg.response)
				}
			},
		},
		{
			name: "file flag",
			args: []string{"--model", "m", "--file", "input.txt"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if cfg.file != "input.txt" {
					t.Errorf("file = %q, want input.txt", cfg.file)
				}
			},
		},
		{
			name: "device flag",
			args: []string{"--model", "m", "--input", "x", "--device", "cuda:0"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if cfg.device != "cuda:0" {
					t.Errorf("device = %q, want cuda:0", cfg.device)
				}
			},
		},
		{
			name: "equals syntax",
			args: []string{"--model=test.gguf", "--input=hello"},
			validate: func(t *testing.T, cfg *guardConfig) {
				if cfg.model != "test.gguf" {
					t.Errorf("model = %q, want test.gguf", cfg.model)
				}
				if cfg.input != "hello" {
					t.Errorf("input = %q, want hello", cfg.input)
				}
			},
		},
		{
			name:    "missing model value",
			args:    []string{"--model"},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := cmd.parseArgs(tt.args)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tt.validate != nil {
				tt.validate(t, cfg)
			}
		})
	}
}

func TestGuardCommand_ParseRisks(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{"harm,jailbreaking,profanity", []string{"harm", "jailbreaking", "profanity"}},
		{"harm", []string{"harm"}},
		{" harm , violence ", []string{"harm", "violence"}},
		{"", nil},
	}
	for _, tt := range tests {
		got := parseRisks(tt.input)
		if tt.want == nil {
			if len(got) != 0 {
				t.Errorf("parseRisks(%q) = %v, want empty", tt.input, got)
			}
			continue
		}
		if len(got) != len(tt.want) {
			t.Errorf("parseRisks(%q) = %v, want %v", tt.input, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("parseRisks(%q)[%d] = %q, want %q", tt.input, i, got[i], tt.want[i])
			}
		}
	}
}

func TestGuardCommand_WriteHumanVerdicts(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	verdicts := []guardian.Verdict{
		{Risk: "harm", Unsafe: true, Confidence: 0.92},
		{Risk: "jailbreaking", Unsafe: false, Confidence: 0.15},
	}

	gi := guardian.GuardianInput{User: "How to hack a computer"}
	err := cmd.writeHumanVerdicts(gi, "granite-guardian-3.3-8b", verdicts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Check model name.
	if !strings.Contains(output, "granite-guardian-3.3-8b") {
		t.Error("output should contain model name")
	}
	// Check input.
	if !strings.Contains(output, "How to hack a computer") {
		t.Error("output should contain input text")
	}
	// Check UNSAFE label.
	if !strings.Contains(output, "UNSAFE") {
		t.Error("output should contain UNSAFE label")
	}
	// Check safe label.
	if !strings.Contains(output, "safe") {
		t.Error("output should contain safe label")
	}
	// Check confidence.
	if !strings.Contains(output, "0.92") {
		t.Error("output should contain confidence 0.92")
	}
	// Check overall flagged.
	if !strings.Contains(output, "FLAGGED") {
		t.Error("output should say FLAGGED")
	}
}

func TestGuardCommand_WriteHumanVerdicts_Safe(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	verdicts := []guardian.Verdict{
		{Risk: "harm", Unsafe: false, Confidence: 0.1},
	}

	gi := guardian.GuardianInput{User: "What is the weather?"}
	err := cmd.writeHumanVerdicts(gi, "model", verdicts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "Overall: SAFE") {
		t.Errorf("expected SAFE, got:\n%s", output)
	}
}

func TestGuardCommand_WriteJSONVerdicts(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	verdicts := []guardian.Verdict{
		{Risk: "harm", Unsafe: true, Confidence: 0.92},
		{Risk: "jailbreaking", Unsafe: false, Confidence: 0.15},
	}

	err := cmd.writeJSONVerdicts(verdicts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var out jsonOutput
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if !out.Flagged {
		t.Error("expected flagged=true")
	}
	if len(out.Verdicts) != 2 {
		t.Fatalf("expected 2 verdicts, got %d", len(out.Verdicts))
	}
	if out.Verdicts[0].Risk != "harm" || !out.Verdicts[0].Unsafe {
		t.Error("first verdict should be harm/unsafe")
	}
	if out.Verdicts[1].Risk != "jailbreaking" || out.Verdicts[1].Unsafe {
		t.Error("second verdict should be jailbreaking/safe")
	}
}

func TestGuardCommand_WriteJSONScan(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	result := &guardian.ScanResult{
		Flagged:     true,
		HighestRisk: "violence",
		Verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: false, Confidence: 0.1},
			{Risk: "violence", Unsafe: true, Confidence: 0.95},
		},
	}

	err := cmd.writeJSONScan(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var out jsonOutput
	if err := json.Unmarshal(buf.Bytes(), &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if !out.Flagged {
		t.Error("expected flagged=true")
	}
}

func TestGuardCommand_ScanUsesAllHarmCategories(t *testing.T) {
	// The --scan flag should use all harm risk categories.
	// We verify by checking the parseArgs output.
	cmd := NewGuardCommand(nil)
	cfg, err := cmd.parseArgs([]string{"--model", "m", "--input", "x", "--scan"})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.scan {
		t.Error("expected scan=true")
	}
	// When scan is true, risks should be empty (Scan method uses HarmRiskCategories internally).
	if len(cfg.risks) != 0 {
		t.Errorf("expected empty risks for scan, got %v", cfg.risks)
	}

	// Verify HarmRiskCategories returns the expected count.
	categories := guardian.HarmRiskCategories()
	if len(categories) != 9 {
		t.Errorf("expected 9 harm categories, got %d", len(categories))
	}
}

func TestGuardCommand_RunMissingModel(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)
	err := cmd.Run(context.Background(), []string{"--input", "hello"})
	if err == nil {
		t.Error("expected error for missing model")
	}
	if !strings.Contains(err.Error(), "--model is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGuardCommand_RunMissingInput(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)
	err := cmd.Run(context.Background(), []string{"--model", "m"})
	if err == nil {
		t.Error("expected error for missing input")
	}
	if !strings.Contains(err.Error(), "--input or --file is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGuardCommand_Usage(t *testing.T) {
	cmd := NewGuardCommand(nil)
	usage := cmd.Usage()
	if !strings.Contains(usage, "--model") {
		t.Error("usage should mention --model")
	}
	if !strings.Contains(usage, "--input") {
		t.Error("usage should mention --input")
	}
	if !strings.Contains(usage, "--scan") {
		t.Error("usage should mention --scan")
	}
	if !strings.Contains(usage, "--json") {
		t.Error("usage should mention --json")
	}
}

func TestGuardCommand_Examples(t *testing.T) {
	cmd := NewGuardCommand(nil)
	examples := cmd.Examples()
	if len(examples) == 0 {
		t.Error("expected at least one example")
	}
}

func TestGuardCommand_WriteHumanScan(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	result := &guardian.ScanResult{
		Flagged:     true,
		HighestRisk: "violence",
		Verdicts: []guardian.Verdict{
			{Risk: "harm", Unsafe: false, Confidence: 0.1},
			{Risk: "violence", Unsafe: true, Confidence: 0.95},
		},
	}
	gi := guardian.GuardianInput{User: "test input"}

	err := cmd.writeHumanScan(gi, result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "full scan") {
		t.Error("scan output should mention full scan")
	}
	if !strings.Contains(output, "FLAGGED") {
		t.Error("output should say FLAGGED")
	}
	if !strings.Contains(output, "violence") {
		t.Error("output should mention highest risk category")
	}
}

func TestGuardCommand_WriteHumanScan_Safe(t *testing.T) {
	var buf bytes.Buffer
	cmd := NewGuardCommand(&buf)

	result := &guardian.ScanResult{
		Flagged:  false,
		Verdicts: []guardian.Verdict{{Risk: "harm", Unsafe: false, Confidence: 0.1}},
	}
	gi := guardian.GuardianInput{User: "hello"}

	err := cmd.writeHumanScan(gi, result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(buf.String(), "Overall: SAFE") {
		t.Error("expected SAFE")
	}
}
