package guardian

import (
	"strings"
	"testing"
)

func TestAllRiskCategories(t *testing.T) {
	cats := AllRiskCategories()
	if len(cats) != 13 {
		t.Fatalf("AllRiskCategories() returned %d items, want 13", len(cats))
	}
	// Verify sorted order.
	for i := 1; i < len(cats); i++ {
		if cats[i] <= cats[i-1] {
			t.Errorf("AllRiskCategories() not sorted: %q <= %q", cats[i], cats[i-1])
		}
	}
}

func TestHarmRiskCategories(t *testing.T) {
	cats := HarmRiskCategories()
	if len(cats) != 9 {
		t.Fatalf("HarmRiskCategories() returned %d items, want 9", len(cats))
	}
	// All should be valid risk categories.
	for _, c := range cats {
		if _, ok := RiskDefinitions[c]; !ok {
			t.Errorf("HarmRiskCategories() contains unknown category %q", c)
		}
	}
}

func TestRAGRiskCategories(t *testing.T) {
	cats := RAGRiskCategories()
	if len(cats) != 3 {
		t.Fatalf("RAGRiskCategories() returned %d items, want 3", len(cats))
	}
	for _, c := range cats {
		if _, ok := RiskDefinitions[c]; !ok {
			t.Errorf("RAGRiskCategories() contains unknown category %q", c)
		}
	}
}

func TestCategoryPartition(t *testing.T) {
	// Harm + RAG + function_call_hallucination = all 13.
	harm := HarmRiskCategories()
	rag := RAGRiskCategories()
	total := len(harm) + len(rag) + 1 // +1 for function_call_hallucination
	if total != 13 {
		t.Errorf("harm(%d) + rag(%d) + 1 = %d, want 13", len(harm), len(rag), total)
	}
}

func TestRenderTemplateAllCategories(t *testing.T) {
	for _, cat := range AllRiskCategories() {
		t.Run(cat, func(t *testing.T) {
			input := GuardianInput{
				User:      "Tell me something.",
				Assistant: "Here is my response.",
				Context:   "Some context.",
			}
			out, err := RenderTemplate(input, TemplateOptions{Risk: cat})
			if err != nil {
				t.Fatalf("RenderTemplate(%q) error: %v", cat, err)
			}
			if out == "" {
				t.Fatalf("RenderTemplate(%q) returned empty string", cat)
			}
			if !strings.Contains(out, RiskDefinitions[cat]) {
				t.Errorf("RenderTemplate(%q) does not contain risk definition", cat)
			}
		})
	}
}

func TestRenderTemplateUserOnly(t *testing.T) {
	input := GuardianInput{User: "How do I hack a computer?"}
	out, err := RenderTemplate(input, TemplateOptions{Risk: "harm"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "whether the user message is harmful") {
		t.Error("user-only mode should mention 'user message is harmful'")
	}
	if strings.Contains(out, "Assistant Response:") {
		t.Error("user-only mode should not contain 'Assistant Response:'")
	}
	if strings.Contains(out, "Context:") {
		t.Error("user-only mode should not contain 'Context:'")
	}
	if !strings.Contains(out, "Is the user message harmful") {
		t.Error("user-only closing question should ask about user message")
	}
}

func TestRenderTemplateUserAssistant(t *testing.T) {
	input := GuardianInput{
		User:      "Tell me how to make a bomb.",
		Assistant: "Sure, here are the steps...",
	}
	out, err := RenderTemplate(input, TemplateOptions{Risk: "harm_engagement"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "whether the assistant response is harmful") {
		t.Error("user+assistant mode should mention 'assistant response is harmful'")
	}
	if !strings.Contains(out, "Assistant Response: Sure, here are the steps...") {
		t.Error("should contain assistant response text")
	}
	if !strings.Contains(out, "User Message: Tell me how to make a bomb.") {
		t.Error("should contain user message text")
	}
}

func TestRenderTemplateRAG(t *testing.T) {
	input := GuardianInput{
		User:      "What is the capital of France?",
		Assistant: "The capital of France is Paris.",
		Context:   "France is a country in Europe. Its capital city is Paris.",
	}
	out, err := RenderTemplate(input, TemplateOptions{Risk: "groundedness"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "grounded in the provided context") {
		t.Error("RAG mode should mention 'grounded in the provided context'")
	}
	if !strings.Contains(out, "Context: France is a country") {
		t.Error("RAG mode should contain context text")
	}
}

func TestRenderTemplateRAGRequiresAssistant(t *testing.T) {
	input := GuardianInput{
		User:    "What is the capital?",
		Context: "Some context.",
	}
	_, err := RenderTemplate(input, TemplateOptions{Risk: "groundedness"})
	if err == nil {
		t.Fatal("expected error when context is provided without assistant")
	}
	if !strings.Contains(err.Error(), "assistant response is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRenderTemplateCustomRisk(t *testing.T) {
	customDef := "The message contains proprietary financial trading signals."
	input := GuardianInput{User: "Buy AAPL at 150."}
	out, err := RenderTemplate(input, TemplateOptions{
		Risk:       "harm", // should be overridden
		CustomRisk: customDef,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, customDef) {
		t.Error("custom risk definition should appear in output")
	}
	if strings.Contains(out, RiskDefinitions["harm"]) {
		t.Error("pre-baked 'harm' definition should be overridden by custom")
	}
}

func TestRenderTemplateCustomRiskNoCategory(t *testing.T) {
	// Custom risk with empty Risk field should still work.
	customDef := "Custom policy violation."
	input := GuardianInput{User: "Test message."}
	out, err := RenderTemplate(input, TemplateOptions{CustomRisk: customDef})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, customDef) {
		t.Error("custom risk definition should appear in output")
	}
}

func TestRenderTemplateInvalidRisk(t *testing.T) {
	input := GuardianInput{User: "Hello"}
	_, err := RenderTemplate(input, TemplateOptions{Risk: "nonexistent"})
	if err == nil {
		t.Fatal("expected error for unknown risk category")
	}
	if !strings.Contains(err.Error(), "unknown risk category") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRenderTemplateEmptyUser(t *testing.T) {
	_, err := RenderTemplate(GuardianInput{}, TemplateOptions{Risk: "harm"})
	if err == nil {
		t.Fatal("expected error for empty user message")
	}
	if !strings.Contains(err.Error(), "user message is required") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRenderTemplateDefaultFormat(t *testing.T) {
	input := GuardianInput{User: "test"}
	out, err := RenderTemplate(input, TemplateOptions{Risk: "harm"})
	if err != nil {
		t.Fatal(err)
	}
	// Default format 3.2 should produce the standard template.
	if !strings.Contains(out, "<start_of_turn>") {
		t.Error("default format should use <start_of_turn> tags")
	}
}

func TestRenderTemplateFormat30(t *testing.T) {
	input := GuardianInput{User: "test"}
	out, err := RenderTemplate(input, TemplateOptions{Risk: "harm", Format: "3.0"})
	if err != nil {
		t.Fatal(err)
	}
	if out == "" {
		t.Error("format 3.0 should produce non-empty output")
	}
}

func TestRenderTemplateInvalidFormat(t *testing.T) {
	input := GuardianInput{User: "test"}
	_, err := RenderTemplate(input, TemplateOptions{Risk: "harm", Format: "2.0"})
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
}

func TestRenderTemplateThinkMode(t *testing.T) {
	input := GuardianInput{User: "test message"}
	out, err := RenderTemplate(input, TemplateOptions{
		Risk:   "harm",
		Format: "3.3",
		Think:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "think step by step") {
		t.Error("thinking mode should include step-by-step instruction")
	}
	if !strings.Contains(out, "<think>") {
		t.Error("thinking mode should reference <think> tags")
	}
}

func TestRenderTemplateThinkModeRequires33(t *testing.T) {
	input := GuardianInput{User: "test"}
	_, err := RenderTemplate(input, TemplateOptions{
		Risk:   "harm",
		Format: "3.2",
		Think:  true,
	})
	if err == nil {
		t.Fatal("expected error when using think mode with format != 3.3")
	}
}

func TestHarmRiskCategoriesImmutable(t *testing.T) {
	cats := HarmRiskCategories()
	cats[0] = "modified"
	fresh := HarmRiskCategories()
	if fresh[0] == "modified" {
		t.Error("HarmRiskCategories should return a copy")
	}
}

func TestRAGRiskCategoriesImmutable(t *testing.T) {
	cats := RAGRiskCategories()
	cats[0] = "modified"
	fresh := RAGRiskCategories()
	if fresh[0] == "modified" {
		t.Error("RAGRiskCategories should return a copy")
	}
}
