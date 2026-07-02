package parity

import (
	"testing"

	"github.com/zerfoo/zerfoo/inference"
)

// TestArchitectureRegistry validates that all expected architectures are registered.
// T32.2: Validate model architecture coverage.
func TestArchitectureRegistry(t *testing.T) {
	// All architectures that should be registered (from GGUF general.architecture values).
	expected := []struct {
		name    string
		aliases []string // alternative GGUF names that map to the same builder
	}{
		{name: "llama", aliases: []string{"llama"}},
		{name: "gemma", aliases: []string{"gemma", "gemma3"}},
		{name: "gemma3n"},
		{name: "qwen2"},
		{name: "mistral"},
		{name: "phi", aliases: []string{"phi", "phi3"}},
		{name: "deepseek_v3", aliases: []string{"deepseek_v3", "deepseek2"}},
		{name: "mamba"},
		{name: "mamba3"},
		{name: "jamba"},
		{name: "granite"},
		{name: "whisper"},
		{name: "bert"},
		{name: "command-r"},
		{name: "falcon"},
		{name: "llama4"},
		{name: "llava"},
		{name: "mixtral"},
		{name: "qwen_vl"},
		{name: "rwkv"},
	}

	registered := inference.ListArchitectures()
	regMap := make(map[string]bool)
	for _, name := range registered {
		regMap[name] = true
	}

	t.Logf("Registered architectures: %d", len(registered))
	for _, name := range registered {
		t.Logf("  %s", name)
	}

	// Verify each expected architecture is registered.
	for _, exp := range expected {
		t.Run(exp.name, func(t *testing.T) {
			builder, ok := inference.GetArchitecture(exp.name)
			if !ok {
				t.Errorf("architecture %q not registered", exp.name)
				return
			}
			if builder == nil {
				t.Errorf("architecture %q has nil builder", exp.name)
			}

			// Check aliases.
			for _, alias := range exp.aliases {
				if _, ok := inference.GetArchitecture(alias); !ok {
					t.Errorf("alias %q for %q not registered", alias, exp.name)
				}
			}
		})
	}

	// Verify minimum count.
	if len(registered) < 20 {
		t.Errorf("expected >= 20 registered architectures, got %d", len(registered))
	}
}

// TestArchitectureBuilderNonNil verifies every registered builder is callable.
func TestArchitectureBuilderNonNil(t *testing.T) {
	for _, name := range inference.ListArchitectures() {
		t.Run(name, func(t *testing.T) {
			builder, ok := inference.GetArchitecture(name)
			if !ok {
				t.Fatalf("GetArchitecture(%q) returned false", name)
			}
			if builder == nil {
				t.Fatalf("GetArchitecture(%q) returned nil builder", name)
			}
		})
	}
}

// TestArchitectureFamilyCoverage validates coverage of major model families.
// Each family represents models commonly found on HuggingFace.
func TestArchitectureFamilyCoverage(t *testing.T) {
	families := map[string][]string{
		"Meta Llama":    {"llama", "llama4"},
		"Google Gemma":  {"gemma", "gemma3", "gemma3n"},
		"Mistral AI":    {"mistral", "mixtral"},
		"Alibaba Qwen":  {"qwen2", "qwen_vl"},
		"Microsoft Phi": {"phi", "phi3"},
		"DeepSeek":      {"deepseek_v3", "deepseek2"},
		"Cohere":        {"command-r"},
		"TII Falcon":    {"falcon"},
		"RWKV":          {"rwkv"},
		"SSM/Hybrid":    {"mamba", "mamba3", "jamba"},
		"IBM Granite":   {"granite"},
		"Multimodal":    {"llava", "qwen_vl", "whisper"},
		"Encoder":       {"bert"},
	}

	for family, archs := range families {
		t.Run(family, func(t *testing.T) {
			for _, arch := range archs {
				if _, ok := inference.GetArchitecture(arch); !ok {
					t.Errorf("family %q: architecture %q not registered", family, arch)
				}
			}
		})
	}
}
