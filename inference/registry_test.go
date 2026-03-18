package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

func dummyBuilder(
	_ map[string]*tensor.TensorNumeric[float32],
	_ *gguf.ModelConfig,
	_ compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	return nil, nil, nil
}

// saveAndResetRegistry saves the current registry state, resets it, and
// returns a function that restores the original state.
func saveAndResetRegistry() func() {
	archRegistry.mu.Lock()
	saved := archRegistry.builders
	archRegistry.builders = make(map[string]ArchBuilder)
	archRegistry.mu.Unlock()
	return func() {
		archRegistry.mu.Lock()
		archRegistry.builders = saved
		archRegistry.mu.Unlock()
	}
}

func TestArchitectureRegistry(t *testing.T) {
	t.Run("register and get", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		RegisterArchitecture("test-arch", dummyBuilder)

		b, ok := GetArchitecture("test-arch")
		if !ok {
			t.Fatal("expected builder to be registered")
		}
		if b == nil {
			t.Fatal("expected non-nil builder")
		}
	})

	t.Run("get unregistered returns false", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		_, ok := GetArchitecture("nonexistent")
		if ok {
			t.Fatal("expected ok=false for unregistered architecture")
		}
	})

	t.Run("list returns sorted names", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		RegisterArchitecture("zeta", dummyBuilder)
		RegisterArchitecture("alpha", dummyBuilder)
		RegisterArchitecture("mid", dummyBuilder)

		names := ListArchitectures()
		if len(names) != 3 {
			t.Fatalf("expected 3 names, got %d", len(names))
		}
		expected := []string{"alpha", "mid", "zeta"}
		for i, name := range names {
			if name != expected[i] {
				t.Errorf("names[%d] = %q, want %q", i, name, expected[i])
			}
		}
	})

	t.Run("list empty registry", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		names := ListArchitectures()
		if len(names) != 0 {
			t.Fatalf("expected empty list, got %d names", len(names))
		}
	})

	t.Run("duplicate registration panics", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		RegisterArchitecture("dup", dummyBuilder)

		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on duplicate registration")
			}
		}()
		RegisterArchitecture("dup", dummyBuilder)
	})

	t.Run("empty name panics", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on empty name")
			}
		}()
		RegisterArchitecture("", dummyBuilder)
	})

	t.Run("nil builder panics", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on nil builder")
			}
		}()
		RegisterArchitecture("nil-builder", nil)
	})

	t.Run("multiple aliases for same builder", func(t *testing.T) {
		restore := saveAndResetRegistry()
		defer restore()
		RegisterArchitecture("gemma", dummyBuilder)
		RegisterArchitecture("gemma3", dummyBuilder)

		b1, ok1 := GetArchitecture("gemma")
		b2, ok2 := GetArchitecture("gemma3")
		if !ok1 || !ok2 {
			t.Fatal("expected both aliases to be registered")
		}
		if b1 == nil || b2 == nil {
			t.Fatal("expected non-nil builders")
		}
	})
}

func TestDefaultArchitecturesRegistered(t *testing.T) {
	// After init(), all built-in architectures should be registered.
	expected := []string{
		"deepseek2", "deepseek_v3",
		"gemma", "gemma3",
		"jamba",
		"llama",
		"mamba",
		"mistral",
		"phi", "phi3",
		"qwen2",
		"whisper",
	}
	for _, name := range expected {
		t.Run(name, func(t *testing.T) {
			b, ok := GetArchitecture(name)
			if !ok {
				t.Fatalf("architecture %q not registered", name)
			}
			if b == nil {
				t.Fatalf("architecture %q has nil builder", name)
			}
		})
	}
}

func TestBuildArchGraphUsesRegistry(t *testing.T) {
	_, _, err := buildArchGraph("totally-unknown-arch-xyz", nil, nil, nil)
	if err == nil {
		t.Fatal("expected error for unknown architecture")
	}
}
