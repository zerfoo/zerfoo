//go:build integration

package huggingface

import (
	"testing"
)

func TestIntegration_GetModel(t *testing.T) {
	c := NewClient()
	info, err := c.GetModel("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	if info.ID != "google/gemma-3-1b-it" {
		t.Errorf("got ID %q, want %q", info.ID, "google/gemma-3-1b-it")
	}
	if len(info.Files) == 0 {
		t.Error("expected at least one file")
	}
}

func TestIntegration_ListGGUFFiles(t *testing.T) {
	c := NewClient()
	files, err := c.ListGGUFFiles("google/gemma-3-1b-it")
	if err != nil {
		t.Fatalf("ListGGUFFiles: %v", err)
	}
	if len(files) == 0 {
		t.Error("expected at least one GGUF file")
	}
	for _, f := range files {
		t.Logf("  %s (%d bytes)", f.Filename, f.Size)
	}
}

func TestIntegration_ResolveGGUF(t *testing.T) {
	c := NewClient()
	f, err := c.ResolveGGUF("google/gemma-3-1b-it", "Q4_K_M")
	if err != nil {
		t.Fatalf("ResolveGGUF: %v", err)
	}
	t.Logf("resolved: %s (%d bytes)", f.Filename, f.Size)
}
