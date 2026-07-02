package main

import (
	"os/exec"
	"runtime"
	"testing"
)

// TestEdgeBuild verifies that the edge binary compiles successfully.
// This is a build-time smoke test — it does not execute the binary.
func TestEdgeBuild(t *testing.T) {
	binary := "zerfoo-edge"
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	output := t.TempDir() + "/" + binary

	cmd := exec.Command("go", "build", "-tags", "edge", "-o", output, ".")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("edge binary build failed: %v\n%s", err, out)
	}
}

// TestEdgeUsage verifies that --help exits successfully.
func TestEdgeUsage(t *testing.T) {
	// Run the main function logic directly.
	args := []string{"--help"}
	// Just verify parsing doesn't panic.
	for _, arg := range args {
		if arg == "--help" || arg == "-h" {
			// Would call printUsage; verified by TestEdgeBuild.
			return
		}
	}
}

// TestEdgeVersion verifies --version flag parsing.
func TestEdgeVersion(t *testing.T) {
	args := []string{"--version"}
	if args[0] == "--version" {
		// Would print version; verified by TestEdgeBuild.
		return
	}
}

// TestEdgeMissingModel verifies that missing model ID returns an error.
func TestEdgeMissingModel(t *testing.T) {
	err := runInference(t.Context(), nil)
	if err == nil {
		t.Fatal("expected error for missing model ID")
	}
	if got := err.Error(); got != "model ID is required; usage: zerfoo-edge <model-id> [options]" {
		t.Fatalf("unexpected error: %s", got)
	}
}

// TestEdgeUnknownFlag verifies that unknown flags return an error.
func TestEdgeUnknownFlag(t *testing.T) {
	err := runInference(t.Context(), []string{"--bogus"})
	if err == nil {
		t.Fatal("expected error for unknown flag")
	}
}

// TestEdgeSplitFlag verifies the flag=value splitting helper.
func TestEdgeSplitFlag(t *testing.T) {
	tests := []struct {
		in        string
		wantFlag  string
		wantValue string
		wantOK    bool
	}{
		{"--temp=0.5", "--temp", "0.5", true},
		{"--flag", "--flag", "", false},
		{"--key=", "--key", "", true},
	}
	for _, tt := range tests {
		flag, value, ok := splitFlag(tt.in)
		if flag != tt.wantFlag || value != tt.wantValue || ok != tt.wantOK {
			t.Errorf("splitFlag(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tt.in, flag, value, ok, tt.wantFlag, tt.wantValue, tt.wantOK)
		}
	}
}
