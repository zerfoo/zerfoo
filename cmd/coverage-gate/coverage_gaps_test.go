package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseCoverageProfile_EdgeCases(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "edge.out")

	content := `mode: set
no_colon_in_path 3 1
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 badnum 1
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 3 badcount
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 3 1
`
	if err := os.WriteFile(profile, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseCoverageProfile(profile)
	if err != nil {
		t.Fatalf("parseCoverageProfile: %v", err)
	}

	// Only the last valid line should be parsed.
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}
