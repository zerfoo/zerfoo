package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseCoverageProfile(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "coverage.out")

	content := `mode: set
github.com/zerfoo/zerfoo/log/logger.go:10.30,15.2 3 1
github.com/zerfoo/zerfoo/log/logger.go:17.30,22.2 3 1
github.com/zerfoo/zerfoo/log/logger.go:24.30,29.2 3 0
github.com/zerfoo/zerfoo/config/loader.go:10.30,15.2 5 1
github.com/zerfoo/zerfoo/config/loader.go:17.30,22.2 5 1
`
	if err := os.WriteFile(profile, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseCoverageProfile(profile)
	if err != nil {
		t.Fatalf("parseCoverageProfile: %v", err)
	}

	// log package: 6 of 9 statements covered = 66.7%
	logCov, ok := results["github.com/zerfoo/zerfoo/log"]
	if !ok {
		t.Fatal("expected log package in results")
	}
	if logCov < 66.0 || logCov > 67.0 {
		t.Errorf("log coverage = %.1f%%, want ~66.7%%", logCov)
	}

	// config package: 10 of 10 statements covered = 100%
	cfgCov, ok := results["github.com/zerfoo/zerfoo/config"]
	if !ok {
		t.Fatal("expected config package in results")
	}
	if cfgCov != 100.0 {
		t.Errorf("config coverage = %.1f%%, want 100.0%%", cfgCov)
	}
}

func TestParseCoverageProfile_FileNotFound(t *testing.T) {
	_, err := parseCoverageProfile("/nonexistent/coverage.out")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestParseCoverageProfile_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "empty.out")
	if err := os.WriteFile(profile, []byte("mode: set\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseCoverageProfile(profile)
	if err != nil {
		t.Fatalf("parseCoverageProfile: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results for empty profile, got %d", len(results))
	}
}

func TestParseCoverageProfile_MalformedLines(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "bad.out")

	content := `mode: set
this is not a valid line
another bad line without proper format
github.com/zerfoo/zerfoo/log/logger.go:10.30,15.2 3 1
`
	if err := os.WriteFile(profile, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	results, err := parseCoverageProfile(profile)
	if err != nil {
		t.Fatalf("parseCoverageProfile: %v", err)
	}

	// Should still parse the valid line
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
}
