package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCoverageProfile(t *testing.T) {
	dir := t.TempDir()
	profile := filepath.Join(dir, "coverage.out")

	content := `mode: set
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 3 1
github.com/zerfoo/ztensor/log/logger.go:17.30,22.2 3 1
github.com/zerfoo/ztensor/log/logger.go:24.30,29.2 3 0
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
	logCov, ok := results["github.com/zerfoo/ztensor/log"]
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
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 3 1
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

func TestIsExcluded(t *testing.T) {
	tests := []struct {
		name     string
		pkg      string
		prefixes []string
		want     bool
	}{
		{"no prefixes", "github.com/zerfoo/ztensor/log", nil, false},
		{"matching prefix", "github.com/zerfoo/zerfoo/internal/cuda", []string{"internal/cuda"}, true},
		{"non-matching prefix", "github.com/zerfoo/ztensor/log", []string{"internal/cuda"}, false},
		{"multiple prefixes match", "github.com/zerfoo/zerfoo/internal/hip", []string{"internal/cuda", "internal/hip"}, true},
		{"whitespace prefix", "github.com/zerfoo/zerfoo/internal/cuda", []string{" internal/cuda "}, true},
		{"empty prefixes", "github.com/zerfoo/ztensor/log", []string{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isExcluded(tt.pkg, tt.prefixes)
			if got != tt.want {
				t.Errorf("isExcluded(%q, %v) = %v, want %v", tt.pkg, tt.prefixes, got, tt.want)
			}
		})
	}
}

func TestRun(t *testing.T) {
	dir := t.TempDir()

	// Create a coverage profile
	profileContent := `mode: set
github.com/zerfoo/ztensor/log/logger.go:10.30,15.2 3 1
github.com/zerfoo/ztensor/log/logger.go:17.30,22.2 3 1
github.com/zerfoo/ztensor/log/logger.go:24.30,29.2 3 0
github.com/zerfoo/zerfoo/config/loader.go:10.30,15.2 5 1
github.com/zerfoo/zerfoo/config/loader.go:17.30,22.2 5 1
`
	profilePath := filepath.Join(dir, "coverage.out")
	if err := os.WriteFile(profilePath, []byte(profileContent), 0o600); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		profile     string
		threshold   float64
		exclude     []string
		wantErr     bool
		errContains string
		outContains string
	}{
		{
			name:        "missing profile file",
			profile:     "/nonexistent/coverage.out",
			threshold:   90,
			wantErr:     true,
			errContains: "error",
		},
		{
			name:        "all pass at low threshold",
			profile:     profilePath,
			threshold:   50,
			wantErr:     false,
			outContains: "All packages at or above",
		},
		{
			name:        "fail at high threshold",
			profile:     profilePath,
			threshold:   95,
			wantErr:     true,
			errContains: "coverage gate failed",
		},
		{
			name:        "exclude failing package",
			profile:     profilePath,
			threshold:   95,
			exclude:     []string{"log"},
			wantErr:     false,
			outContains: "SKIP",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := run(tt.profile, tt.threshold, tt.exclude, &buf)

			if tt.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if tt.errContains != "" && err != nil {
				if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errContains)
				}
			}
			if tt.outContains != "" {
				if !strings.Contains(buf.String(), tt.outContains) {
					t.Errorf("output does not contain %q:\n%s", tt.outContains, buf.String())
				}
			}
		})
	}
}
