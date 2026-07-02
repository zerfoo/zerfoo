package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDeprecationLinter(t *testing.T) {
	tests := []struct {
		name       string
		src        string
		wantCount  int
		wantNames  []string
		wantFields [][]string // expected Missing fields per violation
	}{
		{
			name: "well-formed deprecation",
			src: `package foo

// Deprecated: Use NewFunc instead. Removed in v1.0.
func OldFunc() {}
`,
			wantCount: 0,
		},
		{
			name: "missing version",
			src: `package foo

// Deprecated: Use NewFunc instead.
func NoVersion() {}
`,
			wantCount:  1,
			wantNames:  []string{"NoVersion"},
			wantFields: [][]string{{"version"}},
		},
		{
			name: "missing replacement guidance",
			src: `package foo

// Deprecated: Will be gone in v2.0.
func NoGuidance() {}
`,
			wantCount:  1,
			wantNames:  []string{"NoGuidance"},
			wantFields: [][]string{{"replacement guidance"}},
		},
		{
			name: "missing both",
			src: `package foo

// Deprecated: do not use.
func NoBoth() {}
`,
			wantCount:  1,
			wantNames:  []string{"NoBoth"},
			wantFields: [][]string{{"replacement guidance", "version"}},
		},
		{
			name: "no deprecation comment",
			src: `package foo

// Regular comment.
func Regular() {}
`,
			wantCount: 0,
		},
		{
			name: "type deprecation well-formed",
			src: `package foo

// Deprecated: Use NewType instead. Removed in v1.0.
type OldType struct{}
`,
			wantCount: 0,
		},
		{
			name: "var deprecation missing version",
			src: `package foo

// Deprecated: Use NewVar instead.
var OldVar int
`,
			wantCount:  1,
			wantNames:  []string{"OldVar"},
			wantFields: [][]string{{"version"}},
		},
		{
			name: "multiple violations",
			src: `package foo

// Deprecated: gone.
func Bad1() {}

// Deprecated: old stuff.
func Bad2() {}

// Deprecated: Use Good instead. Since v1.0.
func Ok() {}
`,
			wantCount:  2,
			wantNames:  []string{"Bad1", "Bad2"},
			wantFields: [][]string{{"replacement guidance", "version"}, {"replacement guidance", "version"}},
		},
		{
			name: "replacement with see",
			src: `package foo

// Deprecated: See NewAPI for details. Since v2.1.
func SeeExample() {}
`,
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test.go")
			if err := os.WriteFile(path, []byte(tt.src), 0o644); err != nil {
				t.Fatal(err)
			}

			violations, err := checkFile(path)
			if err != nil {
				t.Fatalf("checkFile: %v", err)
			}

			if got := len(violations); got != tt.wantCount {
				t.Fatalf("got %d violations, want %d", got, tt.wantCount)
			}

			for i, v := range violations {
				if i < len(tt.wantNames) && v.Name != tt.wantNames[i] {
					t.Errorf("violation[%d].Name = %q, want %q", i, v.Name, tt.wantNames[i])
				}
				if i < len(tt.wantFields) {
					if len(v.Missing) != len(tt.wantFields[i]) {
						t.Errorf("violation[%d].Missing = %v, want %v", i, v.Missing, tt.wantFields[i])
						continue
					}
					for j, m := range v.Missing {
						if m != tt.wantFields[i][j] {
							t.Errorf("violation[%d].Missing[%d] = %q, want %q", i, j, m, tt.wantFields[i][j])
						}
					}
				}
			}
		})
	}
}

func TestCheckDir(t *testing.T) {
	dir := t.TempDir()

	// Create a subdirectory with a Go file containing a violation.
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}

	src := `package sub

// Deprecated: old.
func Bad() {}
`
	if err := os.WriteFile(filepath.Join(sub, "bad.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	// Create a vendor dir that should be skipped.
	vendor := filepath.Join(dir, "vendor")
	if err := os.MkdirAll(vendor, 0o755); err != nil {
		t.Fatal(err)
	}
	vendorSrc := `package vendor

// Deprecated: old.
func VendorBad() {}
`
	if err := os.WriteFile(filepath.Join(vendor, "v.go"), []byte(vendorSrc), 0o644); err != nil {
		t.Fatal(err)
	}

	violations, err := checkDir(dir)
	if err != nil {
		t.Fatalf("checkDir: %v", err)
	}

	if got := len(violations); got != 1 {
		t.Fatalf("got %d violations, want 1 (vendor should be skipped)", got)
	}

	if violations[0].Name != "Bad" {
		t.Errorf("violation.Name = %q, want %q", violations[0].Name, "Bad")
	}
}

func TestHasReplacementGuidance(t *testing.T) {
	tests := []struct {
		text string
		want bool
	}{
		{"Use NewFunc instead.", true},
		{"See the new API.", true},
		{"Replace with NewFunc.", true},
		{"Switch to NewFunc.", true},
		{"Migrate to v2 API.", true},
		{"Prefer NewFunc.", true},
		{"Superseded by NewFunc.", true},
		{"Removed in v2.0.", true},
		{"do not use.", false},
		{"old and broken.", false},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			if got := hasReplacementGuidance(tt.text); got != tt.want {
				t.Errorf("hasReplacementGuidance(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

func TestHasVersion(t *testing.T) {
	tests := []struct {
		text string
		want bool
	}{
		{"Removed in v1.0.", true},
		{"Since v2.1.", true},
		{"no version here.", false},
		{"version 2 is out.", false},
		{"v1.0-beta.", true},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			if got := hasVersion(tt.text); got != tt.want {
				t.Errorf("hasVersion(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}
