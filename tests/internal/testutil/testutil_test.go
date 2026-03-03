package testutil

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMeanRelativeError(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
		tol  float64
	}{
		{
			name: "identical slices",
			a:    []float32{1, 2, 3},
			b:    []float32{1, 2, 3},
			want: 0,
			tol:  1e-10,
		},
		{
			name: "known difference",
			a:    []float32{1, 2, 3},
			b:    []float32{1.1, 2.2, 3.3},
			// num = |1-1.1| + |2-2.2| + |3-3.3| ≈ 0.6
			// den = |1.1|+1e-12 + |2.2|+1e-12 + |3.3|+1e-12 ≈ 6.6
			// want ≈ 0.6/6.6 ≈ 0.0909...
			want: 0.6 / (6.6 + 3e-12),
			tol:  1e-4,
		},
		{
			name: "single element",
			a:    []float32{5},
			b:    []float32{10},
			want: 5.0 / (10.0 + 1e-12),
			tol:  1e-10,
		},
		{
			name: "zeros in b",
			a:    []float32{0.1, 0.2},
			b:    []float32{0, 0},
			// den = 0+0 + 2e-12, num = 0.1+0.2 = 0.3
			want: 0.3 / (2e-12),
			tol:  1e6, // very large since denominator is tiny
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := MeanRelativeError(tc.a, tc.b)
			if tc.name == "identical slices" {
				if got != 0 {
					t.Errorf("expected 0, got %f", got)
				}
				return
			}
			if tc.name == "zeros in b" {
				// Just verify it doesn't panic and returns a large value
				if got <= 0 {
					t.Errorf("expected positive value for zero denominator, got %f", got)
				}
				return
			}
			diff := math.Abs(got - tc.want)
			if diff > tc.tol {
				t.Errorf("got %f, want %f (diff=%e)", got, tc.want, diff)
			}
		})
	}
}

func TestMeanRelativeError_Symmetry(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5}
	b := []float32{1.1, 2.2, 3.3, 4.4, 5.5}

	ab := MeanRelativeError(a, b)
	ba := MeanRelativeError(b, a)

	// MRE is not symmetric because denominator uses |b|, so ab != ba in general
	if ab == 0 || ba == 0 {
		t.Error("expected non-zero MRE values")
	}
}

func TestTopKAgreement(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		k    int
		want float64
	}{
		{
			name: "identical slices full agreement",
			a:    []float32{3, 1, 2, 5, 4},
			b:    []float32{3, 1, 2, 5, 4},
			k:    3,
			want: 1.0,
		},
		{
			name: "no overlap",
			a:    []float32{10, 9, 8, 1, 2, 3},
			b:    []float32{1, 2, 3, 10, 9, 8},
			k:    3,
			want: 0.0,
		},
		{
			name: "partial overlap",
			a:    []float32{10, 9, 8, 7, 1, 2},
			b:    []float32{10, 9, 1, 2, 8, 7},
			k:    3,
			// top 3 of a: indices 0,1,2 (values 10,9,8)
			// top 3 of b: indices 0,1,4 (values 10,9,8)
			// overlap: indices 0,1 => 2/3
			want: 2.0 / 3.0,
		},
		{
			name: "k larger than slice",
			a:    []float32{3, 1, 2},
			b:    []float32{3, 1, 2},
			k:    10,
			want: 1.0,
		},
		{
			name: "k equals 1",
			a:    []float32{1, 5, 3},
			b:    []float32{1, 5, 3},
			k:    1,
			want: 1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := TopKAgreement(tc.a, tc.b, tc.k)
			if math.Abs(got-tc.want) > 1e-10 {
				t.Errorf("got %f, want %f", got, tc.want)
			}
		})
	}
}

func TestRelError(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
		tol  float64
	}{
		{
			name: "identical",
			a:    []float64{1, 2, 3},
			b:    []float64{1, 2, 3},
			want: 0,
			tol:  1e-15,
		},
		{
			name: "known values",
			a:    []float64{1.0, 2.0},
			b:    []float64{1.5, 2.5},
			// num = |1.0-1.5| + |2.0-2.5| = 0.5 + 0.5 = 1.0
			// den = |1.5| + |2.5| + 2e-12 = 4.0 + 2e-12
			want: 1.0 / (4.0 + 2e-12),
			tol:  1e-10,
		},
		{
			name: "negative values",
			a:    []float64{-1, -2},
			b:    []float64{-1, -2},
			want: 0,
			tol:  1e-15,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := RelError(tc.a, tc.b)
			diff := math.Abs(got - tc.want)
			if diff > tc.tol {
				t.Errorf("got %f, want %f (diff=%e)", got, tc.want, diff)
			}
		})
	}
}

func TestRequire_OK(t *testing.T) {
	val := Require(t, 42, true, "should not fail")
	if val != 42 {
		t.Errorf("expected 42, got %d", val)
	}
}

func TestRequire_String(t *testing.T) {
	val := Require(t, "hello", true, "should not fail")
	if val != "hello" {
		t.Errorf("expected hello, got %s", val)
	}
}

func TestLoadPrompts_FromFile(t *testing.T) {
	// Create a temp file with known prompts.
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")

	content := "Hello world\nFoo bar\n\nBaz qux\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	prompts := LoadPrompts(path, 10, 1)
	if len(prompts) != 3 {
		t.Fatalf("expected 3 prompts, got %d: %v", len(prompts), prompts)
	}

	if prompts[0] != "Hello world" {
		t.Errorf("prompts[0] = %q, want %q", prompts[0], "Hello world")
	}

	if prompts[1] != "Foo bar" {
		t.Errorf("prompts[1] = %q, want %q", prompts[1], "Foo bar")
	}

	if prompts[2] != "Baz qux" {
		t.Errorf("prompts[2] = %q, want %q", prompts[2], "Baz qux")
	}
}

func TestLoadPrompts_FromFile_Truncated(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")

	content := "A\nB\nC\nD\nE\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	prompts := LoadPrompts(path, 2, 1)
	if len(prompts) != 2 {
		t.Fatalf("expected 2 prompts, got %d", len(prompts))
	}
}

func TestLoadPrompts_NoFile(t *testing.T) {
	prompts := LoadPrompts("/nonexistent/path/file.txt", 5, 42)
	if len(prompts) != 5 {
		t.Fatalf("expected 5 generated prompts, got %d", len(prompts))
	}

	for i, p := range prompts {
		if p == "" {
			t.Errorf("prompt[%d] is empty", i)
		}
	}
}

func TestLoadPrompts_Deterministic(t *testing.T) {
	p1 := LoadPrompts("/nonexistent", 3, 99)
	p2 := LoadPrompts("/nonexistent", 3, 99)

	for i := range p1 {
		if p1[i] != p2[i] {
			t.Errorf("prompt[%d] differs: %q vs %q", i, p1[i], p2[i])
		}
	}
}

func TestLoadPrompts_ZeroN(t *testing.T) {
	prompts := LoadPrompts("/nonexistent", 0, 1)
	// n=0 means "no limit" in LoadPrompts for generated, but the loop generates 0 prompts
	if len(prompts) != 0 {
		t.Errorf("expected 0 prompts for n=0, got %d", len(prompts))
	}
}
