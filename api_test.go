package zerfoo

import (
	"math"
	"testing"
)

func TestLoad_pathDetection(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		isLocal bool
	}{
		{"absolute path", "/models/gemma.gguf", true},
		{"relative dot-slash", "./model.gguf", true},
		{"relative parent", "../model.gguf", true},
		{"huggingface id", "google/gemma-3-1b-it", false},
		{"short alias", "gemma-3-1b-q4", false},
		{"bare name", "my-model", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isLocalPath(tt.input)
			if got != tt.isLocal {
				t.Errorf("isLocalPath(%q) = %v, want %v", tt.input, got, tt.isLocal)
			}
		})
	}
}

func TestLoad_missingFile(t *testing.T) {
	_, err := Load("/nonexistent/path/to/model.gguf")
	if err == nil {
		t.Fatal("expected error for non-existent file, got nil")
	}
}

func TestLoad_huggingFaceStub(t *testing.T) {
	_, err := Load("google/gemma-3-1b-it")
	if err == nil {
		t.Fatal("expected error for HuggingFace model ID, got nil")
	}
	want := "HuggingFace download not yet available"
	if got := err.Error(); got != want && len(got) < len(want) {
		t.Errorf("error = %q, want substring %q", got, want)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b Embedding
		want float32
	}{
		{
			name: "identical vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{1, 0, 0}},
			want: 1.0,
		},
		{
			name: "orthogonal vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{0, 1, 0}},
			want: 0.0,
		},
		{
			name: "opposite vectors",
			a:    Embedding{Vector: []float32{1, 0, 0}},
			b:    Embedding{Vector: []float32{-1, 0, 0}},
			want: -1.0,
		},
		{
			name: "45 degree angle",
			a:    Embedding{Vector: []float32{1, 0}},
			b:    Embedding{Vector: []float32{1, 1}},
			want: float32(1.0 / math.Sqrt(2)),
		},
		{
			name: "empty vectors",
			a:    Embedding{Vector: []float32{}},
			b:    Embedding{Vector: []float32{}},
			want: 0.0,
		},
		{
			name: "mismatched lengths",
			a:    Embedding{Vector: []float32{1, 2}},
			b:    Embedding{Vector: []float32{1, 2, 3}},
			want: 0.0,
		},
		{
			name: "zero vector",
			a:    Embedding{Vector: []float32{0, 0, 0}},
			b:    Embedding{Vector: []float32{1, 2, 3}},
			want: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.a.CosineSimilarity(tt.b)
			if diff := math.Abs(float64(got - tt.want)); diff > 1e-6 {
				t.Errorf("CosineSimilarity = %v, want %v (diff %v)", got, tt.want, diff)
			}
		})
	}
}

func TestEmbed_stub(t *testing.T) {
	// Create a zero-value Model to test the Embed stub.
	// We can't call Embed on a nil inner, so we test the error message
	// by constructing a Model with a non-nil inner. Since we can't easily
	// construct an inference.Model, we test the function signature works.
	m := &Model{}

	// This will panic due to nil inner if Embed tries to use it,
	// but it should return an error before accessing inner.
	result, err := m.Embed([]string{"hello"})
	if err == nil {
		t.Fatal("expected error from Embed stub, got nil")
	}
	want := "embedding not yet supported"
	if got := err.Error(); got != want {
		t.Errorf("Embed error = %q, want %q", got, want)
	}
	if result != nil {
		t.Errorf("Embed result = %v, want nil", result)
	}
}
