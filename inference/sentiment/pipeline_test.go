package sentiment

import (
	"context"
	"fmt"
	"math"
	"testing"
)

// mockEncoder simulates an encoder model for testing.
type mockEncoder struct {
	logits      []float32 // returned by Forward for every call
	outputShape []int
	closed      bool
}

func (m *mockEncoder) Forward(_ context.Context, inputIDs []int) ([]float32, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("empty input")
	}
	return m.logits, nil
}

func (m *mockEncoder) OutputShape() []int { return m.outputShape }

func (m *mockEncoder) Close() error {
	m.closed = true
	return nil
}

// mockTokenizer splits text into fake token IDs.
type mockTokenizer struct{}

func (t *mockTokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}
	// Return one token per character, capped at 10.
	ids := make([]int, 0, len(text))
	for i, r := range text {
		if i >= 10 {
			break
		}
		ids = append(ids, int(r))
	}
	return ids, nil
}

func (t *mockTokenizer) Decode(ids []int) (string, error) {
	return "", nil
}

func TestNewWithEncoder(t *testing.T) {
	enc := &mockEncoder{
		logits:      []float32{1.0, 2.0, 3.0},
		outputShape: []int{1, 3},
	}
	p, err := New("",
		WithEncoder(enc),
		WithLabels([]string{"negative", "neutral", "positive"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer p.Close()

	if p.encoder != enc {
		t.Fatal("expected injected encoder")
	}
	if p.ownsModel {
		t.Fatal("should not own injected encoder")
	}
}

func TestNewRequiresModelPathOrEncoder(t *testing.T) {
	_, err := New("")
	if err == nil {
		t.Fatal("expected error with empty modelPath and no encoder")
	}
}

func TestClassifyTokenized(t *testing.T) {
	enc := &mockEncoder{
		logits:      []float32{-1.0, 2.0, 0.5},
		outputShape: []int{1, 3},
	}
	p, err := New("",
		WithEncoder(enc),
		WithLabels([]string{"negative", "neutral", "positive"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer p.Close()

	results, err := p.ClassifyTokenized(context.Background(), [][]int{
		{101, 2023, 2003, 102},
	})
	if err != nil {
		t.Fatalf("ClassifyTokenized: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}

	r := results[0]
	if r.Label != "neutral" {
		t.Errorf("expected label 'neutral', got %q", r.Label)
	}
	if len(r.Logits) != 3 {
		t.Errorf("expected 3 logits, got %d", len(r.Logits))
	}
	if len(r.Confidence) != 3 {
		t.Errorf("expected 3 confidence scores, got %d", len(r.Confidence))
	}
	// Confidence should sum to ~1.0.
	sum := 0.0
	for _, c := range r.Confidence {
		sum += c
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("confidence sum = %f, want ~1.0", sum)
	}
	// The max confidence should match Score.
	if math.Abs(r.Score-r.Confidence[1]) > 1e-9 {
		t.Errorf("Score = %f, expected confidence[1] = %f", r.Score, r.Confidence[1])
	}
}

func TestClassifyTokenizedEmpty(t *testing.T) {
	enc := &mockEncoder{logits: []float32{1.0, 2.0}, outputShape: []int{1, 2}}
	p, err := New("", WithEncoder(enc))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.ClassifyTokenized(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestClassifyWithTokenizer(t *testing.T) {
	enc := &mockEncoder{
		logits:      []float32{3.0, -1.0},
		outputShape: []int{1, 2},
	}
	p, err := New("",
		WithEncoder(enc),
		WithTokenizer(&mockTokenizer{}),
		WithLabels([]string{"negative", "positive"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer p.Close()

	results, err := p.Classify(context.Background(), []string{"great movie"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Label != "negative" {
		t.Errorf("expected 'negative' (logit 3.0 > -1.0 at index 0), got %q", results[0].Label)
	}
}

func TestClassifyWithoutTokenizer(t *testing.T) {
	enc := &mockEncoder{logits: []float32{1.0, 2.0}, outputShape: []int{1, 2}}
	p, err := New("", WithEncoder(enc))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Classify(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error without tokenizer")
	}
}

func TestBatchProcessing(t *testing.T) {
	enc := &mockEncoder{
		logits:      []float32{0.1, 0.9},
		outputShape: []int{1, 2},
	}
	p, err := New("",
		WithEncoder(enc),
		WithBatchSize(2),
		WithLabels([]string{"negative", "positive"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer p.Close()

	// 5 inputs with batch size 2 -> 3 batches (2+2+1).
	inputs := [][]int{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12},
		{13, 14, 15},
	}
	results, err := p.ClassifyTokenized(context.Background(), inputs)
	if err != nil {
		t.Fatalf("ClassifyTokenized: %v", err)
	}
	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Label != "positive" {
			t.Errorf("result[%d]: expected 'positive', got %q", i, r.Label)
		}
	}
}

func TestContinuousScoring(t *testing.T) {
	// Use logits that produce sigmoid values not summing to 1.0.
	// sigmoid(2.0) ≈ 0.8808, sigmoid(3.0) ≈ 0.9526, sum ≈ 1.833
	enc := &mockEncoder{
		logits:      []float32{2.0, 3.0},
		outputShape: []int{1, 2},
	}
	p, err := New("",
		WithEncoder(enc),
		WithContinuous(),
		WithLabels([]string{"negative", "positive"}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer p.Close()

	results, err := p.ClassifyTokenized(context.Background(), [][]int{{1, 2}})
	if err != nil {
		t.Fatalf("ClassifyTokenized: %v", err)
	}

	r := results[0]
	// In continuous mode, confidence is sigmoid, not softmax.
	expectedSig := 1.0 / (1.0 + math.Exp(-2.0))
	if math.Abs(r.Confidence[0]-expectedSig) > 1e-6 {
		t.Errorf("confidence[0] = %f, want sigmoid(2.0) ≈ %f", r.Confidence[0], expectedSig)
	}
	// Confidence should NOT sum to 1.0 in continuous mode.
	sum := r.Confidence[0] + r.Confidence[1]
	if math.Abs(sum-1.0) < 0.01 {
		t.Errorf("continuous mode confidence should not sum to 1.0, got %f", sum)
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name   string
		logits []float64
		want   []float64
	}{
		{
			name:   "equal logits",
			logits: []float64{1.0, 1.0, 1.0},
			want:   []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
		},
		{
			name:   "single element",
			logits: []float64{5.0},
			want:   []float64{1.0},
		},
		{
			name:   "large difference",
			logits: []float64{100.0, 0.0},
			want:   []float64{1.0, 0.0},
		},
		{
			name:   "nil input",
			logits: nil,
			want:   nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := softmax(tt.logits)
			if tt.want == nil {
				if got != nil {
					t.Errorf("softmax(nil) = %v, want nil", got)
				}
				return
			}
			if len(got) != len(tt.want) {
				t.Fatalf("len(softmax) = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-6 {
					t.Errorf("softmax[%d] = %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestTruncation(t *testing.T) {
	enc := &mockEncoder{
		logits:      []float32{1.0, 2.0},
		outputShape: []int{1, 2},
	}
	p, err := New("",
		WithEncoder(enc),
		WithMaxSeqLen(3),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Input with 6 tokens should be truncated to 3.
	results, err := p.ClassifyTokenized(context.Background(), [][]int{{1, 2, 3, 4, 5, 6}})
	if err != nil {
		t.Fatalf("ClassifyTokenized: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
}

func TestDefaultLabels(t *testing.T) {
	tests := []struct {
		n    int
		want []string
	}{
		{2, []string{"negative", "positive"}},
		{3, []string{"positive", "negative", "neutral"}},
		{5, []string{"class_0", "class_1", "class_2", "class_3", "class_4"}},
	}
	for _, tt := range tests {
		got := defaultLabels(tt.n)
		if len(got) != len(tt.want) {
			t.Errorf("defaultLabels(%d): len=%d, want %d", tt.n, len(got), len(tt.want))
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("defaultLabels(%d)[%d] = %q, want %q", tt.n, i, got[i], tt.want[i])
			}
		}
	}
}

func TestCloseOwned(t *testing.T) {
	enc := &mockEncoder{logits: []float32{1.0}, outputShape: []int{1, 1}}
	p, err := New("", WithEncoder(enc))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// WithEncoder sets ownsModel=false, so Close should not close the encoder.
	if err := p.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if enc.closed {
		t.Error("Close should not close an injected encoder")
	}
}

func TestSigmoid(t *testing.T) {
	got := sigmoid([]float64{0.0})
	if math.Abs(got[0]-0.5) > 1e-9 {
		t.Errorf("sigmoid(0) = %f, want 0.5", got[0])
	}
}
