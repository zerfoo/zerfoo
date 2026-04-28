package mobile

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/registry"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	tokenizer "github.com/zerfoo/ztoken"
)

// fixedLogitsNode always returns logits where a specific token has the highest value.
type fixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
}

func (n *fixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *fixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *fixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *fixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *fixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}

	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

func buildTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: tokenSequence,
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func buildTestTokenizer() tokenizer.Tokenizer {
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7
	return tok
}

const testVocabSize = 10

// newTestEngine builds a mobile.Engine backed by a stub model that always
// generates the token sequence [4, 5] ("hello world") then hits EOS.
func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	tok := buildTestTokenizer()
	// Token sequence: 4=hello, 5=world, then 2=EOS
	g := buildTestGraph(t, testVocabSize, []int{4, 5, 2})
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  testVocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})
	model := inference.NewTestModel(gen, tok, eng, inference.ModelMetadata{
		Architecture:          "test",
		VocabSize:             testVocabSize,
		HiddenSize:            64,
		NumLayers:             1,
		MaxPositionEmbeddings: 32,
		EOSTokenID:            2,
		BOSTokenID:            1,
	}, &registry.ModelInfo{
		ID:   "test-model",
		Path: "/tmp/test",
	})
	return &Engine{model: model}
}

func TestMobile_NewEngine(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{
			name:    "missing file returns error",
			path:    "/nonexistent/model.gguf",
			wantErr: true,
		},
		{
			name:    "empty path returns error",
			path:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng, err := NewEngine(tt.path)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if eng != nil {
					t.Fatal("expected nil engine on error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if eng == nil {
				t.Fatal("expected non-nil engine")
			}
			_ = eng.Close()
		})
	}
}

func TestMobile_Generate(t *testing.T) {
	tests := []struct {
		name      string
		prompt    string
		maxTokens int
		wantErr   bool
	}{
		{
			name:      "basic generation",
			prompt:    "hello",
			maxTokens: 10,
		},
		{
			name:      "zero maxTokens uses default",
			prompt:    "hello",
			maxTokens: 0,
		},
		{
			name:      "closed engine returns error",
			prompt:    "hello",
			maxTokens: 10,
			wantErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestEngine(t)
			if tt.wantErr {
				_ = e.Close()
			}
			result, err := e.Generate(tt.prompt, tt.maxTokens)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == "" {
				t.Fatal("expected non-empty result")
			}
			if !tt.wantErr {
				_ = e.Close()
			}
		})
	}
}

func TestMobile_GenerateWithConfig(t *testing.T) {
	tests := []struct {
		name    string
		prompt  string
		config  *GenerateConfig
		wantErr bool
	}{
		{
			name:   "with temperature and topK",
			prompt: "hello",
			config: &GenerateConfig{
				Temperature: 0.8,
				TopK:        40,
				MaxTokens:   10,
			},
		},
		{
			name:   "nil config uses defaults",
			prompt: "hello",
			config: nil,
		},
		{
			name:   "with topP",
			prompt: "hello",
			config: &GenerateConfig{
				TopP:      0.9,
				MaxTokens: 5,
			},
		},
		{
			name:    "closed engine returns error",
			prompt:  "hello",
			config:  &GenerateConfig{MaxTokens: 10},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestEngine(t)
			if tt.wantErr {
				_ = e.Close()
			}
			result, err := e.GenerateWithConfig(tt.prompt, tt.config)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result == "" {
				t.Fatal("expected non-empty result")
			}
			if !tt.wantErr {
				_ = e.Close()
			}
		})
	}
}

func TestMobile_Tokenize(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		wantErr bool
	}{
		{
			name: "basic tokenization",
			text: "hello world",
		},
		{
			name: "single token",
			text: "hello",
		},
		{
			name:    "closed engine returns error",
			text:    "hello",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestEngine(t)
			if tt.wantErr {
				_ = e.Close()
			}
			result, err := e.Tokenize(tt.text)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Verify result is valid JSON array of ints.
			var ids []int
			if err := json.Unmarshal([]byte(result), &ids); err != nil {
				t.Fatalf("result is not valid JSON int array: %v (got %q)", err, result)
			}
			if len(ids) == 0 {
				t.Fatal("expected non-empty token ID list")
			}
			if !tt.wantErr {
				_ = e.Close()
			}
		})
	}
}

func TestMobile_Close(t *testing.T) {
	tests := []struct {
		name        string
		doubleClose bool
	}{
		{
			name:        "single close",
			doubleClose: false,
		},
		{
			name:        "double close is safe",
			doubleClose: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestEngine(t)
			if err := e.Close(); err != nil {
				t.Fatalf("first close: %v", err)
			}
			if tt.doubleClose {
				if err := e.Close(); err != nil {
					t.Fatalf("second close: %v", err)
				}
			}
		})
	}
}
