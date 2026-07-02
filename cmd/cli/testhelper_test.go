package cli

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	tokenizer "github.com/zerfoo/ztoken"
)

// cliFixedLogitsNode produces logits that select tokens in a fixed sequence.
type cliFixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
}

func (n *cliFixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *cliFixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *cliFixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *cliFixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *cliFixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
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

// buildCLITestModel creates a minimal inference.Model for CLI tests.
func buildCLITestModel(t *testing.T) *inference.Model {
	t.Helper()
	vocabSize := 8
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &cliFixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: []int{6, 7, 2}, // foo, bar, EOS
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})

	return inference.NewTestModel(gen, tok, engine,
		inference.ModelMetadata{
			VocabSize:  vocabSize,
			NumLayers:  1,
			EOSTokenID: 2,
			BOSTokenID: 1,
		},
		&registry.ModelInfo{ID: "test-model", Path: "/tmp/test"},
	)
}
