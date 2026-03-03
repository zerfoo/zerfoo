package tokenizers

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func TestTokenizerNode_OpType(t *testing.T) {
	tn := NewTokenizerNode(map[string]int32{"hello": 1}, 0)
	if tn.OpType() != "Tokenizer" {
		t.Errorf("OpType = %q, want %q", tn.OpType(), "Tokenizer")
	}
}

func TestTokenizerNode_Attributes(t *testing.T) {
	tn := NewTokenizerNode(map[string]int32{"hello": 1}, 0)
	if tn.Attributes() != nil {
		t.Error("expected nil attributes")
	}
}

func TestTokenizerNode_OutputShape(t *testing.T) {
	tn := NewTokenizerNode(map[string]int32{"hello": 1}, 0)
	os := tn.OutputShape()
	if len(os) != 2 || os[0] != 1 || os[1] != -1 {
		t.Errorf("OutputShape = %v, want [1, -1]", os)
	}
}

func TestTokenizerNode_Backward(t *testing.T) {
	tn := NewTokenizerNode(map[string]int32{"hello": 1}, 0)
	_, err := tn.Backward(context.Background(), types.FullBackprop, nil)
	if err == nil {
		t.Error("expected error for non-differentiable operation")
	}
}

func TestTokenizerNode_Forward_InputErrors(t *testing.T) {
	vocab := map[string]int32{"hello": 1, "world": 2}
	tn := NewTokenizerNode(vocab, 0)

	t.Run("zero_inputs", func(t *testing.T) {
		_, err := tn.Forward(context.Background())
		if err == nil {
			t.Error("expected error for 0 inputs")
		}
	})

	t.Run("wrong_type", func(t *testing.T) {
		numTensor, _ := tensor.New[float32]([]int{2}, nil)
		_, err := tn.Forward(context.Background(), numTensor)
		if err == nil {
			t.Error("expected error for non-TensorString input")
		}
	})

	t.Run("wrong_shape", func(t *testing.T) {
		strTensor, _ := tensor.NewString([]int{2}, []string{"hello", "world"})
		_, err := tn.Forward(context.Background(), strTensor)
		if err == nil {
			t.Error("expected error for multi-element string tensor")
		}
	})
}
