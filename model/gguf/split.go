package gguf

import (
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/tensor"
)

// SplitMergedQKV finds merged QKV projection tensors (*.self_attn.qkv_proj.weight)
// in the tensor map and splits each into separate Q, K, V projection tensors.
// This handles architectures like Phi that store merged QKV weights in GGUF.
//
// For MHA (num_heads == num_kv_heads): each projection gets 1/3 of rows.
// For GQA (num_heads > num_kv_heads): Q gets num_heads*head_dim rows,
// K and V each get num_kv_heads*head_dim rows.
func SplitMergedQKV(tensors map[string]*tensor.TensorNumeric[float32], cfg *ModelConfig) error {
	if cfg.NumHeads == 0 {
		return fmt.Errorf("splitMergedQKV: NumHeads is zero")
	}
	if cfg.HiddenSize == 0 {
		return fmt.Errorf("splitMergedQKV: HiddenSize is zero")
	}
	numKVHeads := cfg.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = cfg.NumHeads
	}

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	qRows := cfg.NumHeads * headDim
	kRows := numKVHeads * headDim
	vRows := numKVHeads * headDim

	// Collect keys to split (avoid mutating map during iteration).
	var toSplit []string
	for name := range tensors {
		if strings.Contains(name, "self_attn.qkv_proj.weight") {
			toSplit = append(toSplit, name)
		}
	}

	// Also collect bias tensors.
	var biasToSplit []string
	for name := range tensors {
		if strings.Contains(name, "self_attn.qkv_proj.bias") {
			biasToSplit = append(biasToSplit, name)
		}
	}

	for _, name := range toSplit {
		t := tensors[name]
		if err := splitQKVTensor(tensors, name, t, qRows, kRows, vRows, "weight"); err != nil {
			return err
		}
	}

	for _, name := range biasToSplit {
		t := tensors[name]
		if err := splitQKVBias(tensors, name, t, qRows, kRows, vRows); err != nil {
			return err
		}
	}

	return nil
}

// splitQKVTensor splits a 2D merged QKV weight tensor [qRows+kRows+vRows, hidden]
// into three separate tensors.
func splitQKVTensor(
	tensors map[string]*tensor.TensorNumeric[float32],
	name string,
	t *tensor.TensorNumeric[float32],
	qRows, kRows, vRows int,
	suffix string,
) error {
	shape := t.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("splitMergedQKV: tensor %q has %d dims, want 2", name, len(shape))
	}

	totalRows := shape[0]
	cols := shape[1]
	expectedRows := qRows + kRows + vRows
	if totalRows != expectedRows {
		return fmt.Errorf("splitMergedQKV: tensor %q has %d rows, want %d (q=%d + k=%d + v=%d)",
			name, totalRows, expectedRows, qRows, kRows, vRows)
	}

	data := t.Data()
	prefix := strings.TrimSuffix(name, "qkv_proj."+suffix)

	// Q: rows [0, qRows)
	qData := make([]float32, qRows*cols)
	copy(qData, data[:qRows*cols])
	qT, err := tensor.New[float32]([]int{qRows, cols}, qData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create Q tensor: %w", err)
	}
	tensors[prefix+"q_proj."+suffix] = qT

	// K: rows [qRows, qRows+kRows)
	kData := make([]float32, kRows*cols)
	copy(kData, data[qRows*cols:(qRows+kRows)*cols])
	kT, err := tensor.New[float32]([]int{kRows, cols}, kData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create K tensor: %w", err)
	}
	tensors[prefix+"k_proj."+suffix] = kT

	// V: rows [qRows+kRows, qRows+kRows+vRows)
	vData := make([]float32, vRows*cols)
	copy(vData, data[(qRows+kRows)*cols:])
	vT, err := tensor.New[float32]([]int{vRows, cols}, vData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create V tensor: %w", err)
	}
	tensors[prefix+"v_proj."+suffix] = vT

	// Remove original merged tensor.
	delete(tensors, name)

	return nil
}

// splitQKVBias splits a 1D merged QKV bias tensor [qRows+kRows+vRows]
// into three separate tensors.
func splitQKVBias(
	tensors map[string]*tensor.TensorNumeric[float32],
	name string,
	t *tensor.TensorNumeric[float32],
	qRows, kRows, vRows int,
) error {
	shape := t.Shape()
	if len(shape) != 1 {
		return fmt.Errorf("splitMergedQKV: bias tensor %q has %d dims, want 1", name, len(shape))
	}

	totalElems := shape[0]
	expectedElems := qRows + kRows + vRows
	if totalElems != expectedElems {
		return fmt.Errorf("splitMergedQKV: bias tensor %q has %d elements, want %d",
			name, totalElems, expectedElems)
	}

	data := t.Data()
	prefix := strings.TrimSuffix(name, "qkv_proj.bias")

	qData := make([]float32, qRows)
	copy(qData, data[:qRows])
	qT, err := tensor.New[float32]([]int{qRows}, qData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create Q bias: %w", err)
	}
	tensors[prefix+"q_proj.bias"] = qT

	kData := make([]float32, kRows)
	copy(kData, data[qRows:qRows+kRows])
	kT, err := tensor.New[float32]([]int{kRows}, kData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create K bias: %w", err)
	}
	tensors[prefix+"k_proj.bias"] = kT

	vData := make([]float32, vRows)
	copy(vData, data[qRows+kRows:])
	vT, err := tensor.New[float32]([]int{vRows}, vData)
	if err != nil {
		return fmt.Errorf("splitMergedQKV: create V bias: %w", err)
	}
	tensors[prefix+"v_proj.bias"] = vT

	delete(tensors, name)

	return nil
}
