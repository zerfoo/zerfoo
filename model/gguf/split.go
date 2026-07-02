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

// SplitMergedGateUp finds merged gate+up MLP tensors (*.mlp.up_proj.weight)
// where gate_proj is absent and up_proj has double the expected intermediate size.
// This handles architectures like Phi that concatenate gate and up projections
// into a single tensor: ffn_up has shape [2 * intermediate_size, hidden_size].
// The first half of rows is the gate projection, the second half is the up projection.
func SplitMergedGateUp(tensors map[string]*tensor.TensorNumeric[float32], cfg *ModelConfig) error {
	if cfg.IntermediateSize == 0 {
		return nil // no intermediate size configured, nothing to split
	}

	intermediateSize := cfg.IntermediateSize

	// Collect weight tensors to split (avoid mutating map during iteration).
	var toSplit []string
	for name := range tensors {
		if !strings.Contains(name, "mlp.up_proj.weight") {
			continue
		}
		// Only split if gate_proj is missing.
		gateKey := strings.Replace(name, "mlp.up_proj.weight", "mlp.gate_proj.weight", 1)
		if _, hasGate := tensors[gateKey]; hasGate {
			continue
		}
		// Only split if up_proj has double the expected intermediate size.
		t := tensors[name]
		shape := t.Shape()
		if len(shape) == 2 && shape[0] == 2*intermediateSize {
			toSplit = append(toSplit, name)
		}
	}

	// Collect bias tensors to split.
	var biasToSplit []string
	for name := range tensors {
		if !strings.Contains(name, "mlp.up_proj.bias") {
			continue
		}
		gateBiasKey := strings.Replace(name, "mlp.up_proj.bias", "mlp.gate_proj.bias", 1)
		if _, hasGate := tensors[gateBiasKey]; hasGate {
			continue
		}
		t := tensors[name]
		shape := t.Shape()
		if len(shape) == 1 && shape[0] == 2*intermediateSize {
			biasToSplit = append(biasToSplit, name)
		}
	}

	for _, name := range toSplit {
		t := tensors[name]
		if err := splitGateUpWeight(tensors, name, t, intermediateSize); err != nil {
			return err
		}
	}

	for _, name := range biasToSplit {
		t := tensors[name]
		if err := splitGateUpBias(tensors, name, t, intermediateSize); err != nil {
			return err
		}
	}

	return nil
}

// splitGateUpWeight splits a 2D merged gate+up weight tensor [2*intermediate, hidden]
// into separate gate_proj.weight and up_proj.weight tensors.
func splitGateUpWeight(
	tensors map[string]*tensor.TensorNumeric[float32],
	name string,
	t *tensor.TensorNumeric[float32],
	intermediateSize int,
) error {
	shape := t.Shape()
	cols := shape[1]
	data := t.Data()

	prefix := strings.Replace(name, "mlp.up_proj.weight", "mlp.", 1)

	// Gate: first half of rows [0, intermediateSize)
	gateData := make([]float32, intermediateSize*cols)
	copy(gateData, data[:intermediateSize*cols])
	gateT, err := tensor.New[float32]([]int{intermediateSize, cols}, gateData)
	if err != nil {
		return fmt.Errorf("splitMergedGateUp: create gate tensor: %w", err)
	}
	tensors[prefix+"gate_proj.weight"] = gateT

	// Up: second half of rows [intermediateSize, 2*intermediateSize)
	upData := make([]float32, intermediateSize*cols)
	copy(upData, data[intermediateSize*cols:])
	upT, err := tensor.New[float32]([]int{intermediateSize, cols}, upData)
	if err != nil {
		return fmt.Errorf("splitMergedGateUp: create up tensor: %w", err)
	}
	tensors[name] = upT

	return nil
}

// splitGateUpBias splits a 1D merged gate+up bias tensor [2*intermediate]
// into separate gate_proj.bias and up_proj.bias tensors.
func splitGateUpBias(
	tensors map[string]*tensor.TensorNumeric[float32],
	name string,
	t *tensor.TensorNumeric[float32],
	intermediateSize int,
) error {
	data := t.Data()

	prefix := strings.Replace(name, "mlp.up_proj.bias", "mlp.", 1)

	// Gate: first half [0, intermediateSize)
	gateData := make([]float32, intermediateSize)
	copy(gateData, data[:intermediateSize])
	gateT, err := tensor.New[float32]([]int{intermediateSize}, gateData)
	if err != nil {
		return fmt.Errorf("splitMergedGateUp: create gate bias: %w", err)
	}
	tensors[prefix+"gate_proj.bias"] = gateT

	// Up: second half [intermediateSize, 2*intermediateSize)
	upData := make([]float32, intermediateSize)
	copy(upData, data[intermediateSize:])
	upT, err := tensor.New[float32]([]int{intermediateSize}, upData)
	if err != nil {
		return fmt.Errorf("splitMergedGateUp: create up bias: %w", err)
	}
	tensors[name] = upT

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
