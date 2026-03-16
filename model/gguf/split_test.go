package gguf

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestSplitMergedQKV_MHA(t *testing.T) {
	// MHA: 32 heads, 32 KV heads, head_dim=96, hidden=3072.
	numHeads := 32
	numKVHeads := 32
	headDim := 96
	hidden := numHeads * headDim // 3072

	qRows := numHeads * headDim    // 3072
	kRows := numKVHeads * headDim  // 3072
	vRows := numKVHeads * headDim  // 3072
	totalRows := qRows + kRows + vRows // 9216

	// Create merged QKV tensor [totalRows, hidden] with distinct values.
	data := make([]float32, totalRows*hidden)
	for i := range data {
		data[i] = float32(i)
	}
	merged, err := tensor.New[float32]([]int{totalRows, hidden}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.qkv_proj.weight": merged,
	}

	cfg := &ModelConfig{
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		HiddenSize: hidden,
	}

	if err := SplitMergedQKV(tensors, cfg); err != nil {
		t.Fatalf("SplitMergedQKV: %v", err)
	}

	// Original should be removed.
	if _, ok := tensors["model.layers.0.self_attn.qkv_proj.weight"]; ok {
		t.Error("original qkv_proj.weight not removed")
	}

	// Check Q.
	q, ok := tensors["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatal("q_proj.weight not found")
	}
	if q.Shape()[0] != qRows || q.Shape()[1] != hidden {
		t.Errorf("Q shape = %v, want [%d, %d]", q.Shape(), qRows, hidden)
	}
	if q.Data()[0] != 0 {
		t.Errorf("Q[0] = %v, want 0", q.Data()[0])
	}

	// Check K.
	k, ok := tensors["model.layers.0.self_attn.k_proj.weight"]
	if !ok {
		t.Fatal("k_proj.weight not found")
	}
	if k.Shape()[0] != kRows || k.Shape()[1] != hidden {
		t.Errorf("K shape = %v, want [%d, %d]", k.Shape(), kRows, hidden)
	}
	if k.Data()[0] != float32(qRows*hidden) {
		t.Errorf("K[0] = %v, want %v", k.Data()[0], float32(qRows*hidden))
	}

	// Check V.
	v, ok := tensors["model.layers.0.self_attn.v_proj.weight"]
	if !ok {
		t.Fatal("v_proj.weight not found")
	}
	if v.Shape()[0] != vRows || v.Shape()[1] != hidden {
		t.Errorf("V shape = %v, want [%d, %d]", v.Shape(), vRows, hidden)
	}
	if v.Data()[0] != float32((qRows+kRows)*hidden) {
		t.Errorf("V[0] = %v, want %v", v.Data()[0], float32((qRows+kRows)*hidden))
	}
}

func TestSplitMergedQKV_GQA(t *testing.T) {
	// GQA: 32 heads, 8 KV heads, head_dim=96, hidden=3072.
	numHeads := 32
	numKVHeads := 8
	headDim := 96
	hidden := numHeads * headDim // 3072

	qRows := numHeads * headDim   // 3072
	kRows := numKVHeads * headDim // 768
	vRows := numKVHeads * headDim // 768
	totalRows := qRows + kRows + vRows // 4608

	data := make([]float32, totalRows*hidden)
	for i := range data {
		data[i] = float32(i)
	}
	merged, err := tensor.New[float32]([]int{totalRows, hidden}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.qkv_proj.weight": merged,
	}

	cfg := &ModelConfig{
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		HiddenSize: hidden,
	}

	if err := SplitMergedQKV(tensors, cfg); err != nil {
		t.Fatalf("SplitMergedQKV: %v", err)
	}

	// Original should be removed.
	if _, ok := tensors["model.layers.0.self_attn.qkv_proj.weight"]; ok {
		t.Error("original qkv_proj.weight not removed")
	}

	q := tensors["model.layers.0.self_attn.q_proj.weight"]
	if q.Shape()[0] != qRows || q.Shape()[1] != hidden {
		t.Errorf("Q shape = %v, want [%d, %d]", q.Shape(), qRows, hidden)
	}

	k := tensors["model.layers.0.self_attn.k_proj.weight"]
	if k.Shape()[0] != kRows || k.Shape()[1] != hidden {
		t.Errorf("K shape = %v, want [%d, %d]", k.Shape(), kRows, hidden)
	}

	v := tensors["model.layers.0.self_attn.v_proj.weight"]
	if v.Shape()[0] != vRows || v.Shape()[1] != hidden {
		t.Errorf("V shape = %v, want [%d, %d]", v.Shape(), vRows, hidden)
	}
}

func TestSplitMergedQKV_MissingConfig(t *testing.T) {
	tensors := map[string]*tensor.TensorNumeric[float32]{}

	// NumHeads = 0 should error.
	err := SplitMergedQKV(tensors, &ModelConfig{NumHeads: 0, HiddenSize: 128})
	if err == nil {
		t.Error("expected error for NumHeads=0")
	}

	// HiddenSize = 0 should error.
	err = SplitMergedQKV(tensors, &ModelConfig{NumHeads: 4, HiddenSize: 0})
	if err == nil {
		t.Error("expected error for HiddenSize=0")
	}
}

func TestSplitMergedQKV_Bias(t *testing.T) {
	numHeads := 4
	numKVHeads := 4
	headDim := 8
	hidden := numHeads * headDim // 32

	qRows := numHeads * headDim
	kRows := numKVHeads * headDim
	vRows := numKVHeads * headDim
	totalRows := qRows + kRows + vRows

	// Create merged weight tensor.
	wData := make([]float32, totalRows*hidden)
	for i := range wData {
		wData[i] = float32(i)
	}
	wMerged, _ := tensor.New[float32]([]int{totalRows, hidden}, wData)

	// Create merged bias tensor (1D).
	bData := make([]float32, totalRows)
	for i := range bData {
		bData[i] = float32(i) * 0.1
	}
	bMerged, _ := tensor.New[float32]([]int{totalRows}, bData)

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.qkv_proj.weight": wMerged,
		"model.layers.0.self_attn.qkv_proj.bias":   bMerged,
	}

	cfg := &ModelConfig{
		NumHeads:   numHeads,
		NumKVHeads: numKVHeads,
		HiddenSize: hidden,
	}

	if err := SplitMergedQKV(tensors, cfg); err != nil {
		t.Fatalf("SplitMergedQKV: %v", err)
	}

	// Check bias tensors exist with correct shapes.
	qBias, ok := tensors["model.layers.0.self_attn.q_proj.bias"]
	if !ok {
		t.Fatal("q_proj.bias not found")
	}
	if qBias.Shape()[0] != qRows {
		t.Errorf("Q bias shape = %v, want [%d]", qBias.Shape(), qRows)
	}

	kBias, ok := tensors["model.layers.0.self_attn.k_proj.bias"]
	if !ok {
		t.Fatal("k_proj.bias not found")
	}
	if kBias.Shape()[0] != kRows {
		t.Errorf("K bias shape = %v, want [%d]", kBias.Shape(), kRows)
	}

	vBias, ok := tensors["model.layers.0.self_attn.v_proj.bias"]
	if !ok {
		t.Fatal("v_proj.bias not found")
	}
	if vBias.Shape()[0] != vRows {
		t.Errorf("V bias shape = %v, want [%d]", vBias.Shape(), vRows)
	}

	// Original bias should be removed.
	if _, ok := tensors["model.layers.0.self_attn.qkv_proj.bias"]; ok {
		t.Error("original qkv_proj.bias not removed")
	}
}

func TestSplitMergedQKV_NoQKV(t *testing.T) {
	// No qkv_proj tensors = no-op, no error.
	someTensor, _ := tensor.New[float32]([]int{4, 4}, make([]float32, 16))
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.self_attn.q_proj.weight": someTensor,
	}

	cfg := &ModelConfig{
		NumHeads:   4,
		NumKVHeads: 4,
		HiddenSize: 16,
	}

	if err := SplitMergedQKV(tensors, cfg); err != nil {
		t.Fatalf("SplitMergedQKV: %v", err)
	}

	// Original tensor should be untouched.
	if _, ok := tensors["model.layers.0.self_attn.q_proj.weight"]; !ok {
		t.Error("existing tensor was incorrectly removed")
	}
}
