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

func TestSplitMergedGateUp(t *testing.T) {
	t.Run("split when up_proj has 2x intermediate_size", func(t *testing.T) {
		intermediateSize := 128
		hidden := 64

		// Create merged up_proj with shape [2*intermediateSize, hidden].
		totalRows := 2 * intermediateSize
		data := make([]float32, totalRows*hidden)
		for i := range data {
			data[i] = float32(i)
		}
		merged, err := tensor.New[float32]([]int{totalRows, hidden}, data)
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}

		tensors := map[string]*tensor.TensorNumeric[float32]{
			"model.layers.0.mlp.up_proj.weight": merged,
		}

		cfg := &ModelConfig{IntermediateSize: intermediateSize}

		if err := SplitMergedGateUp(tensors, cfg); err != nil {
			t.Fatalf("SplitMergedGateUp: %v", err)
		}

		// gate_proj should be created with first half of rows.
		gate, ok := tensors["model.layers.0.mlp.gate_proj.weight"]
		if !ok {
			t.Fatal("gate_proj.weight not found")
		}
		if gate.Shape()[0] != intermediateSize || gate.Shape()[1] != hidden {
			t.Errorf("gate shape = %v, want [%d, %d]", gate.Shape(), intermediateSize, hidden)
		}
		if gate.Data()[0] != 0 {
			t.Errorf("gate[0] = %v, want 0", gate.Data()[0])
		}

		// up_proj should be replaced with second half of rows.
		up, ok := tensors["model.layers.0.mlp.up_proj.weight"]
		if !ok {
			t.Fatal("up_proj.weight not found")
		}
		if up.Shape()[0] != intermediateSize || up.Shape()[1] != hidden {
			t.Errorf("up shape = %v, want [%d, %d]", up.Shape(), intermediateSize, hidden)
		}
		if up.Data()[0] != float32(intermediateSize*hidden) {
			t.Errorf("up[0] = %v, want %v", up.Data()[0], float32(intermediateSize*hidden))
		}
	})

	t.Run("no split when gate_proj already exists", func(t *testing.T) {
		intermediateSize := 128
		hidden := 64

		upData := make([]float32, 2*intermediateSize*hidden)
		for i := range upData {
			upData[i] = float32(i)
		}
		upMerged, _ := tensor.New[float32]([]int{2 * intermediateSize, hidden}, upData)

		gateData := make([]float32, intermediateSize*hidden)
		gateTensor, _ := tensor.New[float32]([]int{intermediateSize, hidden}, gateData)

		tensors := map[string]*tensor.TensorNumeric[float32]{
			"model.layers.0.mlp.up_proj.weight":   upMerged,
			"model.layers.0.mlp.gate_proj.weight": gateTensor,
		}

		cfg := &ModelConfig{IntermediateSize: intermediateSize}

		if err := SplitMergedGateUp(tensors, cfg); err != nil {
			t.Fatalf("SplitMergedGateUp: %v", err)
		}

		// up_proj should still have the original 2x size (not split).
		up := tensors["model.layers.0.mlp.up_proj.weight"]
		if up.Shape()[0] != 2*intermediateSize {
			t.Errorf("up_proj should not have been split, shape = %v", up.Shape())
		}
	})

	t.Run("no split when up_proj matches intermediate_size exactly", func(t *testing.T) {
		intermediateSize := 128
		hidden := 64

		upData := make([]float32, intermediateSize*hidden)
		upTensor, _ := tensor.New[float32]([]int{intermediateSize, hidden}, upData)

		tensors := map[string]*tensor.TensorNumeric[float32]{
			"model.layers.0.mlp.up_proj.weight": upTensor,
		}

		cfg := &ModelConfig{IntermediateSize: intermediateSize}

		if err := SplitMergedGateUp(tensors, cfg); err != nil {
			t.Fatalf("SplitMergedGateUp: %v", err)
		}

		// gate_proj should NOT have been created.
		if _, ok := tensors["model.layers.0.mlp.gate_proj.weight"]; ok {
			t.Error("gate_proj.weight should not exist when up_proj is not merged")
		}

		// up_proj should be unchanged.
		up := tensors["model.layers.0.mlp.up_proj.weight"]
		if up.Shape()[0] != intermediateSize {
			t.Errorf("up_proj shape changed unexpectedly: %v", up.Shape())
		}
	})

	t.Run("bias splitting", func(t *testing.T) {
		intermediateSize := 64
		hidden := 32

		// Create merged weight.
		wData := make([]float32, 2*intermediateSize*hidden)
		for i := range wData {
			wData[i] = float32(i)
		}
		wMerged, _ := tensor.New[float32]([]int{2 * intermediateSize, hidden}, wData)

		// Create merged bias (1D).
		bData := make([]float32, 2*intermediateSize)
		for i := range bData {
			bData[i] = float32(i) * 0.1
		}
		bMerged, _ := tensor.New[float32]([]int{2 * intermediateSize}, bData)

		tensors := map[string]*tensor.TensorNumeric[float32]{
			"model.layers.0.mlp.up_proj.weight": wMerged,
			"model.layers.0.mlp.up_proj.bias":   bMerged,
		}

		cfg := &ModelConfig{IntermediateSize: intermediateSize}

		if err := SplitMergedGateUp(tensors, cfg); err != nil {
			t.Fatalf("SplitMergedGateUp: %v", err)
		}

		// Check gate bias.
		gateBias, ok := tensors["model.layers.0.mlp.gate_proj.bias"]
		if !ok {
			t.Fatal("gate_proj.bias not found")
		}
		if gateBias.Shape()[0] != intermediateSize {
			t.Errorf("gate bias shape = %v, want [%d]", gateBias.Shape(), intermediateSize)
		}
		if gateBias.Data()[0] != 0 {
			t.Errorf("gate bias[0] = %v, want 0", gateBias.Data()[0])
		}

		// Check up bias.
		upBias, ok := tensors["model.layers.0.mlp.up_proj.bias"]
		if !ok {
			t.Fatal("up_proj.bias not found")
		}
		if upBias.Shape()[0] != intermediateSize {
			t.Errorf("up bias shape = %v, want [%d]", upBias.Shape(), intermediateSize)
		}
		expectedUpBias0 := float32(intermediateSize) * 0.1
		if upBias.Data()[0] != expectedUpBias0 {
			t.Errorf("up bias[0] = %v, want %v", upBias.Data()[0], expectedUpBias0)
		}

		// Check weight tensors were also split.
		gate, ok := tensors["model.layers.0.mlp.gate_proj.weight"]
		if !ok {
			t.Fatal("gate_proj.weight not found")
		}
		if gate.Shape()[0] != intermediateSize || gate.Shape()[1] != hidden {
			t.Errorf("gate weight shape = %v, want [%d, %d]", gate.Shape(), intermediateSize, hidden)
		}
	})
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
