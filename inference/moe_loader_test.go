package inference

import (
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

func TestSplitMoEWeights_NoExperts(t *testing.T) {
	cfg := &gguf.ModelConfig{NumExperts: 0}
	dm, gpu, cpu, err := SplitMoEWeights(nil, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dm != nil || gpu != nil || cpu != nil {
		t.Error("expected nil results for non-MoE model")
	}
}

func TestSplitMoEWeights_SharedOnGPU_RoutedOnCPU(t *testing.T) {
	// Simulate a model with 4 experts, 1 shared.
	cfg := &gguf.ModelConfig{
		NumExperts:       4,
		NumSharedExperts: 1,
		HiddenSize:       4,
		IntermediateSize: 8,
	}

	// Create dummy tensors.
	dummy := func() *tensor.TensorNumeric[float32] {
		t, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		return t
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		// Shared expert tensors -> GPU
		"blk.0.ffn_shared_expert_gate.weight": dummy(),
		"blk.0.ffn_shared_expert_up.weight":   dummy(),
		"blk.0.ffn_shared_expert_down.weight": dummy(),
		// Stacked expert tensors (contain all routed experts) -> CPU
		"blk.0.ffn_gate_exps.weight": dummy(),
		"blk.0.ffn_up_exps.weight":   dummy(),
		"blk.0.ffn_down_exps.weight": dummy(),
		// Router weight -> CPU (it's a stacked/routing tensor)
		"blk.0.ffn_gate_inp.weight": dummy(),
		// Non-expert tensor -> not in either subset
		"model.embed_tokens.weight": dummy(),
	}

	dm, gpuTensors, cpuTensors, err := SplitMoEWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify device map.
	if dm == nil {
		t.Fatal("device map is nil")
	}
	if len(dm.Experts) != 4 {
		t.Errorf("expected 4 experts in device map, got %d", len(dm.Experts))
	}

	// Expert 0 is shared -> GPU.
	if dm.DeviceForExpert(0) != GPU {
		t.Errorf("shared expert 0 should be on GPU, got %v", dm.DeviceForExpert(0))
	}

	// Experts 1-3 are routed -> CPU.
	for i := 1; i < 4; i++ {
		if dm.DeviceForExpert(i) != CPU {
			t.Errorf("routed expert %d should be on CPU, got %v", i, dm.DeviceForExpert(i))
		}
	}

	// Shared expert tensors should be in GPU set.
	for _, name := range []string{
		"blk.0.ffn_shared_expert_gate.weight",
		"blk.0.ffn_shared_expert_up.weight",
		"blk.0.ffn_shared_expert_down.weight",
	} {
		if _, ok := gpuTensors[name]; !ok {
			t.Errorf("shared expert tensor %q should be in GPU tensors", name)
		}
	}

	// Stacked and router tensors should be in CPU set.
	for _, name := range []string{
		"blk.0.ffn_gate_exps.weight",
		"blk.0.ffn_up_exps.weight",
		"blk.0.ffn_down_exps.weight",
		"blk.0.ffn_gate_inp.weight",
	} {
		if _, ok := cpuTensors[name]; !ok {
			t.Errorf("stacked expert tensor %q should be in CPU tensors", name)
		}
	}

	// Non-expert tensor should not be in either subset.
	if _, ok := gpuTensors["model.embed_tokens.weight"]; ok {
		t.Error("non-expert tensor should not be in GPU tensors")
	}
	if _, ok := cpuTensors["model.embed_tokens.weight"]; ok {
		t.Error("non-expert tensor should not be in CPU tensors")
	}
}

func TestSplitMoEWeights_NoSharedExperts(t *testing.T) {
	cfg := &gguf.ModelConfig{
		NumExperts:       8,
		NumSharedExperts: 0,
		HiddenSize:       4,
	}

	dummy := func() *tensor.TensorNumeric[float32] {
		t, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		return t
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"blk.0.ffn_gate_exps.weight": dummy(),
		"blk.0.ffn_up_exps.weight":   dummy(),
		"blk.0.ffn_down_exps.weight": dummy(),
	}

	dm, _, cpuTensors, err := SplitMoEWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// All experts should be on CPU when there are no shared experts.
	for i := 0; i < 8; i++ {
		if dm.DeviceForExpert(i) != CPU {
			t.Errorf("expert %d should be on CPU when no shared experts, got %v",
				i, dm.DeviceForExpert(i))
		}
	}

	// All stacked tensors on CPU.
	if len(cpuTensors) != 3 {
		t.Errorf("expected 3 CPU tensors, got %d", len(cpuTensors))
	}
}

func TestSplitMoEWeights_PerExpertTensors(t *testing.T) {
	cfg := &gguf.ModelConfig{
		NumExperts:       3,
		NumSharedExperts: 1,
	}

	dummy := func() *tensor.TensorNumeric[float32] {
		t, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		return t
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		// Per-expert GGUF naming: expert 0 is shared -> GPU
		"blk.0.ffn_gate.0.weight": dummy(),
		"blk.0.ffn_up.0.weight":   dummy(),
		"blk.0.ffn_down.0.weight": dummy(),
		// Expert 1 is routed -> CPU
		"blk.0.ffn_gate.1.weight": dummy(),
		"blk.0.ffn_up.1.weight":   dummy(),
		"blk.0.ffn_down.1.weight": dummy(),
		// Expert 2 is routed -> CPU
		"blk.0.ffn_gate.2.weight": dummy(),
	}

	dm, gpuTensors, cpuTensors, err := SplitMoEWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if dm.DeviceForExpert(0) != GPU {
		t.Error("shared expert 0 should be on GPU")
	}
	if dm.DeviceForExpert(1) != CPU {
		t.Error("routed expert 1 should be on CPU")
	}
	if dm.DeviceForExpert(2) != CPU {
		t.Error("routed expert 2 should be on CPU")
	}

	// Expert 0 tensors on GPU.
	for _, name := range []string{
		"blk.0.ffn_gate.0.weight",
		"blk.0.ffn_up.0.weight",
		"blk.0.ffn_down.0.weight",
	} {
		if _, ok := gpuTensors[name]; !ok {
			t.Errorf("shared expert tensor %q should be in GPU set", name)
		}
	}

	// Expert 1 and 2 tensors on CPU.
	for _, name := range []string{
		"blk.0.ffn_gate.1.weight",
		"blk.0.ffn_up.1.weight",
		"blk.0.ffn_down.1.weight",
		"blk.0.ffn_gate.2.weight",
	} {
		if _, ok := cpuTensors[name]; !ok {
			t.Errorf("routed expert tensor %q should be in CPU set", name)
		}
	}
}

func TestSplitMoEWeights_HFStyleTensors(t *testing.T) {
	cfg := &gguf.ModelConfig{
		NumExperts:       3,
		NumSharedExperts: 1,
	}

	dummy := func() *tensor.TensorNumeric[float32] {
		t, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
		return t
	}

	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.layers.0.block_sparse_moe.experts.0.w1.weight": dummy(), // shared -> GPU
		"model.layers.0.block_sparse_moe.experts.1.w1.weight": dummy(), // routed -> CPU
		"model.layers.0.block_sparse_moe.experts.2.w1.weight": dummy(), // routed -> CPU
	}

	dm, gpuTensors, cpuTensors, err := SplitMoEWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if dm.DeviceForExpert(0) != GPU {
		t.Error("shared expert 0 should be on GPU")
	}

	if _, ok := gpuTensors["model.layers.0.block_sparse_moe.experts.0.w1.weight"]; !ok {
		t.Error("HF expert 0 tensor should be in GPU set")
	}
	if _, ok := cpuTensors["model.layers.0.block_sparse_moe.experts.1.w1.weight"]; !ok {
		t.Error("HF expert 1 tensor should be in CPU set")
	}
	if _, ok := cpuTensors["model.layers.0.block_sparse_moe.experts.2.w1.weight"]; !ok {
		t.Error("HF expert 2 tensor should be in CPU set")
	}
}

func TestMoEDeviceMap_Methods(t *testing.T) {
	dm := &MoEDeviceMap{
		Experts: map[int]DeviceType{
			0: GPU,
			1: GPU,
			2: CPU,
			3: CPU,
			4: CPU,
		},
		SharedExperts: []int{0, 1},
		RoutedExperts: []int{2, 3, 4},
	}

	// DeviceForExpert.
	if dm.DeviceForExpert(0) != GPU {
		t.Error("expert 0 should be GPU")
	}
	if dm.DeviceForExpert(4) != CPU {
		t.Error("expert 4 should be CPU")
	}
	if dm.DeviceForExpert(99) != CPU {
		t.Error("unknown expert should default to CPU")
	}

	// GPUExperts.
	gpuIDs := dm.GPUExperts()
	if len(gpuIDs) != 2 {
		t.Errorf("expected 2 GPU experts, got %d", len(gpuIDs))
	}

	// CPUExperts.
	cpuIDs := dm.CPUExperts()
	if len(cpuIDs) != 3 {
		t.Errorf("expected 3 CPU experts, got %d", len(cpuIDs))
	}
}

func TestMoEDeviceMap_Nil(t *testing.T) {
	var dm *MoEDeviceMap
	if dm.DeviceForExpert(0) != CPU {
		t.Error("nil device map should default to CPU")
	}
	if dm.GPUExperts() != nil {
		t.Error("nil device map GPUExperts should return nil")
	}
	if dm.CPUExperts() != nil {
		t.Error("nil device map CPUExperts should return nil")
	}
}

func TestParseExpertTensorName(t *testing.T) {
	tests := []struct {
		name       string
		wantID     int
		wantExpert bool
	}{
		// Per-expert GGUF format.
		{"blk.0.ffn_gate.0.weight", 0, true},
		{"blk.0.ffn_up.3.weight", 3, true},
		{"blk.5.ffn_down.7.weight", 7, true},
		// Stacked expert tensors.
		{"blk.0.ffn_gate_exps.weight", -1, true},
		{"blk.0.ffn_up_exps.weight", -1, true},
		{"blk.0.ffn_down_exps.weight", -1, true},
		{"blk.0.ffn_gate_inp.weight", -1, true},
		// Shared expert tensors.
		{"blk.0.ffn_shared_expert_gate.weight", -1, true},
		{"blk.0.ffn_shared_expert_up.weight", -1, true},
		{"blk.0.ffn_shared_expert_down.weight", -1, true},
		// HF-style per-expert.
		{"model.layers.0.block_sparse_moe.experts.0.w1.weight", 0, true},
		{"model.layers.2.block_sparse_moe.experts.5.w2.weight", 5, true},
		// Non-expert tensors.
		{"model.embed_tokens.weight", -1, false},
		{"model.norm.weight", -1, false},
		{"lm_head.weight", -1, false},
		{"blk.0.attn_q.weight", -1, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotID, gotExpert := parseExpertTensorName(tt.name)
			if gotExpert != tt.wantExpert {
				t.Errorf("parseExpertTensorName(%q) isExpert = %v, want %v", tt.name, gotExpert, tt.wantExpert)
			}
			if gotID != tt.wantID {
				t.Errorf("parseExpertTensorName(%q) expertID = %d, want %d", tt.name, gotID, tt.wantID)
			}
		})
	}
}

func TestIsSharedExpertTensor(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"blk.0.ffn_shared_expert_gate.weight", true},
		{"blk.0.ffn_shared_expert_up.weight", true},
		{"blk.0.ffn_shared_expert_down.weight", true},
		{"blk.0.ffn_gate_exps.weight", false},
		{"model.embed_tokens.weight", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isSharedExpertTensor(tt.name); got != tt.want {
				t.Errorf("isSharedExpertTensor(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
