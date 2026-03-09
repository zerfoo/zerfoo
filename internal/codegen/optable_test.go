package codegen

import (
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
)

func TestEmitterRegistered(t *testing.T) {
	// All Gemma 3 ops should have emitters.
	type opSpec struct {
		name      string
		numInputs int
	}
	gemmaOps := []opSpec{
		{"Add", 2}, {"Sub", 2}, {"Mul", 2}, {"Div", 2}, {"Pow", 2},
		{"Exp", 1}, {"Log", 1}, {"Sqrt", 1}, {"Rsqrt", 1}, {"Tanh", 1},
		{"Neg", 1}, {"Abs", 1}, {"Silu", 1},
		{"AddScalar", 1}, {"MulScalar", 1}, {"SubScalar", 1}, {"DivScalar", 1}, {"PowScalar", 1},
		{"RMSNorm", 2}, {"Softmax", 1},
		{"ReduceSum", 1}, {"ReduceMean", 1},
		{"Slice", 1}, {"Repeat", 1},
		{"MatMul", 2}, {"MatMulNBits", 2},
		{"Gather", 2}, {"Concat", 1}, {"Reshape", 1}, {"Transpose", 1},
		{"KVCacheAppendK", 2}, {"KVCacheAppendV", 2},
		{"KVCacheGetK", 1}, {"KVCacheGetV", 1},
		{"KVCacheSeqLen", 0},
	}
	for _, op := range gemmaOps {
		inputIdx := make([]int, op.numInputs)
		for i := range inputIdx {
			inputIdx[i] = i
		}
		meta := graph.InstructionMeta{OpName: op.name, InputIdx: inputIdx, OutputIdx: 10}
		slots := make([]SlotInfo, op.numInputs)
		for i := range slots {
			slots[i] = SlotInfo{Shape: []int{1, 2048}}
		}
		_, err := Emit(meta, slots)
		if err != nil {
			t.Errorf("op %q: %v", op.name, err)
		}
	}
}

func TestEmitterUnsupportedOp(t *testing.T) {
	meta := graph.InstructionMeta{OpName: "UnknownFancyOp"}
	info := SlotInfo{Shape: []int{1, 4}}
	_, err := Emit(meta, []SlotInfo{info})
	if err == nil {
		t.Error("expected error for unsupported op, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("error should contain 'unsupported': %v", err)
	}
}

func TestEmitterOutputFormat(t *testing.T) {
	tests := []struct {
		op      string
		inputs  int
		wantSub string // substring that should appear in emitted code
	}{
		{"Add", 2, "slot_"},
		{"Exp", 1, "expf"},
		{"MulScalar", 1, "slot_"},
		{"RMSNorm", 2, "dev_rmsnorm"},
		{"Softmax", 1, "dev_softmax"},
		{"ReduceSum", 1, "dev_reduce_sum"},
		{"ReduceMean", 1, "dev_reduce_mean"},
		{"Slice", 1, "dev_slice"},
		{"Repeat", 1, "dev_repeat"},
		{"MatMul", 2, "dev_gemv"},
		{"Gather", 2, "dev_gather"},
		{"KVCacheAppendK", 2, "dev_kv_append"},
		{"KVCacheAppendV", 2, "dev_kv_append"},
		{"KVCacheGetK", 1, "kv_k["},
		{"KVCacheGetV", 1, "kv_v["},
		{"KVCacheSeqLen", 0, "kv_seq_len"},
	}
	for _, tc := range tests {
		meta := graph.InstructionMeta{
			OpName:    tc.op,
			InputIdx:  make([]int, tc.inputs),
			OutputIdx: 10,
		}
		slots := make([]SlotInfo, tc.inputs)
		for i := range slots {
			slots[i] = SlotInfo{Shape: []int{1, 2048}}
		}
		code, err := Emit(meta, slots)
		if err != nil {
			t.Errorf("op %q: %v", tc.op, err)
			continue
		}
		if !strings.Contains(code, tc.wantSub) {
			t.Errorf("op %q: output %q missing %q", tc.op, code, tc.wantSub)
		}
	}
}

func TestKVCacheAppendEmitters(t *testing.T) {
	tests := []struct {
		name    string
		op      string
		layer   int
		wantArr string
	}{
		{"append_k_layer0", "KVCacheAppendK", 0, "kv_k[0]"},
		{"append_k_layer5", "KVCacheAppendK", 5, "kv_k[5]"},
		{"append_v_layer0", "KVCacheAppendV", 0, "kv_v[0]"},
		{"append_v_layer3", "KVCacheAppendV", 3, "kv_v[3]"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			meta := graph.InstructionMeta{
				OpName:    tc.op,
				InputIdx:  []int{2, tc.layer}, // slot 2 is data, InputIdx[1] = layer
				OutputIdx: 5,
			}
			inputs := []SlotInfo{{Shape: []int{8, 128}}} // head_dim = 128
			code, err := Emit(meta, inputs)
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}
			if !strings.Contains(code, tc.wantArr) {
				t.Errorf("want %q in %q", tc.wantArr, code)
			}
			if !strings.Contains(code, "dev_kv_append") {
				t.Errorf("want dev_kv_append in %q", code)
			}
			if !strings.Contains(code, "seq_pos") {
				t.Errorf("want seq_pos in %q", code)
			}
			if !strings.Contains(code, "128") {
				t.Errorf("want head_dim 128 in %q", code)
			}
		})
	}
}

func TestKVCacheGetEmitters(t *testing.T) {
	tests := []struct {
		name    string
		op      string
		layer   int
		wantArr string
	}{
		{"get_k_layer0", "KVCacheGetK", 0, "kv_k[0]"},
		{"get_k_layer7", "KVCacheGetK", 7, "kv_k[7]"},
		{"get_v_layer0", "KVCacheGetV", 0, "kv_v[0]"},
		{"get_v_layer2", "KVCacheGetV", 2, "kv_v[2]"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			meta := graph.InstructionMeta{
				OpName:    tc.op,
				InputIdx:  []int{tc.layer},
				OutputIdx: 8,
			}
			code, err := Emit(meta, nil)
			if err != nil {
				t.Fatalf("Emit: %v", err)
			}
			if !strings.Contains(code, tc.wantArr) {
				t.Errorf("want %q in %q", tc.wantArr, code)
			}
			if !strings.Contains(code, "float* slot_8") {
				t.Errorf("want pointer alias slot_8 in %q", code)
			}
		})
	}
}

func TestKVCacheSeqLenEmitter(t *testing.T) {
	meta := graph.InstructionMeta{
		OpName:    "KVCacheSeqLen",
		InputIdx:  nil,
		OutputIdx: 3,
	}
	code, err := Emit(meta, nil)
	if err != nil {
		t.Fatalf("Emit: %v", err)
	}
	if !strings.Contains(code, "kv_seq_len") {
		t.Errorf("want kv_seq_len in %q", code)
	}
	if !strings.Contains(code, "seq_len_3") {
		t.Errorf("want seq_len_3 in %q", code)
	}
}

func TestKVCacheAppendInsufficientInputs(t *testing.T) {
	meta := graph.InstructionMeta{
		OpName:    "KVCacheAppendK",
		InputIdx:  []int{0}, // only 1 input, need 2
		OutputIdx: 1,
	}
	_, err := Emit(meta, nil)
	if err == nil {
		t.Fatal("expected error for insufficient inputs")
	}
}

func TestKVCacheGetInsufficientInputs(t *testing.T) {
	meta := graph.InstructionMeta{
		OpName:    "KVCacheGetK",
		InputIdx:  nil, // no inputs
		OutputIdx: 1,
	}
	_, err := Emit(meta, nil)
	if err == nil {
		t.Fatal("expected error for insufficient inputs")
	}
}
