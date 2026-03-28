package integration

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// TestBitNet_GGUFLoadPreservesTernaryStorage verifies that the GGUF loader
// creates TernaryStorage for TQ2_0 tensors so that downstream MatMul layers
// can dispatch to the ternary GEMV kernel.
func TestBitNet_GGUFLoadPreservesTernaryStorage(t *testing.T) {
	values := []int8{1, 0, -1, 1, -1, 1, 0, 0}
	ts := tensor.NewTernaryStorageFrom(values)
	raw := ts.RawBytes()

	tensors := []gguf.TensorInfo{{
		Name:       "model.layers.0.self_attn.q_proj.weight",
		Dimensions: []uint64{4, 2}, // GGUF order: cols=4, rows=2
		Type:       gguf.GGMLTypeTQ2_0,
		Offset:     0,
	}}

	r := buildTestGGUF(t, tensors, raw)
	f, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	loaded, err := gguf.LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	tns := loaded["model.layers.0.self_attn.q_proj.weight"]
	if tns == nil {
		t.Fatal("ternary tensor not found in loaded map")
	}

	if _, ok := tns.GetStorage().(*tensor.TernaryStorage); !ok {
		t.Fatalf("expected TernaryStorage, got %T — ternary GEMV dispatch will fail", tns.GetStorage())
	}

	if tns.Shape()[0] != 2 || tns.Shape()[1] != 4 {
		t.Fatalf("shape = %v, want [2 4]", tns.Shape())
	}
}

// TestBitNet_GGUFTernaryThroughMatMul verifies the end-to-end path:
// GGUF load -> TernaryStorage preserved -> MatMul dispatches to ternary GEMV.
func TestBitNet_GGUFTernaryThroughMatMul(t *testing.T) {
	values := []int8{1, 0, -1, 1, -1, 1, 0, 0}
	ts := tensor.NewTernaryStorageFrom(values)
	raw := ts.RawBytes()

	tensors := []gguf.TensorInfo{{
		Name:       "w",
		Dimensions: []uint64{4, 2},
		Type:       gguf.GGMLTypeTQ2_0,
		Offset:     0,
	}}

	r := buildTestGGUF(t, tensors, raw)
	f, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	loaded, err := gguf.LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	w := loaded["w"]

	a, err := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	layer := core.NewMatMul[float32](engine)

	result, err := layer.Forward(context.Background(), a, w)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// C = A * W^T where W = [[1,0,-1,1],[-1,1,0,0]]
	// row0: 1*1 + 2*0 + 3*(-1) + 4*1 = 2
	// row1: 1*(-1) + 2*1 + 3*0 + 4*0 = 1
	want := []float32{2, 1}
	got := result.Data()
	if len(got) != len(want) {
		t.Fatalf("output length: got %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if math.Abs(float64(got[i]-w)) > 1e-6 {
			t.Errorf("output[%d]: got %f, want %f", i, got[i], w)
		}
	}
}

// TestBitNet_F32WeightsUseStandardPath confirms that non-ternary GGUF tensors
// use the standard MatMul path without hitting the ternary dispatcher.
func TestBitNet_F32WeightsUseStandardPath(t *testing.T) {
	f32Data := []float32{1, 0, 0, 0, 1, 0}
	rawBytes := make([]byte, len(f32Data)*4)
	for i, v := range f32Data {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(rawBytes[i*4:], bits)
	}

	tensors := []gguf.TensorInfo{{
		Name:       "w_f32",
		Dimensions: []uint64{3, 2}, // shape [2, 3]
		Type:       gguf.GGMLTypeF32,
		Offset:     0,
	}}

	r := buildTestGGUF(t, tensors, rawBytes)
	f, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	loaded, err := gguf.LoadTensors(f, r)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	w := loaded["w_f32"]
	if _, ok := w.GetStorage().(*tensor.TernaryStorage); ok {
		t.Fatal("F32 tensor should not have TernaryStorage")
	}

	a, err := tensor.New[float32]([]int{1, 3}, []float32{5, 6, 7})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	layer := core.NewMatMul[float32](engine)

	result, err := layer.Forward(context.Background(), a, w)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// W = [[1,0,0],[0,1,0]], A = [5,6,7], C = A * W^T
	// row0 = 5, row1 = 6
	want := []float32{5, 6}
	got := result.Data()
	for i, w := range want {
		if math.Abs(float64(got[i]-w)) > 1e-6 {
			t.Errorf("output[%d]: got %f, want %f", i, got[i], w)
		}
	}
}

// buildTestGGUF creates a minimal GGUF v3 file in memory.
func buildTestGGUF(t *testing.T, tensors []gguf.TensorInfo, tensorData []byte) *bytes.Reader {
	t.Helper()
	var buf bytes.Buffer

	bw := func(data any) {
		if err := binary.Write(&buf, binary.LittleEndian, data); err != nil {
			t.Fatalf("binary.Write: %v", err)
		}
	}
	writeStr := func(s string) {
		bw(uint64(len(s)))
		buf.WriteString(s)
	}

	bw(gguf.Magic)
	bw(uint32(3))             // version
	bw(uint64(len(tensors)))  // tensor count
	bw(uint64(0))             // metadata count

	for _, ti := range tensors {
		writeStr(ti.Name)
		bw(uint32(len(ti.Dimensions)))
		for _, d := range ti.Dimensions {
			bw(d)
		}
		bw(uint32(ti.Type))
		bw(ti.Offset)
	}

	// Pad to 32-byte alignment.
	pos := buf.Len()
	const alignment = 32
	padded := (pos + alignment - 1) / alignment * alignment
	for range padded - pos {
		buf.WriteByte(0)
	}

	buf.Write(tensorData)
	return bytes.NewReader(buf.Bytes())
}
