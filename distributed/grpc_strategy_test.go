package distributed

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/distributed/pb"
	"github.com/zerfoo/zerfoo/tensor"
)

// --- Tensor conversion tests ---

func TestTensorToProto_Float32(t *testing.T) {
	tn, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}

	p := tensorToProto(tn)
	if p == nil {
		t.Fatal("expected non-nil proto")
	}
	if len(p.Shape) != 2 || p.Shape[0] != 2 || p.Shape[1] != 3 {
		t.Errorf("shape = %v, want [2, 3]", p.Shape)
	}
	want := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range p.Data {
		if v != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestTensorToProto_Nil(t *testing.T) {
	p := tensorToProto[float32](nil)
	if p != nil {
		t.Error("expected nil proto for nil tensor")
	}
}

func TestProtoToTensor_Float32(t *testing.T) {
	p := &pb.Tensor{Shape: []int32{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
	tn, err := protoToTensor[float32](p)
	if err != nil {
		t.Fatal(err)
	}
	shape := tn.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 3 {
		t.Errorf("shape = %v, want [2, 3]", shape)
	}
	data := tn.Data()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i, v := range data {
		if v != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestProtoToTensor_Nil(t *testing.T) {
	_, err := protoToTensor[float32](nil)
	if err == nil {
		t.Error("expected error for nil proto")
	}
}

func TestTensorConversion_RoundTrip(t *testing.T) {
	original, err := tensor.New[float32]([]int{3}, []float32{1.5, 2.5, 3.5})
	if err != nil {
		t.Fatal(err)
	}

	p := tensorToProto(original)
	restored, err := protoToTensor[float32](p)
	if err != nil {
		t.Fatal(err)
	}

	origData := original.Data()
	restoredData := restored.Data()
	if len(origData) != len(restoredData) {
		t.Fatalf("data length mismatch: %d vs %d", len(origData), len(restoredData))
	}
	for i := range origData {
		if origData[i] != restoredData[i] {
			t.Errorf("data[%d] = %f, want %f", i, restoredData[i], origData[i])
		}
	}
}

func TestUpdateTensorFromProto(t *testing.T) {
	tn, err := tensor.New[float32]([]int{3}, []float32{0, 0, 0})
	if err != nil {
		t.Fatal(err)
	}
	p := &pb.Tensor{Shape: []int32{3}, Data: []float32{10, 20, 30}}
	updateTensorFromProto(tn, p)

	data := tn.Data()
	want := []float32{10, 20, 30}
	for i, v := range data {
		if v != want[i] {
			t.Errorf("data[%d] = %f, want %f", i, v, want[i])
		}
	}
}

func TestUpdateTensorFromProto_NilSafe(t *testing.T) {
	// Should not panic.
	updateTensorFromProto[float32](nil, &pb.Tensor{Shape: []int32{1}, Data: []float32{1}})
	tn, _ := tensor.New[float32]([]int{1}, []float32{0})
	updateTensorFromProto(tn, nil)
}

// --- NewGrpcStrategy tests ---

func TestNewGrpcStrategy_Defaults(t *testing.T) {
	s := NewGrpcStrategy[float32](GrpcStrategyConfig{
		WorkerAddress: "localhost:0",
	})
	if s == nil {
		t.Fatal("expected non-nil strategy")
	}
	if s.logger == nil {
		t.Error("expected non-nil logger")
	}
	if s.collector == nil {
		t.Error("expected non-nil collector")
	}
}

func TestGrpcStrategy_RankSize(t *testing.T) {
	s := &GrpcStrategy[float32]{rank: 2, size: 5}
	if s.Rank() != 2 {
		t.Errorf("Rank() = %d, want 2", s.Rank())
	}
	if s.Size() != 5 {
		t.Errorf("Size() = %d, want 5", s.Size())
	}
}

func TestGrpcStrategy_ShutdownIdempotent(t *testing.T) {
	s := NewGrpcStrategy[float32](GrpcStrategyConfig{
		WorkerAddress: "localhost:0",
	})
	// Double shutdown should not panic.
	s.Shutdown()
	s.Shutdown()
}

func TestGrpcStrategy_CloseCallsShutdown(t *testing.T) {
	s := NewGrpcStrategy[float32](GrpcStrategyConfig{
		WorkerAddress: "localhost:0",
	})
	if err := s.Close(context.Background()); err != nil {
		t.Errorf("Close() error = %v", err)
	}
}
