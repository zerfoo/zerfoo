package reducesum

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestBuildReduceSum_WithInt64Axes(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"axes":     []int64{0, 1},
		"keepdims": int64(1),
	}

	node, err := BuildReduceSum(engine, ops, "", nil, attrs)
	if err != nil {
		t.Fatalf("BuildReduceSum failed: %v", err)
	}

	rs, ok := node.(*ReduceSum[float32])
	if !ok {
		t.Fatalf("BuildReduceSum returned %T, want *ReduceSum[float32]", node)
	}

	if len(rs.axes) != 2 || rs.axes[0] != 0 || rs.axes[1] != 1 {
		t.Errorf("axes = %v, want [0, 1]", rs.axes)
	}
	if !rs.keepDims {
		t.Error("keepDims = false, want true")
	}
}

func TestBuildReduceSum_WithAnyAxes(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"axes":     []any{int64(2)},
		"keepdims": int64(0),
	}

	node, err := BuildReduceSum(engine, ops, "", nil, attrs)
	if err != nil {
		t.Fatalf("BuildReduceSum failed: %v", err)
	}

	rs := node.(*ReduceSum[float32])
	if len(rs.axes) != 1 || rs.axes[0] != 2 {
		t.Errorf("axes = %v, want [2]", rs.axes)
	}
	if rs.keepDims {
		t.Error("keepDims = true, want false")
	}
}

func TestBuildReduceSum_NoAxes(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{}

	node, err := BuildReduceSum(engine, ops, "", nil, attrs)
	if err != nil {
		t.Fatalf("BuildReduceSum failed: %v", err)
	}

	rs := node.(*ReduceSum[float32])
	if len(rs.axes) != 0 {
		t.Errorf("axes = %v, want []", rs.axes)
	}
}

func TestBuildReduceSum_NoKeepdims(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"axes": []int64{0},
	}

	node, err := BuildReduceSum(engine, ops, "", nil, attrs)
	if err != nil {
		t.Fatalf("BuildReduceSum failed: %v", err)
	}

	rs := node.(*ReduceSum[float32])
	// Default keepdims is 1 (true)
	if !rs.keepDims {
		t.Error("keepDims default should be true (keepdims=1)")
	}
}

func TestBuildReduceSum_UnsupportedAxesType(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	attrs := map[string]interface{}{
		"axes": "invalid",
	}

	_, err := BuildReduceSum(engine, ops, "", nil, attrs)
	if err == nil {
		t.Error("BuildReduceSum with string axes should fail")
	}
}

func TestBuildReduceSum_NodeInterface(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	node, err := BuildReduceSum(engine, ops, "", nil, map[string]interface{}{})
	if err != nil {
		t.Fatalf("BuildReduceSum failed: %v", err)
	}

	var _ = node
}
