package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
)

func TestBuildSparseRoutedAttention(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]interface{}{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
		"max_seq_len":  64,
	}

	params := map[string]*graph.Parameter[float32]{}

	node, err := BuildSparseRoutedAttention(engine, ops, "sra", params, attrs)
	if err != nil {
		t.Fatalf("BuildSparseRoutedAttention failed: %v", err)
	}

	if node.OpType() != "SparseRoutedAttention" {
		t.Errorf("OpType() = %q, want %q", node.OpType(), "SparseRoutedAttention")
	}

	// Verify the built layer can run Forward.
	Q, K, V := makeQKV(t, 1, 4, 2, 2, 8, 8)
	out, err := node.Forward(context.Background(), Q, K, V)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	wantShape := []int{1, 4, 2, 8}
	gotShape := out.Shape()
	for i := range wantShape {
		if gotShape[i] != wantShape[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, gotShape[i], wantShape[i])
		}
	}
}

func TestBuildSparseRoutedAttention_CustomRopeBase(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]interface{}{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
		"max_seq_len":  64,
		"rope_base":    50000.0,
	}

	params := map[string]*graph.Parameter[float32]{}

	node, err := BuildSparseRoutedAttention(engine, ops, "sra", params, attrs)
	if err != nil {
		t.Fatalf("BuildSparseRoutedAttention with custom rope_base failed: %v", err)
	}

	Q, K, V := makeQKV(t, 1, 4, 2, 2, 8, 8)
	_, err = node.Forward(context.Background(), Q, K, V)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
}

func TestBuildSparseRoutedAttention_MissingAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	fullAttrs := map[string]interface{}{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
		"max_seq_len":  64,
	}

	params := map[string]*graph.Parameter[float32]{}

	requiredAttrs := []string{"num_heads", "num_kv_heads", "head_dim", "segment_size", "top_k", "max_seq_len"}
	for _, key := range requiredAttrs {
		t.Run("missing_"+key, func(t *testing.T) {
			attrs := make(map[string]interface{})
			for k, v := range fullAttrs {
				if k != key {
					attrs[k] = v
				}
			}
			_, err := BuildSparseRoutedAttention(engine, ops, "sra", params, attrs)
			if err == nil {
				t.Errorf("expected error for missing attribute %q", key)
			}
		})
	}
}

func TestBuildSparseRoutedAttention_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]interface{}{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
		"max_seq_len":  64,
	}

	params := map[string]*graph.Parameter[float32]{}

	node, err := BuildSparseRoutedAttention(engine, ops, "sra", params, attrs)
	if err != nil {
		t.Fatalf("BuildSparseRoutedAttention failed: %v", err)
	}

	got := node.Attributes()
	checks := map[string]int{
		"num_heads":    4,
		"num_kv_heads": 2,
		"head_dim":     8,
		"segment_size": 4,
		"top_k":        2,
	}
	for k, want := range checks {
		v, ok := got[k]
		if !ok {
			t.Errorf("missing attribute %q", k)
			continue
		}
		if v.(int) != want {
			t.Errorf("attribute %q: got %v, want %v", k, v, want)
		}
	}
}
