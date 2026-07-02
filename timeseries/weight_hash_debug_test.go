package timeseries

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestHashParamTensors(t *testing.T) {
	a, err := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("new a: %v", err)
	}
	b, err := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("new b: %v", err)
	}
	c, err := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 5})
	if err != nil {
		t.Fatalf("new c: %v", err)
	}

	ha := hashFloat32Slice(a.Data())
	hb := hashFloat32Slice(b.Data())
	hc := hashFloat32Slice(c.Data())

	if ha != hb {
		t.Errorf("identical data should hash equal: ha=%x hb=%x", ha, hb)
	}
	if ha == hc {
		t.Errorf("different data should hash differently: ha=%x hc=%x", ha, hc)
	}

	// Exercise the public helper path (no-op without env var; must not panic).
	HashParamTensors("test", []*tensor.TensorNumeric[float32]{a, b, c, nil})

	// With env var set, hashing must still succeed (output goes to log).
	t.Setenv("ZERFOO_DEBUG_WEIGHT_HASH", "1")
	if !weightHashDebugEnabled() {
		t.Fatalf("debug flag should be enabled")
	}
	HashParamTensors("test-on", []*tensor.TensorNumeric[float32]{a, c})
}
