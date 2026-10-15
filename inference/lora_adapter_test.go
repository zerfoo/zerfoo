package inference

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

func TestApplyLoRAAdapter(t *testing.T) {
	baseWeight, err := tensor.New([]int{2, 2}, []float32{1, 0, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	base := map[string]*tensor.TensorNumeric[float32]{"model.layer.proj.weight": baseWeight}
	path := filepath.Join(t.TempDir(), "adapter.gguf")
	w := ztensorgguf.NewWriter()
	w.AddMetadataString("general.architecture", "lora")
	w.AddMetadataUint32("lora.rank", 1)
	w.AddMetadataFloat32("lora.alpha", 2)
	w.AddTensorF32("lora.model.layer.proj.weight_a", []int{1, 2}, []float32{1, 2})
	w.AddTensorF32("lora.model.layer.proj.weight_b", []int{2, 1}, []float32{3, 4})
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(f); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ApplyLoRAAdapter(base, path); err != nil {
		t.Fatal(err)
	}
	want := []float32{7, 12, 8, 17}
	for i, got := range base["model.layer.proj.weight"].Data() {
		if math.Abs(float64(got-want[i])) > 1e-6 {
			t.Fatalf("merged weight %d = %v, want %v", i, got, want[i])
		}
	}
	if got := baseWeight.Data(); got[0] != 1 || got[3] != 1 {
		t.Fatalf("base tensor mutated: %v", got)
	}
}
