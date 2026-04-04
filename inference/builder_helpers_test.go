package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestTensorLookup_Lookup(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"model.embed_tokens.weight": dummy,
	}
	tl := newTensorLookup(tensors)

	got, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != dummy {
		t.Fatal("Lookup returned wrong tensor")
	}
}

func TestTensorLookup_LookupMissing(t *testing.T) {
	tensors := map[string]*tensor.TensorNumeric[float32]{}
	tl := newTensorLookup(tensors)

	_, err := tl.Lookup("nonexistent")
	if err == nil {
		t.Fatal("expected error for missing tensor")
	}
	want := `missing tensor "nonexistent"`
	if err.Error() != want {
		t.Fatalf("error = %q, want %q", err.Error(), want)
	}
}

func TestTensorLookup_Optional(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"lm_head.weight": dummy,
	}
	tl := newTensorLookup(tensors)

	got, ok := tl.Optional("lm_head.weight")
	if !ok {
		t.Fatal("expected ok=true for existing tensor")
	}
	if got != dummy {
		t.Fatal("Optional returned wrong tensor")
	}

	got, ok = tl.Optional("missing")
	if ok {
		t.Fatal("expected ok=false for missing tensor")
	}
	if got != nil {
		t.Fatal("expected nil tensor for missing key")
	}
}

func TestTensorLookup_Has(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{1}, nil)
	if err != nil {
		t.Fatal(err)
	}
	tensors := map[string]*tensor.TensorNumeric[float32]{
		"weight": dummy,
	}
	tl := newTensorLookup(tensors)

	if !tl.Has("weight") {
		t.Fatal("expected Has to return true for existing key")
	}
	if tl.Has("missing") {
		t.Fatal("expected Has to return false for missing key")
	}
}

func TestParamWrapper_Wrap(t *testing.T) {
	dummy, err := tensor.New[float32]([]int{3, 4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	pw := newParamWrapper[float32]()

	p := pw.Wrap("layer.0.weight", dummy)
	if p.Name != "layer.0.weight" {
		t.Fatalf("Name = %q, want %q", p.Name, "layer.0.weight")
	}
	if p.Value != dummy {
		t.Fatal("Value does not match input tensor")
	}
	if p.Gradient != nil {
		t.Fatal("expected nil Gradient for newly wrapped parameter")
	}
}
