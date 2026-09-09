package dsl

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestGGUFPreservesCustomGraphAndSharedWeights(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	e, err := Compile(ctx, residualDefinition(), engine, 42)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range e.Parameters() {
		values := p.Value.Data()
		for i := range values {
			values[i] = float32(i+1) / 10
		}
		p.Value.SetData(values)
	}
	x, err := tensor.New([]int{1, 2}, []float32{2, -1})
	if err != nil {
		t.Fatal(err)
	}
	before, err := e.Forward(ctx, map[string]*tensor.TensorNumeric[float32]{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	var raw bytes.Buffer
	if err := e.WriteGGUF(ctx, &raw, map[string]string{"preprocessing": "external-hash"}); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(raw.Bytes())
	id := hex.EncodeToString(hash[:])
	restored, metadata, err := ReadGGUF(ctx, bytes.NewReader(raw.Bytes()), id, engine)
	if err != nil {
		t.Fatal(err)
	}
	after, err := restored.Forward(ctx, map[string]*tensor.TensorNumeric[float32]{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	if restored.ID() != e.ID() || len(restored.Parameters()) != 2 || metadata["preprocessing"] != "external-hash" {
		t.Fatal("architecture, sharing or context changed")
	}
	for name, value := range before {
		if !reflect.DeepEqual(value.Data(), after[name].Data()) {
			t.Fatalf("output %s changed", name)
		}
	}
	for _, test := range []struct {
		name string
		data []byte
		pin  string
	}{{"truncated", raw.Bytes()[:raw.Len()/2], ""}, {"wrong_identity", raw.Bytes(), "wrong"}} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := ReadGGUF(ctx, bytes.NewReader(test.data), test.pin, engine); err == nil {
				t.Fatal("corruption accepted")
			}
		})
	}
}
