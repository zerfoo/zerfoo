package lora

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestLoraCheckpoint(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	ctx := context.Background()

	dIn, dOut := 8, 4
	rank := 2
	alpha := float32(4.0)

	// Build model with two LoRA-injected layers.
	m := newTestModel[float32]()
	qProj, err := newNamedStubLinear[float32]("q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create q_proj: %v", err)
	}
	vProj, err := newNamedStubLinear[float32]("v_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create v_proj: %v", err)
	}
	m.AddLayer(qProj)
	m.AddLayer(vProj)

	if err := InjectLoRA[float32](m, rank, alpha, []string{"q_proj", "v_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA: %v", err)
	}

	// Set known A/B values so we can verify round-trip.
	for _, layer := range m.Layers() {
		ll, ok := layer.(*LoraLinear[float32])
		if !ok {
			continue
		}
		aData := make([]float32, rank*dIn)
		bData := make([]float32, dOut*rank)
		for i := range aData {
			aData[i] = float32(i+1) * 0.1
		}
		for i := range bData {
			bData[i] = float32(i+1) * 0.05
		}
		aTensor, err := tensor.New[float32]([]int{rank, dIn}, aData)
		if err != nil {
			t.Fatalf("create A tensor: %v", err)
		}
		bTensor, err := tensor.New[float32]([]int{dOut, rank}, bData)
		if err != nil {
			t.Fatalf("create B tensor: %v", err)
		}
		ll.A.Value = aTensor
		ll.B.Value = bTensor
	}

	// Forward pass before save.
	x, err := tensor.New[float32]([]int{1, dIn}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	if err != nil {
		t.Fatalf("create input: %v", err)
	}
	var outputsBefore [][]float32
	for _, layer := range m.Layers() {
		out, err := layer.Forward(ctx, x)
		if err != nil {
			t.Fatalf("forward before save: %v", err)
		}
		outputsBefore = append(outputsBefore, copySlice(out.Data()))
	}

	// Save checkpoint.
	dir := t.TempDir()
	ckptPath := filepath.Join(dir, "adapter.gguf")
	if err := SaveAdapter[float32](ckptPath, m); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	// Verify file exists and is non-empty.
	info, err := os.Stat(ckptPath)
	if err != nil {
		t.Fatalf("stat checkpoint: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("checkpoint file is empty")
	}

	// Build a fresh model with the same structure and inject LoRA.
	m2 := newTestModel[float32]()
	qProj2, err := newNamedStubLinear[float32]("q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create q_proj2: %v", err)
	}
	vProj2, err := newNamedStubLinear[float32]("v_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create v_proj2: %v", err)
	}
	m2.AddLayer(qProj2)
	m2.AddLayer(vProj2)

	if err := InjectLoRA[float32](m2, rank, alpha, []string{"q_proj", "v_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA for m2: %v", err)
	}

	// Load checkpoint into m2.
	if err := LoadAdapter[float32](ckptPath, m2); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	// Forward pass after load — should produce identical output.
	var outputsAfter [][]float32
	for _, layer := range m2.Layers() {
		out, err := layer.Forward(ctx, x)
		if err != nil {
			t.Fatalf("forward after load: %v", err)
		}
		outputsAfter = append(outputsAfter, copySlice(out.Data()))
	}

	if len(outputsBefore) != len(outputsAfter) {
		t.Fatalf("layer count mismatch: %d vs %d", len(outputsBefore), len(outputsAfter))
	}

	for i := range outputsBefore {
		before := outputsBefore[i]
		after := outputsAfter[i]
		if len(before) != len(after) {
			t.Fatalf("layer %d output length mismatch: %d vs %d", i, len(before), len(after))
		}
		for j := range before {
			if math.Abs(float64(before[j]-after[j])) > 1e-6 {
				t.Errorf("layer %d output[%d]: before=%f, after=%f", i, j, before[j], after[j])
			}
		}
	}
}

func TestLoraCheckpoint_NoAdapters(t *testing.T) {
	m := newTestModel[float32]()
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.gguf")

	err := SaveAdapter[float32](path, m)
	if err == nil {
		t.Error("expected error when saving model with no adapters")
	}
}

func TestLoraCheckpoint_LoadMismatch(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dIn, dOut := 8, 4
	rank := 2
	alpha := float32(4.0)

	// Build model and save checkpoint.
	m := newTestModel[float32]()
	qProj, err := newNamedStubLinear[float32]("q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create q_proj: %v", err)
	}
	m.AddLayer(qProj)

	if err := InjectLoRA[float32](m, rank, alpha, []string{"q_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "adapter.gguf")
	if err := SaveAdapter[float32](path, m); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	// Try to load into a model with different layer names.
	m2 := newTestModel[float32]()
	kProj, err := newNamedStubLinear[float32]("k_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create k_proj: %v", err)
	}
	m2.AddLayer(kProj)

	if err := InjectLoRA[float32](m2, rank, alpha, []string{"k_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA: %v", err)
	}

	err = LoadAdapter[float32](path, m2)
	if err == nil {
		t.Error("expected error when loading checkpoint with mismatched layer names")
	}
}

func TestLoraCheckpoint_RoundTripValues(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	dIn, dOut := 4, 3
	rank := 2
	alpha := float32(2.0)

	m := newTestModel[float32]()
	layer, err := newNamedStubLinear[float32]("attn.q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create layer: %v", err)
	}
	m.AddLayer(layer)

	if err := InjectLoRA[float32](m, rank, alpha, []string{"q_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA: %v", err)
	}

	// Set specific values.
	ll := m.Layers()[0].(*LoraLinear[float32])
	aData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	bData := []float32{-0.1, 0.2, -0.3, 0.4, -0.5, 0.6}
	aTensor, _ := tensor.New[float32]([]int{rank, dIn}, aData)
	bTensor, _ := tensor.New[float32]([]int{dOut, rank}, bData)
	ll.A.Value = aTensor
	ll.B.Value = bTensor

	dir := t.TempDir()
	path := filepath.Join(dir, "adapter.gguf")
	if err := SaveAdapter[float32](path, m); err != nil {
		t.Fatalf("SaveAdapter: %v", err)
	}

	// Reload.
	m2 := newTestModel[float32]()
	layer2, err := newNamedStubLinear[float32]("attn.q_proj", engine, dIn, dOut)
	if err != nil {
		t.Fatalf("create layer2: %v", err)
	}
	m2.AddLayer(layer2)

	if err := InjectLoRA[float32](m2, rank, alpha, []string{"q_proj"}, engine); err != nil {
		t.Fatalf("InjectLoRA m2: %v", err)
	}

	if err := LoadAdapter[float32](path, m2); err != nil {
		t.Fatalf("LoadAdapter: %v", err)
	}

	ll2 := m2.Layers()[0].(*LoraLinear[float32])
	aLoaded := ll2.A.Value.Data()
	bLoaded := ll2.B.Value.Data()

	for i, v := range aData {
		if !float32BitsEqual(v, aLoaded[i]) {
			t.Errorf("A[%d]: saved=%v loaded=%v", i, v, aLoaded[i])
		}
	}
	for i, v := range bData {
		if !float32BitsEqual(v, bLoaded[i]) {
			t.Errorf("B[%d]: saved=%v loaded=%v", i, v, bLoaded[i])
		}
	}
}

func copySlice(s []float32) []float32 {
	c := make([]float32, len(s))
	copy(c, s)
	return c
}
