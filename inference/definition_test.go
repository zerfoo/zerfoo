package inference

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestCompileArchitectureReferenceLlama(t *testing.T) {
	cfg := gguf.ModelConfig{Architecture: "llama", VocabSize: 32, HiddenSize: 16, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, IntermediateSize: 32, MaxSeqLen: 64, RopeTheta: 10000}
	tensors := makeLlamaTestTensors(&cfg)
	identity, err := ParameterSetIdentity(tensors)
	if err != nil {
		t.Fatal(err)
	}
	reference := dsl.ArchitectureReference{Version: 1, Architecture: "llama", ComponentVersion: 1, ParametersSHA256: identity, Config: cfg}
	graph, embedding, err := CompileArchitectureReference(context.Background(), reference, tensors, compute.NewCPUEngine(numeric.Float32Ops{}))
	if err != nil {
		t.Fatal(err)
	}
	if graph == nil || embedding == nil {
		t.Fatal("registered builder did not execute")
	}
	assertGraphForwardNonNaN(t, graph, cfg.VocabSize)
	reference.ParametersSHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
	if _, _, err := CompileArchitectureReference(context.Background(), reference, tensors, compute.NewCPUEngine(numeric.Float32Ops{})); err == nil {
		t.Fatal("mismatched parameters accepted")
	}
}
