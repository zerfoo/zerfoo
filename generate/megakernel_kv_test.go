package generate

import (
	"fmt"
	"testing"

	"github.com/zerfoo/ztensor/graph"
)

func TestDetectKVCacheOps_NoKVOps(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
		{OpName: "Mul", InputIdx: []int{2, 3}, OutputIdx: 4},
	}
	detected := detectKVCacheOps(instructions)
	if detected {
		t.Error("expected no KV ops detected")
	}
}

func TestDetectKVCacheOps_WithKVOps(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
		{OpName: "KVCacheAppendK", InputIdx: []int{3}, OutputIdx: 4, ExtraArgs: map[string]any{"layer": 0}},
		{OpName: "KVCacheAppendV", InputIdx: []int{5}, OutputIdx: 6, ExtraArgs: map[string]any{"layer": 0}},
		{OpName: "KVCacheGetK", InputIdx: []int{}, OutputIdx: 7, ExtraArgs: map[string]any{"layer": 0}},
		{OpName: "KVCacheGetV", InputIdx: []int{}, OutputIdx: 8, ExtraArgs: map[string]any{"layer": 0}},
	}
	detected := detectKVCacheOps(instructions)
	if !detected {
		t.Error("expected KV ops detected")
	}
}

func TestExtractKVCacheDims_SingleLayer(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "KVCacheAppendK", InputIdx: []int{3}, OutputIdx: 4, ExtraArgs: map[string]any{"layer": 0}},
		{OpName: "KVCacheAppendV", InputIdx: []int{5}, OutputIdx: 6, ExtraArgs: map[string]any{"layer": 0}},
	}
	slotShapes := [][]int{
		nil, nil, nil,
		{1, 8, 64}, // slot 3: K data shape [batch, numHeads, headDim]
		nil,
		{1, 8, 64}, // slot 5: V data shape
		nil,
	}
	numLayers, numHeads, headDim := extractKVCacheDims(instructions, slotShapes)
	if numLayers != 1 {
		t.Errorf("numLayers = %d, want 1", numLayers)
	}
	if numHeads != 8 {
		t.Errorf("numHeads = %d, want 8", numHeads)
	}
	if headDim != 64 {
		t.Errorf("headDim = %d, want 64", headDim)
	}
}

func TestExtractKVCacheDims_MultipleLayers(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "KVCacheAppendK", InputIdx: []int{0}, OutputIdx: 1, ExtraArgs: map[string]any{"layer": 0}},
		{OpName: "KVCacheAppendK", InputIdx: []int{2}, OutputIdx: 3, ExtraArgs: map[string]any{"layer": 1}},
		{OpName: "KVCacheAppendK", InputIdx: []int{4}, OutputIdx: 5, ExtraArgs: map[string]any{"layer": 3}},
	}
	slotShapes := [][]int{
		{1, 4, 32}, // slot 0
		nil,
		{1, 4, 32}, // slot 2
		nil,
		{1, 4, 32}, // slot 4
		nil,
	}
	numLayers, numHeads, headDim := extractKVCacheDims(instructions, slotShapes)
	if numLayers != 4 { // max layer index 3 + 1
		t.Errorf("numLayers = %d, want 4", numLayers)
	}
	if numHeads != 4 {
		t.Errorf("numHeads = %d, want 4", numHeads)
	}
	if headDim != 32 {
		t.Errorf("headDim = %d, want 32", headDim)
	}
}

func TestExtractKVCacheDims_NoKVOps(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
	}
	numLayers, numHeads, headDim := extractKVCacheDims(instructions, nil)
	if numLayers != 0 || numHeads != 0 || headDim != 0 {
		t.Errorf("expected all zeros, got layers=%d heads=%d dim=%d", numLayers, numHeads, headDim)
	}
}

func TestGPUKVCache_DevicePointerArrays(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 2, 8, 1, 4)
	if err != nil {
		t.Fatalf("NewGPUKVCache: %v", err)
	}
	defer func() { _ = cache.Close() }()

	kPtrs, vPtrs, err := cache.DevicePointerArrays()
	if err != nil {
		t.Fatalf("DevicePointerArrays: %v", err)
	}
	if kPtrs == nil {
		t.Error("expected non-nil kPtrs")
	}
	if vPtrs == nil {
		t.Error("expected non-nil vPtrs")
	}
}

func TestGPUKVCache_DevicePointerArrays_AllocFailure(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 1, 4, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	// Inject alloc failure for pointer array allocation.
	alloc.allocErr = fmt.Errorf("out of GPU memory")
	_, _, err = cache.DevicePointerArrays()
	if err == nil {
		t.Error("expected error from failed alloc")
	}
}

func TestKVCachePosWiring(t *testing.T) {
	// Verify that SeqLen advances with Append and can be used for pos.
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 1, 8, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	if cache.SeqLen() != 0 {
		t.Fatalf("initial SeqLen = %d, want 0", cache.SeqLen())
	}

	tok := []float32{1, 2}
	for pos := range 3 {
		if err := cache.Append(0, tok, tok, pos); err != nil {
			t.Fatalf("Append pos %d: %v", pos, err)
		}
	}

	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen after 3 appends = %d, want 3", cache.SeqLen())
	}

	// Reset should bring pos back to 0.
	cache.Reset()
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen after Reset = %d, want 0", cache.SeqLen())
	}
}
