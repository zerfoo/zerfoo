package generate

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// mockAllocator simulates GPU memory using host-side byte slices.
type mockAllocator struct {
	allocs   map[unsafe.Pointer][]byte
	allocErr error // if set, Alloc returns this error
	freeErr  error // if set, Free returns this error
	cpyErr   error // if set, Memcpy returns this error
}

func newMockAllocator() *mockAllocator {
	return &mockAllocator{allocs: make(map[unsafe.Pointer][]byte)}
}

func (m *mockAllocator) Alloc(size int) (unsafe.Pointer, error) {
	if m.allocErr != nil {
		return nil, m.allocErr
	}
	buf := make([]byte, size)
	ptr := unsafe.Pointer(&buf[0])
	m.allocs[ptr] = buf
	return ptr, nil
}

func (m *mockAllocator) Free(ptr unsafe.Pointer) error {
	if m.freeErr != nil {
		return m.freeErr
	}
	delete(m.allocs, ptr)
	return nil
}

func (m *mockAllocator) Memcpy(dst, src unsafe.Pointer, size int, kind int) error {
	if m.cpyErr != nil {
		return m.cpyErr
	}
	dstSlice := unsafe.Slice((*byte)(dst), size)
	srcSlice := unsafe.Slice((*byte)(src), size)
	copy(dstSlice, srcSlice)
	return nil
}

// readFloat32s reads n float32 values from the mock device pointer.
func readFloat32s(ptr unsafe.Pointer, n int) []float32 {
	byteSlice := unsafe.Slice((*byte)(ptr), n*4)
	out := make([]float32, n)
	for i := range n {
		bits := binary.LittleEndian.Uint32(byteSlice[i*4 : i*4+4])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func TestGPUKVCache_NewAndClose(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 4, 512, 8, 64)
	if err != nil {
		t.Fatalf("NewGPUKVCache: %v", err)
	}
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen = %d, want 0", cache.SeqLen())
	}
	if err := cache.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if len(alloc.allocs) != 0 {
		t.Errorf("leaked %d allocations after Close", len(alloc.allocs))
	}
}

func TestGPUKVCache_NewValidation(t *testing.T) {
	alloc := newMockAllocator()
	tests := []struct {
		name      string
		alloc     GPUAllocator
		layers    int
		maxSeq    int
		heads     int
		headDim   int
		wantError bool
	}{
		{"nil allocator", nil, 4, 512, 8, 64, true},
		{"zero layers", alloc, 0, 512, 8, 64, true},
		{"negative layers", alloc, -1, 512, 8, 64, true},
		{"zero maxSeq", alloc, 4, 0, 8, 64, true},
		{"zero heads", alloc, 4, 512, 0, 64, true},
		{"zero headDim", alloc, 4, 512, 8, 0, true},
		{"valid", alloc, 4, 512, 8, 64, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := NewGPUKVCache(tt.alloc, tt.layers, tt.maxSeq, tt.heads, tt.headDim)
			if tt.wantError {
				if err == nil {
					t.Error("expected error, got nil")
					if c != nil {
						_ = c.Close()
					}
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if c != nil {
					_ = c.Close()
				}
			}
		})
	}
}

func TestGPUKVCache_AppendAndPointers(t *testing.T) {
	alloc := newMockAllocator()
	numLayers := 2
	maxSeq := 8
	numHeads := 2
	headDim := 4
	tokenElems := numHeads * headDim

	cache, err := NewGPUKVCache(alloc, numLayers, maxSeq, numHeads, headDim)
	if err != nil {
		t.Fatalf("NewGPUKVCache: %v", err)
	}
	defer func() { _ = cache.Close() }()

	// Append token at position 0 to both layers.
	k0 := make([]float32, tokenElems)
	v0 := make([]float32, tokenElems)
	for i := range tokenElems {
		k0[i] = float32(i + 1)
		v0[i] = float32(i + 100)
	}

	for layer := range numLayers {
		if err := cache.Append(layer, k0, v0, 0); err != nil {
			t.Fatalf("Append layer %d pos 0: %v", layer, err)
		}
	}

	if cache.SeqLen() != 1 {
		t.Errorf("SeqLen = %d, want 1", cache.SeqLen())
	}

	// Verify data written to layer 0.
	kPtr, vPtr, seqLen := cache.Pointers(0)
	if kPtr == nil || vPtr == nil {
		t.Fatal("Pointers(0) returned nil")
	}
	if seqLen != 1 {
		t.Errorf("Pointers seqLen = %d, want 1", seqLen)
	}

	gotK := readFloat32s(kPtr, tokenElems)
	for i := range tokenElems {
		if gotK[i] != k0[i] {
			t.Errorf("K[%d] = %v, want %v", i, gotK[i], k0[i])
		}
	}

	gotV := readFloat32s(vPtr, tokenElems)
	for i := range tokenElems {
		if gotV[i] != v0[i] {
			t.Errorf("V[%d] = %v, want %v", i, gotV[i], v0[i])
		}
	}
}

func TestGPUKVCache_AppendMultipleTokens(t *testing.T) {
	alloc := newMockAllocator()
	numLayers := 1
	maxSeq := 8
	numHeads := 1
	headDim := 2
	tokenElems := numHeads * headDim

	cache, err := NewGPUKVCache(alloc, numLayers, maxSeq, numHeads, headDim)
	if err != nil {
		t.Fatalf("NewGPUKVCache: %v", err)
	}
	defer func() { _ = cache.Close() }()

	// Append 3 tokens.
	for pos := range 3 {
		k := []float32{float32(pos*2 + 1), float32(pos*2 + 2)}
		v := []float32{float32(pos*2 + 10), float32(pos*2 + 11)}
		if err := cache.Append(0, k, v, pos); err != nil {
			t.Fatalf("Append pos %d: %v", pos, err)
		}
	}

	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen = %d, want 3", cache.SeqLen())
	}

	kPtr, _, _ := cache.Pointers(0)
	gotK := readFloat32s(kPtr, 3*tokenElems)
	wantK := []float32{1, 2, 3, 4, 5, 6}
	for i, w := range wantK {
		if gotK[i] != w {
			t.Errorf("K[%d] = %v, want %v", i, gotK[i], w)
		}
	}
}

func TestGPUKVCache_AppendErrors(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 2, 4, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	good := []float32{1, 2}

	tests := []struct {
		name  string
		layer int
		k, v  []float32
		pos   int
	}{
		{"negative layer", -1, good, good, 0},
		{"layer out of range", 5, good, good, 0},
		{"wrong seqPos", 0, good, good, 1},
		{"k wrong length", 0, []float32{1}, good, 0},
		{"v wrong length", 0, good, []float32{1}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := cache.Append(tt.layer, tt.k, tt.v, tt.pos); err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestGPUKVCache_AppendOverflow(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 1, 2, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	tok := []float32{1, 2}
	// Fill to capacity.
	for pos := range 2 {
		if err := cache.Append(0, tok, tok, pos); err != nil {
			t.Fatalf("Append pos %d: %v", pos, err)
		}
	}
	// Third append should fail.
	if err := cache.Append(0, tok, tok, 2); err == nil {
		t.Error("expected overflow error")
	}
}

func TestGPUKVCache_Reset(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 1, 8, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	tok := []float32{1, 2}
	if err := cache.Append(0, tok, tok, 0); err != nil {
		t.Fatal(err)
	}
	if cache.SeqLen() != 1 {
		t.Fatalf("SeqLen = %d, want 1", cache.SeqLen())
	}

	cache.Reset()
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen after Reset = %d, want 0", cache.SeqLen())
	}

	// Should be able to append at pos 0 again.
	if err := cache.Append(0, tok, tok, 0); err != nil {
		t.Errorf("Append after Reset: %v", err)
	}
}

func TestGPUKVCache_PointersOutOfRange(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 2, 4, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	kPtr, vPtr, seqLen := cache.Pointers(-1)
	if kPtr != nil || vPtr != nil || seqLen != 0 {
		t.Error("Pointers(-1) should return nil, nil, 0")
	}

	kPtr, vPtr, seqLen = cache.Pointers(5)
	if kPtr != nil || vPtr != nil || seqLen != 0 {
		t.Error("Pointers(5) should return nil, nil, 0")
	}
}

func TestGPUKVCache_AllocFailure(t *testing.T) {
	alloc := newMockAllocator()
	alloc.allocErr = fmt.Errorf("out of GPU memory")

	_, err := NewGPUKVCache(alloc, 2, 4, 1, 2)
	if err == nil {
		t.Error("expected error from failed alloc")
	}
}

func TestGPUKVCache_AllocPartialFailure(t *testing.T) {
	// Allocator that fails after N successful allocations.
	callCount := 0
	alloc := &countingAllocator{
		inner:    newMockAllocator(),
		failAt:   3, // fail on 4th alloc (layer 1 V buffer)
		counter:  &callCount,
	}

	_, err := NewGPUKVCache(alloc, 2, 4, 1, 2)
	if err == nil {
		t.Error("expected error from partial alloc failure")
	}
}

type countingAllocator struct {
	inner   *mockAllocator
	failAt  int
	counter *int
}

func (c *countingAllocator) Alloc(size int) (unsafe.Pointer, error) {
	if *c.counter >= c.failAt {
		return nil, fmt.Errorf("alloc #%d failed", *c.counter)
	}
	*c.counter++
	return c.inner.Alloc(size)
}

func (c *countingAllocator) Free(ptr unsafe.Pointer) error {
	return c.inner.Free(ptr)
}

func (c *countingAllocator) Memcpy(dst, src unsafe.Pointer, size int, kind int) error {
	return c.inner.Memcpy(dst, src, size, kind)
}

func TestGPUKVCache_MemcpyFailure(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 1, 4, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cache.Close() }()

	alloc.cpyErr = fmt.Errorf("memcpy failed")
	tok := []float32{1, 2}
	if err := cache.Append(0, tok, tok, 0); err == nil {
		t.Error("expected memcpy error")
	}
}

func TestGPUKVCache_CloseIdempotent(t *testing.T) {
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 2, 4, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	// Second close should be safe (pointers are nil).
	if err := cache.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestGPUKVCache_MemoryBudget(t *testing.T) {
	// Gemma 3 2B: 26 layers, 8 heads, 256 head_dim, 512 tokens
	// Expected: 2 * 26 * 512 * 8 * 256 * 4 = ~104 MB
	alloc := newMockAllocator()
	cache, err := NewGPUKVCache(alloc, 26, 512, 8, 256)
	if err != nil {
		t.Fatalf("NewGPUKVCache: %v", err)
	}
	defer func() { _ = cache.Close() }()

	// 52 allocations: 26 layers * 2 (K + V).
	if len(alloc.allocs) != 52 {
		t.Errorf("allocations = %d, want 52", len(alloc.allocs))
	}

	// Each buffer: 512 * 8 * 256 * 4 = 4,194,304 bytes.
	wantBufBytes := 512 * 8 * 256 * 4
	for ptr, buf := range alloc.allocs {
		if len(buf) != wantBufBytes {
			t.Errorf("buf at %v has %d bytes, want %d", ptr, len(buf), wantBufBytes)
		}
	}
}
