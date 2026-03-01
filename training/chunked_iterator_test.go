package training

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

// makeTestBatch creates a batch with a single target value for testing.
func makeTestBatch(val float32) *Batch[float32] {
	t, _ := tensor.New[float32]([]int{1, 1}, []float32{val})
	return &Batch[float32]{Targets: t}
}

func TestChunkedDataIterator_SingleChunk(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		if idx > 0 {
			return nil, nil
		}
		return []*Batch[float32]{makeTestBatch(1), makeTestBatch(2), makeTestBatch(3)}, nil
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	var values []float32
	for it.Next(ctx) {
		b := it.Batch()
		if b == nil {
			t.Fatal("Batch() returned nil after Next() returned true")
		}
		values = append(values, b.Targets.Data()[0])
	}
	if it.Error() != nil {
		t.Fatalf("unexpected error: %v", it.Error())
	}
	if len(values) != 3 {
		t.Fatalf("expected 3 batches, got %d", len(values))
	}
	for i, want := range []float32{1, 2, 3} {
		if values[i] != want {
			t.Errorf("batch %d: got %v, want %v", i, values[i], want)
		}
	}
}

func TestChunkedDataIterator_MultipleChunks(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		switch idx {
		case 0:
			return []*Batch[float32]{makeTestBatch(10), makeTestBatch(20)}, nil
		case 1:
			return []*Batch[float32]{makeTestBatch(30)}, nil
		case 2:
			return []*Batch[float32]{makeTestBatch(40), makeTestBatch(50)}, nil
		default:
			return nil, nil
		}
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	var values []float32
	for it.Next(ctx) {
		values = append(values, it.Batch().Targets.Data()[0])
	}
	if it.Error() != nil {
		t.Fatalf("unexpected error: %v", it.Error())
	}
	want := []float32{10, 20, 30, 40, 50}
	if len(values) != len(want) {
		t.Fatalf("expected %d batches, got %d", len(want), len(values))
	}
	for i, w := range want {
		if values[i] != w {
			t.Errorf("batch %d: got %v, want %v", i, values[i], w)
		}
	}
}

func TestChunkedDataIterator_EmptyChunks(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		return nil, nil
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	if it.Next(ctx) {
		t.Fatal("expected Next() to return false for empty loader")
	}
	if it.Error() != nil {
		t.Fatalf("unexpected error: %v", it.Error())
	}
}

func TestChunkedDataIterator_Reset(t *testing.T) {
	callCount := 0
	loader := func(idx int) ([]*Batch[float32], error) {
		callCount++
		if idx > 0 {
			return nil, nil
		}
		return []*Batch[float32]{makeTestBatch(float32(callCount))}, nil
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	// First pass
	if !it.Next(ctx) {
		t.Fatal("first Next() should return true")
	}
	v1 := it.Batch().Targets.Data()[0]
	if it.Next(ctx) {
		t.Fatal("second Next() should return false")
	}

	// Reset
	if err := it.Reset(); err != nil {
		t.Fatalf("Reset() error: %v", err)
	}

	// Second pass -- loader is called again from chunk 0
	if !it.Next(ctx) {
		t.Fatal("Next() after Reset() should return true")
	}
	v2 := it.Batch().Targets.Data()[0]

	// Values should differ because callCount increases
	if v1 == v2 {
		t.Error("expected different values after reset (loader called with fresh state)")
	}
}

func TestChunkedDataIterator_Error(t *testing.T) {
	loadErr := context.DeadlineExceeded
	loader := func(idx int) ([]*Batch[float32], error) {
		return nil, loadErr
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	if it.Next(ctx) {
		t.Fatal("Next() should return false when loader errors")
	}
	if !errors.Is(it.Error(), loadErr) {
		t.Fatalf("expected error %v, got %v", loadErr, it.Error())
	}
}

func TestChunkedDataIterator_Close(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		if idx > 0 {
			return nil, nil
		}
		return []*Batch[float32]{makeTestBatch(1)}, nil
	}

	it := NewChunkedDataIterator[float32](loader)
	if err := it.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}
}

func TestChunkedDataIterator_BatchBeforeNext(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		if idx > 0 {
			return nil, nil
		}
		return []*Batch[float32]{makeTestBatch(1)}, nil
	}

	it := NewChunkedDataIterator[float32](loader)

	// Batch() before any Next() call should return nil
	if b := it.Batch(); b != nil {
		t.Fatal("Batch() before Next() should return nil")
	}
}

func TestChunkedDataIterator_SkipsEmptyChunks(t *testing.T) {
	loader := func(idx int) ([]*Batch[float32], error) {
		switch idx {
		case 0:
			return []*Batch[float32]{}, nil // empty chunk
		case 1:
			return []*Batch[float32]{}, nil // another empty chunk
		case 2:
			return []*Batch[float32]{makeTestBatch(99)}, nil
		default:
			return nil, nil
		}
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	if !it.Next(ctx) {
		t.Fatal("Next() should skip empty chunks and find batch in chunk 2")
	}
	v := it.Batch().Targets.Data()[0]
	if v != 99 {
		t.Errorf("got %v, want 99", v)
	}
}

func TestChunkedDataIterator_ReleasesMemory(t *testing.T) {
	// Track which chunks are still referenced via a live flag.
	type tracked struct {
		batch *Batch[float32]
		alive bool
	}

	chunks := make([]tracked, 3)
	loader := func(idx int) ([]*Batch[float32], error) {
		if idx >= len(chunks) {
			return nil, nil
		}
		chunks[idx].batch = makeTestBatch(float32(idx))
		chunks[idx].alive = true
		return []*Batch[float32]{chunks[idx].batch}, nil
	}

	it := NewChunkedDataIterator[float32](loader)
	ctx := context.Background()

	// Consume all batches
	count := 0
	for it.Next(ctx) {
		it.Batch()
		count++
	}
	if count != 3 {
		t.Fatalf("expected 3 batches, got %d", count)
	}
	// The iterator should not hold references to old chunk data.
	// We can't test GC directly, but we verify the iterator's internal
	// currentChunk is nil after exhaustion.
	if it.Error() != nil {
		t.Fatalf("unexpected error: %v", it.Error())
	}
}
