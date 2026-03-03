package training

import (
	"context"

	"github.com/zerfoo/zerfoo/tensor"
)

// ChunkLoader is a callback that returns the next chunk of batches.
// It receives a zero-based chunk index and returns the batches for that chunk.
// Return nil batches (or empty slice) with nil error to signal no more chunks.
type ChunkLoader[T tensor.Numeric] func(chunkIdx int) ([]*Batch[T], error)

// ChunkedDataIterator loads batches in chunks via a callback function.
// Each chunk represents a logical unit of data (e.g., one era, one file shard).
// Only one chunk's batches are held in memory at a time; the previous chunk's
// data is released when the next chunk is loaded.
type ChunkedDataIterator[T tensor.Numeric] struct {
	loader       ChunkLoader[T]
	currentChunk []*Batch[T]
	chunkIdx     int
	batchIdx     int
	err          error
	exhausted    bool
}

// NewChunkedDataIterator creates a new iterator that loads batches in chunks.
// The loader function is called with increasing chunk indices starting from 0.
// When the loader returns nil or an empty slice with no error, iteration ends.
func NewChunkedDataIterator[T tensor.Numeric](loader ChunkLoader[T]) *ChunkedDataIterator[T] {
	return &ChunkedDataIterator[T]{
		loader:   loader,
		chunkIdx: 0,
		batchIdx: -1,
	}
}

// Next advances to the next batch. Returns false when all chunks are exhausted
// or an error occurs. Automatically loads the next chunk when the current one
// is fully consumed.
func (c *ChunkedDataIterator[T]) Next(_ context.Context) bool {
	if c.exhausted {
		return false
	}

	// Try to advance within the current chunk.
	c.batchIdx++
	if c.currentChunk != nil && c.batchIdx < len(c.currentChunk) {
		return true
	}

	// Current chunk exhausted -- load next chunks until we find a non-empty one.
	for {
		// Release the old chunk for GC.
		c.currentChunk = nil

		batches, err := c.loader(c.chunkIdx)
		if err != nil {
			c.err = err
			c.exhausted = true
			return false
		}
		if batches == nil {
			c.exhausted = true
			return false
		}

		c.chunkIdx++
		c.batchIdx = 0

		if len(batches) > 0 {
			c.currentChunk = batches
			return true
		}
		// Empty chunk -- skip and try next.
	}
}

// Batch returns the current batch. Returns nil if called before Next or after
// iteration is exhausted.
func (c *ChunkedDataIterator[T]) Batch() *Batch[T] {
	if c.currentChunk == nil || c.batchIdx < 0 || c.batchIdx >= len(c.currentChunk) {
		return nil
	}
	return c.currentChunk[c.batchIdx]
}

// Error returns any error that occurred during chunk loading.
func (c *ChunkedDataIterator[T]) Error() error {
	return c.err
}

// Close releases resources held by the iterator.
func (c *ChunkedDataIterator[T]) Close() error {
	c.currentChunk = nil
	c.exhausted = true
	return nil
}

// Reset rewinds the iterator to the beginning, allowing re-iteration
// from chunk 0. The loader will be called again starting from index 0.
func (c *ChunkedDataIterator[T]) Reset() error {
	c.currentChunk = nil
	c.chunkIdx = 0
	c.batchIdx = -1
	c.err = nil
	c.exhausted = false
	return nil
}
