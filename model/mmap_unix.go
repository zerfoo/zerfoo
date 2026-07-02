//go:build unix

package model

import (
	"fmt"
	"os"
	"syscall"
)

// MmapReader memory-maps a file and provides access to its contents
// as a byte slice backed by the OS page cache.
type MmapReader struct {
	data []byte
}

// NewMmapReader memory-maps the file at path and returns an MmapReader.
// The caller must call Close when done to release the mapping.
func NewMmapReader(path string) (*MmapReader, error) {
	f, err := os.Open(path) //nolint:gosec // model file path validated by caller
	if err != nil {
		return nil, fmt.Errorf("mmap open %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("mmap stat %q: %w", path, err)
	}

	size := info.Size()
	if size == 0 {
		return &MmapReader{}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %q: %w", path, err)
	}

	return &MmapReader{data: data}, nil
}

// Bytes returns the memory-mapped file contents.
func (r *MmapReader) Bytes() []byte {
	return r.data
}

// Close releases the memory mapping. It is safe to call multiple times.
func (r *MmapReader) Close() error {
	if r.data == nil {
		return nil
	}
	err := syscall.Munmap(r.data)
	r.data = nil
	return err
}
