package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestMmapReader(t *testing.T) {
	tests := []struct {
		name    string
		content []byte
	}{
		{
			name:    "small file",
			content: []byte("hello mmap world"),
		},
		{
			name:    "binary data",
			content: []byte{0x00, 0x01, 0xFF, 0xFE, 0x80, 0x7F},
		},
		{
			name:    "large file",
			content: make([]byte, 1<<20), // 1MB
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "test.bin")
			if err := os.WriteFile(path, tt.content, 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}

			r, err := NewMmapReader(path)
			if err != nil {
				t.Fatalf("NewMmapReader: %v", err)
			}
			defer func() { _ = r.Close() }()

			got := r.Bytes()
			if len(got) != len(tt.content) {
				t.Fatalf("len = %d, want %d", len(got), len(tt.content))
			}
			for i := range got {
				if got[i] != tt.content[i] {
					t.Fatalf("byte %d: got %d, want %d", i, got[i], tt.content[i])
				}
			}
		})
	}
}

func TestMmapReader_MissingFile(t *testing.T) {
	_, err := NewMmapReader("/nonexistent/path/file.bin")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestMmapReader_EmptyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.bin")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	r, err := NewMmapReader(path)
	if err != nil {
		t.Fatalf("NewMmapReader: %v", err)
	}
	defer func() { _ = r.Close() }()

	if len(r.Bytes()) != 0 {
		t.Fatalf("expected empty bytes, got %d", len(r.Bytes()))
	}
}

func TestMmapReader_DoubleClose(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	if err := os.WriteFile(path, []byte("data"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	r, err := NewMmapReader(path)
	if err != nil {
		t.Fatalf("NewMmapReader: %v", err)
	}

	if err := r.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	// Second close should not panic or error.
	if err := r.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}
