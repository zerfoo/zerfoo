package inference

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestSaveLoadTRTEngine(t *testing.T) {
	tmpDir := t.TempDir()
	trtCacheDirOverride = tmpDir
	defer func() { trtCacheDirOverride = "" }()

	data := []byte("fake-engine-data-for-testing")
	key := "test_key_12345"

	if err := SaveTRTEngine(key, data); err != nil {
		t.Fatalf("SaveTRTEngine: %v", err)
	}

	path := filepath.Join(tmpDir, key+".engine")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatalf("expected engine file at %s", path)
	}

	loaded, err := LoadTRTEngine(key)
	if err != nil {
		t.Fatalf("LoadTRTEngine: %v", err)
	}
	if string(loaded) != string(data) {
		t.Errorf("loaded data mismatch: got %q, want %q", loaded, data)
	}
}

func TestLoadTRTEngineMiss(t *testing.T) {
	tmpDir := t.TempDir()
	trtCacheDirOverride = tmpDir
	defer func() { trtCacheDirOverride = "" }()

	data, err := LoadTRTEngine("nonexistent_key")
	if err != nil {
		t.Fatalf("LoadTRTEngine: %v", err)
	}
	if data != nil {
		t.Errorf("expected nil on cache miss, got %d bytes", len(data))
	}
}

func TestTRTCacheKey(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	key1, err := TRTCacheKey("model-a", "fp32")
	if err != nil {
		t.Fatalf("TRTCacheKey: %v", err)
	}
	if key1 == "" {
		t.Fatal("expected non-empty key")
	}

	key2, err := TRTCacheKey("model-a", "fp32")
	if err != nil {
		t.Fatalf("TRTCacheKey: %v", err)
	}
	if key1 != key2 {
		t.Errorf("same inputs produced different keys: %q vs %q", key1, key2)
	}

	key3, err := TRTCacheKey("model-b", "fp32")
	if err != nil {
		t.Fatalf("TRTCacheKey: %v", err)
	}
	if key1 == key3 {
		t.Errorf("different inputs produced same key: %q", key1)
	}

	key4, err := TRTCacheKey("model-a", "fp16")
	if err != nil {
		t.Fatalf("TRTCacheKey: %v", err)
	}
	if key1 == key4 {
		t.Errorf("different precision produced same key: %q", key1)
	}
}
