// Package testutil provides shared helpers for the parity test suite.
//
// These helpers were previously inlined as unexported functions in
// tests/parity/*_test.go; they are extracted here so future parity tests
// can import them cleanly. See T124.6.2.
package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// GoldenDir returns the absolute path to tests/golden/layers/ relative to
// this source file. The previous in-package helper resolved the path via
// runtime.Caller(0) from tests/parity/, so we walk one extra level up
// here to preserve the original target directory.
func GoldenDir() string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(f), "..", "..", "golden", "layers")
}

// LoadGolden loads a golden JSON file from GoldenDir and returns the raw map.
func LoadGolden(t *testing.T, name string) map[string]interface{} {
	t.Helper()
	path := filepath.Join(GoldenDir(), name+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("load golden %s: %v", name, err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("parse golden %s: %v", name, err)
	}
	return m
}

// GetFloat32s extracts a float32 slice from a JSON array.
func GetFloat32s(m map[string]interface{}, key string) []float32 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]float32, len(arr))
	for i, v := range arr {
		out[i] = float32(v.(float64))
	}
	return out
}

// GetFloat64s extracts a float64 slice from a JSON array.
func GetFloat64s(m map[string]interface{}, key string) []float64 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]float64, len(arr))
	for i, v := range arr {
		out[i] = v.(float64)
	}
	return out
}

// GetFloat64s2D extracts a 2D float64 slice from a JSON nested array.
func GetFloat64s2D(m map[string]interface{}, key string) [][]float64 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([][]float64, len(arr))
	for i, row := range arr {
		rowArr := row.([]interface{})
		out[i] = make([]float64, len(rowArr))
		for j, v := range rowArr {
			out[i][j] = v.(float64)
		}
	}
	return out
}

// GetInts extracts an int slice from a JSON array.
func GetInts(m map[string]interface{}, key string) []int {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]int, len(arr))
	for i, v := range arr {
		out[i] = int(v.(float64))
	}
	return out
}

// GetFloat extracts a float64 from a JSON value.
func GetFloat(m map[string]interface{}, key string) float64 {
	return m[key].(float64)
}

// ToFloat64Slice coerces an interface holding a JSON array to []float64.
func ToFloat64Slice(v interface{}) []float64 {
	arr := v.([]interface{})
	out := make([]float64, len(arr))
	for i, x := range arr {
		out[i] = x.(float64)
	}
	return out
}

// ReshapeFloat64 reshapes a flat float64 slice into a 2D slice [rows][cols].
func ReshapeFloat64(flat []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = flat[r*cols : (r+1)*cols]
	}
	return result
}
