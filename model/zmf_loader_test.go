package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// TestLoadZMF tests the successful loading and parsing of a valid .zmf file.
func TestLoadZMF(t *testing.T) {
	// 1. Create a temporary .zmf file for testing.
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "test_model.zmf")

	// Create a sample ZMF Model protobuf message.
	sampleModel := &zmf.Model{
		Metadata: &zmf.Metadata{
			ProducerName:    "zerfoo_test_producer",
			ProducerVersion: "0.1.0",
			OpsetVersion:    1,
		},
		Graph: &zmf.Graph{
			Nodes: []*zmf.Node{
				{
					Name:   "node_1",
					OpType: "TestOp",
					Inputs: []string{"input_tensor"},
				},
			},
		},
	}

	// Marshal the sample model to bytes.
	data, err := proto.Marshal(sampleModel)
	if err != nil {
		t.Fatalf("Failed to marshal sample model: %v", err)
	}

	// Write the bytes to the temporary file.
	if err := os.WriteFile(filePath, data, 0o600); err != nil {
		t.Fatalf("Failed to write temp .zmf file: %v", err)
	}

	// 2. Call the function we are testing.
	loadedModel, err := LoadZMF(filePath)
	if err != nil {
		t.Fatalf("LoadZMF failed: %v", err)
	}

	// 3. Verify the result.
	if loadedModel == nil {
		t.Fatal("LoadZMF returned a nil model")
	}

	// Check a few fields to ensure deserialization was correct.
	if loadedModel.Metadata.ProducerName != "zerfoo_test_producer" {
		t.Errorf("Expected producer name 'zerfoo_test_producer', got '%s'", loadedModel.Metadata.ProducerName)
	}
	if len(loadedModel.Graph.Nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(loadedModel.Graph.Nodes))
	}
	if loadedModel.Graph.Nodes[0].Name != "node_1" {
		t.Errorf("Expected node name 'node_1', got '%s'", loadedModel.Graph.Nodes[0].Name)
	}
}

// TestLoadZMF_FileNotFound tests that LoadZMF returns an error for a non-existent file.
func TestLoadZMF_FileNotFound(t *testing.T) {
	_, err := LoadZMF("non_existent_file.zmf")
	if err == nil {
		t.Fatal("Expected an error for a non-existent file, but got nil")
	}
}

// TestLoadZMF_InvalidData tests that LoadZMF returns an error for a corrupted/invalid file.
func TestLoadZMF_InvalidData(t *testing.T) {
	// 1. Create a temporary file with invalid protobuf data.
	tempDir := t.TempDir()
	filePath := filepath.Join(tempDir, "invalid.zmf")
	invalidData := []byte{0x01, 0x02, 0x03, 0x04} // Not valid protobuf

	if err := os.WriteFile(filePath, invalidData, 0o600); err != nil {
		t.Fatalf("Failed to write temp invalid .zmf file: %v", err)
	}

	// 2. Call the function and expect an error.
	_, err := LoadZMF(filePath)
	if err == nil {
		t.Fatal("Expected an unmarshaling error, but got nil")
	}
}
