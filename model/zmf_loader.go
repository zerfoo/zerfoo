// Package model provides the core structures and loading mechanisms for Zerfoo models.
package model

import (
	"fmt"
	"os"

	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// LoadZMF reads a Zerfoo Model Format (.zmf) file from the specified path,
// deserializes it, and returns the parsed Model object.
func LoadZMF(filePath string) (*zmf.Model, error) {
	// Read the entire file into a byte slice.
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read ZMF file '%s': %w", filePath, err)
	}

	// Create a new Model message to unmarshal into.
	model := &zmf.Model{}

	// Unmarshal the protobuf data.
	if err := proto.Unmarshal(data, model); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ZMF data: %w", err)
	}

	return model, nil
}
