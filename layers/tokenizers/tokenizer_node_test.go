package tokenizers

import (
	"context"
	"reflect"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestTokenizerNode(t *testing.T) {
	vocab := map[string]int32{
		"this": 1,
		"is":   2,
		"a":    3,
		"test": 4,
		"of":   5,
		"the":  6,
		"node": 7,
	}
	unkTokenID := int32(0)

	node := NewTokenizerNode(vocab, unkTokenID)

	tests := []struct {
		name          string
		inputText     string
		expectedIDs   []int32
		expectedShape []int
		expectError   bool
	}{
		{
			name:          "Simple sentence",
			inputText:     "this is a test",
			expectedIDs:   []int32{1, 2, 3, 4},
			expectedShape: []int{1, 4},
			expectError:   false,
		},
		{
			name:          "Sentence with unknown words",
			inputText:     "this is another test",
			expectedIDs:   []int32{1, 2, 0, 4}, // 'another' should be unkTokenID
			expectedShape: []int{1, 4},
			expectError:   false,
		},
		{
			name:          "Sentence with different casing",
			inputText:     "This Is A Test",
			expectedIDs:   []int32{1, 2, 3, 4}, // Should be converted to lowercase
			expectedShape: []int{1, 4},
			expectError:   false,
		},
		{
			name:          "Empty input string",
			inputText:     "",
			expectedIDs:   []int32{},
			expectedShape: []int{1, 0},
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputTextTensor, err := tensor.NewString([]int{1}, []string{tt.inputText})
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}

			outputTensor, err := node.Forward(context.Background(), inputTextTensor)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected an error, but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("Forward() returned an unexpected error: %v", err)
			}

			// Check output type
			output, ok := outputTensor.(*tensor.TensorNumeric[int32])
			if !ok {
				t.Fatalf("Expected output to be *tensor.TensorNumeric[int32], got %T", outputTensor)
			}

			// Check shape
			if !reflect.DeepEqual(output.Shape(), tt.expectedShape) {
				t.Errorf("Expected shape %v, got %v", tt.expectedShape, output.Shape())
			}

			// Check data
			if !reflect.DeepEqual(output.Data(), tt.expectedIDs) {
				t.Errorf("Expected token IDs %v, got %v", tt.expectedIDs, output.Data())
			}
		})
	}
}
