package importer

import (
	"context"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zmf"
	"github.com/zerfoo/zerfoo/tensor"
	"google.golang.org/protobuf/proto"
)

// mockEngine is a simple mock of the compute.Engine for testing purposes.
type mockEngine[T tensor.Numeric] struct{}

func (m *mockEngine[T]) Mul(ctx context.Context, a, b *tensor.Tensor[T], out *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return tensor.New[T](a.Shape(), nil)
}
func (m *mockEngine[T]) Add(ctx context.Context, a, b *tensor.Tensor[T], out *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	return tensor.New[T](a.Shape(), nil)
}
// Add other necessary mock methods...

func TestLoadModel_Comprehensive(t *testing.T) {
	// 1. Create a mock ZMF model protobuf
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Nodes: []*zmf.Node{
				{
					Name:   "token_embedding_1",
					OpType: "TokenEmbedding",
					Inputs: []string{"embedding_table"},
				},
				{
					Name:   "rmsnorm_1",
					OpType: "RMSNorm",
					Inputs: []string{"rmsnorm_1_gain"},
					Attributes: map[string]*zmf.Attribute{
						"epsilon": {Value: &zmf.Attribute_F{F: 1e-6}},
					},
				},
				{
					Name:   "dense_1",
					OpType: "Dense",
					Inputs: []string{"dense_1_weights", "dense_1_bias"},
				},
				{
					Name:   "relu_1",
					OpType: "ReLU",
				},
			},
			Parameters: map[string]*zmf.Tensor{
				"embedding_table": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{1000, 128},
					Data:  make([]byte, 1000*128*4),
				},
				"rmsnorm_1_gain": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{128},
					Data:  make([]byte, 128*4),
				},
				"dense_1_weights": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{128, 256},
					Data:  make([]byte, 128*256*4),
				},
				"dense_1_bias": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{256},
					Data:  make([]byte, 256*4),
				},
			},
		},
	}

	modelBytes, err := proto.Marshal(zmfModel)
	if err != nil {
		t.Fatalf("Failed to marshal mock model: %v", err)
	}

	// 2. Write the mock model to a temporary file
	tmpfile, err := os.CreateTemp("", "test_model.zmf")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write(modelBytes); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}

	// 3. Call LoadModel
	engine := &compute.CPUEngine[float32]{}
	ops := &numeric.Float32Ops{}
	model, err := LoadModel[float32](tmpfile.Name(), engine, ops)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}

	// 4. Verify the model
	if model == nil {
		t.Fatal("LoadModel returned a nil model")
	}
	if model.Embedding == nil {
		t.Fatal("Model embedding is nil")
	}
	if model.Graph == nil {
		t.Fatal("Model graph is nil")
	}
}
