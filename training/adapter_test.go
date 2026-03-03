package training

import (
	"context"
	"errors"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// mockTrainer implements Trainer for testing
type mockTrainer[T tensor.Numeric] struct {
	stepErr bool
}

func (m *mockTrainer[T]) TrainStep(ctx context.Context, g *graph.Graph[T], opt optimizer.Optimizer[T], inputs map[graph.Node[T]]*tensor.TensorNumeric[T], targets *tensor.TensorNumeric[T]) (T, error) {
	if m.stepErr {
		return T(0), errors.New("train step error")
	}
	return T(1), nil
}

// mockGradientStrategy implements GradientStrategy for testing
type mockGradientStrategy[T tensor.Numeric] struct {
	loss T
	err  error
}

func (m *mockGradientStrategy[T]) ComputeGradients(ctx context.Context, g *graph.Graph[T], loss graph.Node[T], batch Batch[T]) (T, error) {
	return m.loss, m.err
}

// mockOpt implements optimizer.Optimizer for testing
type mockOpt[T tensor.Numeric] struct{}

func (m *mockOpt[T]) Step(ctx context.Context, params []*graph.Parameter[T]) error {
	return nil
}

func TestNewTrainerWorkflowAdapter(t *testing.T) {
	trainer := &mockTrainer[float32]{}
	opt := &mockOpt[float32]{}
	adapter := NewTrainerWorkflowAdapter[float32](trainer, opt)

	if adapter == nil {
		t.Fatal("NewTrainerWorkflowAdapter returned nil")
	}
	if adapter.metrics == nil {
		t.Error("metrics map is nil")
	}
}

func TestTrainerWorkflowAdapter_Initialize(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})

	config := WorkflowConfig{
		NumEpochs:    5,
		LearningRate: 0.01,
	}

	err := adapter.Initialize(context.Background(), config)
	if err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}

	if adapter.config.NumEpochs != 5 {
		t.Errorf("config.NumEpochs = %d, want 5", adapter.config.NumEpochs)
	}
}

func TestTrainerWorkflowAdapter_Train(t *testing.T) {
	trainer := &mockTrainer[float32]{}
	opt := &mockOpt[float32]{}
	adapter := NewTrainerWorkflowAdapter[float32](trainer, opt)
	ctx := context.Background()

	config := WorkflowConfig{NumEpochs: 2}
	_ = adapter.Initialize(ctx, config)

	// Create data with batches
	batch := &Batch[float32]{
		Inputs:  make(map[graph.Node[float32]]*tensor.TensorNumeric[float32]),
		Targets: nil,
	}
	dp := NewMockDataProvider[float32]([]*Batch[float32]{batch}, nil)
	mp := NewMockModelProvider[float32](nil)

	result, err := adapter.Train(ctx, dp, mp)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}

	if result.TotalEpochs != 2 {
		t.Errorf("TotalEpochs = %d, want 2", result.TotalEpochs)
	}
}

func TestTrainerWorkflowAdapter_GetMetrics(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	metrics := adapter.GetMetrics()
	if metrics == nil {
		t.Error("GetMetrics returned nil")
	}
}

func TestTrainerWorkflowAdapter_Shutdown(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	adapter.metrics["test"] = 1.0

	err := adapter.Shutdown(context.Background())
	if err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}

	if len(adapter.metrics) != 0 {
		t.Error("metrics should be cleared after shutdown")
	}
}

func TestTrainerWorkflowAdapter_Validate(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	ctx := context.Background()
	_ = adapter.Initialize(ctx, WorkflowConfig{})

	batch := &Batch[float32]{
		Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32]),
	}
	dp := NewMockDataProvider[float32](nil, []*Batch[float32]{batch})
	mp := NewMockModelProvider[float32](nil)

	result, err := adapter.Validate(ctx, dp, mp)
	if err != nil {
		t.Fatalf("Validate failed: %v", err)
	}

	if result.SampleCount != 1 {
		t.Errorf("SampleCount = %d, want 1", result.SampleCount)
	}
}

func TestDataIteratorAdapter_Error(t *testing.T) {
	adapter := NewDataIteratorAdapter[float32](nil)
	if err := adapter.Error(); err != nil {
		t.Errorf("Error() = %v, want nil", err)
	}
}

func TestDataIteratorAdapter_BatchOutOfBounds(t *testing.T) {
	adapter := NewDataIteratorAdapter[float32](nil)
	// Before any Next() call, currentIdx = -1
	if b := adapter.Batch(); b != nil {
		t.Error("Batch() before Next() should return nil")
	}
}

func TestNewSimpleModelProvider(t *testing.T) {
	factory := func(ctx context.Context, config ModelConfig) (*graph.Graph[float32], error) {
		return nil, nil
	}
	info := ModelInfo{Name: "test", Version: "1.0"}

	sp := NewSimpleModelProvider[float32](factory, info)
	if sp == nil {
		t.Fatal("NewSimpleModelProvider returned nil")
	}
}

func TestSimpleModelProvider_CreateModel(t *testing.T) {
	factory := func(ctx context.Context, config ModelConfig) (*graph.Graph[float32], error) {
		return nil, nil
	}
	sp := NewSimpleModelProvider[float32](factory, ModelInfo{})

	_, err := sp.CreateModel(context.Background(), ModelConfig{})
	if err != nil {
		t.Fatalf("CreateModel failed: %v", err)
	}
}

func TestSimpleModelProvider_CreateModel_NilFactory(t *testing.T) {
	sp := NewSimpleModelProvider[float32](nil, ModelInfo{})

	_, err := sp.CreateModel(context.Background(), ModelConfig{})
	if err == nil {
		t.Error("CreateModel with nil factory should fail")
	}
}

func TestSimpleModelProvider_LoadModel(t *testing.T) {
	sp := NewSimpleModelProvider[float32](nil, ModelInfo{})

	_, err := sp.LoadModel(context.Background(), "test.model")
	if err == nil {
		t.Error("LoadModel should return not-implemented error")
	}
}

func TestSimpleModelProvider_SaveModel(t *testing.T) {
	sp := NewSimpleModelProvider[float32](nil, ModelInfo{})

	err := sp.SaveModel(context.Background(), nil, "test.model")
	if err == nil {
		t.Error("SaveModel should return not-implemented error")
	}
}

func TestSimpleModelProvider_GetModelInfo(t *testing.T) {
	info := ModelInfo{Name: "test", Version: "2.0", Architecture: "custom"}
	sp := NewSimpleModelProvider[float32](nil, info)

	got := sp.GetModelInfo()
	if got.Name != "test" || got.Version != "2.0" || got.Architecture != "custom" {
		t.Errorf("GetModelInfo() = %+v, want %+v", got, info)
	}
}

func TestNewGradientStrategyAdapter(t *testing.T) {
	strategy := NewDefaultBackpropStrategy[float32]()
	adapter := NewGradientStrategyAdapter[float32](strategy, nil, nil)

	if adapter == nil {
		t.Fatal("NewGradientStrategyAdapter returned nil")
	}
}

func TestTrainerWorkflowAdapter_Validate_EmptyData(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	ctx := context.Background()
	_ = adapter.Initialize(ctx, WorkflowConfig{})

	dp := NewMockDataProvider[float32](nil, nil)
	mp := NewMockModelProvider[float32](nil)

	result, err := adapter.Validate(ctx, dp, mp)
	if err != nil {
		t.Fatalf("Validate failed: %v", err)
	}
	if result.SampleCount != 0 {
		t.Errorf("SampleCount = %d, want 0", result.SampleCount)
	}
}

func TestTrainerWorkflowAdapter_Train_EmptyData(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	ctx := context.Background()
	_ = adapter.Initialize(ctx, WorkflowConfig{NumEpochs: 1})

	dp := NewMockDataProvider[float32](nil, nil)
	mp := NewMockModelProvider[float32](nil)

	result, err := adapter.Train(ctx, dp, mp)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if result.TotalEpochs != 1 {
		t.Errorf("TotalEpochs = %d, want 1", result.TotalEpochs)
	}
}

func TestTrainerWorkflowAdapter_Train_TrainStepError(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{stepErr: true}, &mockOpt[float32]{})
	ctx := context.Background()
	_ = adapter.Initialize(ctx, WorkflowConfig{NumEpochs: 1})

	batch := &Batch[float32]{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])}
	dp := NewMockDataProvider[float32]([]*Batch[float32]{batch}, nil)
	mp := NewMockModelProvider[float32](nil)

	_, err := adapter.Train(ctx, dp, mp)
	if err == nil {
		t.Error("Train should fail when trainer step fails")
	}
}

func TestGradientStrategyAdapter_ComputeGradientsFromBatch(t *testing.T) {
	// Use a mock strategy that returns a fixed loss
	strategy := &mockGradientStrategy[float32]{loss: 0.5}
	adapter := NewGradientStrategyAdapter[float32](strategy, nil, nil)

	batch := &Batch[float32]{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])}
	loss, err := adapter.ComputeGradientsFromBatch(context.Background(), batch)
	if err != nil {
		t.Fatalf("ComputeGradientsFromBatch failed: %v", err)
	}
	if loss != 0.5 {
		t.Errorf("loss = %v, want 0.5", loss)
	}
}

func TestTrainerWorkflowAdapter_Train_MultipleBatches(t *testing.T) {
	adapter := NewTrainerWorkflowAdapter[float32](&mockTrainer[float32]{}, &mockOpt[float32]{})
	ctx := context.Background()
	_ = adapter.Initialize(ctx, WorkflowConfig{NumEpochs: 2})

	b1 := &Batch[float32]{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])}
	b2 := &Batch[float32]{Inputs: make(map[graph.Node[float32]]*tensor.TensorNumeric[float32])}
	dp := NewMockDataProvider[float32]([]*Batch[float32]{b1, b2}, nil)
	mp := NewMockModelProvider[float32](nil)

	result, err := adapter.Train(ctx, dp, mp)
	if err != nil {
		t.Fatalf("Train failed: %v", err)
	}
	if result.TotalEpochs != 2 {
		t.Errorf("TotalEpochs = %d, want 2", result.TotalEpochs)
	}
	// With 2 epochs, bestEpoch should be 0 (first epoch always best initially)
	if result.BestEpoch != 0 {
		t.Errorf("BestEpoch = %d, want 0", result.BestEpoch)
	}
}
