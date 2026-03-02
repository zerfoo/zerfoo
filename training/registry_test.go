package training

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

func TestPluginRegistry_AllProviderTypes(t *testing.T) {
	reg := NewPluginRegistry[float32]()
	ctx := context.Background()

	// Register all provider types
	_ = reg.RegisterModelProvider("mp", func(ctx context.Context, config map[string]any) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](nil), nil
	})
	_ = reg.RegisterSequenceProvider("sp", func(ctx context.Context, config map[string]any) (SequenceProvider[float32], error) {
		return &mockSequenceProvider[float32]{}, nil
	})
	_ = reg.RegisterMetricComputer("mc", func(ctx context.Context, config map[string]any) (MetricComputer[float32], error) {
		return &mockMetricComputer[float32]{}, nil
	})
	_ = reg.RegisterCrossValidator("cv", func(ctx context.Context, config map[string]any) (CrossValidator[float32], error) {
		return &mockCrossValidator[float32]{}, nil
	})

	// Verify summary
	summary := reg.Summary()
	for _, key := range []string{"modelProviders", "sequenceProviders", "metricComputers", "crossValidators"} {
		if summary[key] != 1 {
			t.Errorf("summary[%s] = %d, want 1", key, summary[key])
		}
	}

	// Test duplicate registration errors
	t.Run("duplicate model provider", func(t *testing.T) {
		err := reg.RegisterModelProvider("mp", func(ctx context.Context, config map[string]any) (ModelProvider[float32], error) {
			return nil, nil
		})
		if err == nil {
			t.Error("expected error")
		}
	})
	t.Run("duplicate sequence provider", func(t *testing.T) {
		err := reg.RegisterSequenceProvider("sp", func(ctx context.Context, config map[string]any) (SequenceProvider[float32], error) {
			return nil, nil
		})
		if err == nil {
			t.Error("expected error")
		}
	})
	t.Run("duplicate metric computer", func(t *testing.T) {
		err := reg.RegisterMetricComputer("mc", func(ctx context.Context, config map[string]any) (MetricComputer[float32], error) {
			return nil, nil
		})
		if err == nil {
			t.Error("expected error")
		}
	})
	t.Run("duplicate cross validator", func(t *testing.T) {
		err := reg.RegisterCrossValidator("cv", func(ctx context.Context, config map[string]any) (CrossValidator[float32], error) {
			return nil, nil
		})
		if err == nil {
			t.Error("expected error")
		}
	})

	// Test retrieval
	t.Run("get model provider", func(t *testing.T) {
		mp, err := reg.GetModelProvider(ctx, "mp", nil)
		if err != nil || mp == nil {
			t.Errorf("GetModelProvider failed: %v", err)
		}
	})
	t.Run("get sequence provider", func(t *testing.T) {
		sp, err := reg.GetSequenceProvider(ctx, "sp", nil)
		if err != nil || sp == nil {
			t.Errorf("GetSequenceProvider failed: %v", err)
		}
	})
	t.Run("get metric computer", func(t *testing.T) {
		mc, err := reg.GetMetricComputer(ctx, "mc", nil)
		if err != nil || mc == nil {
			t.Errorf("GetMetricComputer failed: %v", err)
		}
	})
	t.Run("get cross validator", func(t *testing.T) {
		cv, err := reg.GetCrossValidator(ctx, "cv", nil)
		if err != nil || cv == nil {
			t.Errorf("GetCrossValidator failed: %v", err)
		}
	})

	// Test not found errors
	t.Run("model provider not found", func(t *testing.T) {
		if _, err := reg.GetModelProvider(ctx, "x", nil); err == nil {
			t.Error("expected error")
		}
	})
	t.Run("sequence provider not found", func(t *testing.T) {
		if _, err := reg.GetSequenceProvider(ctx, "x", nil); err == nil {
			t.Error("expected error")
		}
	})
	t.Run("metric computer not found", func(t *testing.T) {
		if _, err := reg.GetMetricComputer(ctx, "x", nil); err == nil {
			t.Error("expected error")
		}
	})
	t.Run("cross validator not found", func(t *testing.T) {
		if _, err := reg.GetCrossValidator(ctx, "x", nil); err == nil {
			t.Error("expected error")
		}
	})

	// Test listing
	if list := reg.ListModelProviders(); len(list) != 1 {
		t.Errorf("ListModelProviders len = %d, want 1", len(list))
	}
	if list := reg.ListSequenceProviders(); len(list) != 1 {
		t.Errorf("ListSequenceProviders len = %d, want 1", len(list))
	}
	if list := reg.ListMetricComputers(); len(list) != 1 {
		t.Errorf("ListMetricComputers len = %d, want 1", len(list))
	}
	if list := reg.ListCrossValidators(); len(list) != 1 {
		t.Errorf("ListCrossValidators len = %d, want 1", len(list))
	}

	// Test unregister
	reg.UnregisterModelProvider("mp")
	reg.UnregisterSequenceProvider("sp")
	reg.UnregisterMetricComputer("mc")
	reg.UnregisterCrossValidator("cv")

	if list := reg.ListModelProviders(); len(list) != 0 {
		t.Error("expected empty after unregister")
	}
	if list := reg.ListSequenceProviders(); len(list) != 0 {
		t.Error("expected empty after unregister")
	}
	if list := reg.ListMetricComputers(); len(list) != 0 {
		t.Error("expected empty after unregister")
	}
	if list := reg.ListCrossValidators(); len(list) != 0 {
		t.Error("expected empty after unregister")
	}
}

func TestPluginRegistry_UnregisterWorkflow(t *testing.T) {
	reg := NewPluginRegistry[float32]()

	_ = reg.RegisterWorkflow("test", func(ctx context.Context, config map[string]any) (TrainingWorkflow[float32], error) {
		return NewMockTrainingWorkflow[float32](), nil
	})
	reg.UnregisterWorkflow("test")

	if len(reg.ListWorkflows()) != 0 {
		t.Error("expected 0 workflows after unregister")
	}
}

func TestPluginRegistry_UnregisterDataProvider(t *testing.T) {
	reg := NewPluginRegistry[float32]()

	_ = reg.RegisterDataProvider("test", func(ctx context.Context, config map[string]any) (DataProvider[float32], error) {
		return NewMockDataProvider[float32](nil, nil), nil
	})
	reg.UnregisterDataProvider("test")

	if len(reg.ListDataProviders()) != 0 {
		t.Error("expected 0 data providers after unregister")
	}
}

func TestPluginRegistry_DataProviderDuplicate(t *testing.T) {
	reg := NewPluginRegistry[float32]()

	factory := func(ctx context.Context, config map[string]any) (DataProvider[float32], error) {
		return NewMockDataProvider[float32](nil, nil), nil
	}

	_ = reg.RegisterDataProvider("test", factory)
	if err := reg.RegisterDataProvider("test", factory); err == nil {
		t.Error("expected error for duplicate data provider")
	}
}

func TestPluginRegistry_DataProviderNotFound(t *testing.T) {
	reg := NewPluginRegistry[float32]()
	_, err := reg.GetDataProvider(context.Background(), "nonexistent", nil)
	if err == nil {
		t.Error("expected error for nonexistent data provider")
	}
}

func TestPluginRegistry_ListDataProviders(t *testing.T) {
	reg := NewPluginRegistry[float32]()
	factory := func(ctx context.Context, config map[string]any) (DataProvider[float32], error) {
		return NewMockDataProvider[float32](nil, nil), nil
	}

	_ = reg.RegisterDataProvider("dp1", factory)
	_ = reg.RegisterDataProvider("dp2", factory)

	list := reg.ListDataProviders()
	if len(list) != 2 {
		t.Errorf("ListDataProviders len = %d, want 2", len(list))
	}
}

// Mock implementations for testing

type mockSequenceProvider[T tensor.Numeric] struct{}

func (m *mockSequenceProvider[T]) GenerateSequences(ctx context.Context, dataset DataProvider[T], config SequenceConfig) ([]DataProvider[T], error) {
	return nil, nil
}

func (m *mockSequenceProvider[T]) GenerateTrainValidationSplit(ctx context.Context, dataset DataProvider[T], config SplitConfig) (DataProvider[T], DataProvider[T], error) {
	return nil, nil, nil
}

func (m *mockSequenceProvider[T]) SetRandomSeed(seed uint64) {}

type mockMetricComputer[T tensor.Numeric] struct{}

func (m *mockMetricComputer[T]) ComputeMetrics(ctx context.Context, predictions, targets *tensor.TensorNumeric[T], metadata map[string]any) (map[string]float64, error) {
	return nil, nil
}

func (m *mockMetricComputer[T]) RegisterMetric(name string, metric MetricFunction[T]) {}
func (m *mockMetricComputer[T]) UnregisterMetric(name string)                         {}
func (m *mockMetricComputer[T]) AvailableMetrics() []string                           { return nil }

type mockCrossValidator[T tensor.Numeric] struct{}

func (m *mockCrossValidator[T]) CreateFolds(ctx context.Context, dataset DataProvider[T], config CVConfig) ([]Fold[T], error) {
	return nil, nil
}

func (m *mockCrossValidator[T]) ValidateModel(ctx context.Context, dataset DataProvider[T], modelProvider ModelProvider[T], config CVConfig) (*CVResult[T], error) {
	return nil, nil
}
