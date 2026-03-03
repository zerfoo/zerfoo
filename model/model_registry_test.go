package model

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// stubSerializer implements ModelSerializer for registry testing.
type stubSerializer[T tensor.Numeric] struct{}

func (s *stubSerializer[T]) Save(_ context.Context, _ ModelInstance[T], _ interface{}) error {
	return nil
}
func (s *stubSerializer[T]) Load(_ context.Context, _ interface{}) (ModelInstance[T], error) {
	return nil, nil
}
func (s *stubSerializer[T]) GetSupportedFormats() []string     { return []string{"zmf"} }
func (s *stubSerializer[T]) GetSerializerInfo() SerializerInfo { return SerializerInfo{Name: "stub"} }

// stubOptimizer implements ModelOptimizer for registry testing.
type stubOptimizer[T tensor.Numeric] struct{}

func (o *stubOptimizer[T]) OptimizeModel(_ context.Context, model ModelInstance[T], _ OptimizationConfig) (ModelInstance[T], error) {
	return model, nil
}
func (o *stubOptimizer[T]) GetOptimizations() []OptimizationStrategy { return nil }
func (o *stubOptimizer[T]) GetOptimizerInfo() OptimizerInfo          { return OptimizerInfo{Name: "stub"} }

// --- Serializer registration ---

func TestModelRegistry_Serializer(t *testing.T) {
	reg := NewModelRegistry[float32]()
	ctx := context.Background()

	factory := func(_ context.Context, _ map[string]any) (ModelSerializer[float32], error) {
		return &stubSerializer[float32]{}, nil
	}

	// Register
	if err := reg.RegisterModelSerializer("s1", factory); err != nil {
		t.Fatalf("RegisterModelSerializer failed: %v", err)
	}

	// Duplicate
	if err := reg.RegisterModelSerializer("s1", factory); err == nil {
		t.Error("expected error for duplicate serializer registration")
	}

	// Get success
	s, err := reg.GetModelSerializer(ctx, "s1", nil)
	if err != nil {
		t.Fatalf("GetModelSerializer failed: %v", err)
	}
	if s == nil {
		t.Error("expected non-nil serializer")
	}

	// Get not found
	_, err = reg.GetModelSerializer(ctx, "missing", nil)
	if err == nil {
		t.Error("expected error for missing serializer")
	}

	// List
	names := reg.ListModelSerializers()
	if len(names) != 1 || names[0] != "s1" {
		t.Errorf("ListModelSerializers: got %v, want [s1]", names)
	}

	// Unregister
	reg.UnregisterModelSerializer("s1")
	names = reg.ListModelSerializers()
	if len(names) != 0 {
		t.Errorf("expected 0 serializers after unregister, got %d", len(names))
	}
}

// --- Loader registration ---

func TestModelRegistry_Loader(t *testing.T) {
	reg := NewModelRegistry[float32]()
	ctx := context.Background()

	factory := func(_ context.Context, _ map[string]any) (ModelLoader[float32], error) {
		return nil, nil
	}

	if err := reg.RegisterModelLoader("l1", factory); err != nil {
		t.Fatalf("RegisterModelLoader failed: %v", err)
	}
	if err := reg.RegisterModelLoader("l1", factory); err == nil {
		t.Error("expected error for duplicate loader registration")
	}

	_, err := reg.GetModelLoader(ctx, "l1", nil)
	if err != nil {
		t.Fatalf("GetModelLoader failed: %v", err)
	}
	_, err = reg.GetModelLoader(ctx, "missing", nil)
	if err == nil {
		t.Error("expected error for missing loader")
	}

	names := reg.ListModelLoaders()
	if len(names) != 1 {
		t.Errorf("ListModelLoaders: got %v, want 1 entry", names)
	}

	reg.UnregisterModelLoader("l1")
	if len(reg.ListModelLoaders()) != 0 {
		t.Error("expected 0 loaders after unregister")
	}
}

// --- Exporter, Validator, Optimizer registration ---

// registryComponentOps holds the register/get/list/unregister operations for a component type.
type registryComponentOps struct {
	register    func(*ModelRegistry[float32]) error
	registerDup func(*ModelRegistry[float32]) error
	get         func(*ModelRegistry[float32]) error
	getMissing  func(*ModelRegistry[float32]) error
	list        func(*ModelRegistry[float32]) int
	unregister  func(*ModelRegistry[float32])
}

// assertRegistryCRUD runs the standard register/duplicate/get/missing/list/unregister
// cycle for any registry component type.
func assertRegistryCRUD(t *testing.T, ops registryComponentOps) {
	t.Helper()
	reg := NewModelRegistry[float32]()

	if err := ops.register(reg); err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	if err := ops.registerDup(reg); err == nil {
		t.Error("expected error for duplicate registration")
	}
	if err := ops.get(reg); err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if err := ops.getMissing(reg); err == nil {
		t.Error("expected error for missing component")
	}
	if n := ops.list(reg); n != 1 {
		t.Errorf("List: got %d, want 1", n)
	}
	ops.unregister(reg)
	if n := ops.list(reg); n != 0 {
		t.Errorf("List after unregister: got %d, want 0", n)
	}
}

func TestModelRegistry_Exporter(t *testing.T) { //nolint:dupl // registry CRUD pattern, structurally similar by design
	assertRegistryCRUD(t, registryComponentOps{
		register: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelExporter("e1", func(_ context.Context, _ map[string]any) (ModelExporter[float32], error) {
				return NewZMFModelExporter[float32](), nil
			})
		},
		registerDup: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelExporter("e1", func(_ context.Context, _ map[string]any) (ModelExporter[float32], error) {
				return nil, nil
			})
		},
		get: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelExporter(context.Background(), "e1", nil)
			return err
		},
		getMissing: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelExporter(context.Background(), "missing", nil)
			return err
		},
		list:       func(r *ModelRegistry[float32]) int { return len(r.ListModelExporters()) },
		unregister: func(r *ModelRegistry[float32]) { r.UnregisterModelExporter("e1") },
	})
}

func TestModelRegistry_Validator(t *testing.T) { //nolint:dupl // registry CRUD pattern, structurally similar by design
	assertRegistryCRUD(t, registryComponentOps{
		register: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelValidator("v1", func(_ context.Context, _ map[string]any) (ModelValidator[float32], error) {
				return NewBasicModelValidator[float32](), nil
			})
		},
		registerDup: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelValidator("v1", func(_ context.Context, _ map[string]any) (ModelValidator[float32], error) {
				return nil, nil
			})
		},
		get: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelValidator(context.Background(), "v1", nil)
			return err
		},
		getMissing: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelValidator(context.Background(), "missing", nil)
			return err
		},
		list:       func(r *ModelRegistry[float32]) int { return len(r.ListModelValidators()) },
		unregister: func(r *ModelRegistry[float32]) { r.UnregisterModelValidator("v1") },
	})
}

func TestModelRegistry_Optimizer(t *testing.T) {
	assertRegistryCRUD(t, registryComponentOps{
		register: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelOptimizer("o1", func(_ context.Context, _ map[string]any) (ModelOptimizer[float32], error) {
				return &stubOptimizer[float32]{}, nil
			})
		},
		registerDup: func(r *ModelRegistry[float32]) error {
			return r.RegisterModelOptimizer("o1", func(_ context.Context, _ map[string]any) (ModelOptimizer[float32], error) {
				return nil, nil
			})
		},
		get: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelOptimizer(context.Background(), "o1", nil)
			return err
		},
		getMissing: func(r *ModelRegistry[float32]) error {
			_, err := r.GetModelOptimizer(context.Background(), "missing", nil)
			return err
		},
		list:       func(r *ModelRegistry[float32]) int { return len(r.ListModelOptimizers()) },
		unregister: func(r *ModelRegistry[float32]) { r.UnregisterModelOptimizer("o1") },
	})
}

// --- Unregister for Provider ---

func TestModelRegistry_UnregisterProvider(t *testing.T) {
	reg := NewModelRegistry[float32]()
	factory := func(_ context.Context, _ map[string]any) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](), nil
	}
	_ = reg.RegisterModelProvider("p1", factory)

	reg.UnregisterModelProvider("p1")
	if len(reg.ListModelProviders()) != 0 {
		t.Error("expected 0 providers after unregister")
	}
}

// --- GetModelProvider not found ---

func TestModelRegistry_GetModelProvider_NotFound(t *testing.T) {
	reg := NewModelRegistry[float32]()
	_, err := reg.GetModelProvider(context.Background(), "nonexistent", nil)
	if err == nil {
		t.Error("expected error for nonexistent provider")
	}
}

// --- FindProviderByCapability ---

func TestModelRegistry_FindProviderByCapability(t *testing.T) {
	reg := NewModelRegistry[float32]()
	ctx := context.Background()

	// Register a provider that supports "mock" and "test" types
	_ = reg.RegisterModelProvider("mockProv", func(_ context.Context, _ map[string]any) (ModelProvider[float32], error) {
		return NewMockModelProvider[float32](), nil
	})

	// Register a provider that returns an error on instantiation (should be skipped)
	_ = reg.RegisterModelProvider("badProv", func(_ context.Context, _ map[string]any) (ModelProvider[float32], error) {
		return nil, fmt.Errorf("factory error")
	})

	// Find matching capability
	matches, err := reg.FindProviderByCapability(ctx, "mock")
	if err != nil {
		t.Fatalf("FindProviderByCapability failed: %v", err)
	}
	if len(matches) != 1 || matches[0] != "mockProv" {
		t.Errorf("expected [mockProv], got %v", matches)
	}

	// No matches
	matches, err = reg.FindProviderByCapability(ctx, "nonexistent_capability")
	if err != nil {
		t.Fatalf("FindProviderByCapability failed: %v", err)
	}
	if len(matches) != 0 {
		t.Errorf("expected no matches, got %v", matches)
	}
}

// --- GetAllRegistrations with all component types ---

func TestModelRegistry_GetAllRegistrations_Full(t *testing.T) {
	reg := NewModelRegistry[float32]()

	_ = reg.RegisterModelProvider("p", func(_ context.Context, _ map[string]any) (ModelProvider[float32], error) {
		return nil, nil
	})
	_ = reg.RegisterModelSerializer("s", func(_ context.Context, _ map[string]any) (ModelSerializer[float32], error) {
		return nil, nil
	})
	_ = reg.RegisterModelLoader("l", func(_ context.Context, _ map[string]any) (ModelLoader[float32], error) {
		return nil, nil
	})
	_ = reg.RegisterModelExporter("e", func(_ context.Context, _ map[string]any) (ModelExporter[float32], error) {
		return nil, nil
	})
	_ = reg.RegisterModelValidator("v", func(_ context.Context, _ map[string]any) (ModelValidator[float32], error) {
		return nil, nil
	})
	_ = reg.RegisterModelOptimizer("o", func(_ context.Context, _ map[string]any) (ModelOptimizer[float32], error) {
		return nil, nil
	})

	regs := reg.GetAllRegistrations()
	checks := map[string]string{
		"providers": "p", "serializers": "s", "loaders": "l",
		"exporters": "e", "validators": "v", "optimizers": "o",
	}
	for key, wantName := range checks {
		names := regs[key]
		if len(names) != 1 || names[0] != wantName {
			t.Errorf("GetAllRegistrations[%s]: got %v, want [%s]", key, names, wantName)
		}
	}

	summary := reg.Summary()
	for key := range checks {
		if summary[key] != 1 {
			t.Errorf("Summary[%s] = %d, want 1", key, summary[key])
		}
	}
}

// --- Ensure stubSerializer and stubOptimizer satisfy interfaces at compile time ---

var _ ModelSerializer[float32] = (*stubSerializer[float32])(nil)
var _ ModelOptimizer[float32] = (*stubOptimizer[float32])(nil)

// --- Layer registry tests ---

func TestUnregisterLayer(t *testing.T) {
	RegisterLayer("TestUnregOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		return nil, nil
	})

	// Should succeed
	_, err := GetLayerBuilder[float32]("TestUnregOp")
	if err != nil {
		t.Fatalf("expected layer builder to exist: %v", err)
	}

	UnregisterLayer("TestUnregOp")

	_, err = GetLayerBuilder[float32]("TestUnregOp")
	if err == nil {
		t.Error("expected error after unregistering layer")
	}
}

func TestGetLayerBuilder_NotFound(t *testing.T) {
	_, err := GetLayerBuilder[float32]("NoSuchOp_12345")
	if err == nil {
		t.Error("expected error for non-existent op type")
	}
}
