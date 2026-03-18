package inference

import (
	"fmt"
	"sort"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// ArchBuilder builds a computation graph for a model architecture from
// pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
type ArchBuilder func(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)

// archRegistry holds the global architecture registry.
var archRegistry = struct {
	mu       sync.RWMutex
	builders map[string]ArchBuilder
}{
	builders: make(map[string]ArchBuilder),
}

// RegisterArchitecture registers an architecture builder under the given name.
// Names correspond to GGUF general.architecture values (e.g. "llama", "gemma").
// Multiple names can map to the same builder (e.g. "gemma" and "gemma3").
// Panics if name is empty or a builder is already registered for that name.
func RegisterArchitecture(name string, builder ArchBuilder) {
	if name == "" {
		panic("inference: RegisterArchitecture called with empty name")
	}
	if builder == nil {
		panic("inference: RegisterArchitecture called with nil builder")
	}
	archRegistry.mu.Lock()
	defer archRegistry.mu.Unlock()
	if _, dup := archRegistry.builders[name]; dup {
		panic(fmt.Sprintf("inference: RegisterArchitecture called twice for %q", name))
	}
	archRegistry.builders[name] = builder
}

// GetArchitecture returns the builder registered for the given architecture
// name. Returns nil, false if no builder is registered.
func GetArchitecture(name string) (ArchBuilder, bool) {
	archRegistry.mu.RLock()
	defer archRegistry.mu.RUnlock()
	b, ok := archRegistry.builders[name]
	return b, ok
}

// ListArchitectures returns a sorted list of all registered architecture names.
func ListArchitectures() []string {
	archRegistry.mu.RLock()
	defer archRegistry.mu.RUnlock()
	names := make([]string, 0, len(archRegistry.builders))
	for name := range archRegistry.builders {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
