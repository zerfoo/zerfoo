package graph

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// TensorReleaser can release tensors back to a pool for reuse.
type TensorReleaser[T tensor.Numeric] interface {
	Release(t *tensor.TensorNumeric[T])
}

// Graph represents a computation graph with a defined execution order.
type Graph[T tensor.Numeric] struct {
	mu          sync.Mutex
	engine      compute.Engine[T]
	engineProxy *compute.EngineProxy[T]
	nodes        []Node[T]
	dependencies map[Node[T]][]Node[T]
	inputs       []Node[T]
	output       Node[T]
	memo         map[Node[T]]*tensor.TensorNumeric[T]
	parallel     bool
	pool         TensorReleaser[T]

	// Cached decode-time allocations to avoid per-forward GC pressure.
	cachedRefCount   map[Node[T]]int // static reference counts (recomputed on clear)
	cachedNodeInputs []*tensor.TensorNumeric[T] // reusable buffer for node inputs
	maxDeps          int                         // max dependencies per node
}

// SetEngineProxy stores a reference to the EngineProxy used by this graph's layers.
func (g *Graph[T]) SetEngineProxy(proxy *compute.EngineProxy[T]) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.engineProxy = proxy
}

// EngineProxy returns the EngineProxy if one was set, or nil.
func (g *Graph[T]) EngineProxy() *compute.EngineProxy[T] {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.engineProxy
}

// WithParallel enables or disables parallel execution of independent nodes.
// When enabled, Forward delegates to ParallelForward for concurrent execution.
// Default is false (sequential) for backward compatibility.
func (g *Graph[T]) WithParallel(enabled bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.parallel = enabled
}

// WithPool sets a tensor pool for intermediate buffer reuse during Forward.
// When set, the executor releases intermediate tensors back to the pool as
// soon as all their consumers have executed.
func (g *Graph[T]) WithPool(pool TensorReleaser[T]) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.pool = pool
}

// Forward executes the forward pass of the entire graph.
// It is safe for concurrent use; callers will be serialized.
// When parallel mode is enabled via WithParallel(true), independent nodes
// are executed concurrently using a goroutine pool.
func (g *Graph[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if g.parallel {
		return ParallelForward(ctx, g, inputs...)
	}
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	// Reuse memo map: clear and repopulate instead of reallocating.
	if g.memo == nil {
		g.memo = make(map[Node[T]]*tensor.TensorNumeric[T], len(g.nodes))
	} else {
		for k := range g.memo {
			delete(g.memo, k)
		}
	}
	for i, n := range g.inputs {
		g.memo[n] = inputs[i]
	}

	// Build reference counts for pool-based intermediate release.
	// Cache the initial refcount template so we only recompute it once.
	var refCount map[Node[T]]int
	if g.pool != nil {
		if g.cachedRefCount == nil {
			g.cachedRefCount = make(map[Node[T]]int, len(g.nodes))
			for _, n := range g.nodes {
				for _, dep := range g.dependencies[n] {
					g.cachedRefCount[dep]++
				}
			}
			for _, n := range g.inputs {
				g.cachedRefCount[n] = -1
			}
			g.cachedRefCount[g.output] = -1
			for _, n := range g.nodes {
				if isConstantNode[T](n) {
					g.cachedRefCount[n] = -1
				}
			}
		}
		// Copy cached template into working refCount.
		refCount = make(map[Node[T]]int, len(g.cachedRefCount))
		for k, v := range g.cachedRefCount {
			refCount[k] = v
		}
	}

	// Pre-allocate nodeInputs buffer to max dependency count.
	if g.maxDeps == 0 {
		for _, n := range g.nodes {
			if d := len(g.dependencies[n]); d > g.maxDeps {
				g.maxDeps = d
			}
		}
	}
	if cap(g.cachedNodeInputs) < g.maxDeps {
		g.cachedNodeInputs = make([]*tensor.TensorNumeric[T], g.maxDeps)
	}

	for nodeIdx, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}

		deps := g.dependencies[n]
		nodeInputs := g.cachedNodeInputs[:len(deps)]
		for i, dep := range deps {
			nodeInputs[i] = g.memo[dep]
		}

		output, err := n.Forward(ctx, nodeInputs...)
		if err != nil {
			// Include node op type and input shapes for debugging.
			var inputShapes [][]int
			var depOps []string
			for j, dep := range g.dependencies[n] {
				depOps = append(depOps, dep.OpType())
				if j < len(nodeInputs) && nodeInputs[j] != nil {
					inputShapes = append(inputShapes, nodeInputs[j].Shape())
				}
			}
			return nil, fmt.Errorf("node[%d] %s: %w (input shapes: %v, dep ops: %v)", nodeIdx, n.OpType(), err, inputShapes, depOps)
		}

		g.memo[n] = output

		// Debug: log node output for ONNX diagnosis.
		if os.Getenv("ZERFOO_DEBUG_ONNX") == "1" && output != nil {
			opType := n.OpType()
			shape := output.Shape()
			// Log first/last few nodes and any Gather/MatMul/AutoPositionIds/AutoAttentionMask nodes.
			logThis := nodeIdx < 5 || nodeIdx >= len(g.nodes)-3 ||
				opType == "Gather" || opType == "MatMul" || opType == "LMHead" ||
				opType == "AutoPositionIds" || opType == "AutoAttentionMask" || opType == "AutoZeroKVCache"
			if logThis {
				var first5 []float64
				data := output.Data()
				for i := 0; i < len(data) && i < 5; i++ {
					first5 = append(first5, float64(data[i]))
				}
				log.Printf("[DEBUG_ONNX] node[%d] %s: shape=%v first5=%v", nodeIdx, opType, shape, first5)
			}
		}

		// Release intermediate tensors whose consumers are all done.
		if refCount != nil {
			for _, dep := range g.dependencies[n] {
				rc := refCount[dep]
				if rc < 0 {
					continue // protected node
				}
				rc--
				refCount[dep] = rc
				if rc == 0 {
					if t := g.memo[dep]; t != nil {
						g.pool.Release(t)
						delete(g.memo, dep)
					}
				}
			}
		}
	}

	return g.memo[g.output], nil
}

// Backward executes the backward pass of the entire graph.
// It is safe for concurrent use; callers will be serialized.
func (g *Graph[T]) Backward(ctx context.Context, mode types.BackwardMode, initialGradient *tensor.TensorNumeric[T]) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	grads := make(map[Node[T]]*tensor.TensorNumeric[T])
	grads[g.output] = initialGradient

	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		if grad, ok := grads[node]; ok {
			nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[node]))
			for j, dep := range g.dependencies[node] {
				nodeInputs[j] = g.memo[dep]
			}

			inputGrads, err := node.Backward(ctx, mode, grad, nodeInputs...)
			if err != nil {
				return err
			}

			for j, dep := range g.dependencies[node] {
				if existingGrad, ok := grads[dep]; !ok {
					grads[dep] = inputGrads[j]
				} else {
					// Accumulate gradients if multiple paths converge to the same node
					addedGrad, err := g.engine.Add(ctx, existingGrad, inputGrads[j])
					if err != nil {
						return fmt.Errorf("error accumulating gradients: %w", err)
					}

					grads[dep] = addedGrad
				}
			}
		}
	}

	return nil
}

// ClearMemo releases intermediate tensors from the last forward pass.
// Call this after Backward to free GPU device memory between training steps.
// Input tensors and parameter values are not released.
func (g *Graph[T]) ClearMemo() {
	g.mu.Lock()
	defer g.mu.Unlock()

	inputSet := make(map[Node[T]]bool, len(g.inputs))
	for _, n := range g.inputs {
		inputSet[n] = true
	}

	for node, t := range g.memo {
		if inputSet[node] {
			continue // Don't release caller-owned input tensors.
		}
		t.Release()
	}
	g.memo = nil
}

// Parameters returns all the trainable parameters in the graph.
func (g *Graph[T]) Parameters() []*Parameter[T] {
	var params []*Parameter[T]
	for _, node := range g.nodes {
		params = append(params, node.Parameters()...)
	}

	return params
}

// Inputs returns the input nodes of the graph.
func (g *Graph[T]) Inputs() []Node[T] {
	return g.inputs
}

// Output returns the output node of the graph.
func (g *Graph[T]) Output() Node[T] {
	return g.output
}

// ConstantTensors returns all constant/parameter weight tensors in the graph.
// Includes tensors from Parameter/Constant nodes, tensors embedded in nodes
// that implement EmbeddedFrozenProvider (e.g. LM head, gather), and all
// Parameter values from every node (e.g. attention and FFN weights).
// Call after graph construction to collect tensors for GPU pre-upload.
func (g *Graph[T]) ConstantTensors() []*tensor.TensorNumeric[T] {
	ctx := context.Background()
	seen := make(map[*tensor.TensorNumeric[T]]bool)
	var tensors []*tensor.TensorNumeric[T]

	add := func(t *tensor.TensorNumeric[T]) {
		if t != nil && !seen[t] {
			seen[t] = true
			tensors = append(tensors, t)
		}
	}

	for _, n := range g.nodes {
		// Collect from Parameter/Constant nodes.
		if isConstantNode[T](n) {
			t, err := n.Forward(ctx)
			if err != nil || t == nil {
				continue
			}
			add(t)
			continue
		}
		// Collect from EmbeddedFrozenProvider nodes (e.g. LM head weight).
		if efp, ok := n.(EmbeddedFrozenProvider[T]); ok {
			for _, t := range efp.EmbeddedFrozen() {
				add(t)
			}
		}
		// Collect Parameter values from all nodes (attention, FFN weights).
		for _, p := range n.Parameters() {
			add(p.Value)
		}
	}
	return tensors
}

// Nodes returns all the nodes in the graph.
func (g *Graph[T]) Nodes() []Node[T] {
	return g.nodes
}

// Dependencies returns the dependencies of a given node.
func (g *Graph[T]) Dependencies(n Node[T]) []Node[T] {
	return g.dependencies[n]
}

// GetNodeMetadata returns metadata for a specific node including its type, attributes, and shape.
func (g *Graph[T]) GetNodeMetadata(n Node[T]) map[string]interface{} {
	metadata := make(map[string]interface{})
	metadata["op_type"] = n.OpType()
	metadata["output_shape"] = n.OutputShape()
	metadata["attributes"] = n.Attributes()
	metadata["parameter_count"] = len(n.Parameters())
	return metadata
}

// GetDependencies returns the dependency map for all nodes in the graph.
func (g *Graph[T]) GetDependencies() map[Node[T]][]Node[T] {
	// Return a copy to prevent external modification
	deps := make(map[Node[T]][]Node[T])
	for node, nodeDeps := range g.dependencies {
		depsCopy := make([]Node[T], len(nodeDeps))
		copy(depsCopy, nodeDeps)
		deps[node] = depsCopy
	}
	return deps
}

// GetAllNodes returns all nodes in the graph in their current order.
func (g *Graph[T]) GetAllNodes() []Node[T] {
	// Return a copy to prevent external modification
	nodes := make([]Node[T], len(g.nodes))
	copy(nodes, g.nodes)
	return nodes
}

// GetTopologicalOrder returns the nodes in topological order for execution.
func (g *Graph[T]) GetTopologicalOrder() ([]Node[T], error) {
	return topologicalSort(g.nodes, g.dependencies)
}

// inputNode is a special node type for graph inputs.
type inputNode[T tensor.Numeric] struct {
	shape []int
}

func (n *inputNode[T]) OpType() string {
	return "Input"
}

func (n *inputNode[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

func (n *inputNode[T]) OutputShape() []int {
	return n.shape
}

func (n *inputNode[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *inputNode[T]) Backward(_ context.Context, mode types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

func (n *inputNode[T]) Parameters() []*Parameter[T] { return nil }

// Statically assert that the type implements the interface.
var _ Node[float32] = (*inputNode[float32])(nil)

func topologicalSort[T tensor.Numeric](nodes []Node[T], deps map[Node[T]][]Node[T]) ([]Node[T], error) {
	var sorted []Node[T]

	visited := make(map[Node[T]]bool)
	recursionStack := make(map[Node[T]]bool)

	var visit func(node Node[T]) error

	visit = func(node Node[T]) error {
		if recursionStack[node] {
			return errors.New("cycle detected in graph")
		}

		if visited[node] {
			return nil
		}

		recursionStack[node] = true
		visited[node] = true

		for _, dep := range deps[node] {
			if err := visit(dep); err != nil {
				return err
			}
		}

		sorted = append(sorted, node)
		delete(recursionStack, node)

		return nil
	}

	for _, node := range nodes {
		if err := visit(node); err != nil {
			return nil, err
		}
	}

	return sorted, nil
}
