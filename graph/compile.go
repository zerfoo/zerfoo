package graph

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// Instruction is a single pre-resolved operation in a compiled execution plan.
// It holds a direct function that calls node.Forward() with pre-computed
// buffer indices, eliminating dependency map lookups and memo operations.
type Instruction[T tensor.Numeric] struct {
	Forward   func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
	InputIdx  []int  // indices into the slot array
	OutputIdx int    // index into the slot array
	OpName    string // for error reporting
}

// ExecutionPlan is a compiled, flat instruction sequence that replaces the
// interpreted node-by-node Forward() loop. Node outputs are stored in an
// indexed slot array instead of a map, eliminating map lookups.
type ExecutionPlan[T tensor.Numeric] struct {
	instructions  []Instruction[T]
	slots         []*tensor.TensorNumeric[T] // indexed output storage
	slotShapes    [][]int                    // shapes from warmup pass
	inputIdx      []int                      // which slots receive graph inputs
	outputIdx     int                        // which slot holds the final output
	frozenIdx     []int                      // slots holding frozen data (params)
	megakernelFn  atomic.Value               // stores func(context.Context, []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) or nil
}

// InstructionMeta is the exported metadata for a single compiled instruction.
// It contains everything needed by a code generator without exposing the
// Forward() closure.
type InstructionMeta struct {
	OpName    string // operation type (e.g. "Add", "MatMulNBits", "RMSNorm")
	InputIdx  []int  // slot indices for inputs
	OutputIdx int    // slot index for the output
}

// FrozenSlot describes a slot that holds frozen (constant) data such as
// model weights. The Data field holds the tensor from the warmup pass.
type FrozenSlot[T tensor.Numeric] struct {
	SlotIdx int
	Data    *tensor.TensorNumeric[T]
}

// Instructions returns exported metadata for each compute instruction in
// the plan. The order matches the execution order.
func (p *ExecutionPlan[T]) Instructions() []InstructionMeta {
	metas := make([]InstructionMeta, len(p.instructions))
	for i, inst := range p.instructions {
		idx := make([]int, len(inst.InputIdx))
		copy(idx, inst.InputIdx)
		metas[i] = InstructionMeta{
			OpName:    inst.OpName,
			InputIdx:  idx,
			OutputIdx: inst.OutputIdx,
		}
	}
	return metas
}

// SlotShapes returns the shape of each slot as determined during compilation.
// Nil entries indicate slots that were not populated during the warmup pass.
func (p *ExecutionPlan[T]) SlotShapes() [][]int {
	out := make([][]int, len(p.slotShapes))
	for i, s := range p.slotShapes {
		if s != nil {
			cp := make([]int, len(s))
			copy(cp, s)
			out[i] = cp
		}
	}
	return out
}

// FrozenSlots returns the frozen (constant/parameter) slots and their data.
func (p *ExecutionPlan[T]) FrozenSlots() []FrozenSlot[T] {
	frozen := make([]FrozenSlot[T], len(p.frozenIdx))
	for i, idx := range p.frozenIdx {
		frozen[i] = FrozenSlot[T]{
			SlotIdx: idx,
			Data:    p.slots[idx],
		}
	}
	return frozen
}

// InputSlots returns the slot indices that receive graph inputs.
func (p *ExecutionPlan[T]) InputSlots() []int {
	idx := make([]int, len(p.inputIdx))
	copy(idx, p.inputIdx)
	return idx
}

// OutputSlot returns the slot index that holds the final output.
func (p *ExecutionPlan[T]) OutputSlot() int {
	return p.outputIdx
}

// SetMegakernelFn sets an optional megakernel function that, when set,
// replaces the per-instruction execution loop in Run(). This allows a fused
// kernel to transparently handle the entire plan execution.
func (p *ExecutionPlan[T]) SetMegakernelFn(fn func(context.Context, []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)) {
	p.megakernelFn.Store(fn)
}

// Run executes the compiled plan. It sets input tensors into the slot array,
// executes each instruction in sequence, and returns the output.
func (p *ExecutionPlan[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if v := p.megakernelFn.Load(); v != nil {
		fn := v.(func(context.Context, []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error))
		return fn(ctx, inputs)
	}

	if len(inputs) != len(p.inputIdx) {
		return nil, fmt.Errorf("compiled plan: expected %d inputs, got %d", len(p.inputIdx), len(inputs))
	}

	// Use local slot copy so concurrent Run() calls are safe.
	slots := make([]*tensor.TensorNumeric[T], len(p.slots))
	copy(slots, p.slots) // copies frozen slot pointers (params)

	for i, idx := range p.inputIdx {
		slots[idx] = inputs[i]
	}

	// Execute each instruction: gather inputs by index, call Forward, store result.
	for i := range p.instructions {
		inst := &p.instructions[i]
		ins := make([]*tensor.TensorNumeric[T], len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			ins[j] = slots[idx]
		}
		result, err := inst.Forward(ctx, ins)
		if err != nil {
			return nil, fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
		slots[inst.OutputIdx] = result
	}

	return slots[p.outputIdx], nil
}

// Compile pre-compiles the graph into a flat ExecutionPlan. It runs one
// Forward() pass to determine tensor shapes, then assigns buffer indices
// and creates instruction kernels for each node.
func (g *Graph[T]) Compile(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*ExecutionPlan[T], error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("compile: expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	// Step 1: Get tensor shapes. Use existing memo from the last Forward()
	// if available (avoids re-running Forward which would corrupt model state
	// like attention KV caches). Otherwise, run one Forward() to populate memo.
	if len(g.memo) == 0 {
		g.memo = make(map[Node[T]]*tensor.TensorNumeric[T])
		for i, n := range g.inputs {
			g.memo[n] = inputs[i]
		}
		for _, n := range g.nodes {
			if _, ok := n.(*inputNode[T]); ok {
				continue
			}
			nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[n]))
			for i, dep := range g.dependencies[n] {
				nodeInputs[i] = g.memo[dep]
			}
			output, err := n.Forward(ctx, nodeInputs...)
			if err != nil {
				return nil, fmt.Errorf("compile forward: node %s: %w", n.OpType(), err)
			}
			g.memo[n] = output
		}
	}

	// Step 2: Assign slot index to each node in topological order.
	nodeIdx := make(map[Node[T]]int, len(g.nodes))
	for i, n := range g.nodes {
		nodeIdx[n] = i
	}

	// Step 3: Create slot array and populate frozen slots (params/constants).
	slots := make([]*tensor.TensorNumeric[T], len(g.nodes))
	var frozenIdx []int
	inputSlots := make([]int, len(g.inputs))
	for i, n := range g.inputs {
		inputSlots[i] = nodeIdx[n]
	}
	for _, n := range g.nodes {
		if isConstantNode[T](n) {
			idx := nodeIdx[n]
			slots[idx] = g.memo[n] // frozen: model weights
			frozenIdx = append(frozenIdx, idx)
		}
	}

	// Step 4: Create instructions for each compute node.
	var instructions []Instruction[T]
	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		if isConstantNode[T](n) {
			continue
		}

		outIdx := nodeIdx[n]
		depIndices := make([]int, len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			depIndices[i] = nodeIdx[dep]
		}

		fwd := func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return n.Forward(ctx, inputs...)
		}
		instructions = append(instructions, Instruction[T]{
			Forward:   fwd,
			InputIdx:  depIndices,
			OutputIdx: outIdx,
			OpName:    n.OpType(),
		})
	}

	// Step 5: Record slot shapes from warmup memo.
	slotShapes := make([][]int, len(g.nodes))
	for n, t := range g.memo {
		if idx, ok := nodeIdx[n]; ok && t != nil {
			slotShapes[idx] = t.Shape()
		}
	}

	return &ExecutionPlan[T]{
		instructions: instructions,
		slots:        slots,
		slotShapes:   slotShapes,
		inputIdx:     inputSlots,
		outputIdx:    nodeIdx[g.output],
		frozenIdx:    frozenIdx,
	}, nil
}

// CompileTraced produces a primitive-op ExecutionPlan by tracing through the
// graph's Forward pass with the EngineProxy recording every engine call.
// Unlike Compile (which creates one instruction per graph node), CompileTraced
// decomposes composite nodes into their constituent engine calls, enabling the
// megakernel emitter to see only primitive operations.
func (g *Graph[T]) CompileTraced(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*ExecutionPlan[T], error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	proxy := g.engineProxy
	if proxy == nil {
		return nil, errors.New("CompileTraced: no EngineProxy set on graph")
	}

	if len(inputs) != len(g.inputs) {
		return nil, fmt.Errorf("CompileTraced: expected %d inputs, got %d", len(g.inputs), len(inputs))
	}

	// Step 1: Collect frozen tensors from constant/parameter nodes.
	var frozenTensors []*tensor.TensorNumeric[T]
	for _, n := range g.nodes {
		if isConstantNode[T](n) {
			t, err := n.Forward(ctx)
			if err != nil || t == nil {
				continue
			}
			frozenTensors = append(frozenTensors, t)
		}
	}

	// Step 2: Create tracer with frozen tensors pre-registered.
	tracer := compute.NewTracer[T](frozenTensors)

	// Register input tensors so the tracer knows their slot IDs.
	for _, in := range inputs {
		tracer.SlotFor(in)
	}

	// Step 3: Enable tracing and run Forward on each node.
	proxy.StartTracing(tracer)

	memo := make(map[Node[T]]*tensor.TensorNumeric[T])
	for i, n := range g.inputs {
		memo[n] = inputs[i]
	}
	for _, n := range g.nodes {
		if _, ok := n.(*inputNode[T]); ok {
			continue
		}
		nodeInputs := make([]*tensor.TensorNumeric[T], len(g.dependencies[n]))
		for i, dep := range g.dependencies[n] {
			nodeInputs[i] = memo[dep]
		}
		output, err := n.Forward(ctx, nodeInputs...)
		if err != nil {
			proxy.StopTracing()
			return nil, fmt.Errorf("CompileTraced forward: node %s: %w", n.OpType(), err)
		}
		memo[n] = output
	}

	proxy.StopTracing()

	// Step 4: Check for opaque ops. If present, fall back to non-traced Compile.
	if tracer.HasOpaqueOps() {
		return nil, errors.New("CompileTraced: trace contains opaque ops (e.g. UnaryOp); use Compile instead")
	}

	// Step 5: Convert traced ops to instructions.
	tracedOps := tracer.TracedOps()
	numSlots := tracer.NextSlot()

	// Populate frozen slot data.
	slots := make([]*tensor.TensorNumeric[T], numSlots)
	frozenSlotIDs := tracer.FrozenSlots()
	frozenSet := make(map[int]bool, len(frozenSlotIDs))
	for _, sid := range frozenSlotIDs {
		frozenSet[sid] = true
	}
	// Map frozen tensors to their slots.
	for _, ft := range frozenTensors {
		sid := tracer.SlotFor(ft)
		slots[sid] = ft
	}

	// Record input slot IDs.
	inputSlots := make([]int, len(inputs))
	for i, in := range inputs {
		inputSlots[i] = tracer.SlotFor(in)
	}

	// Determine output slot: the slot of the final graph output tensor.
	outputTensor := memo[g.output]
	outputSlot := tracer.SlotFor(outputTensor)

	// Build instructions from traced ops.
	engine := proxy.Real()
	instructions := make([]Instruction[T], len(tracedOps))
	for i, op := range tracedOps {
		fwd := makeTracedForward[T](engine, op)
		inputIdx := make([]int, len(op.InputIDs))
		copy(inputIdx, op.InputIDs)
		instructions[i] = Instruction[T]{
			Forward:   fwd,
			InputIdx:  inputIdx,
			OutputIdx: op.OutputID,
			OpName:    op.OpName,
		}
	}

	// Step 5: Record slot shapes.
	slotShapes := make([][]int, numSlots)
	for sid, shape := range tracer.SlotShapes() {
		if sid < numSlots {
			slotShapes[sid] = shape
		}
	}

	return &ExecutionPlan[T]{
		instructions: instructions,
		slots:        slots,
		slotShapes:   slotShapes,
		inputIdx:     inputSlots,
		outputIdx:    outputSlot,
		frozenIdx:    frozenSlotIDs,
	}, nil
}

// makeTracedForward creates a Forward closure for a traced op that replays the
// engine call with the correct method and extra arguments.
func makeTracedForward[T tensor.Numeric](engine compute.Engine[T], op compute.TracedOp) func(context.Context, []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	switch op.OpName {
	case "Add":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Add(ctx, ins[0], ins[1])
		}
	case "Sub":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Sub(ctx, ins[0], ins[1])
		}
	case "Mul":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Mul(ctx, ins[0], ins[1])
		}
	case "Div":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Div(ctx, ins[0], ins[1])
		}
	case "Pow":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Pow(ctx, ins[0], ins[1])
		}
	case "MatMul":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.MatMul(ctx, ins[0], ins[1])
		}
	case "Exp":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Exp(ctx, ins[0])
		}
	case "Log":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Log(ctx, ins[0])
		}
	case "Tanh":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Tanh(ctx, ins[0])
		}
	case "Sqrt":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Sqrt(ctx, ins[0])
		}
	case "Rsqrt":
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Rsqrt(ctx, ins[0])
		}
	case "MulScalar":
		scalar := extractScalar[T](op.ExtraArgs)
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.MulScalar(ctx, ins[0], scalar)
		}
	case "AddScalar":
		scalar := extractScalar[T](op.ExtraArgs)
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.AddScalar(ctx, ins[0], scalar)
		}
	case "DivScalar":
		scalar := extractScalar[T](op.ExtraArgs)
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.DivScalar(ctx, ins[0], scalar)
		}
	case "Softmax":
		axis := extractInt(op.ExtraArgs, "axis")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Softmax(ctx, ins[0], axis)
		}
	case "ReduceSum":
		axis := extractInt(op.ExtraArgs, "axis")
		keepDims := extractBool(op.ExtraArgs, "keepDims")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.ReduceSum(ctx, ins[0], axis, keepDims)
		}
	case "ReduceMean":
		axis := extractInt(op.ExtraArgs, "axis")
		keepDims := extractBool(op.ExtraArgs, "keepDims")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.ReduceMean(ctx, ins[0], axis, keepDims)
		}
	case "Reshape":
		shape := extractIntSlice(op.ExtraArgs, "shape")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Reshape(ctx, ins[0], shape)
		}
	case "Transpose":
		axes := extractIntSlice(op.ExtraArgs, "axes")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Transpose(ctx, ins[0], axes)
		}
	case "Concat":
		axis := extractInt(op.ExtraArgs, "axis")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Concat(ctx, ins, axis)
		}
	case "Repeat":
		axis := extractInt(op.ExtraArgs, "axis")
		reps := extractInt(op.ExtraArgs, "repetitions")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Repeat(ctx, ins[0], axis, reps)
		}
	case "Sum":
		axis := extractInt(op.ExtraArgs, "axis")
		keepDims := extractBool(op.ExtraArgs, "keepDims")
		return func(ctx context.Context, ins []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return engine.Sum(ctx, ins[0], axis, keepDims)
		}
	default:
		opName := op.OpName
		return func(_ context.Context, _ []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			return nil, fmt.Errorf("makeTracedForward: unsupported op %q", opName)
		}
	}
}

// extractScalar extracts a scalar value from ExtraArgs.
func extractScalar[T tensor.Numeric](extra map[string]any) T {
	if extra == nil {
		var zero T
		return zero
	}
	v, ok := extra["scalar"]
	if !ok {
		var zero T
		return zero
	}
	switch val := v.(type) {
	case float64:
		return T(val)
	case float32:
		return T(val)
	case int:
		return T(int64(val))
	case T:
		return val
	default:
		var zero T
		return zero
	}
}

// extractInt extracts an int from ExtraArgs.
func extractInt(extra map[string]any, key string) int {
	if extra == nil {
		return 0
	}
	v, ok := extra[key]
	if !ok {
		return 0
	}
	switch val := v.(type) {
	case int:
		return val
	case int64:
		return int(val)
	case float64:
		return int(val)
	default:
		return 0
	}
}

// extractBool extracts a bool from ExtraArgs.
func extractBool(extra map[string]any, key string) bool {
	if extra == nil {
		return false
	}
	v, ok := extra[key]
	if !ok {
		return false
	}
	b, _ := v.(bool)
	return b
}

// extractIntSlice extracts an int slice from ExtraArgs.
func extractIntSlice(extra map[string]any, key string) []int {
	if extra == nil {
		return nil
	}
	v, ok := extra[key]
	if !ok {
		return nil
	}
	switch val := v.(type) {
	case []int:
		return val
	case []any:
		result := make([]int, len(val))
		for i, item := range val {
			switch iv := item.(type) {
			case int:
				result[i] = iv
			case float64:
				result[i] = int(iv)
			}
		}
		return result
	default:
		return nil
	}
}
