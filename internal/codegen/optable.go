// Package codegen generates CUDA megakernel source code from a compiled
// ExecutionPlan instruction tape. Each primitive op maps to a CUDA device
// function call that operates on register-resident or shared-memory data.
package codegen

import (
	"fmt"

	"github.com/zerfoo/zerfoo/graph"
)

// SlotInfo describes a slot's shape for the emitter.
type SlotInfo struct {
	Shape []int
}

// OpEmitter generates CUDA code for a single instruction. It returns a
// code fragment that will be inserted into the megakernel body.
type OpEmitter func(op graph.InstructionMeta, inputs []SlotInfo) (string, error)

// emitters maps OpName strings to their CUDA code emitters.
var emitters = map[string]OpEmitter{
	// Binary elementwise
	"Add": binaryOp("+"),
	"Sub": binaryOp("-"),
	"Mul": binaryOp("*"),
	"Div": binaryOp("/"),
	"Pow": funcBinaryOp("powf"),

	// Unary elementwise
	"Exp":   unaryOp("expf"),
	"Log":   unaryOp("logf"),
	"Sqrt":  unaryOp("sqrtf"),
	"Rsqrt": unaryOp("rsqrtf"),
	"Tanh":  unaryOp("tanhf"),
	"Neg":   prefixUnaryOp("-"),
	"Abs":   unaryOp("fabsf"),
	"Silu":  siluOp,

	// Scalar ops
	"AddScalar": scalarOp("+"),
	"MulScalar": scalarOp("*"),
	"SubScalar": scalarOp("-"),
	"DivScalar": scalarOp("/"),
	"PowScalar": funcScalarOp("powf"),

	// Reductions
	"RMSNorm":    rmsnormOp,
	"Softmax":    softmaxOp,
	"ReduceSum":  reduceOp("dev_reduce_sum"),
	"ReduceMean": reduceOp("dev_reduce_mean"),
	"Sum":        reduceOp("dev_reduce_sum"),

	// Memory ops
	"MatMul":      gemvOp,
	"MatMulNBits": gemvQ4Op,
	"Gather":      gatherOp,

	// Indexing ops
	"Slice":  sliceOp,
	"Repeat": repeatOp,

	// Shape ops (no-compute in megakernel, just reindex)
	"Concat":    reshapeOp, // reindex in registers
	"Reshape":   reshapeOp, // no-op in flat memory
	"Transpose": transposeOp,

	// KV cache ops
	"KVCacheAppendK": kvCacheAppendOp("kv_k"),
	"KVCacheAppendV": kvCacheAppendOp("kv_v"),
	"KVCacheGetK":    kvCacheGetOp("kv_k"),
	"KVCacheGetV":    kvCacheGetOp("kv_v"),
	"KVCacheSeqLen":  kvCacheSeqLenOp,
}

// Emit generates CUDA code for a single instruction. Returns an error
// if the op is unsupported.
func Emit(op graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	emitter, ok := emitters[op.OpName]
	if !ok {
		return "", fmt.Errorf("unsupported op: %s", op.OpName)
	}
	return emitter(op, inputs)
}

// Supported returns true if the op has a registered emitter.
func Supported(opName string) bool {
	_, ok := emitters[opName]
	return ok
}

// --- Emitter constructors ---

func binaryOp(op string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = slot_%d[tid] %s slot_%d[tid];",
			meta.OutputIdx, meta.InputIdx[0], op, meta.InputIdx[1]), nil
	}
}

func funcBinaryOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = %s(slot_%d[tid], slot_%d[tid]);",
			meta.OutputIdx, fn, meta.InputIdx[0], meta.InputIdx[1]), nil
	}
}

func unaryOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = %s(slot_%d[tid]);",
			meta.OutputIdx, fn, meta.InputIdx[0]), nil
	}
}

func prefixUnaryOp(prefix string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = %s(slot_%d[tid]);",
			meta.OutputIdx, prefix, meta.InputIdx[0]), nil
	}
}

func scalarOp(op string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = slot_%d[tid] %s scalar_%d;",
			meta.OutputIdx, meta.InputIdx[0], op, meta.OutputIdx), nil
	}
}

func funcScalarOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		return fmt.Sprintf("  slot_%d[tid] = %s(slot_%d[tid], scalar_%d);",
			meta.OutputIdx, fn, meta.InputIdx[0], meta.OutputIdx), nil
	}
}

func siluOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  slot_%d[tid] = slot_%d[tid] * (1.0f / (1.0f + expf(-slot_%d[tid])));",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[0]), nil
}

func rmsnormOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  dev_rmsnorm(slot_%d, slot_%d, slot_%d, dim_%d);",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[1], meta.OutputIdx), nil
}

func softmaxOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	cols := 1
	if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
		cols = inputs[0].Shape[len(inputs[0].Shape)-1]
	}
	return fmt.Sprintf("  dev_softmax(slot_%d, slot_%d, 1, %d);",
		meta.OutputIdx, meta.InputIdx[0], cols), nil
}

func gemvOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  dev_gemv_f32(slot_%d, frozen_%d, slot_%d, dim_m_%d, dim_k_%d);",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[1],
		meta.OutputIdx, meta.OutputIdx), nil
}

func gemvQ4Op(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  dev_gemv_q4(slot_%d, frozen_%d, slot_%d, dim_m_%d, dim_k_%d);",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[1],
		meta.OutputIdx, meta.OutputIdx), nil
}

func gatherOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  dev_gather(slot_%d, frozen_%d, slot_%d, dim_%d);",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[1], meta.OutputIdx), nil
}

func reshapeOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  // %s: slot_%d = slot_%d (reindex, no compute)",
		meta.OpName, meta.OutputIdx, meta.InputIdx[0]), nil
}

func transposeOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  dev_transpose(slot_%d, slot_%d, shape_%d, perm_%d);",
		meta.OutputIdx, meta.InputIdx[0], meta.InputIdx[0], meta.OutputIdx), nil
}

func sliceOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dim := 0
	if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
		dim = inputs[0].Shape[len(inputs[0].Shape)-1]
	}
	return fmt.Sprintf("  dev_slice(slot_%d, slot_%d, start_%d, end_%d, axis_%d, %d);",
		meta.OutputIdx, meta.InputIdx[0], meta.OutputIdx, meta.OutputIdx, meta.OutputIdx, dim), nil
}

func repeatOp(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
	dim := 0
	if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
		dim = inputs[0].Shape[len(inputs[0].Shape)-1]
	}
	return fmt.Sprintf("  dev_repeat(slot_%d, slot_%d, axis_%d, reps_%d, %d);",
		meta.OutputIdx, meta.InputIdx[0], meta.OutputIdx, meta.OutputIdx, dim), nil
}

func reduceOp(fn string) OpEmitter {
	return func(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
		dim := 0
		if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
			dim = inputs[0].Shape[len(inputs[0].Shape)-1]
		}
		return fmt.Sprintf("  %s(slot_%d, slot_%d, axis_%d, %d);",
			fn, meta.OutputIdx, meta.InputIdx[0], meta.OutputIdx, dim), nil
	}
}

// kvCacheAppendOp emits a dev_kv_append call that writes new K or V data
// into the layer's KV cache at the current sequence position.
// InputIdx[0] = source data slot, InputIdx[1] = layer index (encoded),
// OutputIdx = destination slot alias.
func kvCacheAppendOp(arrayName string) OpEmitter {
	return func(meta graph.InstructionMeta, inputs []SlotInfo) (string, error) {
		if len(meta.InputIdx) < 2 {
			return "", fmt.Errorf("KVCacheAppend requires 2 inputs (data slot, layer)")
		}
		layer := meta.InputIdx[1]
		headDim := 0
		if len(inputs) > 0 && len(inputs[0].Shape) > 0 {
			headDim = inputs[0].Shape[len(inputs[0].Shape)-1]
		}
		return fmt.Sprintf("  dev_kv_append(%s[%d], slot_%d, seq_pos, %d);",
			arrayName, layer, meta.InputIdx[0], headDim), nil
	}
}

// kvCacheGetOp emits a pointer alias that points into the layer's KV cache.
// InputIdx[0] = layer index (encoded), OutputIdx = destination slot.
func kvCacheGetOp(arrayName string) OpEmitter {
	return func(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
		if len(meta.InputIdx) < 1 {
			return "", fmt.Errorf("KVCacheGet requires 1 input (layer)")
		}
		layer := meta.InputIdx[0]
		return fmt.Sprintf("  float* slot_%d = %s[%d];",
			meta.OutputIdx, arrayName, layer), nil
	}
}

// kvCacheSeqLenOp emits an integer assignment from the kv_seq_len kernel arg.
func kvCacheSeqLenOp(meta graph.InstructionMeta, _ []SlotInfo) (string, error) {
	return fmt.Sprintf("  int seq_len_%d = kv_seq_len;",
		meta.OutputIdx), nil
}
