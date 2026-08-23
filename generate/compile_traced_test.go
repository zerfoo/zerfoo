package generate

import (
	"bytes"
	"context"
	"log"
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// tracelessNode produces its output in plain Go without making a single engine
// call, exactly like the GGUF embedding lookup in inference/arch_llama.go. Its
// output therefore has no producing instruction in a trace.
type tracelessNode struct {
	graph.NoParameters[float32]
	width int
}

func (n *tracelessNode) OpType() string                     { return "Traceless" }
func (n *tracelessNode) Attributes() map[string]interface{} { return nil }
func (n *tracelessNode) OutputShape() []int                 { return []int{1, 1, n.width} }

func (n *tracelessNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *tracelessNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	data := make([]float32, n.width)
	for i := range data {
		data[i] = float32(i + 1)
	}
	return tensor.New([]int{1, 1, n.width}, data)
}

// engineMulNode consumes the traceless output through a real engine call, so a
// trace records an instruction whose input slot has no producer.
type engineMulNode struct {
	graph.NoParameters[float32]
	engine compute.Engine[float32]
	width  int
}

func (n *engineMulNode) OpType() string                     { return "EngineMul" }
func (n *engineMulNode) Attributes() map[string]interface{} { return nil }
func (n *engineMulNode) OutputShape() []int                 { return []int{1, 1, n.width} }

func (n *engineMulNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *engineMulNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.Mul(ctx, inputs[0], inputs[0])
}

// buildUntraceableGraph mirrors the shape of every GGUF architecture graph: an
// EngineProxy is installed, but a node upstream of the first engine op produces
// its tensor outside the engine.
func buildUntraceableGraph(t *testing.T, width int) *graph.Graph[float32] {
	t.Helper()
	proxy := compute.NewEngineProxy[float32](compute.NewCPUEngine(numeric.Float32Ops{}))
	b := graph.NewBuilder[float32](proxy)

	in := b.Input([]int{1, 1, 1})
	emb := &tracelessNode{width: width}
	b.AddNode(emb, in)
	mul := &engineMulNode{engine: proxy, width: width}
	b.AddNode(mul, emb)

	g, err := b.Build(mul)
	if err != nil {
		t.Fatal(err)
	}
	g.SetEngineProxy(proxy)
	return g
}

// TestCompileGraph_TracedPlanOptIn pins issue #994: traced compilation must not
// run by default, because no graph built by the inference package can produce a
// valid traced plan. The failed attempt cost ~0.7s per generator and logged a
// message that reads like a correctness bug.
func TestCompileGraph_TracedPlanOptIn(t *testing.T) {
	tests := []struct {
		name       string
		enabled    bool
		wantTraced bool
	}{
		{name: "disabled by default", enabled: false, wantTraced: false},
		{name: "opt-in still attempts tracing", enabled: true, wantTraced: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prev := tracedPlanEnabled
			tracedPlanEnabled = tt.enabled
			t.Cleanup(func() { tracedPlanEnabled = prev })

			var logBuf bytes.Buffer
			prevOut := log.Writer()
			prevFlags := log.Flags()
			log.SetOutput(&logBuf)
			log.SetFlags(0)
			t.Cleanup(func() {
				log.SetOutput(prevOut)
				log.SetFlags(prevFlags)
			})

			const width = 4
			g := buildUntraceableGraph(t, width)
			gen := NewGenerator[float32](
				g, buildTestTokenizer(), compute.NewCPUEngine(numeric.Float32Ops{}),
				ModelConfig{VocabSize: width, MaxSeqLen: 16, EOSTokenID: 2, NumLayers: 1},
			)

			input, err := tensor.New([]int{1, 1}, []float32{1})
			if err != nil {
				t.Fatal(err)
			}
			// Populate the graph memo so Compile can reuse it.
			if _, err := g.Forward(context.Background(), input); err != nil {
				t.Fatalf("Forward: %v", err)
			}
			gen.compileGraph(context.Background(), input)

			if p := gen.plan.Load(); p == nil {
				t.Fatal("compileGraph produced no plan")
			}

			gotTraced := strings.Contains(logBuf.String(), "CompileTraced")
			if gotTraced != tt.wantTraced {
				t.Errorf("log mentions CompileTraced = %v, want %v; log:\n%s",
					gotTraced, tt.wantTraced, logBuf.String())
			}
		})
	}
}

// TestCompileGraph_TracedPlanCannotValidate red-proofs the change above: it
// asserts the reason the traced path is disabled is still true. If a future
// ztensor release teaches the tracer about tensors produced outside the engine,
// this test fails and the default in compileGraph should be revisited.
func TestCompileGraph_TracedPlanCannotValidate(t *testing.T) {
	const width = 4
	g := buildUntraceableGraph(t, width)

	input, err := tensor.New([]int{1, 1}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	plan, err := g.CompileTraced(ctx, input)
	if err != nil {
		t.Fatalf("CompileTraced: %v", err)
	}
	if _, err := plan.Run(ctx, input); err == nil {
		t.Fatal("traced plan validated unexpectedly; the premise of #994 no longer holds")
	} else if !strings.Contains(err.Error(), "input tensors cannot be nil") {
		t.Fatalf("traced plan failed for an unexpected reason: %v", err)
	}
}
