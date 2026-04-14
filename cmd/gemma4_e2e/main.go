// Command gemma4_e2e runs the Gemma 4 E2B end-to-end integration verification
// as a standalone binary so it can execute inside a Spark pod on the DGX host.
// A 2B-parameter forward pass on CPU exceeds reasonable test timeouts, so the
// forward-pass assertions cannot live in the CPU `go test` path. This binary
// covers T93.4.1: load a real GGUF, build the graph, run one forward pass, and
// verify finite non-zero logits.
//
// Usage (from Spark pod manifest):
//
//	gemma4_e2e -gguf /var/lib/zerfoo/models/gemma-4-E2B-it-Q4_K_M.gguf
//
// Exit codes: 0 = all checks pass; 1 = any check fails.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	ggufPath := flag.String("gguf", "", "path to Gemma 4 E2B GGUF file (required)")
	seqLen := flag.Int("seq", 4, "sequence length for the single forward pass")
	flag.Parse()

	if *ggufPath == "" {
		fmt.Fprintln(os.Stderr, "gemma4_e2e: -gguf is required")
		os.Exit(2)
	}

	fmt.Printf("gemma4_e2e: loading %s\n", *ggufPath)
	mdl, err := inference.LoadGGUF(*ggufPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemma4_e2e: LoadGGUF: %v\n", err)
		os.Exit(1)
	}
	cfg := mdl.Config
	fmt.Printf("gemma4_e2e: arch=%s layers=%d hidden=%d vocab=%d tensors=%d\n",
		cfg.Architecture, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize, len(mdl.Tensors))

	switch cfg.Architecture {
	case "gemma4", "gemma4e", "gemma4moe":
	default:
		fmt.Fprintf(os.Stderr, "gemma4_e2e: unexpected architecture %q\n", cfg.Architecture)
		os.Exit(1)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, _, err := inference.BuildArchGraph(cfg.Architecture, mdl.Tensors, cfg, engine)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemma4_e2e: BuildArchGraph: %v\n", err)
		os.Exit(1)
	}
	if g == nil {
		fmt.Fprintln(os.Stderr, "gemma4_e2e: graph is nil")
		os.Exit(1)
	}
	fmt.Println("gemma4_e2e: graph built")

	tokenIDs := make([]float32, *seqLen)
	for i := range tokenIDs {
		tokenIDs[i] = float32(i + 1)
	}
	input, err := tensor.New([]int{1, *seqLen}, tokenIDs)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemma4_e2e: create input: %v\n", err)
		os.Exit(1)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gemma4_e2e: forward: %v\n", err)
		os.Exit(1)
	}
	shape := output.Shape()
	fmt.Printf("gemma4_e2e: forward complete, output shape=%v\n", shape)
	if len(shape) != 3 || shape[0] != 1 || shape[1] != *seqLen || shape[2] != cfg.VocabSize {
		fmt.Fprintf(os.Stderr, "gemma4_e2e: unexpected shape %v, want [1,%d,%d]\n",
			shape, *seqLen, cfg.VocabSize)
		os.Exit(1)
	}

	data := output.Data()
	hasNonZero := false
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			fmt.Fprintf(os.Stderr, "gemma4_e2e: NaN at logit %d\n", i)
			os.Exit(1)
		}
		if math.IsInf(float64(v), 0) {
			fmt.Fprintf(os.Stderr, "gemma4_e2e: Inf at logit %d\n", i)
			os.Exit(1)
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		fmt.Fprintln(os.Stderr, "gemma4_e2e: all logits are zero")
		os.Exit(1)
	}

	fmt.Println("gemma4_e2e: PASS")
}
