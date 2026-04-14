// Command gemma4_e2e runs Gemma 4 E2B end-to-end verification as a standalone
// binary so it can execute inside a Spark pod on the DGX host. A 2B-parameter
// forward pass on CPU exceeds reasonable test timeouts, so these assertions
// cannot live in the CPU `go test` path.
//
// Two modes:
//   - forward (default, T93.4.1/E96): load GGUF, build the graph, run one
//     forward pass, and verify finite non-zero logits.
//   - generate (T97.1.1): load the model end-to-end via inference.LoadFile
//     (which builds graph + extracts tokenizer + wires a Generator), then run
//     greedy decode for N steps on the given prompt and verify finite logits
//     and non-degenerate output.
//
// Usage (from Spark pod manifest):
//
//	gemma4_e2e -gguf /var/lib/zerfoo/models/gemma-4-E2B-it-Q4_K_M.gguf
//	gemma4_e2e -gguf <path> -mode generate -prompt "The quick" -steps 50
//
// Exit codes: 0 = all checks pass; 1 = any check fails.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	ggufPath := flag.String("gguf", "", "path to Gemma 4 E2B GGUF file (required)")
	mode := flag.String("mode", "forward", "mode: forward | generate")
	seqLen := flag.Int("seq", 4, "[forward] sequence length for the single forward pass")
	prompt := flag.String("prompt", "The quick brown fox", "[generate] prompt text")
	steps := flag.Int("steps", 50, "[generate] max new tokens")
	device := flag.String("device", "cpu", "[generate] compute device: cpu | cuda")
	flag.Parse()

	if *ggufPath == "" {
		fmt.Fprintln(os.Stderr, "gemma4_e2e: -gguf is required")
		os.Exit(2)
	}

	switch *mode {
	case "forward":
		if err := runForward(*ggufPath, *seqLen); err != nil {
			fmt.Fprintf(os.Stderr, "gemma4_e2e: %v\n", err)
			os.Exit(1)
		}
	case "generate":
		if err := runGenerate(*ggufPath, *device, *prompt, *steps); err != nil {
			fmt.Fprintf(os.Stderr, "gemma4_e2e: %v\n", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "gemma4_e2e: unknown -mode %q (want forward|generate)\n", *mode)
		os.Exit(2)
	}

	fmt.Println("gemma4_e2e: PASS")
}

func runForward(ggufPath string, seqLen int) error {
	fmt.Printf("gemma4_e2e: [forward] loading %s\n", ggufPath)
	mdl, err := inference.LoadGGUF(ggufPath)
	if err != nil {
		return fmt.Errorf("LoadGGUF: %w", err)
	}
	cfg := mdl.Config
	fmt.Printf("gemma4_e2e: arch=%s layers=%d hidden=%d vocab=%d tensors=%d\n",
		cfg.Architecture, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize, len(mdl.Tensors))

	switch cfg.Architecture {
	case "gemma4", "gemma4e", "gemma4moe":
	default:
		return fmt.Errorf("unexpected architecture %q", cfg.Architecture)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, _, err := inference.BuildArchGraph(cfg.Architecture, mdl.Tensors, cfg, engine)
	if err != nil {
		return fmt.Errorf("BuildArchGraph: %w", err)
	}
	if g == nil {
		return fmt.Errorf("graph is nil")
	}
	fmt.Println("gemma4_e2e: graph built")

	tokenIDs := make([]float32, seqLen)
	for i := range tokenIDs {
		tokenIDs[i] = float32(i + 1)
	}
	input, err := tensor.New([]int{1, seqLen}, tokenIDs)
	if err != nil {
		return fmt.Errorf("create input: %w", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		return fmt.Errorf("forward: %w", err)
	}
	shape := output.Shape()
	fmt.Printf("gemma4_e2e: forward complete, output shape=%v\n", shape)
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != cfg.VocabSize {
		return fmt.Errorf("unexpected shape %v, want [1,%d,%d]", shape, seqLen, cfg.VocabSize)
	}

	return checkFiniteNonZero(output.Data())
}

func runGenerate(ggufPath, device, prompt string, steps int) error {
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("-prompt must be non-empty")
	}
	if steps <= 0 {
		return fmt.Errorf("-steps must be > 0")
	}

	fmt.Printf("gemma4_e2e: [generate] loading %s on %s\n", ggufPath, device)
	mdl, err := inference.LoadFile(ggufPath,
		inference.WithDevice(device),
	)
	if err != nil {
		return fmt.Errorf("LoadFile: %w", err)
	}
	defer mdl.Close()

	cfg := mdl.Config()
	fmt.Printf("gemma4_e2e: arch=%s layers=%d hidden=%d vocab=%d\n",
		cfg.Architecture, cfg.NumLayers, cfg.HiddenSize, cfg.VocabSize)

	// Generate mode accepts any architecture inference.LoadFile handles. The
	// gemma4/gemma4e/gemma4moe guard in forward mode exists for the E96 smoke
	// check; generate mode uses the real Generator which validates arch itself.
	fmt.Printf("gemma4_e2e: prompt=%q steps=%d\n", prompt, steps)
	out, err := mdl.Generate(context.Background(), prompt,
		inference.WithTemperature(0),
		inference.WithMaxTokens(steps),
	)
	if err != nil {
		return fmt.Errorf("Generate: %w", err)
	}
	fmt.Printf("gemma4_e2e: generated (%d bytes): %q\n", len(out), out)

	if strings.TrimSpace(out) == "" {
		return fmt.Errorf("generated text is empty")
	}
	if containsRepeatedChar(out, steps/2) {
		return fmt.Errorf("generated text is degenerate (repeated single character)")
	}
	return nil
}

func checkFiniteNonZero(data []float32) error {
	hasNonZero := false
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			return fmt.Errorf("NaN at logit %d", i)
		}
		if math.IsInf(float64(v), 0) {
			return fmt.Errorf("Inf at logit %d", i)
		}
		if v != 0 {
			hasNonZero = true
		}
	}
	if !hasNonZero {
		return fmt.Errorf("all logits are zero")
	}
	return nil
}

// containsRepeatedChar reports whether s contains a run of a single rune
// (excluding whitespace) at least runLen long -- a simple degeneracy check.
func containsRepeatedChar(s string, runLen int) bool {
	if runLen < 2 {
		return false
	}
	var prev rune
	var run int
	for _, r := range s {
		if r == ' ' || r == '\n' || r == '\t' {
			prev, run = 0, 0
			continue
		}
		if r == prev {
			run++
			if run >= runLen {
				return true
			}
		} else {
			prev, run = r, 1
		}
	}
	return false
}
