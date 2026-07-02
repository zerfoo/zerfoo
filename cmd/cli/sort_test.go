package cli

import (
	"context"
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/model"
)

func TestCLI_HelpCommandOrderDeterministic(t *testing.T) {
	buildCLI := func() *CLI {
		var buf strings.Builder
		cliApp := newTestCLI(&buf)
		cliApp.RegisterCommand(NewTokenizeCommand())
		cliApp.RegisterCommand(NewPredictCommand(model.Float32ModelRegistry, float32From, float32To))
		return cliApp
	}

	// Run help output twice and verify identical order.
	var outputs [2]string
	for i := range outputs {
		var buf strings.Builder
		cliApp := buildCLI()
		cliApp.out = &buf
		if err := cliApp.Run(context.Background(), nil); err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
		outputs[i] = buf.String()
	}

	if outputs[0] != outputs[1] {
		t.Errorf("help output not deterministic:\nrun 0:\n%s\nrun 1:\n%s", outputs[0], outputs[1])
	}

	// Verify alphabetical order: "predict" before "tokenize".
	predictIdx := strings.Index(outputs[0], "predict")
	tokenizeIdx := strings.Index(outputs[0], "tokenize")
	if predictIdx < 0 || tokenizeIdx < 0 {
		t.Fatal("expected both predict and tokenize in help output")
	}
	if predictIdx > tokenizeIdx {
		t.Error("commands should be in alphabetical order: predict before tokenize")
	}
}

func TestCommandRegistry_ListSorted(t *testing.T) {
	reg := NewCommandRegistry()
	reg.Register(NewTokenizeCommand())
	reg.Register(NewPredictCommand(model.Float32ModelRegistry, float32From, float32To))

	names := reg.List()
	if len(names) != 2 {
		t.Fatalf("expected 2 commands, got %d", len(names))
	}
	if names[0] != "predict" || names[1] != "tokenize" {
		t.Errorf("expected [predict tokenize], got %v", names)
	}
}
