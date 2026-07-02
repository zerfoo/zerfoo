package inference

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// compilePJRT loads the PJRT plugin, creates a client, and compiles the
// graph into a PJRTPlan. The caller must close the returned plan when done.
// Returns nil if pluginPath is empty.
func compilePJRT(pluginPath string, g *graph.Graph[float32], sampleInput *tensor.TensorNumeric[float32]) (*graph.PJRTPlan[float32], error) {
	if pluginPath == "" {
		return nil, nil
	}

	client, err := graph.NewPJRTClient(pluginPath)
	if err != nil {
		return nil, fmt.Errorf("load PJRT plugin %q: %w", pluginPath, err)
	}

	name, _ := client.PlatformName()
	slog.Info("PJRT plugin loaded", "path", pluginPath, "platform", name)

	plan, err := g.CompilePJRT(context.Background(), client, sampleInput)
	if err != nil {
		client.Close()
		return nil, fmt.Errorf("compile graph via PJRT: %w", err)
	}

	slog.Info("PJRT compilation complete", "hasKVCache", plan.HasKVCache(),
		"inputSlots", len(plan.InputSlots), "frozenSlots", len(plan.FrozenSlots))

	return plan, nil
}
