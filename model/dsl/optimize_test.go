package dsl

import (
	"testing"
)

func TestGraphOptimize_ConstantFolding(t *testing.T) {
	tests := []struct {
		name       string
		def        *ModelDef
		wantLayers int
		wantOrder  []string
	}{
		{
			name: "no folding needed",
			def: &ModelDef{
				Name: "no_fold",
				Layers: []LayerDef{
					{Name: "linear1", Type: LayerLinear},
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "linear2", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "linear1", To: "norm"},
					{From: "norm", To: "linear2"},
				},
			},
			wantLayers: 3,
			wantOrder:  []string{"linear1", "norm", "linear2"},
		},
		{
			name: "consecutive rmsnorm folded",
			def: &ModelDef{
				Name: "fold_rmsnorm",
				Layers: []LayerDef{
					{Name: "linear1", Type: LayerLinear},
					{Name: "norm1", Type: LayerRMSNorm},
					{Name: "norm2", Type: LayerRMSNorm},
					{Name: "linear2", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "linear1", To: "norm1"},
					{From: "norm1", To: "norm2"},
					{From: "norm2", To: "linear2"},
				},
			},
			wantLayers: 3,
			wantOrder:  []string{"linear1", "norm1", "linear2"},
		},
		{
			name: "consecutive softmax folded",
			def: &ModelDef{
				Name: "fold_softmax",
				Layers: []LayerDef{
					{Name: "sm1", Type: LayerSoftmax},
					{Name: "sm2", Type: LayerSoftmax},
				},
				Connections: []ConnectionDef{
					{From: "sm1", To: "sm2"},
				},
			},
			wantLayers: 1,
			wantOrder:  []string{"sm1"},
		},
		{
			name: "triple rmsnorm folded to one",
			def: &ModelDef{
				Name: "triple_norm",
				Layers: []LayerDef{
					{Name: "n1", Type: LayerRMSNorm},
					{Name: "n2", Type: LayerRMSNorm},
					{Name: "n3", Type: LayerRMSNorm},
				},
				Connections: []ConnectionDef{
					{From: "n1", To: "n2"},
					{From: "n2", To: "n3"},
				},
			},
			wantLayers: 1,
			wantOrder:  []string{"n1"},
		},
		{
			name: "silu is not idempotent - not folded",
			def: &ModelDef{
				Name: "no_fold_silu",
				Layers: []LayerDef{
					{Name: "s1", Type: LayerSiLU},
					{Name: "s2", Type: LayerSiLU},
				},
				Connections: []ConnectionDef{
					{From: "s1", To: "s2"},
				},
			},
			wantLayers: 2,
			wantOrder:  []string{"s1", "s2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}

			opt := ConstantFolding(g)

			if len(opt.Order()) != tt.wantLayers {
				t.Errorf("got %d layers, want %d; order=%v", len(opt.Order()), tt.wantLayers, opt.Order())
			}
			if tt.wantOrder != nil {
				for i, name := range tt.wantOrder {
					if i >= len(opt.Order()) {
						t.Errorf("order[%d]: missing, want %q", i, name)
						continue
					}
					if opt.Order()[i] != name {
						t.Errorf("order[%d] = %q, want %q", i, opt.Order()[i], name)
					}
				}
			}
		})
	}
}

func TestGraphOptimize_DeadNodeElimination(t *testing.T) {
	tests := []struct {
		name       string
		def        *ModelDef
		wantLayers int
		wantOrder  []string
	}{
		{
			name: "no dead nodes",
			def: &ModelDef{
				Name: "alive",
				Layers: []LayerDef{
					{Name: "a", Type: LayerLinear},
					{Name: "b", Type: LayerRMSNorm},
				},
				Connections: []ConnectionDef{
					{From: "a", To: "b"},
				},
			},
			wantLayers: 2,
			wantOrder:  []string{"a", "b"},
		},
		{
			name: "dead branch removed",
			def: &ModelDef{
				Name: "dead_branch",
				Layers: []LayerDef{
					{Name: "input", Type: LayerLinear},
					{Name: "live_path", Type: LayerRMSNorm},
					{Name: "dead_leaf", Type: LayerSiLU},
					{Name: "output", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "input", To: "live_path"},
					{From: "input", To: "dead_leaf"},
					{From: "live_path", To: "output"},
					// dead_leaf has no path to output
				},
			},
			wantLayers: 3,
			wantOrder:  []string{"input", "live_path", "output"},
		},
		{
			name: "disconnected node removed",
			def: &ModelDef{
				Name: "disconnected",
				Layers: []LayerDef{
					{Name: "main", Type: LayerLinear},
					{Name: "island", Type: LayerSiLU},
				},
				// No connections — both are inputs and outputs.
				// Only the last output (island) is kept, matching Forward semantics.
			},
			wantLayers: 1,
			wantOrder:  []string{"island"},
		},
		{
			name: "chain of dead nodes removed",
			def: &ModelDef{
				Name: "dead_chain",
				Layers: []LayerDef{
					{Name: "root", Type: LayerLinear},
					{Name: "live", Type: LayerRMSNorm},
					{Name: "dead1", Type: LayerSiLU},
					{Name: "dead2", Type: LayerSoftmax},
					{Name: "output", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "root", To: "live"},
					{From: "root", To: "dead1"},
					{From: "dead1", To: "dead2"},
					{From: "live", To: "output"},
				},
			},
			wantLayers: 3,
			wantOrder:  []string{"root", "live", "output"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}

			opt := DeadNodeElimination(g)

			if len(opt.Order()) != tt.wantLayers {
				t.Errorf("got %d layers, want %d; order=%v", len(opt.Order()), tt.wantLayers, opt.Order())
			}
			if tt.wantOrder != nil {
				for i, name := range tt.wantOrder {
					if i >= len(opt.Order()) {
						t.Errorf("order[%d]: missing, want %q", i, name)
						continue
					}
					if opt.Order()[i] != name {
						t.Errorf("order[%d] = %q, want %q", i, opt.Order()[i], name)
					}
				}
			}
		})
	}
}

func TestGraphOptimize_OperatorFusion(t *testing.T) {
	tests := []struct {
		name       string
		def        *ModelDef
		wantLayers int
		wantFused  []string // names of fused layers expected
	}{
		{
			name: "no fusion possible",
			def: &ModelDef{
				Name: "no_fuse",
				Layers: []LayerDef{
					{Name: "a", Type: LayerLinear},
					{Name: "b", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "a", To: "b"},
				},
			},
			wantLayers: 2,
		},
		{
			name: "rmsnorm + silu fused",
			def: &ModelDef{
				Name: "fuse_norm_silu",
				Layers: []LayerDef{
					{Name: "linear1", Type: LayerLinear},
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "act", Type: LayerSiLU},
					{Name: "linear2", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "linear1", To: "norm"},
					{From: "norm", To: "act"},
					{From: "act", To: "linear2"},
				},
			},
			wantLayers: 3, // linear1, norm+act, linear2
			wantFused:  []string{"norm+act"},
		},
		{
			name: "silu + rmsnorm fused",
			def: &ModelDef{
				Name: "fuse_silu_norm",
				Layers: []LayerDef{
					{Name: "act", Type: LayerSiLU},
					{Name: "norm", Type: LayerRMSNorm},
				},
				Connections: []ConnectionDef{
					{From: "act", To: "norm"},
				},
			},
			wantLayers: 1,
			wantFused:  []string{"act+norm"},
		},
		{
			name: "multiple fusions in chain",
			def: &ModelDef{
				Name: "multi_fuse",
				Layers: []LayerDef{
					{Name: "n1", Type: LayerRMSNorm},
					{Name: "s1", Type: LayerSiLU},
					{Name: "linear", Type: LayerLinear},
					{Name: "n2", Type: LayerRMSNorm},
					{Name: "s2", Type: LayerSiLU},
				},
				Connections: []ConnectionDef{
					{From: "n1", To: "s1"},
					{From: "s1", To: "linear"},
					{From: "linear", To: "n2"},
					{From: "n2", To: "s2"},
				},
			},
			wantLayers: 3, // n1+s1, linear, n2+s2
			wantFused:  []string{"n1+s1", "n2+s2"},
		},
		{
			name: "branching prevents fusion",
			def: &ModelDef{
				Name: "no_fuse_branch",
				Layers: []LayerDef{
					{Name: "norm", Type: LayerRMSNorm},
					{Name: "silu", Type: LayerSiLU},
					{Name: "other", Type: LayerLinear},
				},
				Connections: []ConnectionDef{
					{From: "norm", To: "silu"},
					{From: "norm", To: "other"},
				},
			},
			wantLayers: 3, // norm has 2 children, can't fuse
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := Parse(tt.def)
			if err != nil {
				t.Fatalf("Parse failed: %v", err)
			}

			opt := OperatorFusion(g)

			if len(opt.Order()) != tt.wantLayers {
				t.Errorf("got %d layers, want %d; order=%v", len(opt.Order()), tt.wantLayers, opt.Order())
			}

			// Check that fused layers exist and have the right type.
			for _, fusedName := range tt.wantFused {
				idx, ok := opt.layerIndex[fusedName]
				if !ok {
					t.Errorf("expected fused layer %q not found in order %v", fusedName, opt.Order())
					continue
				}
				layer := opt.layers[idx]
				if layer.Type != FusedLayerType {
					t.Errorf("fused layer %q has type %q, want %q", fusedName, layer.Type, FusedLayerType)
				}
				if _, ok := layer.Params["fused_op"]; !ok {
					t.Errorf("fused layer %q missing fused_op param", fusedName)
				}
			}
		})
	}
}

func TestGraphOptimize_CombinedPasses(t *testing.T) {
	// Build a graph with dead nodes AND fusible operators.
	def := &ModelDef{
		Name: "combined",
		Layers: []LayerDef{
			{Name: "input", Type: LayerLinear},
			{Name: "norm", Type: LayerRMSNorm},
			{Name: "act", Type: LayerSiLU},
			{Name: "dead", Type: LayerSoftmax},
			{Name: "output", Type: LayerLinear},
		},
		Connections: []ConnectionDef{
			{From: "input", To: "norm"},
			{From: "norm", To: "act"},
			{From: "input", To: "dead"},
			{From: "act", To: "output"},
		},
	}

	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	opt := Optimize(g, DeadNodeElimination, OperatorFusion)

	// Dead node "dead" removed, then norm+act fused.
	if len(opt.Order()) != 3 {
		t.Errorf("got %d layers, want 3; order=%v", len(opt.Order()), opt.Order())
	}

	// Verify fused node exists.
	found := false
	for _, name := range opt.Order() {
		if name == "norm+act" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected fused layer 'norm+act' in order %v", opt.Order())
	}
}

func TestGraphOptimize_PreservesGraphIntegrity(t *testing.T) {
	// Ensure optimized graphs maintain valid inputs/outputs.
	def := &ModelDef{
		Name: "integrity",
		Layers: []LayerDef{
			{Name: "in", Type: LayerLinear},
			{Name: "norm1", Type: LayerRMSNorm},
			{Name: "norm2", Type: LayerRMSNorm},
			{Name: "out", Type: LayerLinear},
		},
		Connections: []ConnectionDef{
			{From: "in", To: "norm1"},
			{From: "norm1", To: "norm2"},
			{From: "norm2", To: "out"},
		},
	}

	g, err := Parse(def)
	if err != nil {
		t.Fatalf("Parse failed: %v", err)
	}

	opt := ConstantFolding(g)

	// After folding norm2, should have: in -> norm1 -> out
	inputs := opt.Inputs()
	outputs := opt.Outputs()

	if len(inputs) != 1 || inputs[0] != "in" {
		t.Errorf("Inputs() = %v, want [in]", inputs)
	}
	if len(outputs) != 1 || outputs[0] != "out" {
		t.Errorf("Outputs() = %v, want [out]", outputs)
	}
}
