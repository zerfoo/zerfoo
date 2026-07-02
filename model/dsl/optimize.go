package dsl

// OptimizePass is a named graph transformation.
type OptimizePass func(g *ModelGraph) *ModelGraph

// Optimize applies a sequence of optimization passes to a graph,
// returning a new optimized graph. The original graph is not modified.
func Optimize(g *ModelGraph, passes ...OptimizePass) *ModelGraph {
	result := g
	for _, pass := range passes {
		result = pass(result)
	}
	return result
}

// ConstantFolding removes layers whose output can be determined statically.
// In this DSL, identity-like element-wise layers (RMSNorm, SiLU, Softmax) that
// feed into another element-wise layer of the same type are collapsed: the
// duplicate is removed because applying the same normalization/activation twice
// is redundant. Additionally, consecutive identical element-wise layers are
// folded into one.
func ConstantFolding(g *ModelGraph) *ModelGraph {
	// Identify layers to remove: if a layer has exactly one parent and both
	// the layer and its parent are the same element-wise type, the layer is
	// redundant (idempotent folding for RMSNorm and Softmax).
	remove := map[string]bool{}
	for _, name := range g.order {
		parents := g.parents[name]
		if len(parents) != 1 {
			continue
		}
		parent := parents[0]
		layerType := g.layers[g.layerIndex[name]].Type
		parentType := g.layers[g.layerIndex[parent]].Type

		// Idempotent ops: applying twice is same as once.
		if layerType == parentType && isIdempotent(layerType) {
			remove[name] = true
		}
	}

	if len(remove) == 0 {
		return g
	}

	return rebuildWithout(g, remove)
}

// DeadNodeElimination removes layers that do not contribute to the model's
// terminal output. A node is "dead" if no path from it reaches any node that
// is both an output (no children) and reachable from an input (no parents).
// When there are dangling branches (leaf nodes only reachable through a fork),
// they are pruned because they don't contribute to the primary output path.
//
// Specifically, it traces backward from each output that lies on a path
// starting from an input, matching Model.Forward semantics.
func DeadNodeElimination(g *ModelGraph) *ModelGraph {
	// First, find all nodes reachable forward from inputs.
	reachableFromInput := map[string]bool{}
	var walkForward func(name string)
	walkForward = func(name string) {
		if reachableFromInput[name] {
			return
		}
		reachableFromInput[name] = true
		for _, c := range g.children[name] {
			walkForward(c)
		}
	}
	for _, inp := range g.inputs {
		walkForward(inp)
	}

	// Identify "true" outputs: leaf nodes reachable from inputs that are also
	// connected back to an input through parent edges. For graphs with
	// dangling branches, only the main output path matters.
	// We pick the last output in the original outputs list that is reachable
	// from an input — this matches Model.Forward which uses the last output.
	var targetOutputs []string
	for _, out := range g.outputs {
		if reachableFromInput[out] {
			targetOutputs = append(targetOutputs, out)
		}
	}

	// If a graph has branches where one branch leads to the "primary" output
	// and another is a dead end, only keep nodes contributing to the primary
	// output. The primary output is the last one, matching Model.Forward.
	if len(targetOutputs) > 1 {
		// Keep only the last output (matching Forward behavior).
		targetOutputs = targetOutputs[len(targetOutputs)-1:]
	}

	// Walk backward from target outputs to find all live nodes.
	live := map[string]bool{}
	var walk func(name string)
	walk = func(name string) {
		if live[name] {
			return
		}
		live[name] = true
		for _, p := range g.parents[name] {
			walk(p)
		}
	}
	for _, out := range targetOutputs {
		walk(out)
	}

	// Find dead nodes.
	remove := map[string]bool{}
	for _, name := range g.order {
		if !live[name] {
			remove[name] = true
		}
	}

	if len(remove) == 0 {
		return g
	}

	return rebuildWithout(g, remove)
}

// FusedLayerType is the type used for fused operator layers.
const FusedLayerType LayerType = "fused"

// OperatorFusion merges sequences of compatible layers into fused operators.
// Currently supported fusions:
//   - RMSNorm followed by SiLU → fused "rmsnorm_silu"
//   - SiLU followed by RMSNorm → fused "silu_rmsnorm"
func OperatorFusion(g *ModelGraph) *ModelGraph {
	// Look for fusible pairs: A → B where A has exactly one child (B)
	// and B has exactly one parent (A).
	var fusions []fusionPair
	fused := map[string]bool{}

	for _, name := range g.order {
		if fused[name] {
			continue
		}
		children := g.children[name]
		if len(children) != 1 {
			continue
		}
		child := children[0]
		if fused[child] {
			continue
		}
		if len(g.parents[child]) != 1 {
			continue
		}

		lt := g.layers[g.layerIndex[name]].Type
		ct := g.layers[g.layerIndex[child]].Type

		var fusedOp string
		switch {
		case lt == LayerRMSNorm && ct == LayerSiLU:
			fusedOp = "rmsnorm_silu"
		case lt == LayerSiLU && ct == LayerRMSNorm:
			fusedOp = "silu_rmsnorm"
		default:
			continue
		}

		// Merge params from both layers.
		mergedParams := map[string]any{"fused_op": fusedOp}
		for k, v := range g.layers[g.layerIndex[name]].Params {
			mergedParams[k] = v
		}
		for k, v := range g.layers[g.layerIndex[child]].Params {
			mergedParams[k] = v
		}

		fusions = append(fusions, fusionPair{
			first:       name,
			second:      child,
			fusedName:   name + "+" + child,
			fusedParams: mergedParams,
		})
		fused[name] = true
		fused[child] = true
	}

	if len(fusions) == 0 {
		return g
	}

	return rebuildWithFusions(g, fusions)
}

// isIdempotent returns true for ops where applying twice gives the same result
// as applying once.
func isIdempotent(t LayerType) bool {
	switch t {
	case LayerRMSNorm, LayerSoftmax:
		return true
	default:
		return false
	}
}

// rebuildWithout creates a new ModelGraph excluding the specified nodes,
// rewiring edges to skip removed nodes.
func rebuildWithout(g *ModelGraph, remove map[string]bool) *ModelGraph {
	// Build a mapping from removed node to its replacement (its parent).
	// If a removed node's parent is also removed, follow the chain.
	replacement := map[string]string{}
	for _, name := range g.order {
		if !remove[name] {
			continue
		}
		parents := g.parents[name]
		if len(parents) == 0 {
			continue
		}
		rep := parents[0]
		for remove[rep] {
			if p, ok := replacement[rep]; ok {
				rep = p
			} else {
				break
			}
		}
		replacement[name] = rep
	}

	var layers []LayerDef
	layerIndex := map[string]int{}
	for _, name := range g.order {
		if remove[name] {
			continue
		}
		idx := len(layers)
		layers = append(layers, g.layers[g.layerIndex[name]])
		layerIndex[name] = idx
	}

	children := map[string][]string{}
	parents := map[string][]string{}

	// Rebuild edges, rewiring around removed nodes.
	for _, name := range g.order {
		if remove[name] {
			continue
		}
		for _, child := range g.children[name] {
			actual := child
			for remove[actual] {
				// The removed node's children become our children.
				nextChildren := g.children[actual]
				if len(nextChildren) == 0 {
					actual = ""
					break
				}
				actual = nextChildren[0]
			}
			if actual != "" && !remove[actual] {
				children[name] = appendUnique(children[name], actual)
				parents[actual] = appendUnique(parents[actual], name)
			}
		}
	}

	// For removed input nodes, their children need to become inputs themselves.
	for removed := range remove {
		if len(g.parents[removed]) != 0 {
			continue
		}
		for _, child := range g.children[removed] {
			actual := child
			for remove[actual] {
				next := g.children[actual]
				if len(next) == 0 {
					actual = ""
					break
				}
				actual = next[0]
			}
			if actual != "" && !remove[actual] {
				// Remove any parent refs that pointed to removed nodes.
				var cleanParents []string
				for _, p := range parents[actual] {
					if !remove[p] {
						cleanParents = append(cleanParents, p)
					}
				}
				parents[actual] = cleanParents
			}
		}
	}

	var order []string
	for _, name := range g.order {
		if !remove[name] {
			order = append(order, name)
		}
	}

	var inputs, outputs []string
	for _, name := range order {
		if len(parents[name]) == 0 {
			inputs = append(inputs, name)
		}
		if len(children[name]) == 0 {
			outputs = append(outputs, name)
		}
	}

	return &ModelGraph{
		name:       g.name,
		layers:     layers,
		layerIndex: layerIndex,
		children:   children,
		parents:    parents,
		order:      order,
		inputs:     inputs,
		outputs:    outputs,
	}
}

type fusionPair struct {
	first, second string
	fusedName     string
	fusedParams   map[string]any
}

// rebuildWithFusions creates a new ModelGraph where fused pairs are replaced
// by a single fused node.
func rebuildWithFusions(g *ModelGraph, fusions []fusionPair) *ModelGraph {
	// Map from original name to fused name.
	replaceMap := map[string]string{}
	fusedLayers := map[string]LayerDef{}
	removed := map[string]bool{}

	for _, f := range fusions {
		replaceMap[f.first] = f.fusedName
		replaceMap[f.second] = f.fusedName
		removed[f.second] = true // second node is absorbed
		fusedLayers[f.fusedName] = LayerDef{
			Name:   f.fusedName,
			Type:   FusedLayerType,
			Params: f.fusedParams,
		}
	}

	// Build new layer list.
	var layers []LayerDef
	layerIndex := map[string]int{}
	seen := map[string]bool{}

	for _, name := range g.order {
		if removed[name] {
			continue
		}
		outName := name
		if rep, ok := replaceMap[name]; ok {
			outName = rep
		}
		if seen[outName] {
			continue
		}
		seen[outName] = true

		if fl, ok := fusedLayers[outName]; ok {
			layerIndex[outName] = len(layers)
			layers = append(layers, fl)
		} else {
			layerIndex[outName] = len(layers)
			layers = append(layers, g.layers[g.layerIndex[name]])
		}
	}

	// Rebuild edges.
	children := map[string][]string{}
	parents := map[string][]string{}

	resolve := func(name string) string {
		if rep, ok := replaceMap[name]; ok {
			return rep
		}
		return name
	}

	for _, name := range g.order {
		src := resolve(name)
		for _, child := range g.children[name] {
			dst := resolve(child)
			if src == dst {
				continue // internal fused edge
			}
			children[src] = appendUnique(children[src], dst)
			parents[dst] = appendUnique(parents[dst], src)
		}
	}

	// Build order.
	var order []string
	for _, name := range g.order {
		resolved := resolve(name)
		if !seen[resolved] {
			continue
		}
		// Only add once.
		found := false
		for _, o := range order {
			if o == resolved {
				found = true
				break
			}
		}
		if !found {
			order = append(order, resolved)
		}
	}

	var inputs, outputs []string
	for _, name := range order {
		if len(parents[name]) == 0 {
			inputs = append(inputs, name)
		}
		if len(children[name]) == 0 {
			outputs = append(outputs, name)
		}
	}

	return &ModelGraph{
		name:       g.name,
		layers:     layers,
		layerIndex: layerIndex,
		children:   children,
		parents:    parents,
		order:      order,
		inputs:     inputs,
		outputs:    outputs,
	}
}

func appendUnique(slice []string, val string) []string {
	for _, s := range slice {
		if s == val {
			return slice
		}
	}
	return append(slice, val)
}
