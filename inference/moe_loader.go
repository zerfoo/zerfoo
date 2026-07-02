package inference

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// MoEDeviceMap holds the device assignment for each expert in a model with
// Mixture of Experts layers. It is built during GGUF loading by
// [SplitMoEWeights] and consumed by the graph builder to decide whether
// expert FFN weights should be uploaded to GPU or kept in CPU memory.
type MoEDeviceMap struct {
	// Experts maps expert ID to its assigned device.
	Experts map[int]DeviceType

	// SharedExperts lists expert IDs that are always active (shared experts).
	SharedExperts []int

	// RoutedExperts lists expert IDs that are conditionally routed.
	RoutedExperts []int
}

// DeviceForExpert returns the device assignment for the given expert ID.
// If the expert is not in the map, it returns CPU as a safe default.
func (m *MoEDeviceMap) DeviceForExpert(expertID int) DeviceType {
	if m == nil || m.Experts == nil {
		return CPU
	}
	dev, ok := m.Experts[expertID]
	if !ok {
		return CPU
	}
	return dev
}

// GPUExperts returns the expert IDs assigned to GPU.
func (m *MoEDeviceMap) GPUExperts() []int {
	if m == nil {
		return nil
	}
	var ids []int
	for id, dev := range m.Experts {
		if dev == GPU {
			ids = append(ids, id)
		}
	}
	return ids
}

// CPUExperts returns the expert IDs assigned to CPU.
func (m *MoEDeviceMap) CPUExperts() []int {
	if m == nil {
		return nil
	}
	var ids []int
	for id, dev := range m.Experts {
		if dev == CPU {
			ids = append(ids, id)
		}
	}
	return ids
}

// SplitMoEWeights partitions expert weights between GPU and CPU based on the
// model configuration. Shared experts (always active) are assigned to GPU.
// Routed experts are assigned to CPU by default.
//
// The function scans the GGUF tensor map for expert weight patterns (stacked
// tensors like "blk.N.ffn_gate_exps.weight") and builds a device map that
// the graph builder can use for placement decisions.
//
// Parameters:
//   - tensors: the full GGUF tensor map
//   - cfg: model configuration with NumExperts and NumSharedExperts
//
// Returns the device map and two tensor subsets: gpuTensors contains tensor
// names that should be uploaded to GPU, cpuTensors contains names that should
// remain in CPU memory. Non-expert tensors are not included in either subset.
func SplitMoEWeights(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
) (*MoEDeviceMap, map[string]*tensor.TensorNumeric[float32], map[string]*tensor.TensorNumeric[float32], error) {
	if cfg.NumExperts == 0 {
		return nil, nil, nil, nil
	}

	numExperts := cfg.NumExperts
	numShared := cfg.NumSharedExperts

	// Build routing stats: shared experts get frequency 1.0 (always active),
	// routed experts get frequency 0.0 (conditionally active).
	routingStats := make(map[int]float64, numExperts)
	for i := 0; i < numExperts; i++ {
		if i < numShared {
			routingStats[i] = 1.0
		} else {
			routingStats[i] = 0.0
		}
	}

	// Use the placement policy to compute device assignments.
	policy := NewExpertPlacementPolicy(numExperts)
	policy.Assign(routingStats)
	deviceMap := policy.DeviceMap()

	dm := &MoEDeviceMap{
		Experts: deviceMap,
	}
	for id, dev := range deviceMap {
		if dev == GPU {
			dm.SharedExperts = append(dm.SharedExperts, id)
		} else {
			dm.RoutedExperts = append(dm.RoutedExperts, id)
		}
	}

	// Classify expert tensors into GPU and CPU subsets.
	gpuTensors := make(map[string]*tensor.TensorNumeric[float32])
	cpuTensors := make(map[string]*tensor.TensorNumeric[float32])

	for name, t := range tensors {
		expertID, isExpert := parseExpertTensorName(name)
		if !isExpert {
			continue
		}
		if expertID >= 0 && expertID < numExperts {
			if deviceMap[expertID] == GPU {
				gpuTensors[name] = t
			} else {
				cpuTensors[name] = t
			}
			continue
		}
		// Stacked expert tensors (e.g. "blk.N.ffn_gate_exps.weight") and
		// shared expert tensors go to GPU since shared experts are always
		// on GPU and stacked tensors are split later by the graph builder.
		if isSharedExpertTensor(name) {
			gpuTensors[name] = t
		} else {
			// Stacked tensors contain all experts; keep on CPU since the
			// majority of routed experts will run on CPU. The graph builder
			// extracts individual expert slices and uploads shared ones.
			cpuTensors[name] = t
		}
	}

	slog.Info("MoE weight split",
		"total_experts", numExperts,
		"shared_experts", numShared,
		"gpu_tensors", len(gpuTensors),
		"cpu_tensors", len(cpuTensors),
	)

	return dm, gpuTensors, cpuTensors, nil
}

// parseExpertTensorName extracts the expert ID from a per-expert tensor name.
// Returns (expertID, true) for names like "blk.0.ffn_gate.0.weight" where
// the second number is the expert index. Returns (-1, false) for non-expert
// tensors and (-1, true) for stacked expert tensors like "blk.0.ffn_gate_exps.weight".
func parseExpertTensorName(name string) (int, bool) {
	// Per-expert tensor patterns:
	//   "blk.N.ffn_gate.E.weight"
	//   "blk.N.ffn_up.E.weight"
	//   "blk.N.ffn_down.E.weight"
	//   "model.layers.N.block_sparse_moe.experts.E.w1.weight" (HF-style)
	//   "model.layers.N.block_sparse_moe.experts.E.w2.weight"
	//   "model.layers.N.block_sparse_moe.experts.E.w3.weight"

	// Check for stacked expert tensors (no per-expert ID).
	if strings.Contains(name, "ffn_gate_exps") ||
		strings.Contains(name, "ffn_up_exps") ||
		strings.Contains(name, "ffn_down_exps") ||
		strings.Contains(name, "ffn_gate_inp") {
		return -1, true
	}

	// Check for shared expert tensors.
	if isSharedExpertTensor(name) {
		return -1, true
	}

	// Per-expert GGUF format: blk.N.ffn_{gate,up,down}.E.weight
	if strings.HasPrefix(name, "blk.") {
		parts := strings.Split(name, ".")
		// Expected: ["blk", N, "ffn_gate"|"ffn_up"|"ffn_down", E, "weight"]
		if len(parts) == 5 {
			ffnPart := parts[2]
			if ffnPart == "ffn_gate" || ffnPart == "ffn_up" || ffnPart == "ffn_down" {
				var expertID int
				if _, err := fmt.Sscanf(parts[3], "%d", &expertID); err == nil {
					return expertID, true
				}
			}
		}
	}

	// HF-style: model.layers.N.block_sparse_moe.experts.E.w{1,2,3}.weight
	if strings.Contains(name, "block_sparse_moe.experts.") {
		parts := strings.Split(name, ".")
		for i, p := range parts {
			if p == "experts" && i+1 < len(parts) {
				var expertID int
				if _, err := fmt.Sscanf(parts[i+1], "%d", &expertID); err == nil {
					return expertID, true
				}
			}
		}
	}

	return -1, false
}

// isSharedExpertTensor returns true if the tensor name belongs to a shared
// expert (always-active expert that runs on every token).
func isSharedExpertTensor(name string) bool {
	return strings.Contains(name, "ffn_shared_expert")
}
