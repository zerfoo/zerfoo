package autoopt

import (
	"fmt"
	"math"
)

// DeviceCapability wraps a HardwareProfile with computed performance estimates.
type DeviceCapability struct {
	// ID uniquely identifies this device (e.g. "cpu:0", "cuda:0", "cuda:1").
	ID string

	// Profile is the underlying hardware profile for this device.
	Profile *HardwareProfile

	// FLOPS is the estimated peak floating-point operations per second.
	FLOPS float64

	// MemoryBandwidth is the estimated memory bandwidth in bytes per second.
	MemoryBandwidth float64

	// AvailableMemory is the usable memory in bytes for tensor storage.
	AvailableMemory int64

	// IsGPU is true when this device is a GPU accelerator.
	IsGPU bool
}

// NewDeviceCapability computes performance estimates from a HardwareProfile.
// The isGPU flag indicates whether this represents a GPU device.
func NewDeviceCapability(id string, hw *HardwareProfile, isGPU bool) DeviceCapability {
	dev := DeviceCapability{
		ID:      id,
		Profile: hw,
		IsGPU:   isGPU,
	}
	if isGPU && hw.GPUAvailable {
		// Conservative GPU estimates: 10 TFLOPS base, scaled by memory.
		dev.FLOPS = 10e12
		dev.MemoryBandwidth = 500e9 // 500 GB/s
		dev.AvailableMemory = hw.GPUMemory
	} else {
		// CPU estimates: ~50 GFLOPS per core with SIMD.
		coresF := float64(hw.CPUCores)
		if coresF <= 0 {
			coresF = 1
		}
		flopsPerCore := 10e9 // 10 GFLOPS base
		if hw.HasAVX512 {
			flopsPerCore = 80e9
		} else if hw.HasAVX2 {
			flopsPerCore = 50e9
		} else if hw.HasNEON {
			flopsPerCore = 30e9
		}
		dev.FLOPS = coresF * flopsPerCore
		dev.MemoryBandwidth = 50e9 // 50 GB/s typical DDR
		dev.AvailableMemory = hw.TotalRAM
	}
	return dev
}

// Op represents a computation to be scheduled across devices.
type Op struct {
	// Name identifies this operation (for debugging/display).
	Name string

	// Class is the kernel class (GEMM, attention, elementwise, etc.).
	Class KernelClass

	// M, N, K are the dimensions for matrix operations.
	// For elementwise ops, M*N gives the number of elements and K is unused.
	M, N, K int

	// MemoryBytes is the total memory footprint of inputs + outputs.
	MemoryBytes int64

	// OutputDeviceHint, if non-empty, suggests a preferred device for the output
	// (to reduce data transfer for downstream consumers).
	OutputDeviceHint string
}

// FLOPs returns the estimated floating-point operations for this op.
func (op *Op) FLOPs() float64 {
	switch op.Class {
	case KernelGEMM, KernelQuantGEMM:
		// GEMM: 2*M*N*K FLOPs
		return 2.0 * float64(op.M) * float64(op.N) * float64(op.K)
	case KernelGEMV, KernelQuantDot:
		// GEMV: 2*M*K FLOPs
		return 2.0 * float64(op.M) * float64(op.K)
	case KernelAttention:
		// Attention: ~4*M*N*K (QK^T + softmax + AV)
		return 4.0 * float64(op.M) * float64(op.N) * float64(op.K)
	default:
		// Elementwise/normalization: M*N operations
		n := float64(op.M) * float64(op.N)
		if n <= 0 {
			n = 1
		}
		return n
	}
}

// DeviceAssignment maps an op to a device with its estimated execution cost.
type DeviceAssignment struct {
	// Op is the operation being assigned.
	Op Op

	// DeviceID is the ID of the device this op is assigned to.
	DeviceID string

	// EstimatedTimeNs is the estimated execution time in nanoseconds.
	EstimatedTimeNs float64

	// TransferCostNs is the estimated data transfer overhead in nanoseconds.
	TransferCostNs float64
}

// SplitPlan is the result of workload splitting: a set of device assignments.
type SplitPlan struct {
	// Assignments maps each op (by index in the original slice) to a device.
	Assignments []DeviceAssignment

	// TotalEstimatedTimeNs is the estimated wall-clock time assuming
	// operations on different devices run in parallel.
	TotalEstimatedTimeNs float64

	// DeviceMemoryUsed tracks allocated memory per device.
	DeviceMemoryUsed map[string]int64
}

// CostModel estimates the execution time of an Op on a given device.
type CostModel struct {
	// TransferBandwidth is the device-to-device transfer rate in bytes/sec.
	// Defaults to 16 GB/s (PCIe Gen4 x16) if zero.
	TransferBandwidth float64
}

// EstimateTimeNs returns the estimated execution time in nanoseconds
// for the given op on the given device.
func (cm *CostModel) EstimateTimeNs(op *Op, dev *DeviceCapability) float64 {
	flops := op.FLOPs()

	// Compute-bound estimate.
	computeNs := (flops / dev.FLOPS) * 1e9

	// Memory-bound estimate (bytes that must be read/written).
	memNs := float64(op.MemoryBytes) / dev.MemoryBandwidth * 1e9

	// The actual time is dominated by the slower of compute or memory.
	return math.Max(computeNs, memNs)
}

// TransferCostNs estimates the data transfer cost in nanoseconds for
// moving an op's data from one device to another.
func (cm *CostModel) TransferCostNs(memoryBytes int64, srcIsGPU, dstIsGPU bool) float64 {
	// Same device type — no transfer.
	if srcIsGPU == dstIsGPU {
		return 0
	}
	bw := cm.TransferBandwidth
	if bw <= 0 {
		bw = 16e9 // 16 GB/s PCIe Gen4
	}
	return float64(memoryBytes) / bw * 1e9
}

// WorkloadSplitter partitions operations across available devices.
type WorkloadSplitter struct {
	devices []DeviceCapability
	cost    CostModel
}

// NewWorkloadSplitter creates a splitter for the given set of devices.
func NewWorkloadSplitter(devices []DeviceCapability) *WorkloadSplitter {
	return &WorkloadSplitter{
		devices: devices,
		cost:    CostModel{},
	}
}

// NewWorkloadSplitterWithCost creates a splitter with a custom cost model.
func NewWorkloadSplitterWithCost(devices []DeviceCapability, cost CostModel) *WorkloadSplitter {
	return &WorkloadSplitter{
		devices: devices,
		cost:    cost,
	}
}

// Split assigns each op to the device that minimizes estimated execution time,
// respecting device memory constraints and accounting for data transfer costs.
func (ws *WorkloadSplitter) Split(ops []Op) *SplitPlan {
	plan := &SplitPlan{
		Assignments:      make([]DeviceAssignment, 0, len(ops)),
		DeviceMemoryUsed: make(map[string]int64),
	}

	if len(ws.devices) == 0 || len(ops) == 0 {
		return plan
	}

	// Track per-device time for load balancing.
	deviceTime := make(map[string]float64)
	for _, d := range ws.devices {
		deviceTime[d.ID] = 0
		plan.DeviceMemoryUsed[d.ID] = 0
	}

	// Track which device the previous op was assigned to for transfer cost.
	prevDeviceIdx := -1

	for _, op := range ops {
		bestIdx := -1
		bestCost := math.MaxFloat64

		for i := range ws.devices {
			dev := &ws.devices[i]

			// Check memory constraint.
			if dev.AvailableMemory > 0 && plan.DeviceMemoryUsed[dev.ID]+op.MemoryBytes > dev.AvailableMemory {
				continue
			}

			execTime := ws.cost.EstimateTimeNs(&op, dev)

			// Add transfer cost if switching devices.
			var transferCost float64
			if prevDeviceIdx >= 0 && prevDeviceIdx != i {
				prevDev := &ws.devices[prevDeviceIdx]
				transferCost = ws.cost.TransferCostNs(op.MemoryBytes, prevDev.IsGPU, dev.IsGPU)
			}

			totalCost := execTime + transferCost

			if totalCost < bestCost {
				bestCost = totalCost
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			// No device has enough memory; assign to the device with least memory used.
			bestIdx = 0
			leastMem := plan.DeviceMemoryUsed[ws.devices[0].ID]
			for i := 1; i < len(ws.devices); i++ {
				if plan.DeviceMemoryUsed[ws.devices[i].ID] < leastMem {
					leastMem = plan.DeviceMemoryUsed[ws.devices[i].ID]
					bestIdx = i
				}
			}
			bestCost = ws.cost.EstimateTimeNs(&op, &ws.devices[bestIdx])
		}

		dev := &ws.devices[bestIdx]

		var transferCost float64
		if prevDeviceIdx >= 0 && prevDeviceIdx != bestIdx {
			prevDev := &ws.devices[prevDeviceIdx]
			transferCost = ws.cost.TransferCostNs(op.MemoryBytes, prevDev.IsGPU, dev.IsGPU)
		}

		execTime := ws.cost.EstimateTimeNs(&op, dev)

		assignment := DeviceAssignment{
			Op:              op,
			DeviceID:        dev.ID,
			EstimatedTimeNs: execTime,
			TransferCostNs:  transferCost,
		}

		plan.Assignments = append(plan.Assignments, assignment)
		plan.DeviceMemoryUsed[dev.ID] += op.MemoryBytes
		deviceTime[dev.ID] += execTime + transferCost
		prevDeviceIdx = bestIdx
	}

	// Total estimated time is the max across devices (parallel execution).
	for _, t := range deviceTime {
		if t > plan.TotalEstimatedTimeNs {
			plan.TotalEstimatedTimeNs = t
		}
	}

	return plan
}

// String returns a human-readable summary of the split plan.
func (sp *SplitPlan) String() string {
	if len(sp.Assignments) == 0 {
		return "SplitPlan: empty (no ops)"
	}

	deviceOps := make(map[string]int)
	for _, a := range sp.Assignments {
		deviceOps[a.DeviceID]++
	}

	s := fmt.Sprintf("SplitPlan: %d ops across %d devices, est. %.2f ms\n",
		len(sp.Assignments), len(deviceOps), sp.TotalEstimatedTimeNs/1e6)
	for dev, count := range deviceOps {
		memMB := float64(sp.DeviceMemoryUsed[dev]) / (1024 * 1024)
		s += fmt.Sprintf("  %s: %d ops, %.1f MiB\n", dev, count, memMB)
	}
	return s
}
