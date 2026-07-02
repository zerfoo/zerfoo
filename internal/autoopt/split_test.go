package autoopt

import (
	"testing"
)

func makeGPUDevice(id string, flops float64, memBW float64, vram int64) DeviceCapability {
	return DeviceCapability{
		ID: id,
		Profile: &HardwareProfile{
			GPUAvailable: true,
			GPUBackend:   "cuda",
			GPUName:      id,
			GPUMemory:    vram,
		},
		FLOPS:           flops,
		MemoryBandwidth: memBW,
		AvailableMemory: vram,
		IsGPU:           true,
	}
}

func makeCPUDevice(id string, cores int, flops float64, memBW float64, ram int64) DeviceCapability {
	return DeviceCapability{
		ID: id,
		Profile: &HardwareProfile{
			CPUCores: cores,
			HasAVX2:  true,
			TotalRAM: ram,
		},
		FLOPS:           flops,
		MemoryBandwidth: memBW,
		AvailableMemory: ram,
		IsGPU:           false,
	}
}

func TestWorkloadSplit_MultiDevice(t *testing.T) {
	gpu := makeGPUDevice("cuda:0", 10e12, 500e9, 16*1024*1024*1024)
	cpu := makeCPUDevice("cpu:0", 16, 500e9, 50e9, 64*1024*1024*1024)

	tests := []struct {
		name    string
		ops     []Op
		wantMin int // minimum number of distinct devices used
	}{
		{
			name: "large GEMM goes to GPU",
			ops: []Op{
				{Name: "large_gemm", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 256 * 1024 * 1024},
			},
			wantMin: 1,
		},
		{
			name: "mixed ops split across devices",
			ops: []Op{
				{Name: "gemm1", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 256 * 1024 * 1024},
				{Name: "elem1", Class: KernelElementwise, M: 128, N: 128, MemoryBytes: 64 * 1024},
				{Name: "gemm2", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 256 * 1024 * 1024},
				{Name: "norm1", Class: KernelRMSNorm, M: 256, N: 1, MemoryBytes: 1024},
			},
			wantMin: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := NewWorkloadSplitter([]DeviceCapability{gpu, cpu})
			plan := ws.Split(tt.ops)

			if len(plan.Assignments) != len(tt.ops) {
				t.Fatalf("got %d assignments, want %d", len(plan.Assignments), len(tt.ops))
			}

			devices := make(map[string]bool)
			for _, a := range plan.Assignments {
				devices[a.DeviceID] = true
				if a.EstimatedTimeNs <= 0 {
					t.Errorf("op %s: estimated time should be > 0, got %f", a.Op.Name, a.EstimatedTimeNs)
				}
			}

			if len(devices) < tt.wantMin {
				t.Errorf("got %d distinct devices, want >= %d", len(devices), tt.wantMin)
			}

			if plan.TotalEstimatedTimeNs <= 0 {
				t.Error("total estimated time should be > 0")
			}
		})
	}
}

func TestWorkloadSplit_MemoryConstraints(t *testing.T) {
	// Small GPU with only 1 GiB VRAM.
	smallGPU := makeGPUDevice("cuda:0", 10e12, 500e9, 1*1024*1024*1024)
	cpu := makeCPUDevice("cpu:0", 16, 500e9, 50e9, 64*1024*1024*1024)

	tests := []struct {
		name         string
		ops          []Op
		wantOverflow bool // if true, some ops should spill to CPU
	}{
		{
			name: "ops fit in VRAM",
			ops: []Op{
				{Name: "small_gemm", Class: KernelGEMM, M: 512, N: 512, K: 512, MemoryBytes: 100 * 1024 * 1024},
			},
			wantOverflow: false,
		},
		{
			name: "ops exceed VRAM",
			ops: []Op{
				{Name: "big1", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 600 * 1024 * 1024},
				{Name: "big2", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 600 * 1024 * 1024},
			},
			wantOverflow: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := NewWorkloadSplitter([]DeviceCapability{smallGPU, cpu})
			plan := ws.Split(tt.ops)

			if len(plan.Assignments) != len(tt.ops) {
				t.Fatalf("got %d assignments, want %d", len(plan.Assignments), len(tt.ops))
			}

			// Check VRAM is not exceeded.
			gpuMem := plan.DeviceMemoryUsed["cuda:0"]
			if gpuMem > smallGPU.AvailableMemory {
				t.Errorf("GPU memory used %d exceeds available %d", gpuMem, smallGPU.AvailableMemory)
			}

			if tt.wantOverflow {
				cpuOps := 0
				for _, a := range plan.Assignments {
					if a.DeviceID == "cpu:0" {
						cpuOps++
					}
				}
				if cpuOps == 0 {
					t.Error("expected some ops to spill to CPU due to VRAM constraints")
				}
			}
		})
	}
}

func TestWorkloadSplit_CostModel(t *testing.T) {
	gpu := makeGPUDevice("cuda:0", 10e12, 500e9, 16*1024*1024*1024)
	cpu := makeCPUDevice("cpu:0", 16, 500e9, 50e9, 64*1024*1024*1024)

	cm := &CostModel{}

	tests := []struct {
		name       string
		op         Op
		gpuFaster  bool // true if GPU should be faster
	}{
		{
			name:      "large GEMM faster on GPU",
			op:        Op{Name: "large_gemm", Class: KernelGEMM, M: 4096, N: 4096, K: 4096, MemoryBytes: 256 * 1024 * 1024},
			gpuFaster: true,
		},
		{
			name:      "small elementwise comparable",
			op:        Op{Name: "small_elem", Class: KernelElementwise, M: 32, N: 32, MemoryBytes: 4096},
			gpuFaster: true, // GPU FLOPS >> CPU even for small ops in raw compute
		},
		{
			name:      "large attention faster on GPU",
			op:        Op{Name: "attention", Class: KernelAttention, M: 2048, N: 2048, K: 128, MemoryBytes: 512 * 1024 * 1024},
			gpuFaster: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gpuTime := cm.EstimateTimeNs(&tt.op, &gpu)
			cpuTime := cm.EstimateTimeNs(&tt.op, &cpu)

			if gpuTime <= 0 {
				t.Errorf("GPU time should be > 0, got %f", gpuTime)
			}
			if cpuTime <= 0 {
				t.Errorf("CPU time should be > 0, got %f", cpuTime)
			}

			if tt.gpuFaster && gpuTime >= cpuTime {
				t.Errorf("expected GPU (%f ns) to be faster than CPU (%f ns) for %s",
					gpuTime, cpuTime, tt.name)
			}
		})
	}
}

func TestWorkloadSplit_TransferCost(t *testing.T) {
	gpu := makeGPUDevice("cuda:0", 10e12, 500e9, 16*1024*1024*1024)
	cpu := makeCPUDevice("cpu:0", 16, 500e9, 50e9, 64*1024*1024*1024)

	tests := []struct {
		name       string
		memBytes   int64
		srcGPU     bool
		dstGPU     bool
		wantZero   bool
		wantMinNs  float64 // minimum expected transfer time
	}{
		{
			name:     "GPU to GPU no cost",
			memBytes: 1024 * 1024 * 1024,
			srcGPU:   true,
			dstGPU:   true,
			wantZero: true,
		},
		{
			name:     "CPU to CPU no cost",
			memBytes: 1024 * 1024 * 1024,
			srcGPU:   false,
			dstGPU:   false,
			wantZero: true,
		},
		{
			name:      "CPU to GPU has cost",
			memBytes:  1024 * 1024 * 1024, // 1 GiB
			srcGPU:    false,
			dstGPU:    true,
			wantZero:  false,
			wantMinNs: 1e6, // at least 1 ms for 1 GiB
		},
		{
			name:      "GPU to CPU has cost",
			memBytes:  1024 * 1024 * 1024,
			srcGPU:    true,
			dstGPU:    false,
			wantZero:  false,
			wantMinNs: 1e6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm := &CostModel{}
			cost := cm.TransferCostNs(tt.memBytes, tt.srcGPU, tt.dstGPU)

			if tt.wantZero && cost != 0 {
				t.Errorf("expected zero transfer cost, got %f ns", cost)
			}
			if !tt.wantZero && cost <= 0 {
				t.Errorf("expected non-zero transfer cost, got %f ns", cost)
			}
			if tt.wantMinNs > 0 && cost < tt.wantMinNs {
				t.Errorf("expected transfer cost >= %f ns, got %f ns", tt.wantMinNs, cost)
			}
		})
	}

	// Verify that transfer cost affects assignment decisions.
	t.Run("transfer cost influences splitting", func(t *testing.T) {
		// With very high transfer cost, sequential ops should stay on same device.
		highTransferCost := CostModel{TransferBandwidth: 1e6} // very slow: 1 MB/s
		ws := NewWorkloadSplitterWithCost([]DeviceCapability{gpu, cpu}, highTransferCost)

		ops := []Op{
			{Name: "op1", Class: KernelGEMM, M: 1024, N: 1024, K: 1024, MemoryBytes: 100 * 1024 * 1024},
			{Name: "op2", Class: KernelGEMM, M: 1024, N: 1024, K: 1024, MemoryBytes: 100 * 1024 * 1024},
		}

		plan := ws.Split(ops)

		if len(plan.Assignments) != 2 {
			t.Fatalf("got %d assignments, want 2", len(plan.Assignments))
		}

		// With very high transfer cost, both ops should be on the same device.
		if plan.Assignments[0].DeviceID != plan.Assignments[1].DeviceID {
			t.Errorf("with high transfer cost, ops should be on same device; got %s and %s",
				plan.Assignments[0].DeviceID, plan.Assignments[1].DeviceID)
		}
	})
}

func TestWorkloadSplit_EmptyInputs(t *testing.T) {
	tests := []struct {
		name    string
		devices []DeviceCapability
		ops     []Op
	}{
		{
			name:    "no devices",
			devices: nil,
			ops:     []Op{{Name: "op1", Class: KernelGEMM, M: 100, N: 100, K: 100, MemoryBytes: 1024}},
		},
		{
			name:    "no ops",
			devices: []DeviceCapability{makeCPUDevice("cpu:0", 8, 100e9, 50e9, 16*1024*1024*1024)},
			ops:     nil,
		},
		{
			name:    "both empty",
			devices: nil,
			ops:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := NewWorkloadSplitter(tt.devices)
			plan := ws.Split(tt.ops)

			if plan == nil {
				t.Fatal("plan should not be nil")
			}
			if len(plan.Assignments) != 0 {
				t.Errorf("expected 0 assignments, got %d", len(plan.Assignments))
			}
		})
	}
}

func TestNewDeviceCapability(t *testing.T) {
	tests := []struct {
		name  string
		hw    *HardwareProfile
		isGPU bool
	}{
		{
			name: "GPU device",
			hw: &HardwareProfile{
				GPUAvailable: true,
				GPUBackend:   "cuda",
				GPUName:      "RTX 4090",
				GPUMemory:    24 * 1024 * 1024 * 1024,
			},
			isGPU: true,
		},
		{
			name: "CPU device with AVX2",
			hw: &HardwareProfile{
				CPUCores: 16,
				HasAVX2:  true,
				TotalRAM: 64 * 1024 * 1024 * 1024,
			},
			isGPU: false,
		},
		{
			name: "CPU device with NEON",
			hw: &HardwareProfile{
				CPUCores: 10,
				HasNEON:  true,
				TotalRAM: 32 * 1024 * 1024 * 1024,
			},
			isGPU: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dev := NewDeviceCapability("dev:0", tt.hw, tt.isGPU)

			if dev.FLOPS <= 0 {
				t.Errorf("FLOPS should be > 0, got %f", dev.FLOPS)
			}
			if dev.MemoryBandwidth <= 0 {
				t.Errorf("MemoryBandwidth should be > 0, got %f", dev.MemoryBandwidth)
			}
			if dev.IsGPU != tt.isGPU {
				t.Errorf("IsGPU = %v, want %v", dev.IsGPU, tt.isGPU)
			}
		})
	}
}

func TestSplitPlan_String(t *testing.T) {
	plan := &SplitPlan{
		Assignments:          nil,
		TotalEstimatedTimeNs: 0,
		DeviceMemoryUsed:     map[string]int64{},
	}
	s := plan.String()
	if s == "" {
		t.Error("String() should not be empty")
	}

	// Non-empty plan.
	plan.Assignments = []DeviceAssignment{
		{Op: Op{Name: "op1"}, DeviceID: "cuda:0", EstimatedTimeNs: 1e6},
	}
	plan.TotalEstimatedTimeNs = 1e6
	plan.DeviceMemoryUsed["cuda:0"] = 1024 * 1024
	s = plan.String()
	if s == "" {
		t.Error("String() should not be empty for non-empty plan")
	}
}
