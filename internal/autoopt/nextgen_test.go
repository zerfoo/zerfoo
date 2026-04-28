package autoopt

import "testing"

func TestDetectGPUGeneration(t *testing.T) {
	tests := []struct {
		computeCap string
		want       GPUGeneration
	}{
		{"", GPUGenUnknown},
		{"5.0", GPUGenUnknown},
		{"7.0", GPUGenUnknown},
		{"7.5", GPUGenUnknown},
		{"8.0", GPUGenAmpere},
		{"8.6", GPUGenAmpere},
		{"8.9", GPUGenAmpere},
		{"9.0", GPUGenHopper},
		{"9.1", GPUGenHopper},
		{"10.0", GPUGenBlackwell},
		{"10.1", GPUGenBlackwell},
		{"11.0", GPUGenBlackwell},
	}
	for _, tc := range tests {
		got := DetectGPUGeneration(tc.computeCap)
		if got != tc.want {
			t.Errorf("DetectGPUGeneration(%q) = %v, want %v", tc.computeCap, got, tc.want)
		}
	}
}

func TestGPUGeneration_String(t *testing.T) {
	tests := []struct {
		gen  GPUGeneration
		want string
	}{
		{GPUGenAmpere, "Ampere"},
		{GPUGenHopper, "Hopper"},
		{GPUGenBlackwell, "Blackwell"},
		{GPUGenUnknown, "Unknown"},
	}
	for _, tc := range tests {
		if got := tc.gen.String(); got != tc.want {
			t.Errorf("%d.String() = %q, want %q", tc.gen, got, tc.want)
		}
	}
}

func TestDetectHopperCapabilities(t *testing.T) {
	// Hopper should return capabilities.
	caps := DetectHopperCapabilities(GPUGenHopper)
	if caps == nil {
		t.Fatal("expected non-nil capabilities for Hopper")
	}
	if !caps.TMA {
		t.Error("expected TMA support")
	}
	if !caps.WGMMA {
		t.Error("expected WGMMA support")
	}
	if !caps.FP8Native {
		t.Error("expected FP8 native support")
	}
	if caps.ClusterSize != 8 {
		t.Errorf("expected cluster size 8, got %d", caps.ClusterSize)
	}

	// Blackwell inherits Hopper capabilities.
	caps2 := DetectHopperCapabilities(GPUGenBlackwell)
	if caps2 == nil {
		t.Fatal("expected non-nil Hopper capabilities for Blackwell")
	}

	// Ampere and unknown should return nil.
	if DetectHopperCapabilities(GPUGenAmpere) != nil {
		t.Error("expected nil for Ampere")
	}
	if DetectHopperCapabilities(GPUGenUnknown) != nil {
		t.Error("expected nil for Unknown")
	}
}

func TestDetectBlackwellCapabilities(t *testing.T) {
	caps := DetectBlackwellCapabilities(GPUGenBlackwell)
	if caps == nil {
		t.Fatal("expected non-nil capabilities for Blackwell")
	}
	if !caps.FP4Aware {
		t.Error("expected FP4 awareness")
	}
	if !caps.ClusterPrimitives {
		t.Error("expected cluster primitives")
	}
	if caps.MaxClusterSize != 16 {
		t.Errorf("expected max cluster size 16, got %d", caps.MaxClusterSize)
	}
	if !caps.TMA || !caps.WGMMA || !caps.FP8Native {
		t.Error("expected inherited Hopper features (TMA, WGMMA, FP8)")
	}

	// Non-Blackwell should return nil.
	if DetectBlackwellCapabilities(GPUGenHopper) != nil {
		t.Error("expected nil for Hopper")
	}
	if DetectBlackwellCapabilities(GPUGenAmpere) != nil {
		t.Error("expected nil for Ampere")
	}
}

func TestNewNextGenOptimizer_NilProfile(t *testing.T) {
	opt := NewNextGenOptimizer(nil)
	if opt != nil {
		t.Error("expected nil optimizer for nil profile")
	}
}

func TestNewNextGenOptimizer_NonCUDA(t *testing.T) {
	hw := &HardwareProfile{
		GPUAvailable: true,
		GPUBackend:   "rocm",
		GPUName:      "MI250X",
	}
	opt := NewNextGenOptimizer(hw)
	if opt != nil {
		t.Error("expected nil optimizer for non-CUDA GPU")
	}
}

func TestNextGenOptimizer_HopperPath(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      64,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "H100",
		GPUMemory:     80 * 1024 * 1024 * 1024,
		GPUComputeCap: "9.0",
	}
	opt := NewNextGenOptimizer(hw)
	if opt == nil {
		t.Fatal("expected non-nil optimizer for H100")
	}
	if opt.Generation() != GPUGenHopper {
		t.Fatalf("expected Hopper generation, got %v", opt.Generation())
	}

	tests := []struct {
		name string
		op   Op
		want ExecutionPath
	}{
		{
			name: "large GEMM uses TMA+WGMMA",
			op:   Op{Class: KernelGEMM, M: 1024, N: 1024, K: 1024},
			want: PathTMAWGMMA,
		},
		{
			name: "small GEMM uses WGMMA",
			op:   Op{Class: KernelGEMM, M: 32, N: 32, K: 16},
			want: PathWGMMA,
		},
		{
			name: "tiny GEMM uses standard",
			op:   Op{Class: KernelGEMM, M: 8, N: 8, K: 8},
			want: PathStandard,
		},
		{
			name: "attention uses TMA",
			op:   Op{Class: KernelAttention, M: 128, N: 64, K: 64},
			want: PathTMA,
		},
		{
			name: "small attention uses standard",
			op:   Op{Class: KernelAttention, M: 16, N: 16, K: 16},
			want: PathStandard,
		},
		{
			name: "large elementwise uses TMA",
			op:   Op{Class: KernelElementwise, M: 1024, N: 1024, MemoryBytes: 4 * 1024 * 1024},
			want: PathTMA,
		},
		{
			name: "small elementwise uses standard",
			op:   Op{Class: KernelElementwise, M: 16, N: 16, MemoryBytes: 256},
			want: PathStandard,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := opt.SelectOptimalPath(&tc.op)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNextGenOptimizer_BlackwellPath(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      128,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "B200",
		GPUMemory:     192 * 1024 * 1024 * 1024,
		GPUComputeCap: "10.0",
	}
	opt := NewNextGenOptimizer(hw)
	if opt == nil {
		t.Fatal("expected non-nil optimizer for B200")
	}
	if opt.Generation() != GPUGenBlackwell {
		t.Fatalf("expected Blackwell generation, got %v", opt.Generation())
	}

	tests := []struct {
		name string
		op   Op
		want ExecutionPath
	}{
		{
			name: "large quant GEMM uses FP4 cluster",
			op:   Op{Class: KernelQuantGEMM, M: 256, N: 256, K: 128},
			want: PathFP4Cluster,
		},
		{
			name: "large GEMM uses TMA+WGMMA",
			op:   Op{Class: KernelGEMM, M: 1024, N: 1024, K: 1024},
			want: PathTMAWGMMA,
		},
		{
			name: "large attention uses FP4 cluster",
			op:   Op{Class: KernelAttention, M: 256, N: 64, K: 64},
			want: PathFP4Cluster,
		},
		{
			name: "medium attention uses TMA",
			op:   Op{Class: KernelAttention, M: 64, N: 64, K: 64},
			want: PathTMA,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := opt.SelectOptimalPath(&tc.op)
			if got != tc.want {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

func TestNextGenOptimizer_FallbackToAmpere(t *testing.T) {
	hw := &HardwareProfile{
		CPUCores:      32,
		HasAVX2:       true,
		GPUAvailable:  true,
		GPUBackend:    "cuda",
		GPUName:       "A100",
		GPUMemory:     80 * 1024 * 1024 * 1024,
		GPUComputeCap: "8.0",
	}
	opt := NewNextGenOptimizer(hw)
	if opt == nil {
		t.Fatal("expected non-nil optimizer for A100")
	}
	if opt.Generation() != GPUGenAmpere {
		t.Fatalf("expected Ampere generation, got %v", opt.Generation())
	}

	// All operations should fall back to standard path on Ampere.
	ops := []Op{
		{Class: KernelGEMM, M: 1024, N: 1024, K: 1024},
		{Class: KernelAttention, M: 256, N: 64, K: 64},
		{Class: KernelQuantGEMM, M: 256, N: 256, K: 128},
		{Class: KernelElementwise, M: 1024, N: 1024, MemoryBytes: 4 * 1024 * 1024},
	}
	for _, op := range ops {
		got := opt.SelectOptimalPath(&op)
		if got != PathStandard {
			t.Errorf("op %s: expected standard path on Ampere, got %v", op.Class, got)
		}
	}
}

func TestNextGenOptimizer_NilSelectOptimalPath(t *testing.T) {
	var opt *NextGenOptimizer
	op := Op{Class: KernelGEMM, M: 1024, N: 1024, K: 1024}
	got := opt.SelectOptimalPath(&op)
	if got != PathStandard {
		t.Errorf("expected standard path for nil optimizer, got %v", got)
	}
}

func TestNextGenOptimizer_Describe(t *testing.T) {
	tests := []struct {
		name       string
		hw         *HardwareProfile
		wantSubstr string
	}{
		{
			name:       "nil",
			hw:         nil,
			wantSubstr: "disabled",
		},
		{
			name: "Ampere",
			hw: &HardwareProfile{
				GPUAvailable: true, GPUBackend: "cuda",
				GPUComputeCap: "8.0",
			},
			wantSubstr: "Ampere",
		},
		{
			name: "Hopper",
			hw: &HardwareProfile{
				GPUAvailable: true, GPUBackend: "cuda",
				GPUComputeCap: "9.0",
			},
			wantSubstr: "Hopper",
		},
		{
			name: "Blackwell",
			hw: &HardwareProfile{
				GPUAvailable: true, GPUBackend: "cuda",
				GPUComputeCap: "10.0",
			},
			wantSubstr: "Blackwell",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			opt := NewNextGenOptimizer(tc.hw)
			var desc string
			if opt == nil {
				desc = (*NextGenOptimizer)(nil).Describe()
			} else {
				desc = opt.Describe()
			}
			if len(desc) == 0 {
				t.Error("expected non-empty description")
			}
			found := false
			for i := 0; i <= len(desc)-len(tc.wantSubstr); i++ {
				if desc[i:i+len(tc.wantSubstr)] == tc.wantSubstr {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("description %q does not contain %q", desc, tc.wantSubstr)
			}
		})
	}
}

func TestTMAConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TMAConfig
		wantErr bool
	}{
		{
			name: "valid 2D",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 64, BoxDimY: 64,
				ElementSizeBytes: 2, Swizzle: Swizzle128B,
			},
			wantErr: false,
		},
		{
			name: "valid 3D",
			cfg: TMAConfig{
				Dim: TMA3D, BoxDimX: 32, BoxDimY: 32, BoxDimZ: 8,
				ElementSizeBytes: 4, Swizzle: Swizzle64B,
			},
			wantErr: false,
		},
		{
			name: "invalid dimension",
			cfg: TMAConfig{
				Dim: 5, BoxDimX: 64, BoxDimY: 64,
				ElementSizeBytes: 2,
			},
			wantErr: true,
		},
		{
			name: "zero element size",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 64, BoxDimY: 64,
				ElementSizeBytes: 0,
			},
			wantErr: true,
		},
		{
			name: "BoxDimX too large",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 512, BoxDimY: 64,
				ElementSizeBytes: 2,
			},
			wantErr: true,
		},
		{
			name: "BoxDimY zero",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 64, BoxDimY: 0,
				ElementSizeBytes: 2,
			},
			wantErr: true,
		},
		{
			name: "3D with zero BoxDimZ",
			cfg: TMAConfig{
				Dim: TMA3D, BoxDimX: 32, BoxDimY: 32, BoxDimZ: 0,
				ElementSizeBytes: 2,
			},
			wantErr: true,
		},
		{
			name: "box exceeds 128 KiB",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 256, BoxDimY: 256,
				ElementSizeBytes: 4,
			},
			wantErr: true,
		},
		{
			name: "row not aligned to 16 bytes",
			cfg: TMAConfig{
				Dim: TMA2D, BoxDimX: 3, BoxDimY: 64,
				ElementSizeBytes: 2,
			},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestTMAConfig_BoxBytes(t *testing.T) {
	cfg2d := TMAConfig{Dim: TMA2D, BoxDimX: 64, BoxDimY: 32, ElementSizeBytes: 2}
	if got := cfg2d.BoxBytes(); got != 64*32*2 {
		t.Errorf("2D BoxBytes = %d, want %d", got, 64*32*2)
	}

	cfg3d := TMAConfig{Dim: TMA3D, BoxDimX: 16, BoxDimY: 16, BoxDimZ: 4, ElementSizeBytes: 4}
	if got := cfg3d.BoxBytes(); got != 16*16*4*4 {
		t.Errorf("3D BoxBytes = %d, want %d", got, 16*16*4*4)
	}
}

func TestIsTMACompatible(t *testing.T) {
	tests := []struct {
		name             string
		rows, cols, elem int
		want             bool
	}{
		{"valid FP16 tile", 64, 64, 2, true},
		{"valid FP32 tile", 32, 32, 4, true},
		{"unaligned row", 64, 3, 2, false},
		{"too large", 256, 256, 4, false},
		{"zero rows", 0, 64, 2, false},
		{"zero cols", 64, 0, 2, false},
		{"zero elem", 64, 64, 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsTMACompatible(tc.rows, tc.cols, tc.elem)
			if got != tc.want {
				t.Errorf("IsTMACompatible(%d, %d, %d) = %v, want %v",
					tc.rows, tc.cols, tc.elem, got, tc.want)
			}
		})
	}
}

func TestRecommendSwizzle(t *testing.T) {
	tests := []struct {
		elemSize  int
		tileWidth int
		want      SwizzlePattern
	}{
		{2, 64, Swizzle128B},
		{2, 32, Swizzle64B},
		{2, 16, Swizzle32B},
		{1, 8, SwizzleNone},
		{4, 32, Swizzle128B},
		{4, 16, Swizzle64B},
	}
	for _, tc := range tests {
		got := RecommendSwizzle(tc.elemSize, tc.tileWidth)
		if got != tc.want {
			t.Errorf("RecommendSwizzle(%d, %d) = %v, want %v",
				tc.elemSize, tc.tileWidth, got, tc.want)
		}
	}
}

func TestSwizzlePattern_String(t *testing.T) {
	tests := []struct {
		s    SwizzlePattern
		want string
	}{
		{SwizzleNone, "none"},
		{Swizzle32B, "32B"},
		{Swizzle64B, "64B"},
		{Swizzle128B, "128B"},
		{SwizzlePattern(99), "unknown"},
	}
	for _, tc := range tests {
		if got := tc.s.String(); got != tc.want {
			t.Errorf("SwizzlePattern(%d).String() = %q, want %q", tc.s, got, tc.want)
		}
	}
}

func TestWGMMAConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     WGMMAConfig
		wantErr bool
	}{
		{
			name:    "valid FP16",
			cfg:     WGMMAConfig{M: 64, N: 128, K: 16, InputType: WGMMAFP16},
			wantErr: false,
		},
		{
			name:    "valid FP8",
			cfg:     WGMMAConfig{M: 64, N: 256, K: 32, InputType: WGMMAFP8E4M3},
			wantErr: false,
		},
		{
			name:    "valid BF16 small N",
			cfg:     WGMMAConfig{M: 64, N: 8, K: 16, InputType: WGMMABF16},
			wantErr: false,
		},
		{
			name:    "invalid M",
			cfg:     WGMMAConfig{M: 32, N: 128, K: 16, InputType: WGMMAFP16},
			wantErr: true,
		},
		{
			name:    "invalid N",
			cfg:     WGMMAConfig{M: 64, N: 13, K: 16, InputType: WGMMAFP16},
			wantErr: true,
		},
		{
			name:    "invalid K for FP16",
			cfg:     WGMMAConfig{M: 64, N: 128, K: 32, InputType: WGMMAFP16},
			wantErr: true,
		},
		{
			name:    "invalid K for FP8",
			cfg:     WGMMAConfig{M: 64, N: 128, K: 16, InputType: WGMMAFP8E4M3},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if (err != nil) != tc.wantErr {
				t.Errorf("Validate() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestWGMMAConfig_OutputElements(t *testing.T) {
	cfg := WGMMAConfig{M: 64, N: 128, K: 16}
	if got := cfg.OutputElements(); got != 64*128 {
		t.Errorf("OutputElements() = %d, want %d", got, 64*128)
	}
}

func TestSelectWGMMATile(t *testing.T) {
	tests := []struct {
		name    string
		m, n, k int
		dt      WGMMADataType
		wantN   int
		wantK   int
	}{
		{"large FP16", 1024, 1024, 1024, WGMMAFP16, 256, 16},
		{"medium BF16", 256, 128, 256, WGMMABF16, 128, 16},
		{"small FP8", 64, 32, 64, WGMMAFP8E4M3, 32, 32},
		{"tiny problem", 64, 4, 32, WGMMAINT8, 8, 32},
		{"N not multiple of 8", 64, 100, 64, WGMMAFP16, 96, 16},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := SelectWGMMATile(tc.m, tc.n, tc.k, tc.dt)
			if cfg.M != 64 {
				t.Errorf("M = %d, want 64", cfg.M)
			}
			if cfg.N != tc.wantN {
				t.Errorf("N = %d, want %d", cfg.N, tc.wantN)
			}
			if cfg.K != tc.wantK {
				t.Errorf("K = %d, want %d", cfg.K, tc.wantK)
			}
			if !cfg.AccumulatorFP32 {
				t.Error("expected FP32 accumulators")
			}
		})
	}
}

func TestEstimateWGMMAIterations(t *testing.T) {
	cfg := &WGMMAConfig{M: 64, N: 128, K: 16}

	// Exact fit: 1024/64=16 M-iters, 1024/128=8 N-iters, 1024/16=64 K-iters = 8192
	got := EstimateWGMMAIterations(1024, 1024, 1024, cfg)
	if got != 16*8*64 {
		t.Errorf("got %d iterations, want %d", got, 16*8*64)
	}

	// Rounded up: 65/64=2, 129/128=2, 17/16=2 = 8
	got = EstimateWGMMAIterations(65, 129, 17, cfg)
	if got != 8 {
		t.Errorf("got %d iterations, want 8", got)
	}

	// Zero config should return 0.
	got = EstimateWGMMAIterations(1024, 1024, 1024, &WGMMAConfig{})
	if got != 0 {
		t.Errorf("got %d iterations for zero config, want 0", got)
	}
}

func TestWGMMADataType_String(t *testing.T) {
	tests := []struct {
		dt   WGMMADataType
		want string
	}{
		{WGMMAFP16, "fp16"},
		{WGMMABF16, "bf16"},
		{WGMMAFP8E4M3, "fp8_e4m3"},
		{WGMMAFP8E5M2, "fp8_e5m2"},
		{WGMMAINT8, "int8"},
		{WGMMADataType(99), "unknown"},
	}
	for _, tc := range tests {
		if got := tc.dt.String(); got != tc.want {
			t.Errorf("WGMMADataType(%d).String() = %q, want %q", tc.dt, got, tc.want)
		}
	}
}
