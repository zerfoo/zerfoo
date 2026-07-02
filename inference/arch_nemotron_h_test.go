package inference

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// nemotronHTestConfig returns a minimal NemotronHConfig for testing.
func nemotronHTestConfig() NemotronHConfig {
	return NemotronHConfig{
		NumLayers:        4,
		HiddenSize:       16,
		IntermediateSize: 32,
		AttnHeads:        4,
		KVHeads:          2,
		SSMStateSize:     4,
		SSMConvKernel:    4,
		SSMNumHeads:      2,
		RMSEps:           1e-5,
		VocabSize:        32,
		MaxSeqLen:        64,
		RopeTheta:        10000,
	}
}

// fillTensor creates a tensor with deterministic non-zero values.
func fillTensor(shape []int, scale float32) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = scale * float32(math.Sin(float64(i)*0.01))
	}
	t, _ := tensor.New(shape, data)
	return t
}

// onesTensor creates a tensor filled with ones.
func onesTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = 1.0
	}
	t, _ := tensor.New(shape, data)
	return t
}

// logInitTensor creates a tensor with log-space initialization.
func logInitTensor(shape []int) *tensor.TensorNumeric[float32] {
	size := 1
	for _, d := range shape {
		size *= d
	}
	data := make([]float32, size)
	cols := shape[len(shape)-1]
	for i := range data {
		data[i] = float32(math.Log(float64(i%cols + 1)))
	}
	t, _ := tensor.New(shape, data)
	return t
}

// makeNemotronHDenseTestTensors creates synthetic tensors for a 4-layer
// Nemotron-H dense model: layers 0,1 are Mamba, layer 2 is Attention,
// layer 3 is Dense FFN.
func makeNemotronHDenseTestTensors(nc NemotronHConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	dInner := nc.IntermediateSize
	if dInner == 0 {
		dInner = nc.HiddenSize * 2
	}
	kvDim := (nc.HiddenSize / nc.AttnHeads) * nc.KVHeads
	dtRank := int(math.Ceil(float64(nc.HiddenSize) / 16))

	// Global tensors.
	tensors["token_embd.weight"] = fillTensor([]int{nc.VocabSize, nc.HiddenSize}, 0.02)
	tensors["output.weight"] = fillTensor([]int{nc.VocabSize, nc.HiddenSize}, 0.02)
	tensors["output_norm.weight"] = onesTensor([]int{nc.HiddenSize})

	for i := 0; i < nc.NumLayers; i++ {
		prefix := "blk." + itoa(i) + "."
		tensors[prefix+"attn_norm.weight"] = onesTensor([]int{nc.HiddenSize})

		switch {
		case i < 2:
			// Mamba-2 layers.
			tensors[prefix+"ssm_in.weight"] = fillTensor([]int{2 * dInner, nc.HiddenSize}, 0.02)
			tensors[prefix+"ssm_conv1d.weight"] = fillTensor([]int{dInner, 1, nc.SSMConvKernel}, 0.02)
			tensors[prefix+"ssm_dt.weight"] = fillTensor([]int{dInner, dtRank}, 0.02)
			tensors[prefix+"ssm_A.weight"] = logInitTensor([]int{dInner, nc.SSMStateSize})
			tensors[prefix+"ssm_D.weight"] = onesTensor([]int{dInner})
			tensors[prefix+"ssm_out.weight"] = fillTensor([]int{nc.HiddenSize, dInner}, 0.02)

		case i == 2:
			// Attention layer.
			tensors[prefix+"attn_q.weight"] = fillTensor([]int{nc.HiddenSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_k.weight"] = fillTensor([]int{kvDim, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_v.weight"] = fillTensor([]int{kvDim, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_output.weight"] = fillTensor([]int{nc.HiddenSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_norm.weight"] = onesTensor([]int{nc.HiddenSize})
			tensors[prefix+"ffn_gate.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_up.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_down.weight"] = fillTensor([]int{nc.HiddenSize, nc.IntermediateSize}, 0.02)

		case i == 3:
			// Dense FFN layer.
			tensors[prefix+"ffn_gate.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_up.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_down.weight"] = fillTensor([]int{nc.HiddenSize, nc.IntermediateSize}, 0.02)
		}
	}

	return tensors
}

// makeNemotronHMoETestTensors creates synthetic tensors for a 4-layer
// Nemotron-H MoE model: layer 0 is Mamba, layer 1 is Attention,
// layers 2-3 are MoE.
func makeNemotronHMoETestTensors(nc NemotronHConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	dInner := nc.IntermediateSize
	if dInner == 0 {
		dInner = nc.HiddenSize * 2
	}
	kvDim := (nc.HiddenSize / nc.AttnHeads) * nc.KVHeads
	dtRank := int(math.Ceil(float64(nc.HiddenSize) / 16))
	numExperts := nc.NumExperts
	if numExperts == 0 {
		numExperts = 4
	}

	// Global tensors.
	tensors["token_embd.weight"] = fillTensor([]int{nc.VocabSize, nc.HiddenSize}, 0.02)
	tensors["output.weight"] = fillTensor([]int{nc.VocabSize, nc.HiddenSize}, 0.02)
	tensors["output_norm.weight"] = onesTensor([]int{nc.HiddenSize})

	for i := 0; i < nc.NumLayers; i++ {
		prefix := "blk." + itoa(i) + "."
		tensors[prefix+"attn_norm.weight"] = onesTensor([]int{nc.HiddenSize})

		switch {
		case i == 0:
			// Mamba-2 layer.
			tensors[prefix+"ssm_in.weight"] = fillTensor([]int{2 * dInner, nc.HiddenSize}, 0.02)
			tensors[prefix+"ssm_conv1d.weight"] = fillTensor([]int{dInner, 1, nc.SSMConvKernel}, 0.02)
			tensors[prefix+"ssm_dt.weight"] = fillTensor([]int{dInner, dtRank}, 0.02)
			tensors[prefix+"ssm_A.weight"] = logInitTensor([]int{dInner, nc.SSMStateSize})
			tensors[prefix+"ssm_D.weight"] = onesTensor([]int{dInner})
			tensors[prefix+"ssm_out.weight"] = fillTensor([]int{nc.HiddenSize, dInner}, 0.02)

		case i == 1:
			// Attention layer.
			tensors[prefix+"attn_q.weight"] = fillTensor([]int{nc.HiddenSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_k.weight"] = fillTensor([]int{kvDim, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_v.weight"] = fillTensor([]int{kvDim, nc.HiddenSize}, 0.02)
			tensors[prefix+"attn_output.weight"] = fillTensor([]int{nc.HiddenSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_norm.weight"] = onesTensor([]int{nc.HiddenSize})
			tensors[prefix+"ffn_gate.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_up.weight"] = fillTensor([]int{nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_down.weight"] = fillTensor([]int{nc.HiddenSize, nc.IntermediateSize}, 0.02)

		default:
			// MoE layers.
			tensors[prefix+"ffn_gate_inp.weight"] = fillTensor([]int{numExperts, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_gate_exps.weight"] = fillTensor([]int{numExperts, nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_up_exps.weight"] = fillTensor([]int{numExperts, nc.IntermediateSize, nc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_down_exps.weight"] = fillTensor([]int{numExperts, nc.HiddenSize, nc.IntermediateSize}, 0.02)
		}
	}

	return tensors
}

func TestBuildNemotronH_Dense(t *testing.T) {
	nc := nemotronHTestConfig()
	tensors := makeNemotronHDenseTestTensors(nc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildNemotronH(nc, tensors, engine, false)
	if err != nil {
		t.Fatalf("BuildNemotronH: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildNemotronH_MoE(t *testing.T) {
	nc := nemotronHTestConfig()
	nc.NumExperts = 4
	nc.NumExpertsPerToken = 2
	nc.NumSharedExperts = 0
	tensors := makeNemotronHMoETestTensors(nc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildNemotronH(nc, tensors, engine, true)
	if err != nil {
		t.Fatalf("BuildNemotronH MoE: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildNemotronH_DenseForwardNonNaN(t *testing.T) {
	nc := nemotronHTestConfig()
	tensors := makeNemotronHDenseTestTensors(nc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildNemotronH(nc, tensors, engine, false)
	if err != nil {
		t.Fatalf("BuildNemotronH: %v", err)
	}

	tokenIDs := []float32{1, 5, 10, 3}
	seqLen := len(tokenIDs)

	input, err := tensor.New([]int{1, seqLen}, tokenIDs)
	if err != nil {
		t.Fatalf("create input tensor: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != nc.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, nc.VocabSize)
	}

	data := output.Data()
	for i, v := range data {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			t.Fatalf("Inf at index %d", i)
		}
	}
}

func TestDetectNemotronHLayerType(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		want    nemotronHLayerType
		wantStr string
	}{
		{
			name:    "mamba layer",
			key:     "blk.0.ssm_in.weight",
			want:    nemotronHLayerMamba,
			wantStr: "mamba",
		},
		{
			name:    "attention layer",
			key:     "blk.1.attn_q.weight",
			want:    nemotronHLayerAttn,
			wantStr: "attention",
		},
		{
			name:    "moe layer",
			key:     "blk.2.ffn_gate_inp.weight",
			want:    nemotronHLayerMoE,
			wantStr: "moe",
		},
		{
			name:    "ffn layer",
			key:     "blk.3.ffn_gate.weight",
			want:    nemotronHLayerFFN,
			wantStr: "ffn",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensors := map[string]*tensor.TensorNumeric[float32]{
				tt.key: onesTensor([]int{1}),
			}
			// Extract layer index from test key.
			var layerIdx int
			switch tt.wantStr {
			case "mamba":
				layerIdx = 0
			case "attention":
				layerIdx = 1
			case "moe":
				layerIdx = 2
			case "ffn":
				layerIdx = 3
			}
			got := detectNemotronHLayerType(tensors, layerIdx)
			if got != tt.want {
				t.Errorf("detectNemotronHLayerType = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestNemotronHConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want NemotronHConfig
	}{
		{
			name: "standard fields",
			cfg: &gguf.ModelConfig{
				NumLayers:          32,
				HiddenSize:         4096,
				IntermediateSize:   8192,
				NumHeads:           32,
				NumKVHeads:         8,
				VocabSize:          131072,
				MaxSeqLen:          8192,
				RopeTheta:          500000,
				RMSNormEps:         1e-6,
				SSMStateSize:       128,
				SSMConvKernel:      4,
				SSMNumHeads:        64,
				NumExperts:         128,
				NumExpertsPerToken: 6,
				ExpertSharedCount:  2,
			},
			want: NemotronHConfig{
				NumLayers:          32,
				HiddenSize:         4096,
				IntermediateSize:   8192,
				AttnHeads:          32,
				KVHeads:            8,
				SSMStateSize:       128,
				SSMConvKernel:      4,
				SSMNumHeads:        64,
				RMSEps:             1e-6,
				VocabSize:          131072,
				MaxSeqLen:          8192,
				RopeTheta:          500000,
				NumExperts:         128,
				NumExpertsPerToken: 6,
				NumSharedExperts:   2,
			},
		},
		{
			name: "defaults",
			cfg: &gguf.ModelConfig{
				NumLayers:  8,
				HiddenSize: 256,
				NumHeads:   4,
				VocabSize:  1000,
			},
			want: NemotronHConfig{
				NumLayers:    8,
				HiddenSize:   256,
				AttnHeads:    4,
				KVHeads:      4,
				SSMStateSize: 16,
				SSMConvKernel: 4,
				SSMNumHeads:  1,
				RMSEps:       1e-5,
				VocabSize:    1000,
				RopeTheta:    10000,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NemotronHConfigFromGGUF(tt.cfg)
			if got.NumLayers != tt.want.NumLayers {
				t.Errorf("NumLayers = %d, want %d", got.NumLayers, tt.want.NumLayers)
			}
			if got.HiddenSize != tt.want.HiddenSize {
				t.Errorf("HiddenSize = %d, want %d", got.HiddenSize, tt.want.HiddenSize)
			}
			if got.AttnHeads != tt.want.AttnHeads {
				t.Errorf("AttnHeads = %d, want %d", got.AttnHeads, tt.want.AttnHeads)
			}
			if got.KVHeads != tt.want.KVHeads {
				t.Errorf("KVHeads = %d, want %d", got.KVHeads, tt.want.KVHeads)
			}
			if got.SSMStateSize != tt.want.SSMStateSize {
				t.Errorf("SSMStateSize = %d, want %d", got.SSMStateSize, tt.want.SSMStateSize)
			}
			if got.SSMConvKernel != tt.want.SSMConvKernel {
				t.Errorf("SSMConvKernel = %d, want %d", got.SSMConvKernel, tt.want.SSMConvKernel)
			}
			if got.SSMNumHeads != tt.want.SSMNumHeads {
				t.Errorf("SSMNumHeads = %d, want %d", got.SSMNumHeads, tt.want.SSMNumHeads)
			}
			if got.RMSEps != tt.want.RMSEps {
				t.Errorf("RMSEps = %v, want %v", got.RMSEps, tt.want.RMSEps)
			}
			if got.VocabSize != tt.want.VocabSize {
				t.Errorf("VocabSize = %d, want %d", got.VocabSize, tt.want.VocabSize)
			}
			if got.RopeTheta != tt.want.RopeTheta {
				t.Errorf("RopeTheta = %v, want %v", got.RopeTheta, tt.want.RopeTheta)
			}
			if got.NumExperts != tt.want.NumExperts {
				t.Errorf("NumExperts = %d, want %d", got.NumExperts, tt.want.NumExperts)
			}
			if got.NumExpertsPerToken != tt.want.NumExpertsPerToken {
				t.Errorf("NumExpertsPerToken = %d, want %d", got.NumExpertsPerToken, tt.want.NumExpertsPerToken)
			}
			if got.NumSharedExperts != tt.want.NumSharedExperts {
				t.Errorf("NumSharedExperts = %d, want %d", got.NumSharedExperts, tt.want.NumSharedExperts)
			}
		})
	}
}

func TestNemotronH_Registration(t *testing.T) {
	tests := []struct {
		name string
		arch string
	}{
		{"nemotron_h", "nemotron_h"},
		{"nemotron_h_moe", "nemotron_h_moe"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, ok := GetArchitecture(tt.arch)
			if !ok || b == nil {
				t.Fatalf("architecture %q not registered", tt.arch)
			}
		})
	}
}

func TestBuildNemotronH_MissingTensor(t *testing.T) {
	nc := nemotronHTestConfig()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildNemotronH(nc, tensors, engine, false)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}
