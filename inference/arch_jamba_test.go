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

// makeJambaTestTensors creates a minimal set of Jamba-architecture tensors
// with the naming convention used by GGUF Jamba models. Attention layers
// at indices that are multiples of AttentionLayerOffset use attn_* and ffn_*
// tensor names; all other layers use ssm_* tensor names.
func makeJambaTestTensors(jc JambaConfig) map[string]*tensor.TensorNumeric[float32] {
	tensors := make(map[string]*tensor.TensorNumeric[float32])

	dInner := jc.IntermediateSize
	if dInner == 0 {
		dInner = jc.HiddenSize * 2
	}
	dState := jc.SSMHeads
	if dState == 0 {
		dState = 16
	}
	dConv := jc.DConv
	if dConv == 0 {
		dConv = 4
	}
	dtRank := int(math.Ceil(float64(jc.HiddenSize) / 16))
	kvDim := (jc.HiddenSize / jc.AttnHeads) * jc.KVHeads

	fill := func(shape []int, scale float32) *tensor.TensorNumeric[float32] {
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
	ones := func(shape []int) *tensor.TensorNumeric[float32] {
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
	logInit := func(shape []int) *tensor.TensorNumeric[float32] {
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

	// Global tensors.
	tensors["token_embd.weight"] = fill([]int{jc.VocabSize, jc.HiddenSize}, 0.02)
	tensors["output.weight"] = fill([]int{jc.VocabSize, jc.HiddenSize}, 0.02)
	tensors["output_norm.weight"] = ones([]int{jc.HiddenSize})

	// Per-layer tensors.
	for i := 0; i < jc.NumLayers; i++ {
		prefix := "blk." + itoa(i) + "."

		if jc.isAttentionLayer(i) {
			// Attention layer tensors.
			tensors[prefix+"attn_norm.weight"] = ones([]int{jc.HiddenSize})
			tensors[prefix+"attn_q.weight"] = fill([]int{jc.HiddenSize, jc.HiddenSize}, 0.02)
			tensors[prefix+"attn_k.weight"] = fill([]int{kvDim, jc.HiddenSize}, 0.02)
			tensors[prefix+"attn_v.weight"] = fill([]int{kvDim, jc.HiddenSize}, 0.02)
			tensors[prefix+"attn_output.weight"] = fill([]int{jc.HiddenSize, jc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_norm.weight"] = ones([]int{jc.HiddenSize})
			tensors[prefix+"ffn_gate.weight"] = fill([]int{jc.IntermediateSize, jc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_up.weight"] = fill([]int{jc.IntermediateSize, jc.HiddenSize}, 0.02)
			tensors[prefix+"ffn_down.weight"] = fill([]int{jc.HiddenSize, jc.IntermediateSize}, 0.02)
		} else {
			// SSM layer tensors.
			tensors[prefix+"ssm_norm.weight"] = ones([]int{jc.HiddenSize})
			tensors[prefix+"ssm_in_proj.weight"] = fill([]int{2 * dInner, jc.HiddenSize}, 0.02)
			tensors[prefix+"ssm_conv1d.weight"] = fill([]int{dInner, 1, dConv}, 0.02)
			tensors[prefix+"ssm_x_proj.weight"] = fill([]int{dtRank + 2*dState, dInner}, 0.02)
			tensors[prefix+"ssm_dt_proj.weight"] = fill([]int{dInner, dtRank}, 0.02)
			tensors[prefix+"ssm_A_log"] = logInit([]int{dInner, dState})
			tensors[prefix+"ssm_D"] = ones([]int{dInner})
			tensors[prefix+"ssm_out_proj.weight"] = fill([]int{jc.HiddenSize, dInner}, 0.02)
		}
	}

	return tensors
}

func TestBuildJamba(t *testing.T) {
	jc := JambaConfig{
		NumLayers:            4,
		HiddenSize:           16,
		IntermediateSize:     32,
		AttnHeads:            4,
		KVHeads:              2,
		SSMHeads:             4,
		AttentionLayerOffset: 2, // layers 0,2 are attention; layers 1,3 are SSM
		RMSEps:               1e-5,
		VocabSize:            32,
		MaxSeqLen:            64,
		RopeTheta:            10000,
		DConv:                4,
	}

	tensors := makeJambaTestTensors(jc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, emb, err := BuildJamba(jc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildJamba: %v", err)
	}
	if g == nil {
		t.Fatal("graph is nil")
	}
	if emb == nil {
		t.Fatal("embedding is nil")
	}
}

func TestBuildJamba_ForwardNonNaN(t *testing.T) {
	jc := JambaConfig{
		NumLayers:            4,
		HiddenSize:           16,
		IntermediateSize:     32,
		AttnHeads:            4,
		KVHeads:              2,
		SSMHeads:             4,
		AttentionLayerOffset: 2,
		RMSEps:               1e-5,
		VocabSize:            32,
		MaxSeqLen:            64,
		RopeTheta:            10000,
		DConv:                4,
	}

	tensors := makeJambaTestTensors(jc)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, _, err := BuildJamba(jc, tensors, engine)
	if err != nil {
		t.Fatalf("BuildJamba: %v", err)
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
	if len(shape) != 3 || shape[0] != 1 || shape[1] != seqLen || shape[2] != jc.VocabSize {
		t.Fatalf("unexpected output shape: %v, want [1, %d, %d]", shape, seqLen, jc.VocabSize)
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

func TestBuildJamba_MissingTensor(t *testing.T) {
	jc := JambaConfig{
		NumLayers:            2,
		HiddenSize:           16,
		IntermediateSize:     32,
		AttnHeads:            4,
		KVHeads:              2,
		SSMHeads:             4,
		AttentionLayerOffset: 2,
		RMSEps:               1e-5,
		VocabSize:            32,
		MaxSeqLen:            64,
		RopeTheta:            10000,
		DConv:                4,
	}

	tensors := make(map[string]*tensor.TensorNumeric[float32])
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	_, _, err := BuildJamba(jc, tensors, engine)
	if err == nil {
		t.Fatal("expected error for missing tensors")
	}
}

func TestJambaConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name string
		cfg  *gguf.ModelConfig
		want JambaConfig
	}{
		{
			name: "standard fields",
			cfg: &gguf.ModelConfig{
				NumLayers:        16,
				HiddenSize:       768,
				IntermediateSize: 2048,
				NumHeads:         12,
				NumKVHeads:       4,
				VocabSize:        50280,
				MaxSeqLen:        4096,
				RopeTheta:        10000,
				RMSNormEps:       1e-6,
			},
			want: JambaConfig{
				NumLayers:            16,
				HiddenSize:           768,
				IntermediateSize:     2048,
				AttnHeads:            12,
				KVHeads:              4,
				SSMHeads:             4,
				AttentionLayerOffset: 8,
				RMSEps:               1e-6,
				VocabSize:            50280,
				MaxSeqLen:            4096,
				RopeTheta:            10000,
				DConv:                4,
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
			want: JambaConfig{
				NumLayers:            8,
				HiddenSize:           256,
				AttnHeads:            4,
				KVHeads:              0,
				SSMHeads:             16,
				AttentionLayerOffset: 8,
				RMSEps:               1e-5,
				VocabSize:            1000,
				RopeTheta:            10000,
				DConv:                4,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := JambaConfigFromGGUF(tt.cfg)
			if got.NumLayers != tt.want.NumLayers {
				t.Errorf("NumLayers = %d, want %d", got.NumLayers, tt.want.NumLayers)
			}
			if got.HiddenSize != tt.want.HiddenSize {
				t.Errorf("HiddenSize = %d, want %d", got.HiddenSize, tt.want.HiddenSize)
			}
			if got.IntermediateSize != tt.want.IntermediateSize {
				t.Errorf("IntermediateSize = %d, want %d", got.IntermediateSize, tt.want.IntermediateSize)
			}
			if got.AttnHeads != tt.want.AttnHeads {
				t.Errorf("AttnHeads = %d, want %d", got.AttnHeads, tt.want.AttnHeads)
			}
			if got.KVHeads != tt.want.KVHeads {
				t.Errorf("KVHeads = %d, want %d", got.KVHeads, tt.want.KVHeads)
			}
			if got.SSMHeads != tt.want.SSMHeads {
				t.Errorf("SSMHeads = %d, want %d", got.SSMHeads, tt.want.SSMHeads)
			}
			if got.AttentionLayerOffset != tt.want.AttentionLayerOffset {
				t.Errorf("AttentionLayerOffset = %d, want %d", got.AttentionLayerOffset, tt.want.AttentionLayerOffset)
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
		})
	}
}

func TestJambaIsAttentionLayer(t *testing.T) {
	jc := JambaConfig{AttentionLayerOffset: 4}

	tests := []struct {
		layer int
		want  bool
	}{
		{0, true},
		{1, false},
		{2, false},
		{3, false},
		{4, true},
		{7, false},
		{8, true},
	}

	for _, tt := range tests {
		if got := jc.isAttentionLayer(tt.layer); got != tt.want {
			t.Errorf("isAttentionLayer(%d) = %v, want %v", tt.layer, got, tt.want)
		}
	}
}
