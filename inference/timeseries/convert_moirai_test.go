package timeseries

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestMapMoiraiTensorName(t *testing.T) {
	tests := []struct {
		hfName string
		want   string
	}{
		// Encoder self-attention tensors.
		{"model.encoder.layers.0.self_attn.q_proj.weight", "moirai.enc.layer.0.self_attn.q_proj.weight"},
		{"model.encoder.layers.0.self_attn.k_proj.weight", "moirai.enc.layer.0.self_attn.k_proj.weight"},
		{"model.encoder.layers.0.self_attn.v_proj.weight", "moirai.enc.layer.0.self_attn.v_proj.weight"},
		{"model.encoder.layers.0.self_attn.out_proj.weight", "moirai.enc.layer.0.self_attn.out_proj.weight"},
		{"model.encoder.layers.0.self_attn.out_proj.bias", "moirai.enc.layer.0.self_attn.out_proj.bias"},
		// Encoder MLP tensors.
		{"model.encoder.layers.0.mlp.fc1.weight", "moirai.enc.layer.0.mlp.fc1.weight"},
		{"model.encoder.layers.0.mlp.fc1.bias", "moirai.enc.layer.0.mlp.fc1.bias"},
		{"model.encoder.layers.0.mlp.fc2.weight", "moirai.enc.layer.0.mlp.fc2.weight"},
		{"model.encoder.layers.0.mlp.fc2.bias", "moirai.enc.layer.0.mlp.fc2.bias"},
		// Encoder norm tensors.
		{"model.encoder.layers.0.norm1.weight", "moirai.enc.layer.0.norm1.weight"},
		{"model.encoder.layers.0.norm1.bias", "moirai.enc.layer.0.norm1.bias"},
		{"model.encoder.layers.0.norm2.weight", "moirai.enc.layer.0.norm2.weight"},
		{"model.encoder.layers.0.norm2.bias", "moirai.enc.layer.0.norm2.bias"},
		// Higher layer indices.
		{"model.encoder.layers.11.self_attn.q_proj.weight", "moirai.enc.layer.11.self_attn.q_proj.weight"},
		{"model.encoder.layers.11.mlp.fc1.weight", "moirai.enc.layer.11.mlp.fc1.weight"},
		// Global model tensors (patch embedding, output head, etc.).
		{"model.patch_embedding.weight", "moirai.patch_embedding.weight"},
		{"model.patch_embedding.bias", "moirai.patch_embedding.bias"},
		{"model.output_head.weight", "moirai.output_head.weight"},
		{"model.freq_embedding.weight", "moirai.freq_embedding.weight"},
		// Non-model-prefixed tensors.
		{"mask_token", "moirai.mask_token"},
	}

	for _, tt := range tests {
		got := MapMoiraiTensorName(tt.hfName)
		if got != tt.want {
			t.Errorf("MapMoiraiTensorName(%q) = %q, want %q", tt.hfName, got, tt.want)
		}
	}
}

func TestMoiraiConvertConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     MoiraiConvertConfig
		wantErr bool
	}{
		{
			name: "valid",
			cfg: MoiraiConvertConfig{
				NumLayers:         6,
				HiddenDim:         512,
				NumHeads:          8,
				NumFreqEmbeddings: 32,
			},
		},
		{
			name: "zero layers",
			cfg: MoiraiConvertConfig{
				NumLayers:         0,
				HiddenDim:         512,
				NumHeads:          8,
				NumFreqEmbeddings: 32,
			},
			wantErr: true,
		},
		{
			name: "zero hidden dim",
			cfg: MoiraiConvertConfig{
				NumLayers:         6,
				HiddenDim:         0,
				NumHeads:          8,
				NumFreqEmbeddings: 32,
			},
			wantErr: true,
		},
		{
			name: "zero heads",
			cfg: MoiraiConvertConfig{
				NumLayers:         6,
				HiddenDim:         512,
				NumHeads:          0,
				NumFreqEmbeddings: 32,
			},
			wantErr: true,
		},
		{
			name: "zero freq embeddings",
			cfg: MoiraiConvertConfig{
				NumLayers:         6,
				HiddenDim:         512,
				NumHeads:          8,
				NumFreqEmbeddings: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestConvertMoiraiToGGUF(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "model.safetensors")
	ggufPath := filepath.Join(dir, "model.gguf")

	// Create synthetic SafeTensors with tensors matching a 2-layer Moirai model.
	tensors := map[string][]float32{
		"model.encoder.layers.0.self_attn.q_proj.weight": {1, 2, 3, 4},
		"model.encoder.layers.0.self_attn.k_proj.weight": {5, 6, 7, 8},
		"model.encoder.layers.0.mlp.fc1.weight":          {9, 10, 11, 12},
		"model.encoder.layers.1.self_attn.q_proj.weight": {13, 14, 15, 16},
		"model.encoder.layers.1.mlp.fc1.weight":          {17, 18, 19, 20},
		"model.patch_embedding.weight":                    {21, 22},
		"model.output_head.weight":                        {23, 24},
		"model.freq_embedding.weight":                     {25, 26, 27, 28},
	}
	writeSyntheticSafeTensors(t, stPath, tensors)

	cfg := MoiraiConvertConfig{
		NumLayers:         2,
		HiddenDim:         64,
		NumHeads:          4,
		NumFreqEmbeddings: 16,
		ModelName:         "Moirai-2-test",
	}

	if err := ConvertMoiraiToGGUF(stPath, ggufPath, cfg); err != nil {
		t.Fatalf("ConvertMoiraiToGGUF: %v", err)
	}

	// Parse the output GGUF and verify metadata + tensors.
	f, err := os.Open(ggufPath)
	if err != nil {
		t.Fatalf("open GGUF: %v", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		t.Fatalf("parse GGUF: %v", err)
	}

	// Verify architecture metadata.
	if arch, ok := gf.GetString("general.architecture"); !ok || arch != "moirai" {
		t.Errorf("general.architecture = %q, want %q", arch, "moirai")
	}
	if name, ok := gf.GetString("general.name"); !ok || name != "Moirai-2-test" {
		t.Errorf("general.name = %q, want %q", name, "Moirai-2-test")
	}
	if nl, ok := gf.GetUint32("moirai.num_layers"); !ok || nl != 2 {
		t.Errorf("moirai.num_layers = %d, want 2", nl)
	}
	if hd, ok := gf.GetUint32("moirai.hidden_dim"); !ok || hd != 64 {
		t.Errorf("moirai.hidden_dim = %d, want 64", hd)
	}
	if nh, ok := gf.GetUint32("moirai.num_heads"); !ok || nh != 4 {
		t.Errorf("moirai.num_heads = %d, want 4", nh)
	}
	if nfe, ok := gf.GetUint32("moirai.num_freq_embeddings"); !ok || nfe != 16 {
		t.Errorf("moirai.num_freq_embeddings = %d, want 16", nfe)
	}

	// Verify tensor count.
	if len(gf.Tensors) != len(tensors) {
		t.Fatalf("tensor count = %d, want %d", len(gf.Tensors), len(tensors))
	}

	// Verify tensor names were mapped correctly.
	expectedNames := map[string]bool{
		"moirai.enc.layer.0.self_attn.q_proj.weight": true,
		"moirai.enc.layer.0.self_attn.k_proj.weight": true,
		"moirai.enc.layer.0.mlp.fc1.weight":          true,
		"moirai.enc.layer.1.self_attn.q_proj.weight": true,
		"moirai.enc.layer.1.mlp.fc1.weight":          true,
		"moirai.patch_embedding.weight":               true,
		"moirai.output_head.weight":                   true,
		"moirai.freq_embedding.weight":                true,
	}
	for _, ti := range gf.Tensors {
		if !expectedNames[ti.Name] {
			t.Errorf("unexpected tensor name %q", ti.Name)
		}
		delete(expectedNames, ti.Name)
	}
	for name := range expectedNames {
		t.Errorf("missing expected tensor %q", name)
	}

	// Verify tensor data round-trips correctly.
	loadedTensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	// Spot-check one tensor's data.
	lt, ok := loadedTensors["moirai.enc.layer.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatal("missing moirai.enc.layer.0.self_attn.q_proj.weight in loaded tensors")
	}
	want := []float32{1, 2, 3, 4}
	got := lt.Data()
	if len(got) != len(want) {
		t.Fatalf("tensor data length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tensor data[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
