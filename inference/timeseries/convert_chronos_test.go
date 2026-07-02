package timeseries

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestMapChronosTensorName(t *testing.T) {
	tests := []struct {
		hfName string
		want   string
	}{
		// Global / embedding tensors.
		{"shared.weight", "chronos.token_embd.weight"},
		{"lm_head.weight", "chronos.lm_head.weight"},
		{"encoder.embed_tokens.weight", "chronos.enc.token_embd.weight"},
		{"decoder.embed_tokens.weight", "chronos.dec.token_embd.weight"},

		// Encoder final norm.
		{"encoder.final_layer_norm.weight", "chronos.enc.final_norm.weight"},
		// Decoder final norm.
		{"decoder.final_layer_norm.weight", "chronos.dec.final_norm.weight"},

		// Encoder block 0 self-attention.
		{"encoder.block.0.layer.0.SelfAttention.q.weight", "chronos.enc.block.0.attn.q.weight"},
		{"encoder.block.0.layer.0.SelfAttention.k.weight", "chronos.enc.block.0.attn.k.weight"},
		{"encoder.block.0.layer.0.SelfAttention.v.weight", "chronos.enc.block.0.attn.v.weight"},
		{"encoder.block.0.layer.0.SelfAttention.o.weight", "chronos.enc.block.0.attn.o.weight"},
		{"encoder.block.0.layer.0.layer_norm.weight", "chronos.enc.block.0.attn_norm.weight"},

		// Encoder block 0 FFN.
		{"encoder.block.0.layer.1.DenseReluDense.wi.weight", "chronos.enc.block.0.ffn.wi.weight"},
		{"encoder.block.0.layer.1.DenseReluDense.wo.weight", "chronos.enc.block.0.ffn.wo.weight"},
		{"encoder.block.0.layer.1.layer_norm.weight", "chronos.enc.block.0.ffn_norm.weight"},

		// Encoder higher block.
		{"encoder.block.5.layer.0.SelfAttention.q.weight", "chronos.enc.block.5.attn.q.weight"},
		{"encoder.block.5.layer.1.DenseReluDense.wi.weight", "chronos.enc.block.5.ffn.wi.weight"},

		// Decoder block 0 self-attention.
		{"decoder.block.0.layer.0.SelfAttention.q.weight", "chronos.dec.block.0.self_attn.q.weight"},
		{"decoder.block.0.layer.0.SelfAttention.k.weight", "chronos.dec.block.0.self_attn.k.weight"},
		{"decoder.block.0.layer.0.SelfAttention.v.weight", "chronos.dec.block.0.self_attn.v.weight"},
		{"decoder.block.0.layer.0.SelfAttention.o.weight", "chronos.dec.block.0.self_attn.o.weight"},
		{"decoder.block.0.layer.0.layer_norm.weight", "chronos.dec.block.0.self_attn_norm.weight"},

		// Decoder block 0 cross-attention.
		{"decoder.block.0.layer.1.EncDecAttention.q.weight", "chronos.dec.block.0.cross_attn.q.weight"},
		{"decoder.block.0.layer.1.EncDecAttention.k.weight", "chronos.dec.block.0.cross_attn.k.weight"},
		{"decoder.block.0.layer.1.EncDecAttention.v.weight", "chronos.dec.block.0.cross_attn.v.weight"},
		{"decoder.block.0.layer.1.EncDecAttention.o.weight", "chronos.dec.block.0.cross_attn.o.weight"},
		{"decoder.block.0.layer.1.layer_norm.weight", "chronos.dec.block.0.cross_attn_norm.weight"},

		// Decoder block 0 FFN.
		{"decoder.block.0.layer.2.DenseReluDense.wi.weight", "chronos.dec.block.0.ffn.wi.weight"},
		{"decoder.block.0.layer.2.DenseReluDense.wo.weight", "chronos.dec.block.0.ffn.wo.weight"},
		{"decoder.block.0.layer.2.layer_norm.weight", "chronos.dec.block.0.ffn_norm.weight"},

		// Decoder higher block.
		{"decoder.block.3.layer.0.SelfAttention.v.weight", "chronos.dec.block.3.self_attn.v.weight"},
		{"decoder.block.3.layer.1.EncDecAttention.k.weight", "chronos.dec.block.3.cross_attn.k.weight"},
		{"decoder.block.3.layer.2.DenseReluDense.wo.weight", "chronos.dec.block.3.ffn.wo.weight"},

		// Relative attention bias (encoder block 0 only).
		{"encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", "chronos.enc.attn_rel_bias.weight"},
		// Relative attention bias (decoder block 0).
		{"decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", "chronos.dec.block.0.self_attn_rel_bias.weight"},
	}

	for _, tt := range tests {
		got := MapChronosTensorName(tt.hfName)
		if got != tt.want {
			t.Errorf("MapChronosTensorName(%q) = %q, want %q", tt.hfName, got, tt.want)
		}
	}
}

func TestChronosConvertConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     ChronosConvertConfig
		wantErr bool
	}{
		{
			name: "valid",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 6,
				DModel:           512,
				NumHeads:          8,
				DFF:               2048,
				VocabSize:         32128,
			},
		},
		{
			name: "zero encoder layers",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 0,
				NumDecoderLayers: 6,
				DModel:           512,
				NumHeads:          8,
				DFF:               2048,
				VocabSize:         32128,
			},
			wantErr: true,
		},
		{
			name: "zero decoder layers",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 0,
				DModel:           512,
				NumHeads:          8,
				DFF:               2048,
				VocabSize:         32128,
			},
			wantErr: true,
		},
		{
			name: "zero d_model",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 6,
				DModel:           0,
				NumHeads:          8,
				DFF:               2048,
				VocabSize:         32128,
			},
			wantErr: true,
		},
		{
			name: "zero num_heads",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 6,
				DModel:           512,
				NumHeads:          0,
				DFF:               2048,
				VocabSize:         32128,
			},
			wantErr: true,
		},
		{
			name: "zero d_ff",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 6,
				DModel:           512,
				NumHeads:          8,
				DFF:               0,
				VocabSize:         32128,
			},
			wantErr: true,
		},
		{
			name: "zero vocab_size",
			cfg: ChronosConvertConfig{
				NumEncoderLayers: 6,
				NumDecoderLayers: 6,
				DModel:           512,
				NumHeads:          8,
				DFF:               2048,
				VocabSize:         0,
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

func TestConvertChronosToGGUF(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "model.safetensors")
	ggufPath := filepath.Join(dir, "model.gguf")

	// Create synthetic SafeTensors with tensors matching a 1-encoder, 1-decoder
	// Chronos T5 model.
	tensors := map[string][]float32{
		// Embeddings.
		"shared.weight":              {1, 2, 3, 4},
		"encoder.embed_tokens.weight": {5, 6, 7, 8},
		"decoder.embed_tokens.weight": {9, 10, 11, 12},
		// Encoder block 0 self-attention.
		"encoder.block.0.layer.0.SelfAttention.q.weight": {13, 14},
		"encoder.block.0.layer.0.SelfAttention.k.weight": {15, 16},
		"encoder.block.0.layer.0.SelfAttention.v.weight": {17, 18},
		"encoder.block.0.layer.0.SelfAttention.o.weight": {19, 20},
		"encoder.block.0.layer.0.layer_norm.weight":      {21, 22},
		// Encoder block 0 FFN.
		"encoder.block.0.layer.1.DenseReluDense.wi.weight": {23, 24},
		"encoder.block.0.layer.1.DenseReluDense.wo.weight": {25, 26},
		"encoder.block.0.layer.1.layer_norm.weight":        {27, 28},
		// Encoder final norm.
		"encoder.final_layer_norm.weight": {29, 30},
		// Decoder block 0 self-attention.
		"decoder.block.0.layer.0.SelfAttention.q.weight": {31, 32},
		"decoder.block.0.layer.0.layer_norm.weight":      {33, 34},
		// Decoder block 0 cross-attention.
		"decoder.block.0.layer.1.EncDecAttention.q.weight": {35, 36},
		"decoder.block.0.layer.1.layer_norm.weight":        {37, 38},
		// Decoder block 0 FFN.
		"decoder.block.0.layer.2.DenseReluDense.wi.weight": {39, 40},
		"decoder.block.0.layer.2.layer_norm.weight":        {41, 42},
		// Decoder final norm.
		"decoder.final_layer_norm.weight": {43, 44},
		// LM head.
		"lm_head.weight": {45, 46},
	}
	writeSyntheticSafeTensors(t, stPath, tensors)

	cfg := ChronosConvertConfig{
		NumEncoderLayers: 1,
		NumDecoderLayers: 1,
		DModel:           64,
		NumHeads:          4,
		DFF:               256,
		VocabSize:         4,
		ModelName:         "chronos-test",
	}

	if err := ConvertChronosToGGUF(stPath, ggufPath, cfg); err != nil {
		t.Fatalf("ConvertChronosToGGUF: %v", err)
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
	if arch, ok := gf.GetString("general.architecture"); !ok || arch != "chronos" {
		t.Errorf("general.architecture = %q, want %q", arch, "chronos")
	}
	if name, ok := gf.GetString("general.name"); !ok || name != "chronos-test" {
		t.Errorf("general.name = %q, want %q", name, "chronos-test")
	}
	if bc, ok := gf.GetUint32("chronos.encoder_block_count"); !ok || bc != 1 {
		t.Errorf("chronos.encoder_block_count = %d, want 1", bc)
	}
	if bc, ok := gf.GetUint32("chronos.decoder_block_count"); !ok || bc != 1 {
		t.Errorf("chronos.decoder_block_count = %d, want 1", bc)
	}
	if dm, ok := gf.GetUint32("chronos.d_model"); !ok || dm != 64 {
		t.Errorf("chronos.d_model = %d, want 64", dm)
	}

	// Verify tensor count.
	if len(gf.Tensors) != len(tensors) {
		t.Fatalf("tensor count = %d, want %d", len(gf.Tensors), len(tensors))
	}

	// Verify tensor names were mapped correctly.
	expectedNames := map[string]bool{
		"chronos.token_embd.weight":              true,
		"chronos.enc.token_embd.weight":          true,
		"chronos.dec.token_embd.weight":          true,
		"chronos.enc.block.0.attn.q.weight":      true,
		"chronos.enc.block.0.attn.k.weight":      true,
		"chronos.enc.block.0.attn.v.weight":      true,
		"chronos.enc.block.0.attn.o.weight":      true,
		"chronos.enc.block.0.attn_norm.weight":    true,
		"chronos.enc.block.0.ffn.wi.weight":      true,
		"chronos.enc.block.0.ffn.wo.weight":      true,
		"chronos.enc.block.0.ffn_norm.weight":     true,
		"chronos.enc.final_norm.weight":           true,
		"chronos.dec.block.0.self_attn.q.weight":  true,
		"chronos.dec.block.0.self_attn_norm.weight": true,
		"chronos.dec.block.0.cross_attn.q.weight": true,
		"chronos.dec.block.0.cross_attn_norm.weight": true,
		"chronos.dec.block.0.ffn.wi.weight":       true,
		"chronos.dec.block.0.ffn_norm.weight":      true,
		"chronos.dec.final_norm.weight":            true,
		"chronos.lm_head.weight":                  true,
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

	lt, ok := loadedTensors["chronos.token_embd.weight"]
	if !ok {
		t.Fatal("missing chronos.token_embd.weight in loaded tensors")
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
