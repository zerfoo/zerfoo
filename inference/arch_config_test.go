package inference

import (
	"encoding/json"
	"os"
	"testing"
)

func TestArchConfigRegistry_RegisterAndParse(t *testing.T) {
	reg := newArchConfigRegistry()

	called := false
	reg.Register("test_arch", func(raw map[string]interface{}) (*ModelMetadata, error) {
		called = true
		return &ModelMetadata{Architecture: "test_arch", VocabSize: 100}, nil
	})

	raw := map[string]interface{}{
		"model_type": "test_arch",
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if !called {
		t.Error("expected parser to be called")
	}
	if meta.Architecture != "test_arch" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "test_arch")
	}
	if meta.VocabSize != 100 {
		t.Errorf("VocabSize = %d, want 100", meta.VocabSize)
	}
}

func TestArchConfigRegistry_FallbackForUnknown(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type":               "unknown_arch",
		"vocab_size":               float64(32000),
		"hidden_size":              float64(4096),
		"num_hidden_layers":        float64(32),
		"max_position_embeddings":  float64(8192),
		"eos_token_id":             float64(2),
		"bos_token_id":             float64(1),
		"intermediate_size":        float64(11008),
		"num_key_value_heads":      float64(8),
		"num_attention_heads":      float64(32),
		"rope_theta":               float64(500000),
		"tie_word_embeddings":      true,
		"sliding_window":           float64(4096),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "unknown_arch" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "unknown_arch")
	}
	if meta.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", meta.VocabSize)
	}
	if meta.HiddenSize != 4096 {
		t.Errorf("HiddenSize = %d, want 4096", meta.HiddenSize)
	}
	if meta.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", meta.NumLayers)
	}
	if meta.IntermediateSize != 11008 {
		t.Errorf("IntermediateSize = %d, want 11008", meta.IntermediateSize)
	}
	if meta.NumKeyValueHeads != 8 {
		t.Errorf("NumKeyValueHeads = %d, want 8", meta.NumKeyValueHeads)
	}
	if meta.RopeTheta != 500000 {
		t.Errorf("RopeTheta = %f, want 500000", meta.RopeTheta)
	}
	if !meta.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true")
	}
	if meta.SlidingWindow != 4096 {
		t.Errorf("SlidingWindow = %d, want 4096", meta.SlidingWindow)
	}
}

func TestArchConfigRegistry_FallbackMissingFields(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type": "minimal",
		"vocab_size": float64(1000),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.VocabSize != 1000 {
		t.Errorf("VocabSize = %d, want 1000", meta.VocabSize)
	}
	// Missing fields should be zero-valued.
	if meta.HiddenSize != 0 {
		t.Errorf("HiddenSize = %d, want 0", meta.HiddenSize)
	}
	if meta.NumLayers != 0 {
		t.Errorf("NumLayers = %d, want 0", meta.NumLayers)
	}
}

func TestArchConfigRegistry_NoModelType(t *testing.T) {
	reg := newArchConfigRegistry()

	raw := map[string]interface{}{
		"vocab_size": float64(1000),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "" {
		t.Errorf("Architecture = %q, want empty", meta.Architecture)
	}
}

func TestGemmaConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "gemma2 full config",
			raw: map[string]interface{}{
				"model_type":               "gemma2",
				"vocab_size":               float64(256000),
				"hidden_size":              float64(2304),
				"num_hidden_layers":        float64(26),
				"num_attention_heads":      float64(8),
				"num_key_value_heads":      float64(4),
				"intermediate_size":        float64(9216),
				"max_position_embeddings":  float64(8192),
				"eos_token_id":             float64(1),
				"bos_token_id":             float64(2),
			},
			want: ModelMetadata{
				Architecture:          "gemma2",
				VocabSize:             256000,
				HiddenSize:            2304,
				NumLayers:             26,
				NumQueryHeads:         8,
				NumKeyValueHeads:      4,
				IntermediateSize:      9216,
				MaxPositionEmbeddings: 8192,
				EOSTokenID:            1,
				BOSTokenID:            2,
				RopeTheta:             10000,
			},
		},
		{
			name: "gemma3 with rope_theta",
			raw: map[string]interface{}{
				"model_type":               "gemma3",
				"vocab_size":               float64(262144),
				"hidden_size":              float64(2048),
				"num_hidden_layers":        float64(26),
				"num_attention_heads":      float64(8),
				"num_key_value_heads":      float64(4),
				"intermediate_size":        float64(16384),
				"max_position_embeddings":  float64(8192),
				"eos_token_id":             float64(1),
				"bos_token_id":             float64(2),
				"rope_theta":               float64(10000),
			},
			want: ModelMetadata{
				Architecture:          "gemma3",
				VocabSize:             262144,
				HiddenSize:            2048,
				NumLayers:             26,
				NumQueryHeads:         8,
				NumKeyValueHeads:      4,
				IntermediateSize:      16384,
				MaxPositionEmbeddings: 8192,
				EOSTokenID:            1,
				BOSTokenID:            2,
				RopeTheta:             10000,
			},
		},
		{
			name: "gemma2 minimal",
			raw: map[string]interface{}{
				"model_type":          "gemma2",
				"vocab_size":          float64(256000),
				"num_hidden_layers":   float64(18),
				"num_attention_heads": float64(8),
			},
			want: ModelMetadata{
				Architecture:  "gemma2",
				VocabSize:     256000,
				NumLayers:     18,
				NumQueryHeads: 8,
				RopeTheta:     10000,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseGemmaConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseGemmaConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestDefaultArchConfigRegistry_GemmaRegistered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	for _, modelType := range []string{"gemma", "gemma2", "gemma3"} {
		t.Run(modelType, func(t *testing.T) {
			raw := map[string]interface{}{
				"model_type":          modelType,
				"vocab_size":          float64(256000),
				"num_hidden_layers":   float64(26),
				"num_attention_heads": float64(8),
			}
			meta, err := reg.Parse(raw)
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if meta.Architecture != modelType {
				t.Errorf("Architecture = %q, want %q", meta.Architecture, modelType)
			}
			if meta.VocabSize != 256000 {
				t.Errorf("VocabSize = %d, want 256000", meta.VocabSize)
			}
		})
	}
}

func TestRopeScalingConfig_FromRaw(t *testing.T) {
	raw := map[string]interface{}{
		"model_type": "unknown",
		"rope_scaling": map[string]interface{}{
			"type":                               "yarn",
			"factor":                             float64(4.0),
			"original_max_position_embeddings":   float64(32768),
		},
	}
	reg := newArchConfigRegistry()
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.RopeScaling == nil {
		t.Fatal("RopeScaling should not be nil")
	}
	if meta.RopeScaling.Type != "yarn" {
		t.Errorf("RopeScaling.Type = %q, want %q", meta.RopeScaling.Type, "yarn")
	}
	if meta.RopeScaling.Factor != 4.0 {
		t.Errorf("RopeScaling.Factor = %f, want 4.0", meta.RopeScaling.Factor)
	}
	if meta.RopeScaling.OriginalMaxPositionEmbeddings != 32768 {
		t.Errorf("RopeScaling.OriginalMaxPositionEmbeddings = %d, want 32768",
			meta.RopeScaling.OriginalMaxPositionEmbeddings)
	}
}

func TestLlamaConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "llama3.1 8B full",
			raw: map[string]interface{}{
				"model_type":              "llama",
				"vocab_size":              float64(128256),
				"hidden_size":             float64(4096),
				"num_hidden_layers":       float64(32),
				"num_attention_heads":     float64(32),
				"num_key_value_heads":     float64(8),
				"intermediate_size":       float64(14336),
				"max_position_embeddings": float64(131072),
				"rope_theta":              float64(500000),
				"eos_token_id":            float64(128001),
				"bos_token_id":            float64(128000),
				"tie_word_embeddings":     false,
				"rope_scaling": map[string]interface{}{
					"type":                             "llama3",
					"factor":                           float64(8.0),
					"original_max_position_embeddings": float64(8192),
				},
			},
			want: ModelMetadata{
				Architecture:          "llama",
				VocabSize:             128256,
				HiddenSize:            4096,
				NumLayers:             32,
				NumQueryHeads:         32,
				NumKeyValueHeads:      8,
				IntermediateSize:      14336,
				MaxPositionEmbeddings: 131072,
				RopeTheta:             500000,
				EOSTokenID:            128001,
				BOSTokenID:            128000,
			},
		},
		{
			name: "llama minimal without rope_theta defaults to 500000",
			raw: map[string]interface{}{
				"model_type":          "llama",
				"vocab_size":          float64(32000),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(32),
			},
			want: ModelMetadata{
				Architecture:  "llama",
				VocabSize:     32000,
				NumLayers:     32,
				NumQueryHeads: 32,
				RopeTheta:     500000,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseLlamaConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseLlamaConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestLlamaConfigParser_Fixture(t *testing.T) {
	data, err := os.ReadFile("testdata/llama3_config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	meta, err := parseLlamaConfig(raw)
	if err != nil {
		t.Fatalf("parseLlamaConfig error: %v", err)
	}

	assertMetadataEqual(t, ModelMetadata{
		Architecture:          "llama",
		VocabSize:             128256,
		HiddenSize:            4096,
		NumLayers:             32,
		NumQueryHeads:         32,
		NumKeyValueHeads:      8,
		IntermediateSize:      14336,
		MaxPositionEmbeddings: 131072,
		RopeTheta:             500000,
		EOSTokenID:            128001,
		BOSTokenID:            128000,
	}, *meta)

	// Verify RoPE scaling was parsed.
	if meta.RopeScaling == nil {
		t.Fatal("RopeScaling should not be nil")
	}
	if meta.RopeScaling.Type != "llama3" {
		t.Errorf("RopeScaling.Type = %q, want %q", meta.RopeScaling.Type, "llama3")
	}
	if meta.RopeScaling.Factor != 8.0 {
		t.Errorf("RopeScaling.Factor = %f, want 8.0", meta.RopeScaling.Factor)
	}
	if meta.RopeScaling.OriginalMaxPositionEmbeddings != 8192 {
		t.Errorf("RopeScaling.OriginalMaxPositionEmbeddings = %d, want 8192",
			meta.RopeScaling.OriginalMaxPositionEmbeddings)
	}
}

func TestDefaultArchConfigRegistry_LlamaRegistered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type":          "llama",
		"vocab_size":          float64(128256),
		"num_hidden_layers":   float64(32),
		"num_attention_heads": float64(32),
		"num_key_value_heads": float64(8),
		"rope_theta":          float64(500000),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "llama" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "llama")
	}
	if meta.RopeTheta != 500000 {
		t.Errorf("RopeTheta = %f, want 500000", meta.RopeTheta)
	}
}

func TestMistralConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "mistral 7B full",
			raw: map[string]interface{}{
				"model_type":              "mistral",
				"vocab_size":              float64(32000),
				"hidden_size":             float64(4096),
				"num_hidden_layers":       float64(32),
				"num_attention_heads":     float64(32),
				"num_key_value_heads":     float64(8),
				"intermediate_size":       float64(14336),
				"max_position_embeddings": float64(32768),
				"rope_theta":              float64(10000),
				"eos_token_id":            float64(2),
				"bos_token_id":            float64(1),
				"sliding_window":          float64(4096),
			},
			want: ModelMetadata{
				Architecture:          "mistral",
				VocabSize:             32000,
				HiddenSize:            4096,
				NumLayers:             32,
				NumQueryHeads:         32,
				NumKeyValueHeads:      8,
				IntermediateSize:      14336,
				MaxPositionEmbeddings: 32768,
				RopeTheta:             10000,
				EOSTokenID:            2,
				BOSTokenID:            1,
				SlidingWindow:         4096,
			},
		},
		{
			name: "mistral minimal defaults rope_theta to 10000",
			raw: map[string]interface{}{
				"model_type":          "mistral",
				"vocab_size":          float64(32000),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(32),
			},
			want: ModelMetadata{
				Architecture:  "mistral",
				VocabSize:     32000,
				NumLayers:     32,
				NumQueryHeads: 32,
				RopeTheta:     10000,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseMistralConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseMistralConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestMistralConfigParser_Fixture(t *testing.T) {
	data, err := os.ReadFile("testdata/mistral7b_config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	meta, err := parseMistralConfig(raw)
	if err != nil {
		t.Fatalf("parseMistralConfig error: %v", err)
	}

	assertMetadataEqual(t, ModelMetadata{
		Architecture:          "mistral",
		VocabSize:             32000,
		HiddenSize:            4096,
		NumLayers:             32,
		NumQueryHeads:         32,
		NumKeyValueHeads:      8,
		IntermediateSize:      14336,
		MaxPositionEmbeddings: 32768,
		RopeTheta:             10000,
		EOSTokenID:            2,
		BOSTokenID:            1,
		SlidingWindow:         4096,
	}, *meta)
}

func TestQwenConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "qwen2.5 7B full",
			raw: map[string]interface{}{
				"model_type":              "qwen2",
				"vocab_size":              float64(152064),
				"hidden_size":             float64(3584),
				"num_hidden_layers":       float64(28),
				"num_attention_heads":     float64(28),
				"num_key_value_heads":     float64(4),
				"intermediate_size":       float64(18944),
				"max_position_embeddings": float64(32768),
				"rope_theta":              float64(1000000),
				"eos_token_id":            float64(151645),
				"bos_token_id":            float64(151643),
				"sliding_window":          float64(32768),
				"use_sliding_window":      false,
				"rope_scaling": map[string]interface{}{
					"type":                             "yarn",
					"factor":                           float64(4.0),
					"original_max_position_embeddings": float64(32768),
				},
			},
			want: ModelMetadata{
				Architecture:          "qwen2",
				VocabSize:             152064,
				HiddenSize:            3584,
				NumLayers:             28,
				NumQueryHeads:         28,
				NumKeyValueHeads:      4,
				IntermediateSize:      18944,
				MaxPositionEmbeddings: 32768,
				RopeTheta:             1000000,
				EOSTokenID:            151645,
				BOSTokenID:            151643,
				SlidingWindow:         32768,
				AttentionBias:         true,
			},
		},
		{
			name: "qwen2 minimal defaults",
			raw: map[string]interface{}{
				"model_type":          "qwen2",
				"vocab_size":          float64(151936),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(40),
			},
			want: ModelMetadata{
				Architecture:  "qwen2",
				VocabSize:     151936,
				NumLayers:     32,
				NumQueryHeads: 40,
				RopeTheta:     1000000,
				AttentionBias: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseQwenConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseQwenConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestQwenConfigParser_Fixture(t *testing.T) {
	data, err := os.ReadFile("testdata/qwen25_7b_config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	meta, err := parseQwenConfig(raw)
	if err != nil {
		t.Fatalf("parseQwenConfig error: %v", err)
	}

	assertMetadataEqual(t, ModelMetadata{
		Architecture:          "qwen2",
		VocabSize:             152064,
		HiddenSize:            3584,
		NumLayers:             28,
		NumQueryHeads:         28,
		NumKeyValueHeads:      4,
		IntermediateSize:      18944,
		MaxPositionEmbeddings: 32768,
		RopeTheta:             1000000,
		EOSTokenID:            151645,
		BOSTokenID:            151643,
		SlidingWindow:         32768,
		AttentionBias:         true,
	}, *meta)

	if meta.RopeScaling == nil {
		t.Fatal("RopeScaling should not be nil")
	}
	if meta.RopeScaling.Type != "yarn" {
		t.Errorf("RopeScaling.Type = %q, want %q", meta.RopeScaling.Type, "yarn")
	}
	if meta.RopeScaling.Factor != 4.0 {
		t.Errorf("RopeScaling.Factor = %f, want 4.0", meta.RopeScaling.Factor)
	}
}

func TestPhiConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "phi4 full",
			raw: map[string]interface{}{
				"model_type":              "phi3",
				"vocab_size":              float64(100352),
				"hidden_size":             float64(5120),
				"num_hidden_layers":       float64(40),
				"num_attention_heads":     float64(40),
				"num_key_value_heads":     float64(10),
				"intermediate_size":       float64(17920),
				"max_position_embeddings": float64(16384),
				"rope_theta":              float64(10000),
				"eos_token_id":            float64(100265),
				"bos_token_id":            float64(100257),
				"partial_rotary_factor":   float64(0.5),
				"tie_word_embeddings":     false,
				"sliding_window":          float64(2048),
			},
			want: ModelMetadata{
				Architecture:          "phi3",
				VocabSize:             100352,
				HiddenSize:            5120,
				NumLayers:             40,
				NumQueryHeads:         40,
				NumKeyValueHeads:      10,
				IntermediateSize:      17920,
				MaxPositionEmbeddings: 16384,
				RopeTheta:             10000,
				EOSTokenID:            100265,
				BOSTokenID:            100257,
				PartialRotaryFactor:   0.5,
				SlidingWindow:         2048,
			},
		},
		{
			name: "phi minimal defaults partial_rotary_factor to 1.0",
			raw: map[string]interface{}{
				"model_type":          "phi3",
				"vocab_size":          float64(32064),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(32),
			},
			want: ModelMetadata{
				Architecture:        "phi3",
				VocabSize:           32064,
				NumLayers:           32,
				NumQueryHeads:       32,
				RopeTheta:           10000,
				PartialRotaryFactor: 1.0,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parsePhiConfig(tc.raw)
			if err != nil {
				t.Fatalf("parsePhiConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestPhiConfigParser_Fixture(t *testing.T) {
	data, err := os.ReadFile("testdata/phi4_config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	meta, err := parsePhiConfig(raw)
	if err != nil {
		t.Fatalf("parsePhiConfig error: %v", err)
	}

	assertMetadataEqual(t, ModelMetadata{
		Architecture:          "phi3",
		VocabSize:             100352,
		HiddenSize:            5120,
		NumLayers:             40,
		NumQueryHeads:         40,
		NumKeyValueHeads:      10,
		IntermediateSize:      17920,
		MaxPositionEmbeddings: 16384,
		RopeTheta:             10000,
		EOSTokenID:            100265,
		BOSTokenID:            100257,
		PartialRotaryFactor:   0.5,
		SlidingWindow:         2048,
	}, *meta)
}

func TestDeepSeekConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "deepseek v3 full",
			raw: map[string]interface{}{
				"model_type":              "deepseek_v3",
				"vocab_size":              float64(129280),
				"hidden_size":             float64(7168),
				"num_hidden_layers":       float64(61),
				"num_attention_heads":     float64(128),
				"num_key_value_heads":     float64(128),
				"intermediate_size":       float64(18432),
				"max_position_embeddings": float64(163840),
				"rope_theta":              float64(10000),
				"eos_token_id":            float64(1),
				"bos_token_id":            float64(0),
				"kv_lora_rank":            float64(512),
				"q_lora_rank":             float64(1536),
				"qk_rope_head_dim":        float64(64),
				"n_routed_experts":        float64(256),
				"num_experts_per_tok":     float64(8),
				"n_shared_experts":        float64(1),
			},
			want: ModelMetadata{
				Architecture:          "deepseek_v3",
				VocabSize:             129280,
				HiddenSize:            7168,
				NumLayers:             61,
				NumQueryHeads:         128,
				NumKeyValueHeads:      128,
				IntermediateSize:      18432,
				MaxPositionEmbeddings: 163840,
				RopeTheta:             10000,
				EOSTokenID:            1,
				BOSTokenID:            0,
				KVLoRADim:             512,
				QLoRADim:              1536,
				QKRopeHeadDim:         64,
				NumExperts:            256,
				NumExpertsPerToken:    8,
				NumSharedExperts:      1,
			},
		},
		{
			name: "deepseek minimal defaults rope_theta to 10000",
			raw: map[string]interface{}{
				"model_type":          "deepseek_v3",
				"vocab_size":          float64(129280),
				"num_hidden_layers":   float64(61),
				"num_attention_heads": float64(128),
			},
			want: ModelMetadata{
				Architecture:  "deepseek_v3",
				VocabSize:     129280,
				NumLayers:     61,
				NumQueryHeads: 128,
				RopeTheta:     10000,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseDeepSeekConfig(tc.raw)
			if err != nil {
				t.Fatalf("parseDeepSeekConfig error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestDeepSeekConfigParser_Fixture(t *testing.T) {
	data, err := os.ReadFile("testdata/deepseek_v3_config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal fixture: %v", err)
	}

	meta, err := parseDeepSeekConfig(raw)
	if err != nil {
		t.Fatalf("parseDeepSeekConfig error: %v", err)
	}

	assertMetadataEqual(t, ModelMetadata{
		Architecture:          "deepseek_v3",
		VocabSize:             129280,
		HiddenSize:            7168,
		NumLayers:             61,
		NumQueryHeads:         128,
		NumKeyValueHeads:      128,
		IntermediateSize:      18432,
		MaxPositionEmbeddings: 163840,
		RopeTheta:             10000,
		EOSTokenID:            1,
		BOSTokenID:            0,
		KVLoRADim:             512,
		QLoRADim:              1536,
		QKRopeHeadDim:         64,
		NumExperts:            256,
		NumExpertsPerToken:    8,
		NumSharedExperts:      1,
	}, *meta)
}

func TestDefaultArchConfigRegistry_PhiAndDeepSeekRegistered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	for _, tc := range []struct {
		modelType string
		wantArch  string
	}{
		{"phi3", "phi3"},
		{"phi", "phi"},
		{"deepseek_v3", "deepseek_v3"},
	} {
		t.Run(tc.modelType, func(t *testing.T) {
			raw := map[string]interface{}{
				"model_type":          tc.modelType,
				"vocab_size":          float64(32000),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(32),
			}
			meta, err := reg.Parse(raw)
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if meta.Architecture != tc.wantArch {
				t.Errorf("Architecture = %q, want %q", meta.Architecture, tc.wantArch)
			}
		})
	}
}

func TestDefaultArchConfigRegistry_MistralAndQwenRegistered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	for _, tc := range []struct {
		modelType string
		wantArch  string
	}{
		{"mistral", "mistral"},
		{"qwen2", "qwen2"},
	} {
		t.Run(tc.modelType, func(t *testing.T) {
			raw := map[string]interface{}{
				"model_type":          tc.modelType,
				"vocab_size":          float64(32000),
				"num_hidden_layers":   float64(32),
				"num_attention_heads": float64(32),
			}
			meta, err := reg.Parse(raw)
			if err != nil {
				t.Fatalf("Parse error: %v", err)
			}
			if meta.Architecture != tc.wantArch {
				t.Errorf("Architecture = %q, want %q", meta.Architecture, tc.wantArch)
			}
		})
	}
}

// assertMetadataEqual compares key fields of two ModelMetadata values.
func assertMetadataEqual(t *testing.T, want, got ModelMetadata) {
	t.Helper()
	if got.Architecture != want.Architecture {
		t.Errorf("Architecture = %q, want %q", got.Architecture, want.Architecture)
	}
	if got.VocabSize != want.VocabSize {
		t.Errorf("VocabSize = %d, want %d", got.VocabSize, want.VocabSize)
	}
	if got.HiddenSize != want.HiddenSize {
		t.Errorf("HiddenSize = %d, want %d", got.HiddenSize, want.HiddenSize)
	}
	if got.NumLayers != want.NumLayers {
		t.Errorf("NumLayers = %d, want %d", got.NumLayers, want.NumLayers)
	}
	if got.NumQueryHeads != want.NumQueryHeads {
		t.Errorf("NumQueryHeads = %d, want %d", got.NumQueryHeads, want.NumQueryHeads)
	}
	if got.NumKeyValueHeads != want.NumKeyValueHeads {
		t.Errorf("NumKeyValueHeads = %d, want %d", got.NumKeyValueHeads, want.NumKeyValueHeads)
	}
	if got.IntermediateSize != want.IntermediateSize {
		t.Errorf("IntermediateSize = %d, want %d", got.IntermediateSize, want.IntermediateSize)
	}
	if got.MaxPositionEmbeddings != want.MaxPositionEmbeddings {
		t.Errorf("MaxPositionEmbeddings = %d, want %d", got.MaxPositionEmbeddings, want.MaxPositionEmbeddings)
	}
	if got.EOSTokenID != want.EOSTokenID {
		t.Errorf("EOSTokenID = %d, want %d", got.EOSTokenID, want.EOSTokenID)
	}
	if got.BOSTokenID != want.BOSTokenID {
		t.Errorf("BOSTokenID = %d, want %d", got.BOSTokenID, want.BOSTokenID)
	}
	if got.RopeTheta != want.RopeTheta {
		t.Errorf("RopeTheta = %f, want %f", got.RopeTheta, want.RopeTheta)
	}
	if got.TieWordEmbeddings != want.TieWordEmbeddings {
		t.Errorf("TieWordEmbeddings = %v, want %v", got.TieWordEmbeddings, want.TieWordEmbeddings)
	}
	if got.SlidingWindow != want.SlidingWindow {
		t.Errorf("SlidingWindow = %d, want %d", got.SlidingWindow, want.SlidingWindow)
	}
	if got.AttentionBias != want.AttentionBias {
		t.Errorf("AttentionBias = %v, want %v", got.AttentionBias, want.AttentionBias)
	}
	if got.PartialRotaryFactor != want.PartialRotaryFactor {
		t.Errorf("PartialRotaryFactor = %f, want %f", got.PartialRotaryFactor, want.PartialRotaryFactor)
	}
	if got.KVLoRADim != want.KVLoRADim {
		t.Errorf("KVLoRADim = %d, want %d", got.KVLoRADim, want.KVLoRADim)
	}
	if got.QLoRADim != want.QLoRADim {
		t.Errorf("QLoRADim = %d, want %d", got.QLoRADim, want.QLoRADim)
	}
	if got.QKRopeHeadDim != want.QKRopeHeadDim {
		t.Errorf("QKRopeHeadDim = %d, want %d", got.QKRopeHeadDim, want.QKRopeHeadDim)
	}
	if got.NumExperts != want.NumExperts {
		t.Errorf("NumExperts = %d, want %d", got.NumExperts, want.NumExperts)
	}
	if got.NumExpertsPerToken != want.NumExpertsPerToken {
		t.Errorf("NumExpertsPerToken = %d, want %d", got.NumExpertsPerToken, want.NumExpertsPerToken)
	}
	if got.NumSharedExperts != want.NumSharedExperts {
		t.Errorf("NumSharedExperts = %d, want %d", got.NumSharedExperts, want.NumSharedExperts)
	}
	if got.LayerNormEps != want.LayerNormEps {
		t.Errorf("LayerNormEps = %g, want %g", got.LayerNormEps, want.LayerNormEps)
	}
}

func TestGPT2ConfigParser(t *testing.T) {
	tests := []struct {
		name string
		raw  map[string]interface{}
		want ModelMetadata
	}{
		{
			name: "TinyStories-like full config",
			raw: map[string]interface{}{
				"model_type":          "gpt2",
				"vocab_size":          float64(50257),
				"n_embd":              float64(64),
				"n_layer":             float64(8),
				"n_head":              float64(16),
				"n_positions":         float64(512),
				"n_inner":             float64(256),
				"layer_norm_epsilon":  float64(1e-5),
			},
			want: ModelMetadata{
				Architecture:          "gpt2",
				VocabSize:             50257,
				HiddenSize:            64,
				NumLayers:             8,
				NumQueryHeads:         16,
				NumKeyValueHeads:      16,
				IntermediateSize:      256,
				MaxPositionEmbeddings: 512,
				LayerNormEps:          1e-5,
				TieWordEmbeddings:     true,
			},
		},
		{
			name: "GPT-2 small (124M)",
			raw: map[string]interface{}{
				"model_type":          "gpt2",
				"vocab_size":          float64(50257),
				"n_embd":              float64(768),
				"n_layer":             float64(12),
				"n_head":              float64(12),
				"n_positions":         float64(1024),
				"layer_norm_epsilon":  float64(1e-5),
			},
			want: ModelMetadata{
				Architecture:          "gpt2",
				VocabSize:             50257,
				HiddenSize:            768,
				NumLayers:             12,
				NumQueryHeads:         12,
				NumKeyValueHeads:      12,
				IntermediateSize:      3072, // 4 * 768
				MaxPositionEmbeddings: 1024,
				LayerNormEps:          1e-5,
				TieWordEmbeddings:     true,
			},
		},
		{
			name: "minimal config defaults n_inner to 4*n_embd and layer_norm_epsilon to 1e-5",
			raw: map[string]interface{}{
				"model_type": "gpt2",
				"vocab_size": float64(50257),
				"n_embd":     float64(768),
				"n_layer":    float64(12),
				"n_head":     float64(12),
			},
			want: ModelMetadata{
				Architecture:     "gpt2",
				VocabSize:        50257,
				HiddenSize:       768,
				NumLayers:        12,
				NumQueryHeads:    12,
				NumKeyValueHeads: 12,
				IntermediateSize: 3072,
				LayerNormEps:     1e-5,
				TieWordEmbeddings: true,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseGPT2Config(tc.raw)
			if err != nil {
				t.Fatalf("parseGPT2Config error: %v", err)
			}
			assertMetadataEqual(t, tc.want, *got)
		})
	}
}

func TestDefaultArchConfigRegistry_GPT2Registered(t *testing.T) {
	reg := DefaultArchConfigRegistry()

	raw := map[string]interface{}{
		"model_type": "gpt2",
		"vocab_size": float64(50257),
		"n_embd":     float64(768),
		"n_layer":    float64(12),
		"n_head":     float64(12),
		"n_positions": float64(1024),
	}
	meta, err := reg.Parse(raw)
	if err != nil {
		t.Fatalf("Parse error: %v", err)
	}
	if meta.Architecture != "gpt2" {
		t.Errorf("Architecture = %q, want %q", meta.Architecture, "gpt2")
	}
	if meta.HiddenSize != 768 {
		t.Errorf("HiddenSize = %d, want 768", meta.HiddenSize)
	}
	if !meta.TieWordEmbeddings {
		t.Error("TieWordEmbeddings = false, want true")
	}
	if meta.NumQueryHeads != meta.NumKeyValueHeads {
		t.Errorf("GPT-2 is MHA: NumQueryHeads (%d) must equal NumKeyValueHeads (%d)",
			meta.NumQueryHeads, meta.NumKeyValueHeads)
	}
}
