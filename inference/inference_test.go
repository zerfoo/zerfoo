package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	tokenizer "github.com/zerfoo/ztoken"
)

// mockDTypeSetter records the DType set on it.
type mockDTypeSetter struct {
	compute.Engine[float32]
	dtype compute.DType
}

func (m *mockDTypeSetter) SetDType(d compute.DType) { m.dtype = d }

func TestApplyDType(t *testing.T) {
	tests := []struct {
		dtype string
		want  compute.DType
	}{
		{"fp16", compute.DTypeFP16},
		{"fp8", compute.DTypeFP8},
	}
	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			mock := &mockDTypeSetter{}
			applyDType(mock, tt.dtype)
			if mock.dtype != tt.want {
				t.Errorf("applyDType(%q) set dtype=%d, want %d", tt.dtype, mock.dtype, tt.want)
			}
		})
	}
}

func TestApplyDType_NoOp(t *testing.T) {
	tests := []string{"", "fp32"}
	for _, dtype := range tests {
		t.Run(dtype, func(t *testing.T) {
			mock := &mockDTypeSetter{}
			applyDType(mock, dtype)
			if mock.dtype != compute.DTypeF32 {
				t.Errorf("applyDType(%q) should be no-op, got dtype=%d", dtype, mock.dtype)
			}
		})
	}
}

// fixedLogitsNode always returns logits where a specific token has the highest value.
type fixedLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize     int
	tokenSequence []int
	callCount     int
}

func (n *fixedLogitsNode) OpType() string                     { return "FixedLogits" }
func (n *fixedLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *fixedLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *fixedLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

func (n *fixedLogitsNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	seqLen := 1
	if len(inputs) > 0 {
		shape := inputs[0].Shape()
		if len(shape) >= 2 {
			seqLen = shape[1]
		}
	}

	data := make([]float32, seqLen*n.vocabSize)
	for pos := range seqLen {
		targetToken := n.tokenSequence[n.callCount%len(n.tokenSequence)]
		offset := pos * n.vocabSize
		for j := range n.vocabSize {
			data[offset+j] = -10.0
		}
		if targetToken >= 0 && targetToken < n.vocabSize {
			data[offset+targetToken] = 10.0
		}
		if pos == seqLen-1 {
			n.callCount++
		}
	}

	return tensor.New([]int{1, seqLen, n.vocabSize}, data)
}

func buildTestGraph(t *testing.T, vocabSize int, tokenSequence []int) *graph.Graph[float32] {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &fixedLogitsNode{
		vocabSize:     vocabSize,
		tokenSequence: tokenSequence,
	}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func buildTestTokenizer() tokenizer.Tokenizer {
	tok := tokenizer.NewWhitespaceTokenizer()
	tok.AddToken("hello") // 4
	tok.AddToken("world") // 5
	tok.AddToken("foo")   // 6
	tok.AddToken("bar")   // 7
	return tok
}

func buildTestModel(t *testing.T, vocabSize int, tokenSequence []int) *Model {
	t.Helper()
	tok := buildTestTokenizer()
	g := buildTestGraph(t, vocabSize, tokenSequence)
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config: ModelMetadata{
			Architecture:          "test",
			VocabSize:             vocabSize,
			HiddenSize:            64,
			NumLayers:             1,
			MaxPositionEmbeddings: 32,
			EOSTokenID:            2,
			BOSTokenID:            1,
			ChatTemplate:          "gemma",
		},
		info: &registry.ModelInfo{
			ID:   "test-model",
			Path: "/tmp/test",
		},
	}
}

// --- loadMetadata tests ---

func TestLoadMetadata(t *testing.T) {
	t.Run("gemma config with model_type", func(t *testing.T) {
		dir := t.TempDir()
		raw := `{
			"model_type": "gemma3",
			"vocab_size": 262144,
			"hidden_size": 2048,
			"num_hidden_layers": 26,
			"num_attention_heads": 8,
			"num_key_value_heads": 4,
			"max_position_embeddings": 8192,
			"eos_token_id": 1,
			"bos_token_id": 2,
			"chat_template": "gemma"
		}`
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}

		got, err := loadMetadata(path)
		if err != nil {
			t.Fatalf("loadMetadata error: %v", err)
		}
		if got.VocabSize != 262144 {
			t.Errorf("VocabSize = %d, want 262144", got.VocabSize)
		}
		if got.Architecture != "gemma3" {
			t.Errorf("Architecture = %q, want %q", got.Architecture, "gemma3")
		}
		if got.NumLayers != 26 {
			t.Errorf("NumLayers = %d, want 26", got.NumLayers)
		}
		if got.ChatTemplate != "gemma" {
			t.Errorf("ChatTemplate = %q, want %q", got.ChatTemplate, "gemma")
		}
	})

	t.Run("llama config from fixture", func(t *testing.T) {
		got, err := loadMetadata("testdata/llama3_config.json")
		if err != nil {
			t.Fatalf("loadMetadata error: %v", err)
		}
		if got.Architecture != "llama" {
			t.Errorf("Architecture = %q, want %q", got.Architecture, "llama")
		}
		if got.VocabSize != 128256 {
			t.Errorf("VocabSize = %d, want 128256", got.VocabSize)
		}
		if got.HiddenSize != 4096 {
			t.Errorf("HiddenSize = %d, want 4096", got.HiddenSize)
		}
		if got.NumLayers != 32 {
			t.Errorf("NumLayers = %d, want 32", got.NumLayers)
		}
		if got.NumQueryHeads != 32 {
			t.Errorf("NumQueryHeads = %d, want 32", got.NumQueryHeads)
		}
		if got.NumKeyValueHeads != 8 {
			t.Errorf("NumKeyValueHeads = %d, want 8", got.NumKeyValueHeads)
		}
		if got.RopeTheta != 500000 {
			t.Errorf("RopeTheta = %f, want 500000", got.RopeTheta)
		}
	})

	t.Run("unknown model_type uses fallback", func(t *testing.T) {
		dir := t.TempDir()
		raw := `{
			"model_type": "future_model",
			"vocab_size": 50000,
			"num_hidden_layers": 12,
			"hidden_size": 768
		}`
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}

		got, err := loadMetadata(path)
		if err != nil {
			t.Fatalf("loadMetadata error: %v", err)
		}
		if got.Architecture != "future_model" {
			t.Errorf("Architecture = %q, want %q", got.Architecture, "future_model")
		}
		if got.VocabSize != 50000 {
			t.Errorf("VocabSize = %d, want 50000", got.VocabSize)
		}
	})

	t.Run("file not found", func(t *testing.T) {
		_, err := loadMetadata("/nonexistent/config.json")
		if err == nil {
			t.Error("expected error for nonexistent file")
		}
	})

	t.Run("invalid json", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.json")
		if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
			t.Fatal(err)
		}
		_, err := loadMetadata(path)
		if err == nil {
			t.Error("expected error for invalid JSON")
		}
	})
}

// --- Option tests ---

func TestOptions(t *testing.T) {
	t.Run("WithCacheDir", func(t *testing.T) {
		o := &loadOptions{}
		WithCacheDir("/tmp/cache")(o)
		if o.cacheDir != "/tmp/cache" {
			t.Errorf("cacheDir = %q, want %q", o.cacheDir, "/tmp/cache")
		}
	})

	t.Run("WithDevice", func(t *testing.T) {
		o := &loadOptions{}
		WithDevice("cuda")(o)
		if o.device != "cuda" {
			t.Errorf("device = %q, want %q", o.device, "cuda")
		}
	})

	t.Run("WithMaxSeqLen", func(t *testing.T) {
		o := &loadOptions{}
		WithMaxSeqLen(4096)(o)
		if o.maxSeqLen != 4096 {
			t.Errorf("maxSeqLen = %d, want 4096", o.maxSeqLen)
		}
	})

	t.Run("WithRegistry", func(t *testing.T) {
		reg := &mockRegistry{models: map[string]*registry.ModelInfo{}}
		o := &loadOptions{}
		WithRegistry(reg)(o)
		if o.registry == nil {
			t.Error("registry not set")
		}
	})

	t.Run("WithBackend", func(t *testing.T) {
		o := &loadOptions{}
		WithBackend("tensorrt")(o)
		if o.backend != "tensorrt" {
			t.Errorf("backend = %q, want %q", o.backend, "tensorrt")
		}
	})

	t.Run("WithPrecision", func(t *testing.T) {
		o := &loadOptions{}
		WithPrecision("fp16")(o)
		if o.precision != "fp16" {
			t.Errorf("precision = %q, want %q", o.precision, "fp16")
		}
	})

	t.Run("WithMmap_disable", func(t *testing.T) {
		o := &loadOptions{mmap: true}
		WithMmap(false)(o)
		if o.mmap {
			t.Error("mmap should be disabled")
		}
	})

	t.Run("WithQuaRot", func(t *testing.T) {
		o := &loadOptions{}
		WithQuaRot(true)(o)
		if !o.quarot {
			t.Error("quarot not set")
		}
	})

	t.Run("WithQuaRot_false", func(t *testing.T) {
		o := &loadOptions{quarot: true}
		WithQuaRot(false)(o)
		if o.quarot {
			t.Error("quarot should be false")
		}
	})
}

// --- GenerateOption tests ---

func TestGenerateOptions(t *testing.T) {
	t.Run("WithTemperature", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTemperature(0.7)(&sc)
		if sc.Temperature != 0.7 {
			t.Errorf("Temperature = %f, want 0.7", sc.Temperature)
		}
	})

	t.Run("WithTopK", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTopK(50)(&sc)
		if sc.TopK != 50 {
			t.Errorf("TopK = %d, want 50", sc.TopK)
		}
	})

	t.Run("WithTopP", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithTopP(0.9)(&sc)
		if sc.TopP != 0.9 {
			t.Errorf("TopP = %f, want 0.9", sc.TopP)
		}
	})

	t.Run("WithMaxTokens", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithMaxTokens(100)(&sc)
		if sc.MaxNewTokens != 100 {
			t.Errorf("MaxNewTokens = %d, want 100", sc.MaxNewTokens)
		}
	})

	t.Run("WithRepetitionPenalty", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithRepetitionPenalty(1.2)(&sc)
		if sc.RepetitionPenalty != 1.2 {
			t.Errorf("RepetitionPenalty = %f, want 1.2", sc.RepetitionPenalty)
		}
	})

	t.Run("WithStopStrings", func(t *testing.T) {
		sc := generate.DefaultSamplingConfig()
		WithStopStrings("stop1", "stop2")(&sc)
		if len(sc.StopStrings) != 2 {
			t.Fatalf("StopStrings len = %d, want 2", len(sc.StopStrings))
		}
		if sc.StopStrings[0] != "stop1" || sc.StopStrings[1] != "stop2" {
			t.Errorf("StopStrings = %v, want [stop1 stop2]", sc.StopStrings)
		}
	})
}

func TestBuildSamplingConfig(t *testing.T) {
	t.Run("defaults with no options", func(t *testing.T) {
		sc := buildSamplingConfig(nil)
		def := generate.DefaultSamplingConfig()
		if sc.Temperature != def.Temperature {
			t.Errorf("Temperature = %f, want %f", sc.Temperature, def.Temperature)
		}
		if sc.MaxNewTokens != def.MaxNewTokens {
			t.Errorf("MaxNewTokens = %d, want %d", sc.MaxNewTokens, def.MaxNewTokens)
		}
	})

	t.Run("applies options in order", func(t *testing.T) {
		sc := buildSamplingConfig([]GenerateOption{
			WithTemperature(0.5),
			WithMaxTokens(10),
		})
		if sc.Temperature != 0.5 {
			t.Errorf("Temperature = %f, want 0.5", sc.Temperature)
		}
		if sc.MaxNewTokens != 10 {
			t.Errorf("MaxNewTokens = %d, want 10", sc.MaxNewTokens)
		}
	})
}

// --- Generate tests ---

func TestModel_Generate(t *testing.T) {
	vocabSize := 8
	// Produce tokens 6 (foo), 7 (bar), then EOS (2).
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	result, err := m.Generate(context.Background(), "hello world",
		WithTemperature(0),
		WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if result != "foo bar" {
		t.Errorf("Generate = %q, want %q", result, "foo bar")
	}
}

func TestModel_Generate_MaxTokens(t *testing.T) {
	vocabSize := 8
	// Never produce EOS.
	m := buildTestModel(t, vocabSize, []int{6})

	result, err := m.Generate(context.Background(), "hello",
		WithTemperature(0),
		WithMaxTokens(3),
	)
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	if result != "foo foo foo" {
		t.Errorf("Generate = %q, want %q", result, "foo foo foo")
	}
}

// --- GenerateStream tests ---

func TestModel_GenerateStream(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	var tokens []string
	err := m.GenerateStream(context.Background(), "hello",
		generate.TokenStreamFunc(func(token string, done bool) error {
			if !done {
				tokens = append(tokens, token)
			}
			return nil
		}),
		WithTemperature(0),
		WithMaxTokens(10),
	)
	if err != nil {
		t.Fatalf("GenerateStream error: %v", err)
	}
	got := strings.Join(tokens, "")
	if got != "foobar" && got != "foo bar" {
		t.Errorf("streamed tokens = %q, want foo+bar", got)
	}
}

// --- Chat tests ---

func TestModel_Chat(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 7, 2})

	resp, err := m.Chat(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Content != "foo bar" {
		t.Errorf("Chat content = %q, want %q", resp.Content, "foo bar")
	}
	if resp.TokensUsed <= 0 {
		t.Errorf("TokensUsed = %d, want > 0", resp.TokensUsed)
	}
}

// --- formatMessages tests ---

func TestFormatMessages(t *testing.T) {
	singleUser := []Message{{Role: "user", Content: "Hello"}}
	multiTurn := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi!"},
		{Role: "user", Content: "How are you?"},
	}

	tests := []struct {
		name     string
		template string
		messages []Message
		want     string
	}{
		{
			name:     "gemma single",
			template: "gemma",
			messages: singleUser,
			want:     "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name:     "gemma multi",
			template: "gemma",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "model", Content: "Hi there"},
			},
			want: "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name:     "llama single",
			template: "llama",
			messages: singleUser,
			want:     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
		},
		{
			name:     "llama multi with system",
			template: "llama",
			messages: multiTurn,
			want: "<|begin_of_text|>" +
				"<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>" +
				"<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" +
				"<|start_header_id|>assistant<|end_header_id|>\n\nHi!<|eot_id|>" +
				"<|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|>" +
				"<|start_header_id|>assistant<|end_header_id|>\n\n",
		},
		{
			name:     "mistral single",
			template: "mistral",
			messages: singleUser,
			want:     "[INST] Hello [/INST]",
		},
		{
			name:     "mistral multi with system",
			template: "mistral",
			messages: multiTurn,
			want:     "[INST] You are helpful.\n\nHello [/INST]Hi![INST] How are you? [/INST]",
		},
		{
			name:     "qwen single",
			template: "qwen2",
			messages: singleUser,
			want:     "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
		},
		{
			name:     "qwen multi with system",
			template: "qwen2",
			messages: multiTurn,
			want: "<|im_start|>system\nYou are helpful.<|im_end|>\n" +
				"<|im_start|>user\nHello<|im_end|>\n" +
				"<|im_start|>assistant\nHi!<|im_end|>\n" +
				"<|im_start|>user\nHow are you?<|im_end|>\n" +
				"<|im_start|>assistant\n",
		},
		{
			name:     "deepseek single",
			template: "deepseek",
			messages: singleUser,
			want:     "<|begin_of_sentence|>User: Hello\n\nAssistant:",
		},
		{
			name:     "deepseek multi with system",
			template: "deepseek",
			messages: multiTurn,
			want:     "<|begin_of_sentence|>You are helpful.\n\nUser: Hello\n\nAssistant: Hi!\n\nUser: How are you?\n\nAssistant:",
		},
		{
			name:     "deepseek system only",
			template: "deepseek",
			messages: []Message{{Role: "system", Content: "You are a coding assistant."}},
			want:     "<|begin_of_sentence|>You are a coding assistant.\n\nAssistant:",
		},
		{
			name:     "deepseek system and user",
			template: "deepseek",
			messages: []Message{
				{Role: "system", Content: "You are helpful."},
				{Role: "user", Content: "What is Go?"},
			},
			want: "<|begin_of_sentence|>You are helpful.\n\nUser: What is Go?\n\nAssistant:",
		},
		{
			name:     "deepseek user and assistant no system",
			template: "deepseek",
			messages: []Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
				{Role: "user", Content: "Tell me a joke."},
			},
			want: "<|begin_of_sentence|>User: Hello\n\nAssistant: Hi there!\n\nUser: Tell me a joke.\n\nAssistant:",
		},
		{
			name:     "deepseek long multi-turn",
			template: "deepseek",
			messages: []Message{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "Hi"},
				{Role: "assistant", Content: "Hello!"},
				{Role: "user", Content: "What is 2+2?"},
				{Role: "assistant", Content: "4"},
				{Role: "user", Content: "Thanks"},
			},
			want: "<|begin_of_sentence|>Be concise.\n\nUser: Hi\n\nAssistant: Hello!\n\nUser: What is 2+2?\n\nAssistant: 4\n\nUser: Thanks\n\nAssistant:",
		},
		{
			name:     "phi single",
			template: "phi",
			messages: singleUser,
			want:     "<|user|>\nHello<|end|>\n<|assistant|>\n",
		},
		{
			name:     "phi multi with system",
			template: "phi",
			messages: multiTurn,
			want: "<|system|>\nYou are helpful.<|end|>\n" +
				"<|user|>\nHello<|end|>\n" +
				"<|assistant|>\nHi!<|end|>\n" +
				"<|user|>\nHow are you?<|end|>\n" +
				"<|assistant|>\n",
		},
		{
			name:     "empty template defaults to gemma",
			template: "",
			messages: singleUser,
			want:     "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
		},
		{
			name:     "unknown template uses generic fallback",
			template: "unknown_model",
			messages: singleUser,
			want:     "user: Hello\nassistant: ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{config: ModelMetadata{ChatTemplate: tt.template}}
			got := m.formatMessages(tt.messages)
			if got != tt.want {
				t.Errorf("formatMessages =\n%q\nwant\n%q", got, tt.want)
			}
		})
	}
}

// --- Embed tests ---

func TestModel_Embed_NotSupported(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6, 2})

	// Embed requires embedding weights to be set via SetEmbeddingWeights.
	// A freshly built test model has no embedding weights, so this should error.
	_, err := m.Embed("hello")
	if err == nil {
		t.Error("expected error from Embed when embedding weights not set")
	}
	if !strings.Contains(err.Error(), "embedding weights") {
		t.Errorf("error = %q, want 'embedding weights'", err.Error())
	}
}

func TestModel_Embed_EmptyText(t *testing.T) {
	vocabSize := 8
	m := buildTestModel(t, vocabSize, []int{6})

	_, err := m.Embed("")
	if err == nil {
		t.Error("expected error for empty text")
	}
}

// --- Config and Info tests ---

func TestModel_Config(t *testing.T) {
	m := &Model{
		config: ModelMetadata{
			Architecture: "gemma3",
			VocabSize:    256000,
		},
	}
	cfg := m.Config()
	if cfg.Architecture != "gemma3" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "gemma3")
	}
	if cfg.VocabSize != 256000 {
		t.Errorf("VocabSize = %d, want 256000", cfg.VocabSize)
	}
}

func TestModel_Info(t *testing.T) {
	info := &registry.ModelInfo{ID: "test-model", Path: "/tmp/test"}
	m := &Model{info: info}
	got := m.Info()
	if got.ID != "test-model" {
		t.Errorf("Info().ID = %q, want %q", got.ID, "test-model")
	}
}

// --- mockRegistry for Load tests ---

type mockRegistry struct {
	models map[string]*registry.ModelInfo
}

func (r *mockRegistry) Get(id string) (*registry.ModelInfo, bool) {
	info, ok := r.models[id]
	return info, ok
}

func (r *mockRegistry) Pull(_ context.Context, id string) (*registry.ModelInfo, error) {
	info, ok := r.models[id]
	if !ok {
		return nil, os.ErrNotExist
	}
	return info, nil
}

func (r *mockRegistry) List() []registry.ModelInfo {
	var result []registry.ModelInfo
	for _, info := range r.models {
		result = append(result, *info)
	}
	return result
}

func (r *mockRegistry) Delete(id string) error {
	delete(r.models, id)
	return nil
}

// --- Load tests ---

func TestLoad_RegistryNotFound(t *testing.T) {
	reg := &mockRegistry{models: map[string]*registry.ModelInfo{}}
	_, err := Load("nonexistent-model", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for nonexistent model")
	}
}

func TestLoad_MissingConfig(t *testing.T) {
	dir := t.TempDir()
	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_MissingTokenizer(t *testing.T) {
	dir := t.TempDir()
	// Write config.json but no GGUF file.
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1}
	data, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_MissingZMF(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_InvalidZMF(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	// Write an invalid ZMF file (not a valid protobuf).
	if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("not a protobuf"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_PullPath(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	// pullRegistry.Get always returns false to force the Pull path.
	reg := &pullRegistry{
		info: &registry.ModelInfo{ID: "test", Path: dir},
	}
	_, err := Load("test", WithRegistry(reg))
	// Will fail at tokenizer load but exercises the Pull path.
	if err == nil {
		t.Error("expected error")
	}
	if !reg.pulled {
		t.Error("expected Pull to be called")
	}
}

func TestLoad_DefaultCacheDir(t *testing.T) {
	// Load without WithCacheDir exercises the empty-cacheDir registry creation path.
	// Will fail immediately because the model doesn't exist in default registry.
	_, err := Load("nonexistent-model-12345")
	if err == nil {
		t.Error("expected error")
	}
}

func TestLoad_WithCacheDir(t *testing.T) {
	dir := t.TempDir()
	// Load with WithCacheDir exercises the cacheDir registry creation path.
	_, err := Load("nonexistent-model-12345", WithCacheDir(dir))
	if err == nil {
		t.Error("expected error")
	}
}

func TestLoad_MaxSeqLenOverride(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 8192}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}

	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("bad"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	// WithMaxSeqLen won't be reached because ZMF load fails first,
	// but it exercises the option path.
	_, err := Load("test", WithRegistry(reg), WithMaxSeqLen(2048))
	if err == nil {
		t.Error("expected error")
	}
}

// pullRegistry always forces the Pull path.
type pullRegistry struct {
	info   *registry.ModelInfo
	pulled bool
}

func (r *pullRegistry) Get(_ string) (*registry.ModelInfo, bool) {
	return nil, false
}

func (r *pullRegistry) Pull(_ context.Context, _ string) (*registry.ModelInfo, error) {
	r.pulled = true
	return r.info, nil
}

func (r *pullRegistry) List() []registry.ModelInfo { return nil }
func (r *pullRegistry) Delete(_ string) error      { return nil }

// --- Chat error path ---

// errorNode always returns an error on Forward.
type errorNode struct {
	graph.NoParameters[float32]
}

func (n *errorNode) OpType() string                     { return "Error" }
func (n *errorNode) Attributes() map[string]interface{} { return nil }
func (n *errorNode) OutputShape() []int                 { return []int{1, 1, 8} }
func (n *errorNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *errorNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return nil, os.ErrInvalid
}

func buildErrorModel(t *testing.T) *Model {
	t.Helper()
	tok := buildTestTokenizer()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](engine)
	in := b.Input([]int{1, 1, 1})
	node := &errorNode{}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	gen := generate.NewGenerator(g, tok, engine, generate.ModelConfig{
		VocabSize:  8,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		NumLayers:  0,
	})
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    engine,
		config:    ModelMetadata{ChatTemplate: "gemma"},
	}
}

func TestModel_Chat_GenerateError(t *testing.T) {
	m := buildErrorModel(t)
	_, err := m.Chat(context.Background(), []Message{
		{Role: "user", Content: "hello"},
	}, WithTemperature(0), WithMaxTokens(5))
	if err == nil {
		t.Error("expected error from Chat when Generate fails")
	}
}

// --- assembleModel tests ---

func TestAssembleModel(t *testing.T) {
	t.Run("uses default maxSeqLen from metadata", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := 8
		g := buildTestGraph(t, vocabSize, []int{6, 2})
		eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
		meta := &ModelMetadata{
			VocabSize:             vocabSize,
			MaxPositionEmbeddings: 4096,
			EOSTokenID:            2,
			BOSTokenID:            1,
			NumLayers:             1,
		}
		info := &registry.ModelInfo{ID: "test", Path: "/tmp/test"}

		m := assembleModel(g, tok, eng, meta, info, 0, "")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
		if m.config.VocabSize != vocabSize {
			t.Errorf("VocabSize = %d, want %d", m.config.VocabSize, vocabSize)
		}
		if m.config.MaxPositionEmbeddings != 4096 {
			t.Errorf("MaxPositionEmbeddings = %d, want 4096", m.config.MaxPositionEmbeddings)
		}
		if m.info.ID != "test" {
			t.Errorf("Info.ID = %q, want %q", m.info.ID, "test")
		}
	})

	t.Run("overrides maxSeqLen when positive", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := 8
		g := buildTestGraph(t, vocabSize, []int{6, 2})
		eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
		meta := &ModelMetadata{
			VocabSize:             vocabSize,
			MaxPositionEmbeddings: 4096,
			EOSTokenID:            2,
			NumLayers:             1,
		}
		info := &registry.ModelInfo{ID: "test", Path: "/tmp/test"}

		m := assembleModel(g, tok, eng, meta, info, 2048, "")
		if m == nil {
			t.Fatal("expected non-nil model")
		}
		// Verify the model can generate (proves the generator was wired correctly).
		result, err := m.Generate(context.Background(), "hello",
			WithTemperature(0), WithMaxTokens(5))
		if err != nil {
			t.Fatalf("Generate error: %v", err)
		}
		if result == "" {
			t.Error("expected non-empty result")
		}
	})

	t.Run("generates correctly with assembled model", func(t *testing.T) {
		tok := buildTestTokenizer()
		vocabSize := 8
		g := buildTestGraph(t, vocabSize, []int{6, 7, 2})
		eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
		meta := &ModelMetadata{
			VocabSize:             vocabSize,
			MaxPositionEmbeddings: 32,
			EOSTokenID:            2,
			BOSTokenID:            1,
			NumLayers:             0,
			ChatTemplate:          "gemma",
		}
		info := &registry.ModelInfo{ID: "test", Path: "/tmp/test"}

		m := assembleModel(g, tok, eng, meta, info, 0, "")
		result, err := m.Generate(context.Background(), "hello world",
			WithTemperature(0), WithMaxTokens(10))
		if err != nil {
			t.Fatalf("Generate error: %v", err)
		}
		if result != "foo bar" {
			t.Errorf("Generate = %q, want %q", result, "foo bar")
		}
	})
}

// --- Embed forward error test ---

func TestModel_Embed_ForwardError(t *testing.T) {
	m := buildErrorModel(t)
	// buildErrorModel does not set embedding weights, so Embed errors before forward.
	_, err := m.Embed("hello")
	if err == nil {
		t.Error("expected error from Embed when embedding weights not set")
	}
	if !strings.Contains(err.Error(), "embedding weights") {
		t.Errorf("error = %q, want 'embedding weights'", err.Error())
	}
}

// --- parseDevice tests ---

func TestParseDevice(t *testing.T) {
	tests := []struct {
		input    string
		wantType string
		wantID   int
		wantErr  bool
	}{
		{input: "", wantType: "cpu", wantID: 0},
		{input: "cpu", wantType: "cpu", wantID: 0},
		{input: "CPU", wantType: "cpu", wantID: 0},
		{input: " cpu ", wantType: "cpu", wantID: 0},
		{input: "cuda", wantType: "cuda", wantID: 0},
		{input: "CUDA", wantType: "cuda", wantID: 0},
		{input: "cuda:0", wantType: "cuda", wantID: 0},
		{input: "cuda:1", wantType: "cuda", wantID: 1},
		{input: "cuda:7", wantType: "cuda", wantID: 7},
		{input: "cuda:-1", wantErr: true},
		{input: "cuda:abc", wantErr: true},
		{input: "rocm", wantType: "rocm", wantID: 0},
		{input: "ROCM", wantType: "rocm", wantID: 0},
		{input: "rocm:0", wantType: "rocm", wantID: 0},
		{input: "rocm:1", wantType: "rocm", wantID: 1},
		{input: "rocm:3", wantType: "rocm", wantID: 3},
		{input: "rocm:-1", wantErr: true},
		{input: "rocm:abc", wantErr: true},
		{input: "opencl", wantType: "opencl", wantID: 0},
		{input: "OPENCL", wantType: "opencl", wantID: 0},
		{input: "opencl:0", wantType: "opencl", wantID: 0},
		{input: "opencl:1", wantType: "opencl", wantID: 1},
		{input: "opencl:3", wantType: "opencl", wantID: 3},
		{input: "opencl:-1", wantErr: true},
		{input: "opencl:abc", wantErr: true},
		{input: "tpu", wantErr: true},
		{input: "metal:0", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			gotType, gotID, err := parseDevice(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Errorf("parseDevice(%q) expected error", tc.input)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseDevice(%q) error: %v", tc.input, err)
			}
			if gotType != tc.wantType {
				t.Errorf("parseDevice(%q) type = %q, want %q", tc.input, gotType, tc.wantType)
			}
			if gotID != tc.wantID {
				t.Errorf("parseDevice(%q) id = %d, want %d", tc.input, gotID, tc.wantID)
			}
		})
	}
}

// --- createEngine tests (non-CUDA build) ---

func TestCreateEngine_CPU(t *testing.T) {
	eng, err := createEngine("cpu")
	if err != nil {
		t.Fatalf("createEngine(cpu) error: %v", err)
	}
	if eng == nil {
		t.Fatal("expected non-nil engine")
	}
}

func TestCreateEngine_InvalidDevice(t *testing.T) {
	_, err := createEngine("tpu")
	if err == nil {
		t.Error("expected error for unsupported device")
	}
}

func TestCreateEngine_GPUDevices(t *testing.T) {
	// CUDA succeeds when runtime is available, errors otherwise.
	for _, dev := range []string{"cuda", "cuda:0"} {
		t.Run(dev, func(t *testing.T) {
			eng, err := createEngine(dev)
			if cuda.Available() {
				if err != nil {
					t.Errorf("expected success with CUDA available, got: %v", err)
				}
				if closer, ok := any(eng).(interface{ Close() error }); ok {
					closer.Close()
				}
			} else {
				if err == nil {
					t.Error("expected error without CUDA")
				} else if !strings.Contains(err.Error(), "CUDA") {
					t.Errorf("error = %q, want to contain %q", err.Error(), "CUDA")
				}
			}
		})
	}
	// ROCm and OpenCL always error (no runtime detection yet).
	tests := []struct {
		device  string
		wantErr string
	}{
		{"rocm", "ROCm"},
		{"rocm:0", "ROCm"},
		{"opencl", "OpenCL"},
		{"opencl:0", "OpenCL"},
	}
	for _, tc := range tests {
		t.Run(tc.device, func(t *testing.T) {
			_, err := createEngine(tc.device)
			if err == nil {
				t.Errorf("expected error for %s device", tc.device)
				return
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want to contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestCreateEngine_Default(t *testing.T) {
	// Empty string should default to CPU.
	eng, err := createEngine("")
	if err != nil {
		t.Fatalf("createEngine('') error: %v", err)
	}
	if eng == nil {
		t.Fatal("expected non-nil engine")
	}
}

// --- Model.Close tests ---

func TestModel_Close_CPUEngine(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 2})
	// CPU engine doesn't implement io.Closer, so Close should be a no-op.
	if err := m.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
}

func TestModel_Close_NilEngine(t *testing.T) {
	m := &Model{}
	if err := m.Close(); err != nil {
		t.Errorf("Close() error with nil engine: %v", err)
	}
}

type closableEngine struct {
	compute.Engine[float32]
	closed bool
}

func (c *closableEngine) Close() error {
	c.closed = true
	return nil
}

func TestModel_Close_WithCloser(t *testing.T) {
	ce := &closableEngine{Engine: compute.NewCPUEngine[float32](numeric.Float32Ops{})}
	m := &Model{engine: ce}
	if err := m.Close(); err != nil {
		t.Errorf("Close() error: %v", err)
	}
	if !ce.closed {
		t.Error("expected engine Close() to be called")
	}
}

func TestGetInt_EdgeCases(t *testing.T) {
	raw := map[string]interface{}{
		"float_val":   float64(42),
		"int_val":     7,
		"string_val":  "hello",
		"missing_key": nil,
	}

	if v := getInt(raw, "float_val"); v != 42 {
		t.Errorf("getInt(float_val) = %d, want 42", v)
	}
	if v := getInt(raw, "int_val"); v != 7 {
		t.Errorf("getInt(int_val) = %d, want 7", v)
	}
	if v := getInt(raw, "string_val"); v != 0 {
		t.Errorf("getInt(string_val) = %d, want 0", v)
	}
	if v := getInt(raw, "nonexistent"); v != 0 {
		t.Errorf("getInt(nonexistent) = %d, want 0", v)
	}
}

func TestGetFloat_EdgeCases(t *testing.T) {
	raw := map[string]interface{}{
		"float_val":  float64(3.14),
		"int_val":    7,
		"string_val": "hello",
	}

	if v := getFloat(raw, "float_val"); v != 3.14 {
		t.Errorf("getFloat(float_val) = %f, want 3.14", v)
	}
	if v := getFloat(raw, "int_val"); v != 7.0 {
		t.Errorf("getFloat(int_val) = %f, want 7.0", v)
	}
	if v := getFloat(raw, "string_val"); v != 0 {
		t.Errorf("getFloat(string_val) = %f, want 0", v)
	}
	if v := getFloat(raw, "nonexistent"); v != 0 {
		t.Errorf("getFloat(nonexistent) = %f, want 0", v)
	}
}

// --- Load with invalid device ---

func TestNewTestModel(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	meta := ModelMetadata{VocabSize: 100, NumLayers: 1}
	info := &registry.ModelInfo{ID: "test-model"}

	m := NewTestModel(nil, nil, eng, meta, info)
	if m == nil {
		t.Fatal("expected non-nil model")
	}
	if m.Config().VocabSize != 100 {
		t.Errorf("VocabSize = %d, want 100", m.Config().VocabSize)
	}
	if m.Info().ID != "test-model" {
		t.Errorf("ID = %q, want test-model", m.Info().ID)
	}
}

func TestLoad_InvalidDevice(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}
	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg), WithDevice("tpu"))
	if err == nil {
		t.Error("expected error for unsupported device or no GGUF")
	}
	if !strings.Contains(err.Error(), "create engine") && !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'create engine' or 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_MmapMissingZMF(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}
	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg), WithMmap(true))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

func TestLoad_MmapInvalidZMF(t *testing.T) {
	dir := t.TempDir()
	cfg := ModelMetadata{VocabSize: 100, NumLayers: 1, MaxPositionEmbeddings: 128}
	cfgData, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgData, 0o600); err != nil {
		t.Fatal(err)
	}
	tokJSON := `{"model":{"type":"BPE","vocab":{"hello":0},"merges":[]},"added_tokens":[]}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tokJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("not a protobuf"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test": {ID: "test", Path: dir},
		},
	}
	_, err := Load("test", WithRegistry(reg), WithMmap(true))
	if err == nil {
		t.Error("expected error for directory without GGUF")
	}
	if !strings.Contains(err.Error(), "no GGUF file") {
		t.Errorf("error = %q, want 'no GGUF file' substring", err.Error())
	}
}

// --- Alias tests ---

func TestResolveAlias(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"gemma-3-1b-q4", "google/gemma-3-1b-it-qat-q4_0-gguf"},
		{"llama-3-1b-q4", "meta-llama/Llama-3.2-1B-Instruct-GGUF"},
		{"unknown-model", "unknown-model"},
		{"google/some-repo", "google/some-repo"},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := ResolveAlias(tt.input)
			if got != tt.want {
				t.Errorf("ResolveAlias(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestRegisterAlias(t *testing.T) {
	RegisterAlias("my-model", "my-org/my-model-gguf")
	got := ResolveAlias("my-model")
	if got != "my-org/my-model-gguf" {
		t.Errorf("ResolveAlias after RegisterAlias = %q, want %q", got, "my-org/my-model-gguf")
	}
	// Clean up.
	modelAliasesMu.Lock()
	delete(modelAliases, "my-model")
	modelAliasesMu.Unlock()
}

func TestAliasConcurrentAccess(t *testing.T) {
	const goroutines = 100
	const iterations = 200

	var wg sync.WaitGroup
	wg.Add(goroutines * 2)

	// Half the goroutines write aliases.
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				name := fmt.Sprintf("race-model-%d-%d", id, j)
				RegisterAlias(name, "org/repo-"+name)
			}
		}(i)
	}

	// Half the goroutines read aliases.
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				name := fmt.Sprintf("race-model-%d-%d", id, j)
				_ = ResolveAlias(name)
				// Also resolve a built-in alias to exercise reads on existing keys.
				_ = ResolveAlias("gemma-3-1b-q4")
			}
		}(i)
	}

	wg.Wait()

	// Verify a written alias is readable.
	got := ResolveAlias("race-model-0-0")
	if got != "org/repo-race-model-0-0" {
		t.Errorf("ResolveAlias(race-model-0-0) = %q, want %q", got, "org/repo-race-model-0-0")
	}

	// Clean up all test aliases.
	modelAliasesMu.Lock()
	for i := 0; i < goroutines; i++ {
		for j := 0; j < iterations; j++ {
			delete(modelAliases, fmt.Sprintf("race-model-%d-%d", i, j))
		}
	}
	modelAliasesMu.Unlock()
}

// --- findGGUF tests ---

func TestFindGGUF(t *testing.T) {
	t.Run("finds gguf file", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("data"), 0o600); err != nil {
			t.Fatal(err)
		}
		got := findGGUF(dir)
		if got != filepath.Join(dir, "model.gguf") {
			t.Errorf("findGGUF = %q, want %q", got, filepath.Join(dir, "model.gguf"))
		}
	})

	t.Run("returns empty for no gguf", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "model.zmf"), []byte("data"), 0o600); err != nil {
			t.Fatal(err)
		}
		got := findGGUF(dir)
		if got != "" {
			t.Errorf("findGGUF = %q, want empty", got)
		}
	})

	t.Run("returns empty for nonexistent dir", func(t *testing.T) {
		got := findGGUF("/nonexistent/path")
		if got != "" {
			t.Errorf("findGGUF = %q, want empty", got)
		}
	})
}

// --- Load with alias resolution ---

func TestLoad_ResolvesAlias(t *testing.T) {
	// Verify that Load resolves aliases before checking the registry.
	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"google/gemma-3-1b-it-qat-q4_0-gguf": {ID: "google/gemma-3-1b-it-qat-q4_0-gguf", Path: t.TempDir()},
		},
	}
	// This will fail on model loading (no files), but we verify alias resolution
	// by checking that the registry was queried with the resolved ID.
	_, err := Load("gemma-3-1b-q4", WithRegistry(reg))
	// Should fail trying to load config.json, not "model not found"
	if err == nil {
		t.Error("expected error")
	}
	if strings.Contains(err.Error(), "pull model") {
		t.Errorf("error = %q; alias was not resolved (tried to pull)", err.Error())
	}
}

// --- Load with GGUF auto-detect ---

func TestLoad_DetectsGGUFFile(t *testing.T) {
	dir := t.TempDir()
	// Create a fake .gguf file (will fail GGUF parsing, but tests the detection path).
	if err := os.WriteFile(filepath.Join(dir, "model.gguf"), []byte("not-gguf"), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := &mockRegistry{
		models: map[string]*registry.ModelInfo{
			"test-gguf": {ID: "test-gguf", Path: dir},
		},
	}
	_, err := Load("test-gguf", WithRegistry(reg))
	if err == nil {
		t.Error("expected error from GGUF parsing")
	}
	// The error should come from GGUF loading, not ZMF loading.
	if strings.Contains(err.Error(), "load model") || strings.Contains(err.Error(), "config.json") {
		t.Errorf("error = %q; should have tried GGUF path, not ZMF path", err.Error())
	}
}

func TestModel_ConcurrentGenerate_NoRace(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 7, 2})

	const numClients = 4
	var wg sync.WaitGroup
	wg.Add(numClients)
	errs := make([]error, numClients)
	results := make([]string, numClients)

	for i := range numClients {
		go func(idx int) {
			defer wg.Done()
			result, err := m.Generate(context.Background(), "hello world",
				WithTemperature(0), WithMaxTokens(10))
			errs[idx] = err
			results[idx] = result
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("client %d error: %v", i, err)
		}
	}

	// With a shared graph, concurrent sessions may produce varying output
	// (including empty on EOS). The key assertion is no errors and no races.
	nonEmpty := 0
	for _, result := range results {
		if result != "" {
			nonEmpty++
		}
	}
	t.Logf("%d/%d clients produced non-empty output", nonEmpty, numClients)
}

// panicLogitsNode panics in Forward to simulate a generation failure.
type panicLogitsNode struct {
	graph.NoParameters[float32]
	vocabSize int
}

func (n *panicLogitsNode) OpType() string                     { return "PanicLogits" }
func (n *panicLogitsNode) Attributes() map[string]interface{} { return nil }
func (n *panicLogitsNode) OutputShape() []int                 { return []int{1, 1, n.vocabSize} }
func (n *panicLogitsNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}
func (n *panicLogitsNode) Forward(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	panic("simulated generation panic")
}

func buildPanicModel(t *testing.T) *Model {
	t.Helper()
	vocabSize := 8
	tok := buildTestTokenizer()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	b := graph.NewBuilder[float32](eng)
	in := b.Input([]int{1, 1, 1})
	node := &panicLogitsNode{vocabSize: vocabSize}
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}
	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  vocabSize,
		MaxSeqLen:  32,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  0,
	})
	pool := make(chan *generate.InferenceSession[float32], 4)
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config: ModelMetadata{
			Architecture: "test",
			VocabSize:    vocabSize,
			EOSTokenID:   2,
			BOSTokenID:   1,
		},
		info: &registry.ModelInfo{
			ID:   "panic-test",
			Path: "/tmp/test",
		},
		sessionPool: pool,
	}
}

func TestModel_Generate_SessionReturnedOnPanic(t *testing.T) {
	m := buildPanicModel(t)

	// Pre-fill pool with one session so we can track it.
	sess := m.acquireSession()
	m.releaseSession(sess)
	if len(m.sessionPool) != 1 {
		t.Fatalf("pool should have 1 session before test, got %d", len(m.sessionPool))
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { recover() }()
		m.Generate(context.Background(), "hello", WithMaxTokens(1))
	}()
	<-done

	if len(m.sessionPool) != 1 {
		t.Errorf("session pool should have 1 session after panic, got %d", len(m.sessionPool))
	}
}

func TestModel_GenerateStream_SessionReturnedOnPanic(t *testing.T) {
	m := buildPanicModel(t)

	sess := m.acquireSession()
	m.releaseSession(sess)
	if len(m.sessionPool) != 1 {
		t.Fatalf("pool should have 1 session before test, got %d", len(m.sessionPool))
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		defer func() { recover() }()
		m.GenerateStream(context.Background(), "hello",
			generate.TokenStreamFunc(func(string, bool) error { return nil }),
			WithMaxTokens(1),
		)
	}()
	<-done

	if len(m.sessionPool) != 1 {
		t.Errorf("session pool should have 1 session after panic, got %d", len(m.sessionPool))
	}
}
