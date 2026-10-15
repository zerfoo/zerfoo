package inference

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	tokenizer "github.com/zerfoo/ztoken"
)

// EmbeddingModel runs a full contextual decoder pass and pools its final
// normalized hidden state. Query and document instructions remain the caller's
// responsibility; this type has no skill-specific behavior.
type EmbeddingModel struct {
	graph     *graph.Graph[float32]
	tokenizer tokenizer.Tokenizer
	engine    compute.Engine[float32]
	closer    io.Closer
	hidden    int
	vocabSize int
	maxTokens int
	endToken  int
}

// LoadEmbeddingFile loads a GGUF decoder for contextual embedding. The first
// supported architecture is Qwen3. This constructor does not use the public
// Model.Embed token-table average or build an LM head.
func LoadEmbeddingFile(path string, opts ...Option) (_ *EmbeddingModel, err error) {
	return loadEmbeddingFile(path, "", opts...)
}

// LoadEmbeddingFileWithAdapter applies a GGUF LoRA adapter before building the
// contextual graph. Adapter weights are merged in memory; the base file stays
// unchanged.
func LoadEmbeddingFileWithAdapter(path, adapterPath string, opts ...Option) (_ *EmbeddingModel, err error) {
	if adapterPath == "" {
		return nil, fmt.Errorf("embedding adapter path is empty")
	}
	return loadEmbeddingFile(path, adapterPath, opts...)
}

func loadEmbeddingFile(path, adapterPath string, opts ...Option) (_ *EmbeddingModel, err error) {
	o := &loadOptions{device: "cpu", mmap: true}
	for _, opt := range opts {
		opt(o)
	}
	if strings.HasPrefix(o.device, "cuda") {
		o.mmap = false
	}
	var gm *GGUFModel
	var closer io.Closer
	if o.mmap {
		gm, closer, err = LoadGGUFMmap(path)
	} else {
		gm, err = LoadGGUF(path)
	}
	if err != nil {
		return nil, fmt.Errorf("load embedding GGUF: %w", err)
	}
	defer func() {
		if err != nil && closer != nil {
			_ = closer.Close()
		}
	}()
	if gm.Config.Architecture != "qwen3" {
		return nil, fmt.Errorf("contextual embedding unsupported for architecture %q", gm.Config.Architecture)
	}
	if adapterPath != "" {
		if err := ApplyLoRAAdapter(gm.Tensors, adapterPath); err != nil {
			return nil, fmt.Errorf("apply embedding adapter: %w", err)
		}
	}
	tok, err := gguf.ExtractTokenizer(gm.File)
	if err != nil {
		return nil, fmt.Errorf("extract embedding tokenizer: %w", err)
	}
	endToken := tok.SpecialTokens().EOS
	if endToken <= 0 {
		return nil, fmt.Errorf("embedding tokenizer has no end token")
	}
	eng, err := createEngine(o.device)
	if err != nil {
		return nil, fmt.Errorf("create embedding engine: %w", err)
	}
	defer func() {
		if err != nil {
			if c, ok := eng.(io.Closer); ok {
				_ = c.Close()
			}
		}
	}()
	applyDType(eng, o.dtype)
	g, embedWeight, err := buildQwen3EmbeddingGraph(gm.Tensors, gm.Config, eng)
	if err != nil {
		return nil, fmt.Errorf("build embedding graph: %w", err)
	}
	if uploader, ok := eng.(compute.WeightUploader); ok {
		weights := append(g.ConstantTensors(), embedWeight)
		for _, p := range g.Parameters() {
			if p.Value != nil {
				weights = append(weights, p.Value)
			}
		}
		if err := uploader.UploadWeights(weights); err != nil {
			return nil, fmt.Errorf("upload embedding weights: %w", err)
		}
	}
	maxTokens := gm.Config.MaxSeqLen
	if o.maxSeqLen > 0 {
		if maxTokens > 0 && o.maxSeqLen > maxTokens {
			return nil, fmt.Errorf("embedding max sequence %d exceeds model limit %d", o.maxSeqLen, maxTokens)
		}
		maxTokens = o.maxSeqLen
	}
	return &EmbeddingModel{graph: g, tokenizer: tok, engine: eng, closer: closer, hidden: gm.Config.HiddenSize, vocabSize: embedWeight.Shape()[0], maxTokens: maxTokens, endToken: endToken}, nil
}

// EmbedText tokenizes text, truncates on the right when configured, and runs
// the contextual model. The caller supplies any query instruction prefix.
func (m *EmbeddingModel) EmbedText(ctx context.Context, text string) ([]float32, error) {
	ids, err := m.TokenIDs(text)
	if err != nil {
		return nil, err
	}
	return m.EmbedIDs(ctx, ids)
}

// TokenIDs returns the exact token sequence used by EmbedText, including the
// terminal token. It supports independent tokenizer parity checks.
func (m *EmbeddingModel) TokenIDs(text string) ([]int, error) {
	ids, err := m.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("tokenize embedding text: %w", err)
	}
	// Qwen3-Embedding's tokenizer appends EOS, which is also its padding ID.
	// Include it in the truncation budget because final-token pooling selects it.
	if m.maxTokens > 0 && len(ids)+1 > m.maxTokens {
		ids = ids[:m.maxTokens-1]
	}
	ids = append(ids, m.endToken)
	return ids, nil
}

// EmbedIDs runs the decoder and L2-normalizes the final token's hidden state.
// This entry point also makes tokenizer parity independently testable.
func (m *EmbeddingModel) EmbedIDs(ctx context.Context, ids []int) ([]float32, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("embedding input has no tokens")
	}
	if m.maxTokens > 0 && len(ids) > m.maxTokens {
		return nil, fmt.Errorf("embedding input has %d tokens, maximum %d", len(ids), m.maxTokens)
	}
	data := make([]float32, len(ids))
	for i, id := range ids {
		if id < 0 || id >= m.vocabSize {
			return nil, fmt.Errorf("embedding token %d has invalid ID %d", i, id)
		}
		data[i] = float32(id)
	}
	input, err := tensor.New([]int{1, len(ids)}, data)
	if err != nil {
		return nil, fmt.Errorf("create embedding input: %w", err)
	}
	states, err := m.graph.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("contextual embedding forward: %w", err)
	}
	shape := states.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != len(ids) || shape[2] != m.hidden {
		return nil, fmt.Errorf("embedding hidden shape %v, want [1 %d %d]", shape, len(ids), m.hidden)
	}
	all := states.Data()
	start := (len(ids) - 1) * m.hidden
	vec := append([]float32(nil), all[start:start+m.hidden]...)
	var norm float64
	for _, x := range vec {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, fmt.Errorf("non-finite contextual embedding")
		}
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return nil, fmt.Errorf("zero contextual embedding")
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range vec {
		vec[i] *= scale
	}
	return vec, nil
}

// Close releases the engine and any mapped GGUF file.
func (m *EmbeddingModel) Close() error {
	var first error
	if c, ok := m.engine.(io.Closer); ok {
		first = c.Close()
	}
	if m.closer != nil {
		if err := m.closer.Close(); err != nil && first == nil {
			first = err
		}
		m.closer = nil
	}
	return first
}
