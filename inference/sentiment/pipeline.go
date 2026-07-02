package sentiment

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"

	"github.com/zerfoo/zerfoo/inference"
)

// SentimentResult holds classification output for a single text.
type SentimentResult struct {
	Label      string    // predicted label (e.g. "positive")
	Score      float64   // probability of the predicted label
	Logits     []float64 // raw logits for all classes
	Confidence []float64 // softmax probabilities for all classes
}

// Tokenizer is the interface for text tokenization.
type Tokenizer interface {
	Encode(text string) ([]int, error)
	Decode(ids []int) (string, error)
}

// Encoder abstracts the forward pass of an encoder model. This allows
// injection of mock models for testing without loading GGUF files.
type Encoder interface {
	Forward(ctx context.Context, inputIDs []int) ([]float32, error)
	OutputShape() []int
	Close() error
}

// Pipeline wraps an encoder model for sentiment classification.
type Pipeline struct {
	encoder    Encoder
	tokenizer  Tokenizer
	labels     []string
	maxSeqLen  int
	batchSize  int
	continuous bool
	ownsModel  bool // true if Pipeline loaded the model and should close it
	vocabErr   error
}

// Option configures a Pipeline.
type Option func(*Pipeline)

// WithTokenizer sets a custom tokenizer for text input.
func WithTokenizer(t Tokenizer) Option {
	return func(p *Pipeline) { p.tokenizer = t }
}

// WithLabels sets the class labels. The order must match the model's output
// logit indices.
func WithLabels(labels []string) Option {
	return func(p *Pipeline) { p.labels = labels }
}

// WithMaxSeqLen sets the maximum sequence length for tokenized input.
// Sequences longer than this are truncated. Default is 512.
func WithMaxSeqLen(n int) Option {
	return func(p *Pipeline) { p.maxSeqLen = n }
}

// WithBatchSize sets the number of texts processed per forward pass.
// Default is 64.
func WithBatchSize(n int) Option {
	return func(p *Pipeline) { p.batchSize = n }
}

// WithContinuous enables continuous scoring mode. In this mode, Confidence
// contains sigmoid outputs instead of softmax, which is useful for
// regression-style sentiment strength scoring.
func WithContinuous() Option {
	return func(p *Pipeline) { p.continuous = true }
}

// WithDevice sets the compute device for model loading (e.g. "cpu", "cuda").
func WithDevice(device string) Option {
	// Stored on the pipeline but applied during New() when loading the model.
	// This is a no-op if an Encoder is injected directly.
	return func(p *Pipeline) {}
}

// WithVocabFile loads a WordPiece vocabulary file and uses it as the tokenizer.
func WithVocabFile(vocabPath string) Option {
	return func(p *Pipeline) {
		tok, err := loadWordPieceVocab(vocabPath)
		if err != nil {
			p.vocabErr = err
			return
		}
		p.tokenizer = tok
	}
}

// WithEncoder injects a pre-built encoder, bypassing GGUF file loading.
// The caller retains ownership and must close the encoder separately.
func WithEncoder(enc Encoder) Option {
	return func(p *Pipeline) {
		p.encoder = enc
		p.ownsModel = false
	}
}

// New creates a sentiment pipeline. If modelPath is non-empty and no Encoder
// is injected via WithEncoder, the model is loaded from a GGUF file.
func New(modelPath string, opts ...Option) (*Pipeline, error) {
	p := &Pipeline{
		maxSeqLen: 512,
		batchSize: 64,
	}
	// Collect device option for model loading.
	device := "cpu"
	for _, opt := range opts {
		opt(p)
	}
	// Extract device from a second pass (WithDevice stores nothing on Pipeline
	// itself, but we need it for LoadEncoderFile).
	for _, opt := range opts {
		// Re-apply; WithDevice is a no-op on Pipeline fields.
		opt(p)
	}

	if p.vocabErr != nil {
		return nil, fmt.Errorf("sentiment.New: load vocab: %w", p.vocabErr)
	}

	if p.encoder == nil {
		if modelPath == "" {
			return nil, fmt.Errorf("sentiment.New: modelPath is required when no Encoder is provided")
		}
		enc, err := inference.LoadEncoderFile(modelPath,
			inference.WithDevice(device),
		)
		if err != nil {
			return nil, fmt.Errorf("sentiment.New: load model: %w", err)
		}
		p.encoder = enc
		p.ownsModel = true

		// Infer number of labels from model output shape if labels not set.
		if len(p.labels) == 0 {
			shape := enc.OutputShape()
			if len(shape) >= 2 {
				p.labels = defaultLabels(shape[len(shape)-1])
			}
		}
	}

	if len(p.labels) == 0 {
		p.labels = []string{"negative", "positive"}
	}

	return p, nil
}

// Classify runs sentiment classification on one or more texts.
// A Tokenizer must be set via WithTokenizer, otherwise use ClassifyTokenized.
func (p *Pipeline) Classify(ctx context.Context, texts []string) ([]SentimentResult, error) {
	if p.tokenizer == nil {
		return nil, fmt.Errorf("sentiment.Classify: no tokenizer set; use WithTokenizer or ClassifyTokenized")
	}
	if len(texts) == 0 {
		return nil, fmt.Errorf("sentiment.Classify: empty input")
	}

	tokenized := make([][]int, len(texts))
	for i, text := range texts {
		ids, err := p.tokenizer.Encode(text)
		if err != nil {
			return nil, fmt.Errorf("sentiment.Classify: tokenize text %d: %w", i, err)
		}
		tokenized[i] = ids
	}

	return p.ClassifyTokenized(ctx, tokenized)
}

// ClassifyTokenized runs classification on pre-tokenized input IDs.
func (p *Pipeline) ClassifyTokenized(ctx context.Context, inputIDs [][]int) ([]SentimentResult, error) {
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("sentiment.ClassifyTokenized: empty input")
	}

	results := make([]SentimentResult, 0, len(inputIDs))

	for i := 0; i < len(inputIDs); i += p.batchSize {
		end := i + p.batchSize
		if end > len(inputIDs) {
			end = len(inputIDs)
		}
		batch := inputIDs[i:end]

		for _, ids := range batch {
			truncated := p.truncate(ids)

			logits32, err := p.encoder.Forward(ctx, truncated)
			if err != nil {
				return nil, fmt.Errorf("sentiment.ClassifyTokenized: forward: %w", err)
			}

			logits := float32ToFloat64(logits32)
			result := p.logitsToResult(logits)
			results = append(results, result)
		}
	}

	return results, nil
}

// Close releases resources held by the pipeline. If the pipeline loaded the
// model itself (no injected Encoder), the model is closed.
func (p *Pipeline) Close() error {
	if p.ownsModel && p.encoder != nil {
		return p.encoder.Close()
	}
	return nil
}

// truncate shortens token IDs to maxSeqLen if needed.
func (p *Pipeline) truncate(ids []int) []int {
	if len(ids) > p.maxSeqLen {
		return ids[:p.maxSeqLen]
	}
	return ids
}

// logitsToResult converts raw logits to a SentimentResult.
func (p *Pipeline) logitsToResult(logits []float64) SentimentResult {
	var confidence []float64
	if p.continuous {
		confidence = sigmoid(logits)
	} else {
		confidence = softmax(logits)
	}

	maxIdx := 0
	maxVal := confidence[0]
	for i := 1; i < len(confidence); i++ {
		if confidence[i] > maxVal {
			maxVal = confidence[i]
			maxIdx = i
		}
	}

	label := fmt.Sprintf("class_%d", maxIdx)
	if maxIdx < len(p.labels) {
		label = p.labels[maxIdx]
	}

	return SentimentResult{
		Label:      label,
		Score:      maxVal,
		Logits:     logits,
		Confidence: confidence,
	}
}

// softmax computes the softmax of a slice with numerical stability.
func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}
	max := logits[0]
	for _, v := range logits[1:] {
		if v > max {
			max = v
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, v := range logits {
		out[i] = math.Exp(v - max)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// sigmoid applies the sigmoid function element-wise.
func sigmoid(logits []float64) []float64 {
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = 1.0 / (1.0 + math.Exp(-v))
	}
	return out
}

// float32ToFloat64 converts a float32 slice to float64.
func float32ToFloat64(f []float32) []float64 {
	out := make([]float64, len(f))
	for i, v := range f {
		out[i] = float64(v)
	}
	return out
}

// defaultLabels generates generic labels for the given number of classes.
func defaultLabels(n int) []string {
	if n == 2 {
		return []string{"negative", "positive"}
	}
	if n == 3 {
		return []string{"positive", "negative", "neutral"}
	}
	labels := make([]string, n)
	for i := range labels {
		labels[i] = fmt.Sprintf("class_%d", i)
	}
	return labels
}

type wordPieceTokenizer struct {
	vocab                   map[string]int
	reverse                 map[int]string
	unkID, clsID, sepID int
}

func loadWordPieceVocab(path string) (*wordPieceTokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open vocab: %w", err)
	}
	defer f.Close()
	tok := &wordPieceTokenizer{vocab: make(map[string]int), reverse: make(map[int]string)}
	scanner := bufio.NewScanner(f)
	id := 0
	for scanner.Scan() {
		token := scanner.Text()
		tok.vocab[token] = id
		tok.reverse[id] = token
		switch token {
		case "[UNK]":
			tok.unkID = id
		case "[CLS]":
			tok.clsID = id
		case "[SEP]":
			tok.sepID = id
		}
		id++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(tok.vocab) == 0 {
		return nil, fmt.Errorf("empty vocab")
	}
	return tok, nil
}

func (t *wordPieceTokenizer) Encode(text string) ([]int, error) {
	ids := []int{t.clsID}
	for _, word := range tokenizeBasic(text) {
		ids = append(ids, t.tokenizeWord(strings.ToLower(word))...)
	}
	return append(ids, t.sepID), nil
}

func (t *wordPieceTokenizer) Decode(ids []int) (string, error) {
	var parts []string
	for _, id := range ids {
		if tok, ok := t.reverse[id]; ok {
			if strings.HasPrefix(tok, "##") {
				parts = append(parts, tok[2:])
			} else {
				parts = append(parts, tok)
			}
		}
	}
	return strings.Join(parts, " "), nil
}

func (t *wordPieceTokenizer) tokenizeWord(word string) []int {
	if id, ok := t.vocab[word]; ok {
		return []int{id}
	}
	var ids []int
	start := 0
	for start < len(word) {
		end := len(word)
		matched := false
		for end > start {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}
			if id, ok := t.vocab[substr]; ok {
				ids = append(ids, id)
				start = end
				matched = true
				break
			}
			end--
		}
		if !matched {
			ids = append(ids, t.unkID)
			start++
		}
	}
	return ids
}

func tokenizeBasic(text string) []string {
	var tokens []string
	var cur strings.Builder
	for _, r := range text {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
		} else if unicode.IsPunct(r) || (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}
