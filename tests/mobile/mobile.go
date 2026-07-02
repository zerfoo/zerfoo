// Package mobile provides gomobile-compatible bindings for zerfoo inference.
//
// All exported types use only gomobile-safe types (no slices, maps, or channels).
// Token IDs from Tokenize are returned as a JSON-encoded string.
package mobile

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/zerfoo/zerfoo/inference"
)

// GenerateConfig holds sampling parameters for text generation.
// All fields use gomobile-compatible types.
type GenerateConfig struct {
	Temperature float64
	TopP        float64
	TopK        int
	MaxTokens   int
}

// Engine is an opaque handle to a loaded GGUF model.
// It wraps inference.Model and exposes a gomobile-safe API.
type Engine struct {
	mu    sync.Mutex
	model *inference.Model
}

// NewEngine loads a GGUF model from the given file path and returns an Engine.
func NewEngine(modelPath string) (*Engine, error) {
	m, err := inference.LoadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}
	return &Engine{model: m}, nil
}

// Generate produces text from the given prompt using default sampling parameters.
func (e *Engine) Generate(prompt string, maxTokens int) (string, error) {
	if e.model == nil {
		return "", fmt.Errorf("engine is closed")
	}
	if maxTokens <= 0 {
		maxTokens = 256
	}
	return e.model.Generate(context.Background(), prompt, inference.WithMaxTokens(maxTokens))
}

// GenerateWithConfig produces text using the provided sampling configuration.
func (e *Engine) GenerateWithConfig(prompt string, config *GenerateConfig) (string, error) {
	if e.model == nil {
		return "", fmt.Errorf("engine is closed")
	}
	if config == nil {
		return e.Generate(prompt, 256)
	}

	var opts []inference.GenerateOption
	if config.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(config.MaxTokens))
	} else {
		opts = append(opts, inference.WithMaxTokens(256))
	}
	if config.Temperature > 0 {
		opts = append(opts, inference.WithTemperature(config.Temperature))
	}
	if config.TopP > 0 {
		opts = append(opts, inference.WithTopP(config.TopP))
	}
	if config.TopK > 0 {
		opts = append(opts, inference.WithTopK(config.TopK))
	}
	return e.model.Generate(context.Background(), prompt, opts...)
}

// Tokenize encodes the given text into token IDs and returns them as a
// JSON array string (e.g. "[1,2,3]"). This avoids returning a slice,
// which is not supported by gomobile.
func (e *Engine) Tokenize(text string) (string, error) {
	if e.model == nil {
		return "", fmt.Errorf("engine is closed")
	}
	tok := e.model.Tokenizer()
	ids, err := tok.Encode(text)
	if err != nil {
		return "", fmt.Errorf("tokenize: %w", err)
	}
	data, err := json.Marshal(ids)
	if err != nil {
		return "", fmt.Errorf("marshal token IDs: %w", err)
	}
	return string(data), nil
}

// Close releases all resources held by the engine.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.model == nil {
		return nil
	}
	err := e.model.Close()
	e.model = nil
	return err
}
