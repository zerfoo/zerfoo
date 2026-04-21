package inference

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// GGUFModel holds a loaded GGUF model's configuration and tensors.
// This is an intermediate representation; full inference requires an
// architecture-specific graph builder to convert these into a computation graph.
type GGUFModel struct {
	Config  *gguf.ModelConfig
	Tensors map[string]*tensor.TensorNumeric[float32]
	File    *gguf.File
}

// LoadGGUF loads a GGUF model file and returns its configuration and tensors.
// Tensor names are mapped from GGUF convention (blk.N.attn_q.weight) to
// Zerfoo canonical names (model.layers.N.self_attn.q_proj.weight).
//
// Supports both single-file and split (multi-shard) GGUF files. Split files
// are detected automatically from the filename pattern (e.g., Model-00001-of-00003.gguf).
func LoadGGUF(path string) (*GGUFModel, error) {
	// Try split-file loading first.
	sf, err := gguf.ParseSplit(path)
	if err != nil {
		return nil, fmt.Errorf("parse split GGUF: %w", err)
	}
	if sf != nil {
		return loadGGUFSplit(sf)
	}

	// Single file path.
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := gguf.ExtractModelConfig(gf)
	if err != nil {
		return nil, fmt.Errorf("extract model config: %w", err)
	}

	rawTensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load tensors: %w", err)
	}

	return finishGGUFLoad(cfg, rawTensors, gf, true)
}

// loadGGUFSplit loads tensors from a split (multi-shard) GGUF file using heap allocation.
func loadGGUFSplit(sf *gguf.SplitFile) (*GGUFModel, error) {
	cfg, err := gguf.ExtractModelConfig(sf.File)
	if err != nil {
		return nil, fmt.Errorf("extract model config: %w", err)
	}

	// Open all shard files for reading.
	readers := make([]*os.File, len(sf.ShardPaths))
	for i, p := range sf.ShardPaths {
		f, err := os.Open(filepath.Clean(p))
		if err != nil {
			for j := range i {
				_ = readers[j].Close()
			}
			return nil, fmt.Errorf("open shard %d: %w", i, err)
		}
		readers[i] = f
	}
	defer func() {
		for _, f := range readers {
			_ = f.Close()
		}
	}()

	rawTensors, err := gguf.LoadTensorsSplit(sf, readers)
	if err != nil {
		return nil, fmt.Errorf("load split tensors: %w", err)
	}

	slog.Info("loaded split GGUF", "shards", len(sf.Shards), "tensors", len(rawTensors))
	return finishGGUFLoad(cfg, rawTensors, sf.File, true)
}

// finishGGUFLoad applies name mapping, tensor splitting, and precision upgrades.
func finishGGUFLoad(
	cfg *gguf.ModelConfig,
	rawTensors map[string]*tensor.TensorNumeric[float32],
	gf *gguf.File,
	upgradeEmbed bool,
) (*GGUFModel, error) {
	// Map GGUF tensor names to canonical Zerfoo names.
	mapped := make(map[string]*tensor.TensorNumeric[float32], len(rawTensors))
	for name, t := range rawTensors {
		canonical := gguf.MapTensorName(cfg.Architecture, name)
		mapped[canonical] = t
	}

	// Split merged QKV tensors (e.g., Phi attn_qkv.weight) into separate Q/K/V.
	if err := gguf.SplitMergedQKV(mapped, cfg); err != nil {
		return nil, fmt.Errorf("split merged QKV: %w", err)
	}

	// Split merged gate+up MLP tensors (e.g., Phi ffn_up with gate+up concatenated).
	if err := gguf.SplitMergedGateUp(mapped, cfg); err != nil {
		return nil, fmt.Errorf("split merged gate+up: %w", err)
	}

	// Upgrade embedding and output projection tensors from Q4 to Q8 precision.
	if upgradeEmbed {
		upgradeEmbeddingPrecision(mapped)
	}

	return &GGUFModel{
		Config:  cfg,
		Tensors: mapped,
		File:    gf,
	}, nil
}

// mmapCloser wraps the mmap cleanup function as an io.Closer so that the
// Model can release the mapping when it is closed.
type mmapCloser struct {
	fn func() error
}

func (c *mmapCloser) Close() error { return c.fn() }

// LoadGGUFMmap loads a GGUF model file using memory-mapped I/O. Instead of
// reading tensor data into heap-allocated Go slices, the entire file is mmap'd
// and tensors reference slices of the mapped region via MmapStorage. This gives
// near-instant startup and keeps tensor data out of the Go heap.
//
// Supports both single-file and split (multi-shard) GGUF files. For split files,
// each shard is independently mmap'd, allowing models larger than physical RAM
// to be loaded — the OS pages data in from disk on demand.
//
// The returned io.Closer must be kept alive and closed when the model is no
// longer needed -- it releases the memory mapping(s).
func LoadGGUFMmap(path string) (*GGUFModel, io.Closer, error) {
	// Try split-file loading first.
	sf, err := gguf.ParseSplit(path)
	if err != nil {
		return nil, nil, fmt.Errorf("parse split GGUF: %w", err)
	}
	if sf != nil {
		return loadGGUFMmapSplit(sf)
	}

	// Single file path.
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := gguf.ExtractModelConfig(gf)
	if err != nil {
		return nil, nil, fmt.Errorf("extract model config: %w", err)
	}

	// Memory-map the entire file.
	mapped, cleanup, err := tensor.MmapFile(filepath.Clean(path))
	if err != nil {
		return nil, nil, fmt.Errorf("mmap GGUF file: %w", err)
	}

	slog.Info("mmap'd GGUF file", "path", path, "size_mb", len(mapped)/(1024*1024))

	// Create tensors backed by mmap'd regions (zero-copy).
	rawTensors, err := gguf.LoadTensorsMmap(gf, mapped)
	if err != nil {
		_ = cleanup()
		return nil, nil, fmt.Errorf("load tensors (mmap): %w", err)
	}

	model, err := finishGGUFLoad(cfg, rawTensors, gf, false)
	if err != nil {
		_ = cleanup()
		return nil, nil, err
	}

	// Switch madvise to random access for inference.
	_ = tensor.MadviseRandom(mapped)

	return model, &mmapCloser{fn: cleanup}, nil
}

// loadGGUFMmapSplit loads a split GGUF model using mmap for all shards.
func loadGGUFMmapSplit(sf *gguf.SplitFile) (*GGUFModel, io.Closer, error) {
	cfg, err := gguf.ExtractModelConfig(sf.File)
	if err != nil {
		return nil, nil, fmt.Errorf("extract model config: %w", err)
	}

	// Mmap all shards.
	mappedShards := make([][]byte, len(sf.ShardPaths))
	cleanupFns := make([]func() error, len(sf.ShardPaths))
	var totalSize int64

	cleanupAll := func() error {
		var firstErr error
		for _, fn := range cleanupFns {
			if fn != nil {
				if err := fn(); err != nil && firstErr == nil {
					firstErr = err
				}
			}
		}
		return firstErr
	}

	for i, p := range sf.ShardPaths {
		mapped, cleanup, err := tensor.MmapFile(filepath.Clean(p))
		if err != nil {
			_ = cleanupAll()
			return nil, nil, fmt.Errorf("mmap shard %d (%s): %w", i, p, err)
		}
		mappedShards[i] = mapped
		cleanupFns[i] = cleanup
		totalSize += int64(len(mapped))
	}

	slog.Info("mmap'd split GGUF",
		"shards", len(sf.ShardPaths),
		"total_size_gb", fmt.Sprintf("%.1f", float64(totalSize)/(1024*1024*1024)),
	)

	// Create tensors backed by mmap'd regions from the correct shards.
	rawTensors, err := gguf.LoadTensorsMmapSplit(sf, mappedShards)
	if err != nil {
		_ = cleanupAll()
		return nil, nil, fmt.Errorf("load split tensors (mmap): %w", err)
	}

	model, err := finishGGUFLoad(cfg, rawTensors, sf.File, false)
	if err != nil {
		_ = cleanupAll()
		return nil, nil, err
	}

	// Switch all shards to random access for inference.
	for _, mapped := range mappedShards {
		_ = tensor.MadviseRandom(mapped)
	}

	return model, &mmapCloser{fn: cleanupAll}, nil
}

// upgradeEmbeddingPrecision re-quantizes embedding and output projection tensors
// from Q4_0 to Q8_0. These tensors use gather operations (index lookup), not GEMV,
// so the Q4_0 GEMV speed advantage does not apply. The extra precision from Q8
// (256 quantization levels vs 16) prevents the cumulative numerical errors that
// cause garbage output in Q4_K models like Mistral 7B and Llama 3.x.
//
// T99.2.2.7 (Gemma 4 Edge PLE): when ZERFOO_GEMMA4_PLE_EMBED_Q8=1, also upgrade
// `model.ple_embed_tokens.weight` (Q4_K re-quantized to Q4_0 at decode) to Q8.
// This is the gather table for the PLE tokenSlice path; H12+H20 joint tests
// whether lower gather noise plus the tokenSlice RMSNorm (H20) restores Q4_K_M
// decode coherence. Baseline (flag unset) is bit-identical to the pre-T99.2.2.7
// load path.
func upgradeEmbeddingPrecision(tensors map[string]*tensor.TensorNumeric[float32]) {
	targets := []string{
		"model.embed_tokens.weight",
		"lm_head.weight",
	}
	if os.Getenv("ZERFOO_GEMMA4_PLE_EMBED_Q8") == "1" {
		targets = append(targets, "model.ple_embed_tokens.weight")
	}
	for _, name := range targets {
		t, ok := tensors[name]
		if !ok {
			continue
		}
		if _, isQ4 := t.GetStorage().(*tensor.Q4Storage); !isQ4 {
			continue
		}
		// Upgrade Q4 to Q8 for embedding tensors. Q8 has 256 quantization
		// levels vs Q4's 16, recovering precision lost during Q6_K→Q4_0
		// re-quantization. Q8 storage is 2x smaller than F32 and the Q8
		// GEMV kernel is used for the output projection (LM head).
		f32 := t.Data()
		q8 := tensor.QuantizeQ8(f32)
		upgraded, err := tensor.NewWithStorage[float32](t.Shape(), q8)
		if err != nil {
			slog.Warn("failed to upgrade tensor precision", "tensor", name, "error", err)
			continue
		}
		tensors[name] = upgraded
		slog.Info("upgraded tensor from Q4 to Q8", "tensor", name, "shape", t.Shape())
	}
}

// ToModelMetadata converts a GGUF model config to inference.ModelMetadata.
func (m *GGUFModel) ToModelMetadata() *ModelMetadata {
	return &ModelMetadata{
		Architecture:          m.Config.Architecture,
		VocabSize:             m.Config.VocabSize,
		HiddenSize:            m.Config.HiddenSize,
		NumLayers:             m.Config.NumLayers,
		MaxPositionEmbeddings: m.Config.MaxSeqLen,
		NumQueryHeads:         m.Config.NumHeads,
		NumKeyValueHeads:      m.Config.NumKVHeads,
		IntermediateSize:      m.Config.IntermediateSize,
		RopeTheta:             m.Config.RopeTheta,
		ChatTemplate:          chatTemplateForArch(m.Config.Architecture),
		AudioNumMels:          m.Config.AudioNumMels,
	}
}

// chatTemplateForArch returns the chat template name for a GGUF architecture.
func chatTemplateForArch(arch string) string {
	switch arch {
	case "gemma", "gemma3":
		return "gemma"
	case "llama":
		return "llama"
	case "mistral":
		return "mistral"
	case "qwen2":
		return "qwen2"
	case "deepseek":
		return "deepseek"
	case "phi3":
		return "phi"
	default:
		return ""
	}
}
