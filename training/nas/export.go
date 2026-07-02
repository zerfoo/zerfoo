package nas

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/zerfoo/zerfoo/model/gguf"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// ExportConfig holds configuration for exporting a NAS-discovered architecture
// to GGUF format.
type ExportConfig struct {
	// ModelName is the human-readable model name stored in general.name.
	ModelName string
	// HiddenDim is the hidden dimension of the discovered architecture.
	HiddenDim int
	// NumLayers is the number of stacked cells in the architecture.
	NumLayers int
	// InputFeatures is the number of input features (for time-series models).
	InputFeatures int
	// PatchLen is the patch length for patch-based architectures.
	PatchLen int
	// HorizonLen is the forecast horizon length.
	HorizonLen int
}

// ExportGGUF writes a NAS-discovered architecture and its trained weights to
// a GGUF v3 file. The architecture topology is encoded in GGUF metadata under
// nas.* keys, and model hyperparameters are stored under ts.signal.* keys for
// compatibility with the standard time-series inference path.
//
// The weights map keys are tensor names (e.g., "blk.0.attn_q.weight") and
// values are flat float32 slices. Each tensor's shape is provided in the
// shapes map with the same keys. Shapes use row-major (PyTorch) convention;
// they are reversed to GGML order on write.
func ExportGGUF(w io.Writer, arch *DiscretizedArch, cfg ExportConfig, weights map[string][]float32, shapes map[string][]int) error {
	if arch == nil {
		return fmt.Errorf("nas: ExportGGUF requires a non-nil DiscretizedArch")
	}
	if !arch.Cell.Valid() {
		return fmt.Errorf("nas: ExportGGUF: discretized cell is not valid")
	}

	// Sort tensor names for deterministic output.
	names := make([]string, 0, len(weights))
	for name := range weights {
		names = append(names, name)
	}
	sort.Strings(names)

	// Validate that every weight tensor has a corresponding shape entry.
	for _, name := range names {
		if _, ok := shapes[name]; !ok {
			return fmt.Errorf("nas: ExportGGUF: tensor %q has weights but no shape", name)
		}
	}

	gw := ztensorgguf.NewWriter()

	// Add metadata.
	addMetadata(gw, arch, cfg)

	// Add tensors in sorted order.
	for _, name := range names {
		gw.AddTensorF32(name, shapes[name], weights[name])
	}

	if err := gw.Write(w); err != nil {
		return fmt.Errorf("nas: ExportGGUF: %w", err)
	}

	return nil
}

// addMetadata populates the GGUF writer with NAS metadata entries encoding
// the architecture topology and model config.
func addMetadata(w *ztensorgguf.Writer, arch *DiscretizedArch, cfg ExportConfig) {
	// General metadata.
	w.AddMetadataString("general.architecture", "nas")
	if cfg.ModelName != "" {
		w.AddMetadataString("general.name", cfg.ModelName)
	}

	// NAS cell topology.
	w.AddMetadataUint32("nas.cell.num_nodes", uint32(arch.Cell.NumNodes))
	w.AddMetadataUint32("nas.cell.num_edges", uint32(len(arch.Cell.Edges)))
	w.AddMetadataInt64("nas.total_params", arch.TotalParams)

	// Encode edges as indexed metadata keys. Each edge stores from, to, and op.
	for i, e := range arch.Cell.Edges {
		prefix := fmt.Sprintf("nas.cell.edge.%d.", i)
		w.AddMetadataUint32(prefix+"from", uint32(e.From))
		w.AddMetadataUint32(prefix+"to", uint32(e.To))
		w.AddMetadataString(prefix+"op", string(e.Op))
	}

	// Time-series signal config for inference path compatibility.
	if cfg.HiddenDim > 0 {
		w.AddMetadataUint32("ts.signal.hidden_dim", uint32(cfg.HiddenDim))
	}
	if cfg.NumLayers > 0 {
		w.AddMetadataUint32("ts.signal.num_layers", uint32(cfg.NumLayers))
	}
	if cfg.InputFeatures > 0 {
		w.AddMetadataUint32("ts.signal.input_features", uint32(cfg.InputFeatures))
	}
	if cfg.PatchLen > 0 {
		w.AddMetadataUint32("ts.signal.patch_len", uint32(cfg.PatchLen))
	}
	if cfg.HorizonLen > 0 {
		w.AddMetadataUint32("ts.signal.horizon_len", uint32(cfg.HorizonLen))
	}
}

// LoadNASArchFromGGUF reads a NAS-exported GGUF file and reconstructs the
// DiscretizedArch and ExportConfig from its metadata. This enables round-trip
// verification: export then load back and confirm the architecture matches.
func LoadNASArchFromGGUF(f *gguf.File) (*DiscretizedArch, ExportConfig, error) {
	var cfg ExportConfig

	if name, ok := f.GetString("general.name"); ok {
		cfg.ModelName = name
	}

	numNodes, ok := f.GetUint32("nas.cell.num_nodes")
	if !ok {
		return nil, cfg, fmt.Errorf("nas: missing nas.cell.num_nodes metadata")
	}
	numEdges, ok := f.GetUint32("nas.cell.num_edges")
	if !ok {
		return nil, cfg, fmt.Errorf("nas: missing nas.cell.num_edges metadata")
	}

	// Read total params.
	var totalParams int64
	if v, ok := f.Metadata["nas.total_params"]; ok {
		switch n := v.(type) {
		case int64:
			totalParams = n
		case uint64:
			totalParams = int64(n)
		case uint32:
			totalParams = int64(n)
		}
	}

	// Reconstruct edges.
	edges := make([]Edge, numEdges)
	for i := range numEdges {
		prefix := fmt.Sprintf("nas.cell.edge.%d.", i)

		from, ok := f.GetUint32(prefix + "from")
		if !ok {
			return nil, cfg, fmt.Errorf("nas: missing %sfrom metadata", prefix)
		}
		to, ok := f.GetUint32(prefix + "to")
		if !ok {
			return nil, cfg, fmt.Errorf("nas: missing %sto metadata", prefix)
		}
		op, ok := f.GetString(prefix + "op")
		if !ok {
			return nil, cfg, fmt.Errorf("nas: missing %sop metadata", prefix)
		}

		edges[i] = Edge{
			From: int(from),
			To:   int(to),
			Op:   OpType(op),
		}
	}

	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: int(numNodes),
			Edges:    edges,
		},
		TotalParams: totalParams,
	}

	if !arch.Cell.Valid() {
		return nil, cfg, fmt.Errorf("nas: loaded cell is not valid")
	}

	// Read time-series signal config.
	if v, ok := f.GetUint32("ts.signal.hidden_dim"); ok {
		cfg.HiddenDim = int(v)
	}
	if v, ok := f.GetUint32("ts.signal.num_layers"); ok {
		cfg.NumLayers = int(v)
	}
	if v, ok := f.GetUint32("ts.signal.input_features"); ok {
		cfg.InputFeatures = int(v)
	}
	if v, ok := f.GetUint32("ts.signal.patch_len"); ok {
		cfg.PatchLen = int(v)
	}
	if v, ok := f.GetUint32("ts.signal.horizon_len"); ok {
		cfg.HorizonLen = int(v)
	}

	return arch, cfg, nil
}

// ValidateExportRoundTrip exports a NAS architecture to GGUF, parses it back,
// and verifies the architecture matches. Returns an error if any field differs.
func ValidateExportRoundTrip(arch *DiscretizedArch, cfg ExportConfig, weights map[string][]float32, shapes map[string][]int) error {
	var buf bytes.Buffer
	if err := ExportGGUF(&buf, arch, cfg, weights, shapes); err != nil {
		return fmt.Errorf("export: %w", err)
	}

	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		return fmt.Errorf("parse exported GGUF: %w", err)
	}

	loaded, loadedCfg, err := LoadNASArchFromGGUF(gf)
	if err != nil {
		return fmt.Errorf("load NAS arch: %w", err)
	}

	// Verify architecture.
	if loaded.Cell.NumNodes != arch.Cell.NumNodes {
		return fmt.Errorf("NumNodes mismatch: got %d, want %d", loaded.Cell.NumNodes, arch.Cell.NumNodes)
	}
	if len(loaded.Cell.Edges) != len(arch.Cell.Edges) {
		return fmt.Errorf("edge count mismatch: got %d, want %d", len(loaded.Cell.Edges), len(arch.Cell.Edges))
	}
	for i, e := range loaded.Cell.Edges {
		orig := arch.Cell.Edges[i]
		if e.From != orig.From || e.To != orig.To || e.Op != orig.Op {
			return fmt.Errorf("edge %d mismatch: got {%d,%d,%s}, want {%d,%d,%s}",
				i, e.From, e.To, e.Op, orig.From, orig.To, orig.Op)
		}
	}
	if loaded.TotalParams != arch.TotalParams {
		return fmt.Errorf("TotalParams mismatch: got %d, want %d", loaded.TotalParams, arch.TotalParams)
	}

	// Verify config.
	if loadedCfg.ModelName != cfg.ModelName {
		return fmt.Errorf("ModelName mismatch: got %q, want %q", loadedCfg.ModelName, cfg.ModelName)
	}
	if loadedCfg.HiddenDim != cfg.HiddenDim {
		return fmt.Errorf("HiddenDim mismatch: got %d, want %d", loadedCfg.HiddenDim, cfg.HiddenDim)
	}
	if loadedCfg.NumLayers != cfg.NumLayers {
		return fmt.Errorf("NumLayers mismatch: got %d, want %d", loadedCfg.NumLayers, cfg.NumLayers)
	}

	// Verify tensors are present and data matches.
	if len(gf.Tensors) != len(weights) {
		return fmt.Errorf("tensor count mismatch: got %d, want %d", len(gf.Tensors), len(weights))
	}

	tensors, err := gguf.LoadTensors(gf, r)
	if err != nil {
		return fmt.Errorf("load tensors: %w", err)
	}

	for name, origData := range weights {
		loaded, ok := tensors[name]
		if !ok {
			return fmt.Errorf("tensor %q missing from loaded GGUF", name)
		}
		loadedData := loaded.Data()
		if len(loadedData) != len(origData) {
			return fmt.Errorf("tensor %q length mismatch: got %d, want %d", name, len(loadedData), len(origData))
		}
		for j, v := range origData {
			if math.Float32bits(loadedData[j]) != math.Float32bits(v) {
				return fmt.Errorf("tensor %q element %d mismatch: got %v, want %v", name, j, loadedData[j], v)
			}
		}
	}

	return nil
}
