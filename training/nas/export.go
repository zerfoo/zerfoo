package nas

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/zerfoo/zerfoo/model/gguf"
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

	// Build metadata key-value pairs.
	metadata := buildMetadata(arch, cfg)

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

	var buf bytes.Buffer

	// Write GGUF v3 header.
	binary.Write(&buf, binary.LittleEndian, gguf.Magic)
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(len(names)))
	binary.Write(&buf, binary.LittleEndian, uint64(len(metadata)))

	// Write metadata key-value pairs.
	for _, kv := range metadata {
		writeString(&buf, kv.key)
		binary.Write(&buf, binary.LittleEndian, kv.valueType)
		switch kv.valueType {
		case gguf.TypeString:
			writeString(&buf, kv.value.(string))
		case gguf.TypeUint32:
			binary.Write(&buf, binary.LittleEndian, kv.value.(uint32))
		case gguf.TypeFloat32:
			binary.Write(&buf, binary.LittleEndian, kv.value.(float32))
		case gguf.TypeInt64:
			binary.Write(&buf, binary.LittleEndian, kv.value.(int64))
		}
	}

	// Write tensor info entries.
	var offset uint64
	for _, name := range names {
		data := weights[name]
		shape := shapes[name]

		writeString(&buf, name)

		// Number of dimensions.
		binary.Write(&buf, binary.LittleEndian, uint32(len(shape)))

		// Dimensions in GGML order (innermost-first): reverse of row-major.
		for i := len(shape) - 1; i >= 0; i-- {
			binary.Write(&buf, binary.LittleEndian, uint64(shape[i]))
		}

		// Type = F32.
		binary.Write(&buf, binary.LittleEndian, uint32(gguf.GGMLTypeF32))

		// Offset relative to data start.
		binary.Write(&buf, binary.LittleEndian, offset)
		offset += uint64(len(data)) * 4
	}

	// Align to 32-byte boundary before tensor data.
	const alignment = 32
	pos := buf.Len()
	padding := (alignment - (pos % alignment)) % alignment
	for range padding {
		buf.WriteByte(0)
	}

	// Write header + metadata + tensor info.
	if _, err := w.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("nas: ExportGGUF: write header: %w", err)
	}

	// Write tensor data.
	for _, name := range names {
		data := weights[name]
		for _, v := range data {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return fmt.Errorf("nas: ExportGGUF: write tensor %q: %w", name, err)
			}
		}
	}

	return nil
}

// metadataKV represents a single GGUF metadata key-value pair.
type metadataKV struct {
	key       string
	valueType uint32
	value     any
}

// buildMetadata constructs the ordered list of GGUF metadata entries for a NAS
// export. It encodes the architecture topology and model config.
func buildMetadata(arch *DiscretizedArch, cfg ExportConfig) []metadataKV {
	var kvs []metadataKV

	// General metadata.
	kvs = append(kvs, metadataKV{"general.architecture", gguf.TypeString, "nas"})
	if cfg.ModelName != "" {
		kvs = append(kvs, metadataKV{"general.name", gguf.TypeString, cfg.ModelName})
	}

	// NAS cell topology.
	kvs = append(kvs, metadataKV{"nas.cell.num_nodes", gguf.TypeUint32, uint32(arch.Cell.NumNodes)})
	kvs = append(kvs, metadataKV{"nas.cell.num_edges", gguf.TypeUint32, uint32(len(arch.Cell.Edges))})
	kvs = append(kvs, metadataKV{"nas.total_params", gguf.TypeInt64, arch.TotalParams})

	// Encode edges as indexed metadata keys. Each edge stores from, to, and op.
	for i, e := range arch.Cell.Edges {
		prefix := fmt.Sprintf("nas.cell.edge.%d.", i)
		kvs = append(kvs, metadataKV{prefix + "from", gguf.TypeUint32, uint32(e.From)})
		kvs = append(kvs, metadataKV{prefix + "to", gguf.TypeUint32, uint32(e.To)})
		kvs = append(kvs, metadataKV{prefix + "op", gguf.TypeString, string(e.Op)})
	}

	// Time-series signal config for inference path compatibility.
	if cfg.HiddenDim > 0 {
		kvs = append(kvs, metadataKV{"ts.signal.hidden_dim", gguf.TypeUint32, uint32(cfg.HiddenDim)})
	}
	if cfg.NumLayers > 0 {
		kvs = append(kvs, metadataKV{"ts.signal.num_layers", gguf.TypeUint32, uint32(cfg.NumLayers)})
	}
	if cfg.InputFeatures > 0 {
		kvs = append(kvs, metadataKV{"ts.signal.input_features", gguf.TypeUint32, uint32(cfg.InputFeatures)})
	}
	if cfg.PatchLen > 0 {
		kvs = append(kvs, metadataKV{"ts.signal.patch_len", gguf.TypeUint32, uint32(cfg.PatchLen)})
	}
	if cfg.HorizonLen > 0 {
		kvs = append(kvs, metadataKV{"ts.signal.horizon_len", gguf.TypeUint32, uint32(cfg.HorizonLen)})
	}

	return kvs
}

// writeString writes a GGUF-format string (uint64 length + bytes).
func writeString(buf *bytes.Buffer, s string) {
	binary.Write(buf, binary.LittleEndian, uint64(len(s)))
	buf.WriteString(s)
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
