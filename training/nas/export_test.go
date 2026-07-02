package nas

import (
	"bytes"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestExportGGUFRoundTrip(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpSkipConnect},
				{From: 1, To: 2, Op: OpZero},
			},
		},
		TotalParams: 36864,
	}

	cfg := ExportConfig{
		ModelName:     "nas-test-model",
		HiddenDim:     128,
		NumLayers:     4,
		InputFeatures: 16,
		PatchLen:      32,
		HorizonLen:    8,
	}

	weights := map[string][]float32{
		"blk.0.weight": {1.0, 2.0, 3.0, 4.0},
		"blk.1.weight": {5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
	}
	shapes := map[string][]int{
		"blk.0.weight": {2, 2},
		"blk.1.weight": {2, 3},
	}

	if err := ValidateExportRoundTrip(arch, cfg, weights, shapes); err != nil {
		t.Fatalf("round-trip validation failed: %v", err)
	}
}

func TestExportGGUFNilArch(t *testing.T) {
	var buf bytes.Buffer
	err := ExportGGUF(&buf, nil, ExportConfig{}, nil, nil)
	if err == nil {
		t.Fatal("expected error for nil arch")
	}
}

func TestExportGGUFInvalidCell(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 1, // invalid: needs at least 2
			Edges:    nil,
		},
	}
	var buf bytes.Buffer
	err := ExportGGUF(&buf, arch, ExportConfig{}, nil, nil)
	if err == nil {
		t.Fatal("expected error for invalid cell")
	}
}

func TestExportGGUFMissingShape(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpZero},
				{From: 1, To: 2, Op: OpSkipConnect},
			},
		},
	}

	weights := map[string][]float32{
		"tensor.weight": {1.0, 2.0},
	}
	shapes := map[string][]int{} // missing shape for tensor.weight

	var buf bytes.Buffer
	err := ExportGGUF(&buf, arch, ExportConfig{}, weights, shapes)
	if err == nil {
		t.Fatal("expected error for missing shape")
	}
}

func TestExportGGUFEmptyWeights(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpZero},
				{From: 1, To: 2, Op: OpSkipConnect},
			},
		},
	}

	// No weights — valid GGUF with zero tensors.
	weights := map[string][]float32{}
	shapes := map[string][]int{}

	var buf bytes.Buffer
	if err := ExportGGUF(&buf, arch, ExportConfig{ModelName: "empty"}, weights, shapes); err != nil {
		t.Fatalf("ExportGGUF with empty weights: %v", err)
	}

	// Parse and verify metadata.
	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	archStr, ok := gf.GetString("general.architecture")
	if !ok || archStr != "nas" {
		t.Errorf("general.architecture: got %q, want %q", archStr, "nas")
	}

	name, ok := gf.GetString("general.name")
	if !ok || name != "empty" {
		t.Errorf("general.name: got %q, want %q", name, "empty")
	}
}

func TestExportGGUFArchMetadata(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 4,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpSepConv5x5},
				{From: 0, To: 3, Op: OpAvgPool3x3},
				{From: 1, To: 2, Op: OpMaxPool3x3},
				{From: 1, To: 3, Op: OpConv5x5},
				{From: 2, To: 3, Op: OpSkipConnect},
			},
		},
		TotalParams: 200000,
	}

	cfg := ExportConfig{
		ModelName:     "nas-4node",
		HiddenDim:     256,
		NumLayers:     6,
		InputFeatures: 8,
		PatchLen:      16,
		HorizonLen:    4,
	}

	var buf bytes.Buffer
	if err := ExportGGUF(&buf, arch, cfg, map[string][]float32{}, map[string][]int{}); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	loaded, loadedCfg, err := LoadNASArchFromGGUF(gf)
	if err != nil {
		t.Fatalf("LoadNASArchFromGGUF: %v", err)
	}

	// Verify cell topology.
	if loaded.Cell.NumNodes != 4 {
		t.Errorf("NumNodes: got %d, want 4", loaded.Cell.NumNodes)
	}
	if len(loaded.Cell.Edges) != 6 {
		t.Errorf("num edges: got %d, want 6", len(loaded.Cell.Edges))
	}

	wantOps := []OpType{OpConv3x3, OpSepConv5x5, OpAvgPool3x3, OpMaxPool3x3, OpConv5x5, OpSkipConnect}
	for i, e := range loaded.Cell.Edges {
		if e.Op != wantOps[i] {
			t.Errorf("edge %d op: got %s, want %s", i, e.Op, wantOps[i])
		}
	}

	if loaded.TotalParams != 200000 {
		t.Errorf("TotalParams: got %d, want 200000", loaded.TotalParams)
	}

	// Verify config round-trip.
	if loadedCfg.ModelName != "nas-4node" {
		t.Errorf("ModelName: got %q, want %q", loadedCfg.ModelName, "nas-4node")
	}
	if loadedCfg.HiddenDim != 256 {
		t.Errorf("HiddenDim: got %d, want 256", loadedCfg.HiddenDim)
	}
	if loadedCfg.NumLayers != 6 {
		t.Errorf("NumLayers: got %d, want 6", loadedCfg.NumLayers)
	}
	if loadedCfg.InputFeatures != 8 {
		t.Errorf("InputFeatures: got %d, want 8", loadedCfg.InputFeatures)
	}
	if loadedCfg.PatchLen != 16 {
		t.Errorf("PatchLen: got %d, want 16", loadedCfg.PatchLen)
	}
	if loadedCfg.HorizonLen != 4 {
		t.Errorf("HorizonLen: got %d, want 4", loadedCfg.HorizonLen)
	}
}

func TestLoadNASArchFromGGUFMissingFields(t *testing.T) {
	tests := []struct {
		name     string
		metadata map[string]any
	}{
		{
			name:     "missing num_nodes",
			metadata: map[string]any{"nas.cell.num_edges": uint32(1)},
		},
		{
			name:     "missing num_edges",
			metadata: map[string]any{"nas.cell.num_nodes": uint32(3)},
		},
		{
			name: "missing edge from",
			metadata: map[string]any{
				"nas.cell.num_nodes": uint32(3),
				"nas.cell.num_edges": uint32(1),
				"nas.cell.edge.0.to": uint32(1),
				"nas.cell.edge.0.op": "conv_3x3",
			},
		},
		{
			name: "missing edge op",
			metadata: map[string]any{
				"nas.cell.num_nodes":   uint32(3),
				"nas.cell.num_edges":   uint32(1),
				"nas.cell.edge.0.from": uint32(0),
				"nas.cell.edge.0.to":   uint32(1),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gf := &gguf.File{
				Version:  3,
				Metadata: tt.metadata,
			}
			_, _, err := LoadNASArchFromGGUF(gf)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestExportGGUFTensorDataIntegrity(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpZero},
				{From: 1, To: 2, Op: OpSkipConnect},
			},
		},
	}

	// Create a larger tensor to test data integrity.
	size := 256
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.01
	}

	weights := map[string][]float32{"layer.weight": data}
	shapes := map[string][]int{"layer.weight": {16, 16}}

	var buf bytes.Buffer
	if err := ExportGGUF(&buf, arch, ExportConfig{}, weights, shapes); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	tensors, err := gguf.LoadTensors(gf, r)
	if err != nil {
		t.Fatalf("load tensors: %v", err)
	}

	loaded, ok := tensors["layer.weight"]
	if !ok {
		t.Fatal("tensor layer.weight not found")
	}

	loadedData := loaded.Data()
	if len(loadedData) != size {
		t.Fatalf("data length: got %d, want %d", len(loadedData), size)
	}

	for i, v := range data {
		if loadedData[i] != v {
			t.Errorf("element %d: got %v, want %v", i, loadedData[i], v)
			break
		}
	}

	// Verify shape was stored correctly (GGML reverses dimensions).
	wantShape := []int{16, 16}
	gotShape := loaded.Shape()
	if len(gotShape) != len(wantShape) {
		t.Fatalf("shape dims: got %d, want %d", len(gotShape), len(wantShape))
	}
	for i, v := range wantShape {
		if gotShape[i] != v {
			t.Errorf("shape[%d]: got %d, want %d", i, gotShape[i], v)
		}
	}
}

func TestExportGGUFNoModelName(t *testing.T) {
	arch := &DiscretizedArch{
		Cell: Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpZero},
				{From: 1, To: 2, Op: OpSkipConnect},
			},
		},
	}

	// No model name — general.name should be absent.
	var buf bytes.Buffer
	if err := ExportGGUF(&buf, arch, ExportConfig{}, map[string][]float32{}, map[string][]int{}); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	r := bytes.NewReader(buf.Bytes())
	gf, err := gguf.Parse(r)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	if _, ok := gf.GetString("general.name"); ok {
		t.Error("general.name should not be present when ModelName is empty")
	}
}
