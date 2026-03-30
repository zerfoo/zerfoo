package timeseries

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestMapTiRexTensorName(t *testing.T) {
	tests := []struct {
		hfName string
		want   string
	}{
		// Block-level sLSTM tensors.
		{"blocks.0.slstm.weight_ih", "tirex.block.0.slstm.weight_ih"},
		{"blocks.0.slstm.weight_hh", "tirex.block.0.slstm.weight_hh"},
		{"blocks.0.slstm.bias", "tirex.block.0.slstm.bias"},
		// Block-level mLSTM tensors.
		{"blocks.1.mlstm.weight_q", "tirex.block.1.mlstm.weight_q"},
		{"blocks.1.mlstm.weight_k", "tirex.block.1.mlstm.weight_k"},
		{"blocks.1.mlstm.weight_v", "tirex.block.1.mlstm.weight_v"},
		{"blocks.2.mlstm.bias", "tirex.block.2.mlstm.bias"},
		// Higher layer indices.
		{"blocks.15.slstm.weight_ih", "tirex.block.15.slstm.weight_ih"},
		// Global tensors.
		{"input_proj.weight", "tirex.input_proj.weight"},
		{"input_proj.bias", "tirex.input_proj.bias"},
		{"output_head.weight", "tirex.output_head.weight"},
		{"output_head.bias", "tirex.output_head.bias"},
		{"norm.weight", "tirex.norm.weight"},
		{"norm.bias", "tirex.norm.bias"},
		// Block-level norm tensors.
		{"blocks.0.norm.weight", "tirex.block.0.norm.weight"},
	}

	for _, tt := range tests {
		got := MapTiRexTensorName(tt.hfName)
		if got != tt.want {
			t.Errorf("MapTiRexTensorName(%q) = %q, want %q", tt.hfName, got, tt.want)
		}
	}
}

func TestTiRexConvertConfigValidate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     TiRexConvertConfig
		wantErr bool
	}{
		{
			name: "valid",
			cfg: TiRexConvertConfig{
				NumLayers:  3,
				HiddenDim:  128,
				BlockTypes: []string{"slstm", "mlstm", "slstm"},
			},
		},
		{
			name: "zero layers",
			cfg: TiRexConvertConfig{
				NumLayers:  0,
				HiddenDim:  128,
				BlockTypes: []string{},
			},
			wantErr: true,
		},
		{
			name: "zero hidden dim",
			cfg: TiRexConvertConfig{
				NumLayers:  2,
				HiddenDim:  0,
				BlockTypes: []string{"slstm", "mlstm"},
			},
			wantErr: true,
		},
		{
			name: "mismatched block types length",
			cfg: TiRexConvertConfig{
				NumLayers:  3,
				HiddenDim:  128,
				BlockTypes: []string{"slstm", "mlstm"},
			},
			wantErr: true,
		},
		{
			name: "invalid block type",
			cfg: TiRexConvertConfig{
				NumLayers:  2,
				HiddenDim:  128,
				BlockTypes: []string{"slstm", "gru"},
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

// writeSyntheticSafeTensors creates a minimal SafeTensors file with the given
// tensor names and float32 data. Each tensor is a 1D vector.
func writeSyntheticSafeTensors(t *testing.T, path string, tensors map[string][]float32) {
	t.Helper()

	// Build header and data.
	type stMeta struct {
		DType       string  `json:"dtype"`
		Shape       []int   `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	header := make(map[string]stMeta, len(tensors))
	var dataBlob []byte

	// Sort names for deterministic layout.
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sortStrings(names)

	var offset int64
	for _, name := range names {
		data := tensors[name]
		rawBytes := make([]byte, len(data)*4)
		for i, v := range data {
			binary.LittleEndian.PutUint32(rawBytes[i*4:], math.Float32bits(v))
		}
		header[name] = stMeta{
			DType:       "F32",
			Shape:       []int{len(data)},
			DataOffsets: [2]int64{offset, offset + int64(len(rawBytes))},
		}
		dataBlob = append(dataBlob, rawBytes...)
		offset += int64(len(rawBytes))
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal safetensors header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create safetensors file: %v", err)
	}
	defer func() { _ = f.Close() }()

	// Write 8-byte header length + JSON header + tensor data.
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerJSON))); err != nil {
		t.Fatalf("write header length: %v", err)
	}
	if _, err := f.Write(headerJSON); err != nil {
		t.Fatalf("write header JSON: %v", err)
	}
	if _, err := f.Write(dataBlob); err != nil {
		t.Fatalf("write tensor data: %v", err)
	}
}

// sortStrings sorts a slice of strings in place.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func TestConvertTiRexToGGUF(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "model.safetensors")
	ggufPath := filepath.Join(dir, "model.gguf")

	// Create synthetic SafeTensors with tensors matching a 2-layer TiRex model.
	tensors := map[string][]float32{
		"blocks.0.slstm.weight_ih": {1, 2, 3, 4},
		"blocks.0.slstm.weight_hh": {5, 6, 7, 8},
		"blocks.1.mlstm.weight_q":  {9, 10, 11, 12},
		"blocks.1.mlstm.weight_k":  {13, 14, 15, 16},
		"input_proj.weight":        {17, 18},
		"output_head.weight":       {19, 20},
	}
	writeSyntheticSafeTensors(t, stPath, tensors)

	cfg := TiRexConvertConfig{
		NumLayers:  2,
		HiddenDim:  64,
		BlockTypes: []string{"slstm", "mlstm"},
		ModelName:  "TiRex-test",
	}

	if err := ConvertTiRexToGGUF(stPath, ggufPath, cfg); err != nil {
		t.Fatalf("ConvertTiRexToGGUF: %v", err)
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
	if arch, ok := gf.GetString("general.architecture"); !ok || arch != "tirex" {
		t.Errorf("general.architecture = %q, want %q", arch, "tirex")
	}
	if name, ok := gf.GetString("general.name"); !ok || name != "TiRex-test" {
		t.Errorf("general.name = %q, want %q", name, "TiRex-test")
	}
	if bc, ok := gf.GetUint32("tirex.block_count"); !ok || bc != 2 {
		t.Errorf("tirex.block_count = %d, want 2", bc)
	}
	if hd, ok := gf.GetUint32("tirex.hidden_dim"); !ok || hd != 64 {
		t.Errorf("tirex.hidden_dim = %d, want 64", hd)
	}

	// Verify tensor count.
	if len(gf.Tensors) != len(tensors) {
		t.Fatalf("tensor count = %d, want %d", len(gf.Tensors), len(tensors))
	}

	// Verify tensor names were mapped correctly.
	expectedNames := map[string]bool{
		"tirex.block.0.slstm.weight_ih": true,
		"tirex.block.0.slstm.weight_hh": true,
		"tirex.block.1.mlstm.weight_q":  true,
		"tirex.block.1.mlstm.weight_k":  true,
		"tirex.input_proj.weight":        true,
		"tirex.output_head.weight":       true,
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

	// Verify tensor data round-trips correctly by loading tensors.
	loadedTensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	// Spot-check one tensor's data.
	lt, ok := loadedTensors["tirex.block.0.slstm.weight_ih"]
	if !ok {
		t.Fatal("missing tirex.block.0.slstm.weight_ih in loaded tensors")
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

func TestConvertTiRexToGGUF_F64Conversion(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "model.safetensors")
	ggufPath := filepath.Join(dir, "model.gguf")

	// Build a SafeTensors file with F64 tensors manually.
	type stMeta struct {
		DType       string  `json:"dtype"`
		Shape       []int   `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	f64Data := []float64{1.5, 2.5, 3.5}
	rawBytes := make([]byte, len(f64Data)*8)
	for i, v := range f64Data {
		binary.LittleEndian.PutUint64(rawBytes[i*8:], math.Float64bits(v))
	}

	header := map[string]stMeta{
		"input_proj.weight": {
			DType:       "F64",
			Shape:       []int{3},
			DataOffsets: [2]int64{0, int64(len(rawBytes))},
		},
	}

	headerJSON, _ := json.Marshal(header)

	sf, err := os.Create(stPath)
	if err != nil {
		t.Fatal(err)
	}
	_ = binary.Write(sf, binary.LittleEndian, uint64(len(headerJSON)))
	_, _ = sf.Write(headerJSON)
	_, _ = sf.Write(rawBytes)
	_ = sf.Close()

	cfg := TiRexConvertConfig{
		NumLayers:  1,
		HiddenDim:  3,
		BlockTypes: []string{"slstm"},
	}

	if err := ConvertTiRexToGGUF(stPath, ggufPath, cfg); err != nil {
		t.Fatalf("ConvertTiRexToGGUF: %v", err)
	}

	// Verify the converted tensor is F32.
	gf2, err := os.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = gf2.Close() }()

	parsed, err := gguf.Parse(gf2)
	if err != nil {
		t.Fatalf("parse GGUF: %v", err)
	}

	loadedTensors, err := gguf.LoadTensors(parsed, gf2)
	if err != nil {
		t.Fatalf("LoadTensors: %v", err)
	}

	lt, ok := loadedTensors["tirex.input_proj.weight"]
	if !ok {
		t.Fatal("missing tirex.input_proj.weight")
	}

	got := lt.Data()
	want := []float32{1.5, 2.5, 3.5}
	if len(got) != len(want) {
		t.Fatalf("tensor data length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("data[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestParseSafeTensorsHeader(t *testing.T) {
	// Build a minimal SafeTensors file with __metadata__ entry.
	type stMeta struct {
		DType       string  `json:"dtype"`
		Shape       []int   `json:"shape"`
		DataOffsets [2]int64 `json:"data_offsets"`
	}

	raw := map[string]interface{}{
		"__metadata__": map[string]string{"format": "pt"},
		"weight": stMeta{
			DType:       "F32",
			Shape:       []int{2},
			DataOffsets: [2]int64{0, 8},
		},
	}

	headerJSON, _ := json.Marshal(raw)

	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON)))
	buf.Write(headerJSON)
	// Write 8 bytes of dummy tensor data.
	buf.Write(make([]byte, 8))

	r := bytes.NewReader(buf.Bytes())
	header, dataOffset, err := parseSafeTensorsHeader(r)
	if err != nil {
		t.Fatalf("parseSafeTensorsHeader: %v", err)
	}

	// __metadata__ should be skipped.
	if _, ok := header["__metadata__"]; ok {
		t.Error("__metadata__ should be filtered from header")
	}

	if _, ok := header["weight"]; !ok {
		t.Error("missing 'weight' tensor in header")
	}

	expectedOffset := int64(8 + len(headerJSON))
	if dataOffset != expectedOffset {
		t.Errorf("dataOffset = %d, want %d", dataOffset, expectedOffset)
	}
}
