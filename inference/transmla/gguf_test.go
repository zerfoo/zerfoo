package transmla

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// buildTestGGUF creates an in-memory GGUF file with the given tensors and metadata
// using the ztensor/gguf writer, then returns the bytes.
func buildTestGGUF(t *testing.T, metadata map[string]any, tensors []testTensor) []byte {
	t.Helper()
	w := ztensorgguf.NewWriter()

	for k, v := range metadata {
		switch val := v.(type) {
		case string:
			w.AddMetadataString(k, val)
		case uint32:
			w.AddMetadataUint32(k, val)
		case float32:
			w.AddMetadataFloat32(k, val)
		}
	}

	for _, tt := range tensors {
		w.AddTensorF32(tt.name, tt.shape, tt.data)
	}

	var buf bytes.Buffer
	if err := w.Write(&buf); err != nil {
		t.Fatalf("write test GGUF: %v", err)
	}
	return buf.Bytes()
}

type testTensor struct {
	name  string
	shape []int
	data  []float32
}

func TestConvertGGUF_Basic(t *testing.T) {
	// Build a minimal GGUF with 2 layers, each having k_proj and v_proj weights.
	// Also include a non-attention tensor to verify passthrough.
	dModel := 4
	dK := 2
	dV := 2
	rank := 2

	var tensors []testTensor
	metadata := map[string]any{
		"general.architecture": "llama",
		"llama.block_count":    uint32(2),
	}

	for layer := 0; layer < 2; layer++ {
		// K projection: [dK, dModel]
		kData := make([]float32, dK*dModel)
		for i := range kData {
			kData[i] = float32(layer*100 + i + 1)
		}
		tensors = append(tensors, testTensor{
			name:  fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer),
			shape: []int{dK, dModel},
			data:  kData,
		})

		// V projection: [dV, dModel]
		vData := make([]float32, dV*dModel)
		for i := range vData {
			vData[i] = float32(layer*100 + i + 50)
		}
		tensors = append(tensors, testTensor{
			name:  fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer),
			shape: []int{dV, dModel},
			data:  vData,
		})

		// Q projection (passthrough)
		qData := make([]float32, dK*dModel)
		for i := range qData {
			qData[i] = float32(i + 1)
		}
		tensors = append(tensors, testTensor{
			name:  fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer),
			shape: []int{dK, dModel},
			data:  qData,
		})
	}

	// Add embedding tensor.
	embedData := make([]float32, 8)
	for i := range embedData {
		embedData[i] = float32(i) * 0.1
	}
	tensors = append(tensors, testTensor{
		name:  "model.embed_tokens.weight",
		shape: []int{2, 4},
		data:  embedData,
	})

	srcBytes := buildTestGGUF(t, metadata, tensors)

	// Convert.
	src := bytes.NewReader(srcBytes)
	var dst bytes.Buffer
	err := ConvertGGUF(src, &dst, ConvertGGUFOptions{
		Rank:       rank,
		SourceArch: "llama",
	})
	if err != nil {
		t.Fatalf("ConvertGGUF: %v", err)
	}

	// Parse the output GGUF.
	outReader := bytes.NewReader(dst.Bytes())
	outFile, err := gguf.Parse(outReader)
	if err != nil {
		t.Fatalf("parse output GGUF: %v", err)
	}

	// Verify metadata.
	if v, ok := outFile.GetUint32("transmla.kv_lora_dim"); !ok || v != uint32(rank) {
		t.Errorf("transmla.kv_lora_dim = %d, want %d", v, rank)
	}
	if v, ok := outFile.GetString("transmla.source_arch"); !ok || v != "llama" {
		t.Errorf("transmla.source_arch = %q, want %q", v, "llama")
	}
	if v, ok := outFile.GetString("general.architecture"); !ok || v != "llama" {
		t.Errorf("general.architecture = %q, want %q", v, "llama")
	}

	// Build a map of output tensor names.
	tensorNames := make(map[string]gguf.TensorInfo)
	for _, ti := range outFile.Tensors {
		tensorNames[ti.Name] = ti
	}

	// Verify k_proj and v_proj are removed.
	for layer := 0; layer < 2; layer++ {
		kName := fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
		vName := fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
		if _, ok := tensorNames[kName]; ok {
			t.Errorf("k_proj tensor %q should be removed", kName)
		}
		if _, ok := tensorNames[vName]; ok {
			t.Errorf("v_proj tensor %q should be removed", vName)
		}
	}

	// Verify TransMLA tensors exist with correct shapes.
	for layer := 0; layer < 2; layer++ {
		prefix := fmt.Sprintf("transmla.%d.", layer)

		wDKV, ok := tensorNames[prefix+"wDKV"]
		if !ok {
			t.Fatalf("missing tensor %swDKV", prefix)
		}
		// wDKV shape: [dModel, rank] = [4, 2], stored in GGUF as reversed: [rank, dModel]
		if got := reverseDims(wDKV.Dimensions); got[0] != uint64(dModel) || got[1] != uint64(rank) {
			t.Errorf("%swDKV shape = %v, want [%d, %d]", prefix, got, dModel, rank)
		}

		wUK, ok := tensorNames[prefix+"wUK"]
		if !ok {
			t.Fatalf("missing tensor %swUK", prefix)
		}
		if got := reverseDims(wUK.Dimensions); got[0] != uint64(dK) || got[1] != uint64(rank) {
			t.Errorf("%swUK shape = %v, want [%d, %d]", prefix, got, dK, rank)
		}

		wUV, ok := tensorNames[prefix+"wUV"]
		if !ok {
			t.Fatalf("missing tensor %swUV", prefix)
		}
		if got := reverseDims(wUV.Dimensions); got[0] != uint64(dV) || got[1] != uint64(rank) {
			t.Errorf("%swUV shape = %v, want [%d, %d]", prefix, got, dV, rank)
		}
	}

	// Verify passthrough tensors are present.
	for layer := 0; layer < 2; layer++ {
		qName := fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
		if _, ok := tensorNames[qName]; !ok {
			t.Errorf("passthrough tensor %q missing", qName)
		}
	}
	if _, ok := tensorNames["model.embed_tokens.weight"]; !ok {
		t.Error("passthrough tensor model.embed_tokens.weight missing")
	}
}

func TestConvertGGUF_ReconstructionAccuracy(t *testing.T) {
	// Create a rank-1 KV matrix and decompose at rank 1 — reconstruction should
	// be nearly exact.
	dModel := 4
	dK := 2
	dV := 2

	// Build a rank-1 matrix: all rows are multiples of [1,2,3,4].
	kData := []float32{1, 2, 3, 4, 2, 4, 6, 8}
	vData := []float32{3, 6, 9, 12, 4, 8, 12, 16}

	tensors := []testTensor{
		{name: "model.layers.0.self_attn.k_proj.weight", shape: []int{dK, dModel}, data: kData},
		{name: "model.layers.0.self_attn.v_proj.weight", shape: []int{dV, dModel}, data: vData},
	}
	metadata := map[string]any{"general.architecture": "llama"}
	srcBytes := buildTestGGUF(t, metadata, tensors)

	src := bytes.NewReader(srcBytes)
	var dst bytes.Buffer
	err := ConvertGGUF(src, &dst, ConvertGGUFOptions{Rank: 1, SourceArch: "llama"})
	if err != nil {
		t.Fatalf("ConvertGGUF: %v", err)
	}

	// Parse output and read the decomposed tensors.
	outReader := bytes.NewReader(dst.Bytes())
	outFile, err := gguf.Parse(outReader)
	if err != nil {
		t.Fatalf("parse output: %v", err)
	}

	// Read wDKV, wUK, wUV tensor data.
	wDKVData := readF32Tensor(t, outReader, outFile, "transmla.0.wDKV")
	wUKData := readF32Tensor(t, outReader, outFile, "transmla.0.wUK")
	wUVData := readF32Tensor(t, outReader, outFile, "transmla.0.wUV")

	// Reconstruct W_K = wUK * wDKV^T and check against original.
	rank := 1
	for i := 0; i < dK; i++ {
		for j := 0; j < dModel; j++ {
			var val float64
			for r := 0; r < rank; r++ {
				val += float64(wUKData[i*rank+r]) * float64(wDKVData[j*rank+r])
			}
			orig := float64(kData[i*dModel+j])
			if diff := math.Abs(val - orig); diff > 1e-4 {
				t.Errorf("wK_reconstructed[%d][%d] = %g, want %g (diff %g)", i, j, val, orig, diff)
			}
		}
	}

	// Reconstruct W_V = wUV * wDKV^T and check.
	for i := 0; i < dV; i++ {
		for j := 0; j < dModel; j++ {
			var val float64
			for r := 0; r < rank; r++ {
				val += float64(wUVData[i*rank+r]) * float64(wDKVData[j*rank+r])
			}
			orig := float64(vData[i*dModel+j])
			if diff := math.Abs(val - orig); diff > 1e-4 {
				t.Errorf("wV_reconstructed[%d][%d] = %g, want %g (diff %g)", i, j, val, orig, diff)
			}
		}
	}
}

func TestConvertGGUF_InvalidRank(t *testing.T) {
	tensors := []testTensor{
		{name: "model.layers.0.self_attn.k_proj.weight", shape: []int{2, 4}, data: make([]float32, 8)},
		{name: "model.layers.0.self_attn.v_proj.weight", shape: []int{2, 4}, data: make([]float32, 8)},
	}
	srcBytes := buildTestGGUF(t, nil, tensors)
	src := bytes.NewReader(srcBytes)
	var dst bytes.Buffer
	err := ConvertGGUF(src, &dst, ConvertGGUFOptions{Rank: 0})
	if err == nil {
		t.Error("expected error for rank 0, got nil")
	}
}

func TestConvertGGUF_MissingVProj(t *testing.T) {
	// Only k_proj, no v_proj — should fail.
	tensors := []testTensor{
		{name: "model.layers.0.self_attn.k_proj.weight", shape: []int{2, 4}, data: make([]float32, 8)},
	}
	srcBytes := buildTestGGUF(t, nil, tensors)
	src := bytes.NewReader(srcBytes)
	var dst bytes.Buffer
	err := ConvertGGUF(src, &dst, ConvertGGUFOptions{Rank: 1})
	if err == nil {
		t.Error("expected error for missing v_proj, got nil")
	}
}

func TestConvertGGUF_NoKVTensors(t *testing.T) {
	// GGUF with no K/V tensors — should produce valid output with no transmla tensors.
	tensors := []testTensor{
		{name: "model.embed_tokens.weight", shape: []int{2, 4}, data: make([]float32, 8)},
	}
	metadata := map[string]any{"general.architecture": "llama"}
	srcBytes := buildTestGGUF(t, metadata, tensors)
	src := bytes.NewReader(srcBytes)
	var dst bytes.Buffer
	err := ConvertGGUF(src, &dst, ConvertGGUFOptions{Rank: 2, SourceArch: "llama"})
	if err != nil {
		t.Fatalf("ConvertGGUF: %v", err)
	}

	outReader := bytes.NewReader(dst.Bytes())
	outFile, err := gguf.Parse(outReader)
	if err != nil {
		t.Fatalf("parse output: %v", err)
	}

	// Should have the original tensor plus metadata.
	if len(outFile.Tensors) != 1 {
		t.Errorf("expected 1 tensor, got %d", len(outFile.Tensors))
	}
	if outFile.Tensors[0].Name != "model.embed_tokens.weight" {
		t.Errorf("tensor name = %q, want model.embed_tokens.weight", outFile.Tensors[0].Name)
	}
	if v, ok := outFile.GetUint32("transmla.kv_lora_dim"); !ok || v != 2 {
		t.Errorf("transmla.kv_lora_dim = %d, want 2", v)
	}
}

// Helper to reverse GGUF dimensions (innermost-first -> outermost-first).
func reverseDims(dims []uint64) []uint64 {
	out := make([]uint64, len(dims))
	for i, d := range dims {
		out[len(dims)-1-i] = d
	}
	return out
}

// readF32Tensor reads a float32 tensor's data from a parsed GGUF file.
func readF32Tensor(t *testing.T, r *bytes.Reader, f *gguf.File, name string) []float32 {
	t.Helper()
	for _, ti := range f.Tensors {
		if ti.Name != name {
			continue
		}
		var nElems int64 = 1
		for _, d := range ti.Dimensions {
			nElems *= int64(d)
		}
		offset := f.DataOffset + int64(ti.Offset)
		if _, err := r.Seek(offset, 0); err != nil {
			t.Fatalf("seek %s: %v", name, err)
		}
		data := make([]float32, nElems)
		for i := range nElems {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				t.Fatalf("read %s[%d]: %v", name, i, err)
			}
			data[i] = math.Float32frombits(bits)
		}
		return data
	}
	t.Fatalf("tensor %q not found", name)
	return nil
}
