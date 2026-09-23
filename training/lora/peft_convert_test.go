package lora

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/gguf"
)

func TestConvertPEFTAdapter(t *testing.T) {
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "adapter.safetensors")
	configPath := filepath.Join(dir, "adapter_config.json")
	outputPath := filepath.Join(dir, "adapter.gguf")
	config := `{"peft_type":"LORA","r":1,"lora_alpha":2,"target_modules":["proj"],"base_model_name_or_path":"example/base","revision":"pinned"}`
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	header, err := json.Marshal(map[string]any{
		"base_model.model.model.layer.proj.lora_A.weight": map[string]any{"dtype": "F32", "shape": []int{1, 2}, "data_offsets": []int{0, 8}},
		"base_model.model.model.layer.proj.lora_B.weight": map[string]any{"dtype": "F32", "shape": []int{2, 1}, "data_offsets": []int{8, 16}},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 8+len(header)+16)
	binary.LittleEndian.PutUint64(raw[:8], uint64(len(header)))
	copy(raw[8:], header)
	for i, value := range []float32{1, 2, 3, 4} {
		binary.LittleEndian.PutUint32(raw[8+len(header)+i*4:], math.Float32bits(value))
	}
	if err := os.WriteFile(weightsPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := ConvertPEFTAdapter(weightsPath, configPath, outputPath); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	file, err := gguf.Parse(f)
	if err != nil {
		t.Fatal(err)
	}
	rank, ok := file.GetUint32("lora.rank")
	if !ok || rank != 1 || len(file.Tensors) != 2 {
		t.Fatalf("adapter metadata or tensor count: rank=%d tensors=%d", rank, len(file.Tensors))
	}
	if err := ConvertPEFTAdapter(weightsPath, configPath, outputPath); err == nil {
		t.Fatal("converter overwrote an existing adapter")
	}
}
