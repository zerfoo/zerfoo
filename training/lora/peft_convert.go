package lora

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/zerfoo/zerfoo/model/safetensors"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

type peftConfig struct {
	PEFTType        string         `json:"peft_type"`
	Rank            int            `json:"r"`
	Alpha           float32        `json:"lora_alpha"`
	BaseModel       string         `json:"base_model_name_or_path"`
	Revision        string         `json:"revision"`
	TargetModules   []string       `json:"target_modules"`
	RankPattern     map[string]int `json:"rank_pattern"`
	AlphaPattern    map[string]any `json:"alpha_pattern"`
	Bias            string         `json:"bias"`
	LoRABias        bool           `json:"lora_bias"`
	ModulesToSave   []string       `json:"modules_to_save"`
	TargetParams    []string       `json:"target_parameters"`
	LayerReplicas   any            `json:"layer_replication"`
	UseQALoRA       bool           `json:"use_qalora"`
	UseBDLoRA       bool           `json:"use_bdlora"`
	UseDoRA         bool           `json:"use_dora"`
	UseRSLora       bool           `json:"use_rslora"`
	FanInFanOut     bool           `json:"fan_in_fan_out"`
	LayersToConvert any            `json:"layers_to_transform"`
}

type peftPair struct {
	a, b       safetensors.Tensor
	hasA, hasB bool
}

// ConvertPEFTAdapter writes a standard LoRA GGUF adapter from a PEFT F32
// safetensors file. It handles ordinary A/B LoRA only; unsupported PEFT
// variants fail rather than producing plausible but wrong weights.
func ConvertPEFTAdapter(weightsPath, configPath, outputPath string) error {
	rawConfig, err := os.ReadFile(filepath.Clean(configPath))
	if err != nil {
		return fmt.Errorf("read PEFT config: %w", err)
	}
	var cfg peftConfig
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return fmt.Errorf("decode PEFT config: %w", err)
	}
	if cfg.PEFTType != "LORA" || cfg.Rank <= 0 || cfg.Alpha <= 0 || len(cfg.TargetModules) == 0 || cfg.UseDoRA || cfg.UseRSLora || cfg.UseQALoRA || cfg.UseBDLoRA || cfg.FanInFanOut || cfg.LoRABias || (cfg.Bias != "" && cfg.Bias != "none") || len(cfg.ModulesToSave) > 0 || len(cfg.TargetParams) > 0 || cfg.LayerReplicas != nil || cfg.LayersToConvert != nil || len(cfg.RankPattern) > 0 || len(cfg.AlphaPattern) > 0 {
		return fmt.Errorf("unsupported PEFT adapter configuration")
	}
	weights, err := safetensors.ReadF32(weightsPath)
	if err != nil {
		return fmt.Errorf("read PEFT weights: %w", err)
	}
	pairs := make(map[string]*peftPair)
	seenModule := make(map[string]bool)
	for sourceName, value := range weights {
		name, ok := strings.CutPrefix(sourceName, "base_model.model.")
		if !ok {
			return fmt.Errorf("unexpected PEFT tensor %q", sourceName)
		}
		var layerName string
		var isA bool
		switch {
		case strings.HasSuffix(name, ".lora_A.weight"):
			layerName = strings.TrimSuffix(name, ".lora_A.weight")
			isA = true
		case strings.HasSuffix(name, ".lora_B.weight"):
			layerName = strings.TrimSuffix(name, ".lora_B.weight")
		default:
			return fmt.Errorf("unexpected PEFT tensor %q", sourceName)
		}
		matched := false
		for _, module := range cfg.TargetModules {
			if strings.HasSuffix(layerName, "."+module) {
				seenModule[module] = true
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf("PEFT tensor %q is outside target modules", sourceName)
		}
		pair := pairs[layerName]
		if pair == nil {
			pair = &peftPair{}
			pairs[layerName] = pair
		}
		if isA {
			pair.a, pair.hasA = value, true
		} else {
			pair.b, pair.hasB = value, true
		}
	}
	for _, module := range cfg.TargetModules {
		if !seenModule[module] {
			return fmt.Errorf("PEFT adapter has no %q target", module)
		}
	}
	names := make([]string, 0, len(pairs))
	for name, pair := range pairs {
		if !pair.hasA || !pair.hasB || len(pair.a.Shape) != 2 || len(pair.b.Shape) != 2 || pair.a.Shape[0] != cfg.Rank || pair.b.Shape[1] != cfg.Rank {
			return fmt.Errorf("PEFT layer %q has incomplete or invalid A/B tensors", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	rawWeights, err := os.ReadFile(filepath.Clean(weightsPath))
	if err != nil {
		return fmt.Errorf("hash PEFT weights: %w", err)
	}
	digest := sha256.Sum256(rawWeights)
	w := ztensorgguf.NewWriter()
	w.AddMetadataString("general.architecture", "lora")
	w.AddMetadataUint32("lora.rank", uint32(cfg.Rank))
	w.AddMetadataFloat32("lora.alpha", cfg.Alpha)
	w.AddMetadataString("lora.base_model", cfg.BaseModel)
	w.AddMetadataString("lora.base_revision", cfg.Revision)
	w.AddMetadataString("lora.source_sha256", hex.EncodeToString(digest[:]))
	w.AddMetadataStringArray("lora.target_modules", cfg.TargetModules)
	for _, name := range names {
		pair := pairs[name]
		w.AddTensorF32("lora."+name+".weight_a", pair.a.Shape, pair.a.Data)
		w.AddTensorF32("lora."+name+".weight_b", pair.b.Shape, pair.b.Data)
	}
	out, err := os.OpenFile(filepath.Clean(outputPath), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return fmt.Errorf("create GGUF adapter: %w", err)
	}
	if err := w.Write(out); err != nil {
		_ = out.Close()
		_ = os.Remove(outputPath)
		return fmt.Errorf("write GGUF adapter: %w", err)
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(outputPath)
		return fmt.Errorf("close GGUF adapter: %w", err)
	}
	return nil
}
