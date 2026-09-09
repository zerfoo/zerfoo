package tabular

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/zerfoo/ztensor/compute"
)

// MigrateLegacyClassifier imports bounded ZTAB v1 weights without calling the
// legacy unbounded loader. legacyLabels explicitly maps Long=0, Short=1, Flat=2
// to user labels and must match the supplied dataset's label order. GELU models
// are rejected because the certified classifier recipe currently uses ReLU.
func MigrateLegacyClassifier(ctx context.Context, source, output string, dataset *Dataset,
	legacyLabels []string, engine compute.Engine[float32]) (string, error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if dataset == nil || len(legacyLabels) != 3 || !slices.Equal(legacyLabels, dataset.manifest.Labels) {
		return "", fmt.Errorf("tabular: explicit legacy labels must match the dataset's three-class order")
	}
	root, err := os.OpenRoot(filepath.Dir(source))
	if err != nil {
		return "", fmt.Errorf("tabular: legacy root: %w", err)
	}
	raw, readErr := readBoundedBundleFile(root, filepath.Base(source), 128<<20)
	if err := errors.Join(readErr, root.Close()); err != nil {
		return "", err
	}
	if len(raw) < 12 || string(raw[:4]) != "ZTAB" || binary.LittleEndian.Uint32(raw[4:8]) != 1 {
		return "", fmt.Errorf("tabular: unsupported or truncated legacy header")
	}
	configLength := uint64(binary.LittleEndian.Uint32(raw[8:12]))
	if configLength > 1<<20 || configLength > uint64(len(raw)-12) {
		return "", fmt.Errorf("tabular: invalid legacy config length")
	}
	var legacy ModelConfig
	if err := json.Unmarshal(raw[12:12+int(configLength)], &legacy); err != nil {
		return "", fmt.Errorf("tabular: legacy config: %w", err)
	}
	if legacy.Activation != ActivationReLU || len(legacy.HiddenDims) == 0 {
		return "", fmt.Errorf("tabular: unsupported legacy topology or activation")
	}
	config := ClassifierConfig{InputDim: legacy.InputDim, ClassCount: 3, Labels: slices.Clone(legacyLabels), HiddenDims: legacy.HiddenDims}
	if err := validateClassifierConfig(config); err != nil {
		return "", err
	}
	model, err := NewClassifier(config, engine)
	if err != nil {
		return "", err
	}
	expected := 0
	for _, p := range model.params {
		expected += len(p.Value.Data()) * 4
	}
	offset := 12 + int(configLength)
	if len(raw)-offset != expected {
		return "", fmt.Errorf("tabular: legacy weight length mismatch")
	}
	for _, p := range model.params {
		values := make([]float32, len(p.Value.Data()))
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[offset : offset+4]))
			offset += 4
			if math.IsNaN(float64(values[i])) || math.IsInf(float64(values[i]), 0) {
				return "", fmt.Errorf("tabular: nonfinite legacy weight")
			}
		}
		p.Value.SetData(values)
	}
	return SaveClassifierBundle(ctx, output, model, dataset, "")
}
