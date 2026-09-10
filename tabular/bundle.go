package tabular

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	writer "github.com/zerfoo/ztensor/gguf"
)

// BundleManifest describes an immutable float32 deployment bundle. Identity is
// the SHA-256 of these exact JSON bytes; weights have their own content hash.
type BundleManifest struct {
	DefinitionSHA256    string           `json:"definition_sha256,omitempty"`
	Version             int              `json:"version"`
	Architecture        string           `json:"architecture"`
	Config              ClassifierConfig `json:"config"`
	Features            []string         `json:"features"`
	Preprocessing       Standardization  `json:"preprocessing"`
	DatasetHash         string           `json:"dataset_hash"`
	WeightsSHA256       string           `json:"weights_sha256"`
	EvaluationReference string           `json:"evaluation_reference,omitempty"`
}

// ClassifierBundle contains a validated model and its preprocessing. ID should
// be pinned by callers when loading an artifact selected by an evaluation run.
type ClassifierBundle struct {
	Model    *Classifier[float32]
	Manifest BundleManifest
	ID       string
}

func contentHash(raw []byte) string { sum := sha256.Sum256(raw); return hex.EncodeToString(sum[:]) }

// SaveClassifierBundle atomically publishes a new immutable directory. Existing
// paths are not overwritten. An evaluation reference does not imply qualification.
func SaveClassifierBundle(ctx context.Context, path string, model *Classifier[float32], dataset *Dataset, evaluation string) (id string, err error) {
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if model == nil || dataset == nil {
		return "", fmt.Errorf("tabular: model and dataset are required")
	}
	data := dataset.Manifest()
	if model.config.InputDim != len(data.Options.Features) || !slices.Equal(model.config.Labels, data.Labels) {
		return "", fmt.Errorf("tabular: bundle dataset/model schema mismatch")
	}
	datasetHash, err := dataset.Hash()
	if err != nil {
		return "", err
	}
	architecture := "zerfoo.tabular.linear.v1"
	if len(model.config.HiddenDims) > 0 {
		architecture = "zerfoo.tabular.mlp.v1"
	}
	if model.config.Definition != nil {
		architecture = "zerfoo.dsl.v1"
	}
	gw := writer.NewWriter()
	gw.AddMetadataString("general.architecture", architecture)
	for _, param := range model.params {
		values := param.Value.Data()
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return "", fmt.Errorf("tabular: cannot export nonfinite weights")
			}
		}
		gw.AddTensorF32(param.Name, param.Value.Shape(), values)
	}
	var weights bytes.Buffer
	if err := gw.Write(&weights); err != nil {
		return "", fmt.Errorf("tabular: GGUF export: %w", err)
	}
	manifest := BundleManifest{Version: 1, Architecture: architecture, Config: model.Config(), Features: data.Options.Features, Preprocessing: data.Preprocessing, DatasetHash: datasetHash, WeightsSHA256: contentHash(weights.Bytes()), EvaluationReference: evaluation}
	if model.compiled != nil {
		manifest.DefinitionSHA256 = model.compiled.ID()
	}
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return "", fmt.Errorf("tabular: bundle manifest: %w", err)
	}
	raw = append(raw, '\n')
	id = contentHash(raw)
	lockPath := path + ".lock"
	if err := os.Mkdir(lockPath, 0700); err != nil {
		return "", fmt.Errorf("tabular: acquire output lock: %w", err)
	}
	defer func() { err = errors.Join(err, os.Remove(lockPath)) }()
	if _, statErr := os.Lstat(path); statErr == nil {
		return "", fmt.Errorf("tabular: bundle output already exists")
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return "", fmt.Errorf("tabular: inspect output: %w", statErr)
	}
	stage, err := os.MkdirTemp(filepath.Dir(path), ".classifier-bundle-")
	if err != nil {
		return "", fmt.Errorf("tabular: stage bundle: %w", err)
	}
	defer func() { err = errors.Join(err, os.RemoveAll(stage)) }()
	root, err := os.OpenRoot(stage)
	if err != nil {
		return "", fmt.Errorf("tabular: open staging root: %w", err)
	}
	for _, file := range []struct {
		name string
		data []byte
	}{{"weights.gguf", weights.Bytes()}, {"manifest.json", raw}, {"bundle.sha256", []byte(id + "\n")}} {
		if writeErr := root.WriteFile(file.name, file.data, 0600); writeErr != nil {
			return "", errors.Join(fmt.Errorf("tabular: write %s: %w", file.name, writeErr), root.Close())
		}
	}
	if err := root.Close(); err != nil {
		return "", fmt.Errorf("tabular: close staging root: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.Rename(stage, path); err != nil {
		return "", fmt.Errorf("tabular: publish bundle: %w", err)
	}
	return id, nil
}

func readBoundedBundleFile(root *os.Root, name string, limit int64) ([]byte, error) {
	file, err := root.Open(name)
	if err != nil {
		return nil, fmt.Errorf("tabular: open %s: %w", name, err)
	}
	raw, readErr := io.ReadAll(io.LimitReader(file, limit+1))
	if err := errors.Join(readErr, file.Close()); err != nil {
		return nil, fmt.Errorf("tabular: read %s: %w", name, err)
	}
	if int64(len(raw)) > limit {
		return nil, fmt.Errorf("tabular: %s exceeds size limit", name)
	}
	return raw, nil
}

// LoadClassifierBundle verifies content and tensor contracts before publishing
// the model. expectedID may be empty for inspection; deployments should pin it.
func LoadClassifierBundle(ctx context.Context, path, expectedID string, engine compute.Engine[float32]) (*ClassifierBundle, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(path)
	if err != nil {
		return nil, fmt.Errorf("tabular: open bundle: %w", err)
	}
	files := make(map[string][]byte)
	for _, file := range []struct {
		name  string
		limit int64
	}{{"manifest.json", 1 << 20}, {"bundle.sha256", 128}, {"weights.gguf", 128 << 20}} {
		raw, readErr := readBoundedBundleFile(root, file.name, file.limit)
		if readErr != nil {
			return nil, errors.Join(readErr, root.Close())
		}
		files[file.name] = raw
	}
	if err := root.Close(); err != nil {
		return nil, fmt.Errorf("tabular: close bundle: %w", err)
	}
	id := contentHash(files["manifest.json"])
	if id != strings.TrimSpace(string(files["bundle.sha256"])) || (expectedID != "" && id != expectedID) {
		return nil, fmt.Errorf("tabular: bundle identity mismatch")
	}
	var manifest BundleManifest
	decoder := json.NewDecoder(bytes.NewReader(files["manifest.json"]))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return nil, fmt.Errorf("tabular: decode bundle manifest: %w", err)
	}
	if err := decoder.Decode(new(any)); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("tabular: trailing manifest content")
	}
	if manifest.Version != 1 {
		return nil, fmt.Errorf("tabular: unsupported bundle version %d", manifest.Version)
	}
	if manifest.WeightsSHA256 != contentHash(files["weights.gguf"]) {
		return nil, fmt.Errorf("tabular: weight hash mismatch")
	}
	if err := validateClassifierConfig(manifest.Config); err != nil {
		return nil, err
	}
	if len(manifest.Features) != manifest.Config.InputDim {
		return nil, fmt.Errorf("tabular: bundle feature count mismatch")
	}
	seen := make(map[string]bool)
	for _, name := range manifest.Features {
		if name == "" || seen[name] {
			return nil, fmt.Errorf("tabular: invalid bundle feature names")
		}
		seen[name] = true
	}
	if _, err := manifest.Preprocessing.Transform(make([]float64, manifest.Config.InputDim)); err != nil {
		return nil, err
	}
	expectedArchitecture := "zerfoo.tabular.linear.v1"
	if len(manifest.Config.HiddenDims) > 0 {
		expectedArchitecture = "zerfoo.tabular.mlp.v1"
	}
	if manifest.Config.Definition != nil {
		expectedArchitecture = "zerfoo.dsl.v1"
	}
	if manifest.Config.Definition != nil {
		checked, err := dsl.Validate(*manifest.Config.Definition)
		if err != nil {
			return nil, err
		}
		if checked.ID != manifest.DefinitionSHA256 {
			return nil, fmt.Errorf("tabular: DSL definition identity mismatch")
		}
	} else if manifest.DefinitionSHA256 != "" {
		return nil, fmt.Errorf("tabular: unexpected DSL identity")
	}

	if manifest.Architecture != expectedArchitecture {
		return nil, fmt.Errorf("tabular: unsupported classifier architecture")
	}
	parsed, err := gguf.Parse(bytes.NewReader(files["weights.gguf"]))
	if err != nil {
		return nil, fmt.Errorf("tabular: parse bundle GGUF: %w", err)
	}
	if parsed.Metadata["general.architecture"] != manifest.Architecture {
		return nil, fmt.Errorf("tabular: GGUF architecture mismatch")
	}
	model, err := NewClassifier(manifest.Config, engine)
	if err != nil {
		return nil, err
	}
	if len(parsed.Tensors) != len(model.params) {
		return nil, fmt.Errorf("tabular: unexpected tensor count")
	}
	expectedShapes := make(map[string][]int)
	for _, p := range model.params {
		expectedShapes[p.Name] = p.Value.Shape()
	}
	seenTensors := make(map[string]bool)
	for _, info := range parsed.Tensors {
		shape, ok := expectedShapes[info.Name]
		if !ok || seenTensors[info.Name] || info.Type != gguf.GGMLTypeF32 || len(shape) != len(info.Dimensions) {
			return nil, fmt.Errorf("tabular: invalid tensor %q", info.Name)
		}
		seenTensors[info.Name] = true
		for i, d := range shape {
			if uint64(d) != info.Dimensions[len(shape)-1-i] {
				return nil, fmt.Errorf("tabular: tensor %q shape mismatch", info.Name)
			}
		}
	}
	tensors, err := gguf.LoadTensors(parsed, bytes.NewReader(files["weights.gguf"]))
	if err != nil {
		return nil, fmt.Errorf("tabular: load weights: %w", err)
	}
	for _, p := range model.params {
		values := tensors[p.Name].Data()
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("tabular: nonfinite bundle weights")
			}
		}
		p.Value.SetData(values)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &ClassifierBundle{Model: model, Manifest: manifest, ID: id}, nil
}

// PredictRaw applies saved preprocessing and returns the model prediction.
func (b *ClassifierBundle) PredictRaw(ctx context.Context, values []float64) (Prediction, error) {
	row, err := b.Manifest.Preprocessing.Transform(values)
	if err != nil {
		return Prediction{}, err
	}
	return b.Model.Predict(ctx, row)
}
