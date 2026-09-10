package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// ParameterSetIdentity hashes sorted tensor names, shapes and float32 values.
// This pins the decoded parameter set, not the source GGUF byte representation.
func ParameterSetIdentity(tensors map[string]*tensor.TensorNumeric[float32]) (string, error) {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) == 0 {
		return "", fmt.Errorf("inference: parameter set is empty")
	}
	hash := sha256.New()
	var encoded [4]byte
	for _, name := range names {
		value := tensors[name]
		if value == nil {
			return "", fmt.Errorf("inference: nil tensor %s", name)
		}
		header, err := json.Marshal(struct {
			Name  string
			Shape []int
		}{name, value.Shape()})
		if err != nil {
			return "", err
		}
		if _, err := hash.Write(append(header, '\n')); err != nil {
			return "", err
		}
		for _, v := range value.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return "", fmt.Errorf("inference: nonfinite parameter %s", name)
			}
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(v))
			if _, err := hash.Write(encoded[:]); err != nil {
				return "", err
			}
		}
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// CompileArchitectureReference invokes the existing registered builder after
// validating version, configuration and parameter identity. Current architecture
// builders retain their inference-only contracts; no training claim is added.
func CompileArchitectureReference(ctx context.Context, reference dsl.ArchitectureReference, tensors map[string]*tensor.TensorNumeric[float32], engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if engine == nil {
		return nil, nil, fmt.Errorf("inference: engine required")
	}
	if err := reference.Validate(); err != nil {
		return nil, nil, err
	}
	identity, err := ParameterSetIdentity(tensors)
	if err != nil {
		return nil, nil, err
	}
	if identity != reference.ParametersSHA256 {
		return nil, nil, fmt.Errorf("inference: parameter identity mismatch")
	}
	factory, ok := GetArchitecture(reference.Architecture)
	if !ok {
		return nil, nil, fmt.Errorf("inference: architecture builder unavailable")
	}
	// Own the configuration before passing it to builders that may normalize it.
	raw, err := json.Marshal(reference.Config)
	if err != nil {
		return nil, nil, err
	}
	var owned dsl.ArchitectureReference
	if err := json.Unmarshal(raw, &owned.Config); err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	return factory(tensors, &owned.Config, engine)
}
