package dsl

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	writer "github.com/zerfoo/ztensor/gguf"
)

// WriteGGUF serializes the complete architecture and each shared tensor once.
// Metadata is caller-owned lifecycle context, not a resumable optimizer state.
func (e *Executable) WriteGGUF(ctx context.Context, out io.Writer, metadata map[string]string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	definition, err := json.Marshal(e.validated.Definition)
	if err != nil {
		return err
	}
	meta, err := json.Marshal(metadata)
	if err != nil {
		return err
	}
	if len(meta) > 1<<20 {
		return fmt.Errorf("dsl: metadata exceeds 1 MiB")
	}
	w := writer.NewWriter()
	w.AddMetadataString("general.architecture", "zerfoo.dsl.v1")
	w.AddMetadataString("zerfoo.dsl.definition", string(definition))
	w.AddMetadataString("zerfoo.dsl.identity", e.ID())
	w.AddMetadataString("zerfoo.dsl.context", string(meta))
	for _, p := range e.params {
		for _, v := range p.Value.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("dsl: nonfinite parameter %s", p.Name)
			}
		}
		w.AddTensorF32(p.Name, p.Value.Shape(), p.Value.Data())
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return w.Write(out)
}

// ReadGGUF bounds input, validates definition/shape/name/type/identity contracts,
// and reconstructs through the same compiler. expectedHash pins complete bytes.
func ReadGGUF(ctx context.Context, input io.Reader, expectedHash string, engine compute.Engine[float32]) (*Executable, map[string]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	raw, err := io.ReadAll(io.LimitReader(input, (128<<20)+1))
	if err != nil {
		return nil, nil, err
	}
	if len(raw) > 128<<20 {
		return nil, nil, fmt.Errorf("dsl: GGUF exceeds 128 MiB")
	}
	hash := sha256.Sum256(raw)
	if expectedHash != "" && expectedHash != hex.EncodeToString(hash[:]) {
		return nil, nil, fmt.Errorf("dsl: GGUF identity mismatch")
	}
	parsed, err := gguf.Parse(bytes.NewReader(raw))
	if err != nil {
		return nil, nil, err
	}
	if parsed.Metadata["general.architecture"] != "zerfoo.dsl.v1" {
		return nil, nil, fmt.Errorf("dsl: unsupported GGUF architecture")
	}
	encoded, ok := parsed.Metadata["zerfoo.dsl.definition"].(string)
	if !ok || len(encoded) > 1<<20 {
		return nil, nil, fmt.Errorf("dsl: missing or oversized definition")
	}
	var d Definition
	decoder := json.NewDecoder(bytes.NewBufferString(encoded))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&d); err != nil {
		return nil, nil, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, nil, fmt.Errorf("dsl: trailing definition")
	}
	validated, err := Validate(d)
	if err != nil {
		return nil, nil, err
	}
	if parsed.Metadata["zerfoo.dsl.identity"] != validated.ID {
		return nil, nil, fmt.Errorf("dsl: definition identity mismatch")
	}
	if len(parsed.Tensors) != len(validated.Definition.Parameters) {
		return nil, nil, fmt.Errorf("dsl: tensor count mismatch")
	}
	expected := map[string][]int{}
	for _, p := range validated.Definition.Parameters {
		expected[p.Name] = p.Shape
	}
	seen := map[string]bool{}
	for _, info := range parsed.Tensors {
		shape, ok := expected[info.Name]
		if !ok || seen[info.Name] || info.Type != gguf.GGMLTypeF32 || len(shape) != len(info.Dimensions) {
			return nil, nil, fmt.Errorf("dsl: invalid tensor %q", info.Name)
		}
		seen[info.Name] = true
		for i, n := range shape {
			if uint64(n) != info.Dimensions[len(shape)-1-i] {
				return nil, nil, fmt.Errorf("dsl: tensor shape mismatch")
			}
		}
	}
	encoded, ok = parsed.Metadata["zerfoo.dsl.context"].(string)
	if !ok || len(encoded) > 1<<20 {
		return nil, nil, fmt.Errorf("dsl: missing or oversized context")
	}
	var metadata map[string]string
	if err := json.Unmarshal([]byte(encoded), &metadata); err != nil {
		return nil, nil, err
	}
	values, err := gguf.LoadTensors(parsed, bytes.NewReader(raw))
	if err != nil {
		return nil, nil, err
	}
	for _, value := range values {
		for _, v := range value.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, nil, fmt.Errorf("dsl: nonfinite tensor")
			}
		}
	}
	executable, err := Compile(ctx, validated.Definition, engine, 0)
	if err != nil {
		return nil, nil, err
	}
	for _, p := range executable.params {
		p.Value.SetData(slices.Clone(values[p.Name].Data()))
	}
	return executable, metadata, nil
}
