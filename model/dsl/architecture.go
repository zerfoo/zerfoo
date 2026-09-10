package dsl

import (
	"encoding/hex"
	"fmt"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/model/gguf"
)

// ArchitectureReference selects an existing architecture builder and binds its
// complete configuration to an exact canonical parameter set. It does not
// assert that the architecture supports training from scratch.
type ArchitectureReference struct {
	Version          int              `json:"version"`
	Architecture     string           `json:"architecture"`
	ComponentVersion int              `json:"component_version"`
	ParametersSHA256 string           `json:"parameters_sha256"`
	Config           gguf.ModelConfig `json:"config"`
}

func (r ArchitectureReference) Validate() error {
	if r.Version != 1 || r.ComponentVersion != 1 || r.Architecture != r.Config.Architecture {
		return fmt.Errorf("dsl: incompatible architecture reference")
	}
	if _, _, ok := model.Component("architecture", r.Architecture); !ok {
		return invalid("unsupported_architecture", "architecture", r.Architecture)
	}
	raw, err := hex.DecodeString(r.ParametersSHA256)
	if err != nil || len(raw) != 32 {
		return fmt.Errorf("dsl: exact parameter-set identity required")
	}
	return nil
}
