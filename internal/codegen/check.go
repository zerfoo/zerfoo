package codegen

import (
	"github.com/zerfoo/ztensor/graph"
)

// CheckSupport verifies that all instructions in the tape have emitters.
// Returns the list of unsupported op names (empty if all are supported).
func CheckSupport(instructions []graph.InstructionMeta) []string {
	var unsupported []string
	seen := make(map[string]bool)
	for _, inst := range instructions {
		if !Supported(inst.OpName) && !seen[inst.OpName] {
			unsupported = append(unsupported, inst.OpName)
			seen[inst.OpName] = true
		}
	}
	return unsupported
}
