// Package functional provides stateless, tensor-in tensor-out wrappers for
// common neural-network operations.  Unlike the graph-aware layer types in
// sibling packages (e.g. layers/normalization), the functions here carry no
// parameter state, do no graph registration, and have no backward pass — they
// are pure forward-only computations suitable for scripting, testing, and
// composition.
package functional
