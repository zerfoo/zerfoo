// Package model provides the abstraction layer for representing, loading,
// serializing, and managing neural network models in the Zerfoo framework. (Stability: stable)
//
// # Model Representation
//
// The core [Model] struct pairs a token embedding layer with a computation
// graph. Call [NewModel] to create one, then invoke its Forward method to run
// the embedding lookup followed by a full graph evaluation.
//
// The [ModelInstance] interface generalises this to any implementation that
// supports forward inference, backpropagation, parameter access, and
// training/inference mode toggling. [StandardModelInstance] adapts [Model] to
// this interface and is the default implementation used throughout Zerfoo.
//
// # Provider and Registry
//
// [ModelProvider] is the factory interface for creating model instances.
// [StandardModelProvider] is the built-in implementation that creates models
// from an existing computation graph.
//
// [ModelRegistry] is a thread-safe, generic registry that stores factory
// functions for every pluggable component kind: providers, serializers,
// loaders, exporters, validators, and optimizers. Pre-instantiated registries
// for common numeric types are available as [Float32ModelRegistry] and
// [Float64ModelRegistry]. Components are registered by name and retrieved via
// their corresponding Get* methods (e.g. [ModelRegistry.GetModelProvider]).
//
// # Layer Builder Registry
//
// [RegisterLayer] and [GetLayerBuilder] manage a global map from op-type
// strings to [LayerBuilder] functions. During GGUF model loading (see package
// [github.com/zerfoo/zerfoo/inference]), the loader looks up each operation
// by its op_type and calls the corresponding builder to reconstruct the graph
// node with the correct parameters and attributes.
//
// # Parameter Resolution
//
// [ParamResolver] maps architecture-specific weight names to canonical names
// so that the same graph-building code works across architectures. Call
// [NewParamResolver] with an architecture string (e.g. "phi") to obtain the
// appropriate resolver, then use [ResolveAll] to produce a parameter map that
// supports lookup by both original and canonical names.
//
// # Serialization and Export
//
// [ModelSerializer], [ModelLoader], and [ModelExporter] define generic
// interfaces for model persistence. The [Exporter] interface provides a
// simpler single-method contract for writing a [Model] to a file path.
//
// # Validation and Optimization
//
// [ModelValidator] checks model correctness including graph consistency,
// parameter integrity, and input shape compatibility. [BasicModelValidator] is
// the default implementation. [ModelOptimizer] applies performance or memory
// optimizations to a model instance.
//
// # Memory-Mapped File Access
//
// [MmapReader] memory-maps a model file for zero-copy access to its contents,
// used during GGUF loading to avoid buffering large weight tensors into heap
// memory.
//
// # Integration
//
// Models built by this package are consumed by the inference pipeline
// (package [github.com/zerfoo/zerfoo/inference]) which loads GGUF files and
// constructs architecture-specific graphs, and by the text generation layer
// (package [github.com/zerfoo/zerfoo/generate]) which drives autoregressive
// token generation over a model's forward pass.
// Stability: stable
package model
