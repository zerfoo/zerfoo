// Package training provides neural network training orchestration for the
// Zerfoo ML framework. (Stability: beta)
//
// The package implements a layered design: a core [Trainer] interface for
// single-step parameter updates, a [DefaultTrainer] that wires together a
// computation graph, a loss node, an optimizer, and a pluggable gradient
// strategy, and a higher-level [TrainingWorkflow] interface for full
// training-loop orchestration with data providers and model providers.
//
// # Trainer and DefaultTrainer
//
// [Trainer] is the fundamental training interface. It performs one training
// step: forward pass, loss computation, backward pass, and parameter update.
//
//	trainer := training.NewDefaultTrainer[float32](g, lossNode, opt, nil)
//	loss, err := trainer.TrainStep(ctx, g, opt, inputs, targets)
//
// [DefaultTrainer] is the standard implementation. It delegates gradient
// computation to a [GradientStrategy] and parameter updates to an
// [optimizer.Optimizer]. When no strategy is provided, it defaults to
// [DefaultBackpropStrategy].
//
// # Gradient Strategies
//
// [GradientStrategy] controls how gradients are computed for each training
// step. Two strategies are provided:
//
//   - [DefaultBackpropStrategy] performs standard backpropagation through the
//     full computation graph.
//   - [OneStepApproximationStrategy] performs a single-step gradient
//     approximation, useful for recurrent models where full BPTT is too
//     expensive.
//
// Custom strategies can implement the [GradientStrategy] interface to add
// auxiliary losses, gradient clipping, deep supervision, or other
// specialized gradient computation techniques.
//
// # Optimizers
//
// The optimizer sub-package defines the [optimizer.Optimizer] interface and
// provides several implementations:
//
//   - AdamW — Adam with decoupled weight decay.
//   - SGD — Stochastic gradient descent with optional momentum.
//   - EMA — Exponential moving average of model parameters.
//   - SWA — Stochastic weight averaging.
//
// # Batch and Data Iteration
//
// [Batch] groups inputs and targets for a single training step. Inputs are
// provided as a map from graph input nodes to tensors; targets are a single
// tensor.
//
// [DataIterator] provides sequential access to batches.
// [ChunkedDataIterator] loads batches in chunks via a callback, keeping only
// one chunk in memory at a time for large datasets. [DataIteratorAdapter]
// wraps a static slice of batches as a [DataIterator].
//
// # Model Interface
//
// [Model] defines a trainable model with Forward, Backward, and Parameters
// methods. This is the low-level model interface used by training
// components that need direct forward/backward control.
//
// # Training Workflow and Plugin Registry
//
// For full training-loop orchestration, [TrainingWorkflow] combines data
// providers, model providers, metrics, and the training loop into a single
// interface. [TrainerWorkflowAdapter] bridges the core [Trainer] interface
// to [TrainingWorkflow] for use with the plugin system.
//
// [PluginRegistry] enables runtime registration and lookup of workflows,
// data providers, model providers, sequence providers, metric computers,
// and cross validators. Global registries [Float32Registry] and
// [Float64Registry] are provided for common numeric types.
//
// See interfaces_doc.go for detailed documentation of the plugin
// architecture and workflow interfaces.
// Stability: beta
package training
