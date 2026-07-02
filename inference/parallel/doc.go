// Package parallel provides tensor and pipeline parallelism for distributing
// inference across multiple GPUs. Tensor parallelism splits individual linear
// layers across devices and synchronises with AllReduce; pipeline parallelism
// assigns sequential layer groups to different devices.
// Stability: alpha
package parallel
