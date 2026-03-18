// Package disaggpb defines the gRPC service contracts for disaggregated
// prefill/decode serving. A PrefillWorker runs the prompt-encoding phase and
// streams KV blocks back, while a DecodeWorker consumes those blocks and
// streams generated tokens.
// Stability: alpha
package disaggpb
