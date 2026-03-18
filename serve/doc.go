// Package serve provides an OpenAI-compatible HTTP API server for model inference. (Stability: stable)
//
// The server exposes REST endpoints that follow the OpenAI API specification,
// enabling drop-in compatibility with existing OpenAI client libraries and tools.
//
// # Creating a Server
//
// Use [NewServer] to create a server for a loaded model:
//
//	m, _ := inference.LoadGGUF("model.gguf")
//	srv := serve.NewServer(m,
//		serve.WithLogger(logger),
//		serve.WithMetrics(collector),
//	)
//	http.ListenAndServe(":8080", srv.Handler())
//
// The returned [Server] is configured with functional options:
//
//   - [WithLogger] sets structured request logging.
//   - [WithMetrics] enables Prometheus-compatible metrics collection.
//   - [WithDraftModel] enables speculative decoding with a smaller draft model.
//   - [WithBatchScheduler] routes non-streaming requests through a [BatchScheduler]
//     for higher throughput.
//
// # Endpoints
//
// The server registers the following HTTP routes:
//
//	POST /v1/chat/completions   Chat completion (streaming and non-streaming)
//	POST /v1/completions        Text completion (streaming and non-streaming)
//	POST /v1/embeddings         Text embeddings
//	GET  /v1/models             List loaded models
//	GET  /v1/models/{id}        Get model info
//	DELETE /v1/models/{id}      Unload a model
//	GET  /openapi.yaml          OpenAPI specification
//	GET  /metrics               Prometheus metrics
//
// # SSE Streaming
//
// When a chat or text completion request sets "stream": true, the server responds
// with Server-Sent Events (SSE). Each event contains a JSON chunk with incremental
// tokens. The stream terminates with a "data: [DONE]" sentinel.
//
// # Tool Calling
//
// Chat completion requests may include OpenAI-compatible tool definitions. The server
// validates tool schemas, detects tool calls in model output via [DetectToolCall], and
// returns structured tool_calls in the response. Tool choice can be set to "auto",
// "none", or forced to a specific function.
//
// # Structured Output
//
// Chat completions support response_format with "json_schema" type for
// grammar-constrained decoding, ensuring model output conforms to a provided
// JSON Schema.
//
// # Batch Scheduling
//
// A [BatchScheduler] groups incoming non-streaming requests into batches for
// efficient GPU utilization. Create one with [NewBatchScheduler], configure it
// with a [BatchConfig], and pass it to [WithBatchScheduler]:
//
//	bs := serve.NewBatchScheduler(serve.BatchConfig{
//		MaxBatchSize: 8,
//		BatchTimeout: 10 * time.Millisecond,
//	})
//	bs.Start()
//	defer bs.Stop()
//	srv := serve.NewServer(m, serve.WithBatchScheduler(bs))
//
// # Metrics
//
// The GET /metrics endpoint exposes Prometheus text exposition format metrics:
//
//   - requests_total: total number of completed requests (counter)
//   - tokens_generated_total: total tokens generated (counter)
//   - tokens_per_second: rolling token generation rate (gauge)
//   - request_latency_ms: request latency histogram with configurable buckets
//
// Metrics are collected through the runtime.Collector interface passed via
// [WithMetrics].
//
// # Graceful Shutdown
//
// Call [Server.Close] to gracefully stop the server, which drains the batch
// scheduler if one is attached.
// Stability: stable
package serve
