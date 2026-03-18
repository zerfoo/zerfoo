// Recipe 04: OpenAI-Compatible Server
//
// Serve a GGUF model behind an OpenAI-compatible HTTP API. Clients that work
// with the OpenAI API (curl, Python openai library, LangChain, etc.) can
// connect directly -- just point them at http://localhost:8080.
//
// Endpoints:
//   - POST /v1/chat/completions   (chat)
//   - POST /v1/completions        (text completion)
//   - POST /v1/embeddings         (embeddings)
//   - GET  /v1/models             (model listing)
//   - GET  /health                (health check)
//
// Usage:
//
//	go run ./docs/cookbook/04-openai-server/ --model path/to/model.gguf
//	curl http://localhost:8080/v1/chat/completions -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
)

func main() {
	modelPath := flag.String("model", "", "path to GGUF model file")
	port := flag.String("port", "8080", "listen port")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0"`)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: openai-server --model <model.gguf> [--port 8080] [--device cpu]")
		os.Exit(1)
	}

	// Load the model.
	model, err := inference.LoadFile(*modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Create the OpenAI-compatible server and wire up HTTP.
	srv := serve.NewServer(model)
	httpServer := &http.Server{
		Addr:    net.JoinHostPort("", *port),
		Handler: srv.Handler(),
	}

	// Graceful shutdown on SIGINT / SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() { errCh <- httpServer.ListenAndServe() }()

	fmt.Fprintf(os.Stderr, "Serving on http://localhost:%s\n", *port)

	select {
	case <-ctx.Done():
		fmt.Fprintln(os.Stderr, "\nShutting down...")
		httpServer.Shutdown(context.Background())
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "server: %v\n", err)
			os.Exit(1)
		}
	}
}
