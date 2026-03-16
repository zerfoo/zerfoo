// Command api-server demonstrates starting an OpenAI-compatible inference server.
//
// Usage:
//
//	go build -o api-server ./examples/api-server/
//	./api-server path/to/model.gguf
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
	port := flag.String("port", "8080", "listen port")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0", "rocm"`)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <model.gguf>\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	modelPath := flag.Arg(0)

	// Load the GGUF model file.
	model, err := inference.LoadFile(modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s model (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Create the OpenAI-compatible server.
	srv := serve.NewServer(model)
	httpServer := &http.Server{
		Addr:    net.JoinHostPort("", *port),
		Handler: srv.Handler(),
	}

	// Graceful shutdown on SIGINT/SIGTERM.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	fmt.Fprintf(os.Stderr, "Serving on http://localhost:%s\n", *port)

	select {
	case <-ctx.Done():
		fmt.Fprintln(os.Stderr, "\nShutting down...")
		if err := httpServer.Shutdown(context.Background()); err != nil {
			fmt.Fprintf(os.Stderr, "Shutdown error: %v\n", err)
		}
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
			os.Exit(1)
		}
	}
}
