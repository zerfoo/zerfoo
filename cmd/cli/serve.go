package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
	"github.com/zerfoo/zerfoo/shutdown"
)

// ServeCommand implements the "serve" CLI command for starting
// an OpenAI-compatible HTTP inference server.
type ServeCommand struct {
	out           io.Writer
	shutdownCoord *shutdown.Coordinator
	// loadFn allows injection of a custom model loader for testing.
	loadFn func(modelID string, opts ...inference.Option) (*inference.Model, error)
}

// NewServeCommand creates a new ServeCommand.
func NewServeCommand(coord *shutdown.Coordinator, out io.Writer) *ServeCommand {
	return &ServeCommand{
		out:           out,
		shutdownCoord: coord,
		loadFn:        inference.Load,
	}
}

// Name implements Command.Name.
func (c *ServeCommand) Name() string { return "serve" }

// Description implements Command.Description.
func (c *ServeCommand) Description() string {
	return "Start an OpenAI-compatible inference server"
}

// Run implements Command.Run.
func (c *ServeCommand) Run(ctx context.Context, args []string) error {
	var modelID, cacheDir, port string

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--port":
			if i+1 >= len(args) {
				return errors.New("--port requires a value")
			}
			port = args[i+1]
			i++
		case "--cache-dir":
			if i+1 >= len(args) {
				return errors.New("--cache-dir requires a value")
			}
			cacheDir = args[i+1]
			i++
		default:
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			modelID = args[i]
		}
	}

	if modelID == "" {
		return errors.New("model ID is required")
	}
	if port == "" {
		port = "8080"
	}

	var loadOpts []inference.Option
	if cacheDir != "" {
		loadOpts = append(loadOpts, inference.WithCacheDir(cacheDir))
	}

	li := startLoading(c.out)
	mdl, err := c.loadFn(modelID, loadOpts...)
	li.stop()
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	srv := serve.NewServer(mdl)
	httpServer := &http.Server{
		Addr:    net.JoinHostPort("", port),
		Handler: srv.Handler(),
	}

	if c.shutdownCoord != nil {
		c.shutdownCoord.Register(shutdownAdapter{httpServer})
	}

	_, _ = fmt.Fprintf(c.out, "Serving %s on :%s\n", modelID, port)

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		return httpServer.Shutdown(context.Background())
	case err := <-errCh:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	}
}

// shutdownAdapter adapts *http.Server to the shutdown.Closer interface.
type shutdownAdapter struct {
	srv *http.Server
}

func (a shutdownAdapter) Close(ctx context.Context) error {
	return a.srv.Shutdown(ctx)
}

// Usage implements Command.Usage.
func (c *ServeCommand) Usage() string {
	return `serve [OPTIONS] <model-id>

Start an OpenAI-compatible HTTP inference server.

OPTIONS:
  --port <port>       Listen port (default: 8080)
  --cache-dir <dir>   Override model cache directory

ENDPOINTS:
  POST /v1/chat/completions   Chat completion
  POST /v1/completions        Text completion
  GET  /v1/models             Model info`
}

// Examples implements Command.Examples.
func (c *ServeCommand) Examples() []string {
	return []string{
		"serve google/gemma-3-1b",
		"serve google/gemma-3-1b --port 9090",
	}
}

// Static interface assertion.
var _ Command = (*ServeCommand)(nil)
