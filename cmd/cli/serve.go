package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/serve"
	"github.com/zerfoo/zerfoo/serve/shutdown"
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
	var modelID, cacheDir, port, gpusRaw, apiKey, tlsCert, tlsKey string
	var pjrtPlugin string
	var allowNoAuth bool

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--allow-no-auth":
			allowNoAuth = true
			continue
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
		case "--gpus":
			if i+1 >= len(args) {
				return errors.New("--gpus requires a value")
			}
			gpusRaw = args[i+1]
			i++
		case "--api-key":
			if i+1 >= len(args) {
				return errors.New("--api-key requires a value")
			}
			apiKey = args[i+1]
			i++
		case "--tls-cert":
			if i+1 >= len(args) {
				return errors.New("--tls-cert requires a value")
			}
			tlsCert = args[i+1]
			i++
		case "--tls-key":
			if i+1 >= len(args) {
				return errors.New("--tls-key requires a value")
			}
			tlsKey = args[i+1]
			i++
		case "--pjrt":
			if i+1 >= len(args) {
				return errors.New("--pjrt requires a value")
			}
			pjrtPlugin = args[i+1]
			i++
		default:
			if modelID != "" {
				return fmt.Errorf("unexpected argument: %s", args[i])
			}
			modelID = args[i]
		}
	}

	// Fall back to ZERFOO_API_KEY env var when --api-key is not provided.
	if apiKey == "" {
		apiKey = os.Getenv("ZERFOO_API_KEY")
	}

	if modelID == "" {
		return errors.New("model ID is required")
	}
	if port == "" {
		port = "8080"
	}

	if (tlsCert != "") != (tlsKey != "") {
		return errors.New("both --tls-cert and --tls-key are required")
	}

	var loadOpts []inference.Option
	if cacheDir != "" {
		loadOpts = append(loadOpts, inference.WithCacheDir(cacheDir))
	}
	if pjrtPlugin != "" {
		loadOpts = append(loadOpts, inference.WithPJRT(pjrtPlugin))
	}

	var gpuIDs []int
	if gpusRaw != "" {
		var err error
		gpuIDs, err = parseGPUList(gpusRaw)
		if err != nil {
			return fmt.Errorf("invalid --gpus value: %w", err)
		}
	}

	// Require explicit opt-in when no API key is configured.
	if apiKey == "" && !allowNoAuth {
		return errors.New("set --api-key, ZERFOO_API_KEY, or --allow-no-auth")
	}
	if apiKey == "" && allowNoAuth {
		_, _ = fmt.Fprintf(c.out, "WARN: serve: no API key configured, all endpoints are public\n")
	}

	li := startLoading(c.out)
	mdl, err := c.loadFn(modelID, loadOpts...)
	li.stop()
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	var serverOpts []serve.ServerOption
	if len(gpuIDs) > 0 {
		serverOpts = append(serverOpts, serve.WithGPUs(gpuIDs))
	}
	if apiKey != "" {
		serverOpts = append(serverOpts, serve.WithAPIKey(apiKey))
	}
	srv := serve.NewServer(mdl, serverOpts...)
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
		if tlsCert != "" && tlsKey != "" {
			errCh <- httpServer.ListenAndServeTLS(tlsCert, tlsKey)
		} else {
			errCh <- httpServer.ListenAndServe()
		}
	}()

	select {
	case <-ctx.Done():
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutCancel()
		if err := httpServer.Shutdown(shutCtx); err != nil {
			_, _ = fmt.Fprintf(c.out, "WARN: serve: graceful shutdown timed out after 30s: %v\n", err)
			return err
		}
		return nil
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

// parseGPUList parses a comma-separated list of GPU IDs (e.g. "0,1,2,3")
// and returns them as a sorted, deduplicated slice of non-negative integers.
func parseGPUList(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	seen := make(map[int]bool, len(parts))
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return nil, fmt.Errorf("empty GPU ID in list")
		}
		id, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("non-numeric GPU ID: %q", p)
		}
		if id < 0 {
			return nil, fmt.Errorf("negative GPU ID: %d", id)
		}
		if seen[id] {
			return nil, fmt.Errorf("duplicate GPU ID: %d", id)
		}
		seen[id] = true
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("empty GPU list")
	}
	return ids, nil
}

// Usage implements Command.Usage.
func (c *ServeCommand) Usage() string {
	return `serve [OPTIONS] <model-id>

Start an OpenAI-compatible HTTP inference server.

OPTIONS:
  --port <port>       Listen port (default: 8080)
  --cache-dir <dir>   Override model cache directory
  --gpus <ids>        Comma-separated GPU IDs to distribute model across (e.g. 0,1,2,3)
  --api-key <key>     Require Bearer token auth (env: ZERFOO_API_KEY)
  --allow-no-auth     Allow starting without an API key (public endpoints)
  --tls-cert <path>   Path to TLS certificate file (requires --tls-key)
  --tls-key <path>    Path to TLS private key file (requires --tls-cert)
  --pjrt <path>       Path to PJRT plugin .so for accelerator backend

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
		"serve google/gemma-3-1b --gpus 0,1,2,3",
		"serve google/gemma-3-1b --pjrt /usr/lib/pjrt_cpu.so",
	}
}

// Static interface assertion.
var _ Command = (*ServeCommand)(nil)
