// Command embedding demonstrates embedding Zerfoo inference inside a Go HTTP handler.
//
// Usage:
//
//	go build -o embedding ./examples/embedding/
//	./embedding path/to/model.gguf
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"

	"github.com/zerfoo/zerfoo/inference"
)

func main() {
	port := flag.String("port", "8080", "listen port")
	device := flag.String("device", "cpu", `compute device: "cpu", "cuda", "cuda:0", "rocm"`)
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")

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

	// Load the model once at startup.
	model, err := inference.LoadFile(modelPath, inference.WithDevice(*device))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "Loaded %s model (%d layers, vocab %d)\n",
		cfg.Architecture, cfg.NumLayers, cfg.VocabSize)

	// Register an HTTP handler that uses the loaded model for inference.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /generate", handleGenerate(model, *maxTokens))
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	addr := net.JoinHostPort("", *port)
	fmt.Fprintf(os.Stderr, "Listening on http://localhost:%s\n", *port)

	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
		os.Exit(1)
	}
}

type generateRequest struct {
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
}

type generateResponse struct {
	Text string `json:"text"`
}

func handleGenerate(model *inference.Model, defaultMaxTokens int) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req generateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
			return
		}
		if req.Prompt == "" {
			http.Error(w, `{"error":"prompt is required"}`, http.StatusBadRequest)
			return
		}

		var opts []inference.GenerateOption
		if req.MaxTokens != nil {
			opts = append(opts, inference.WithMaxTokens(*req.MaxTokens))
		} else {
			opts = append(opts, inference.WithMaxTokens(defaultMaxTokens))
		}
		if req.Temperature != nil {
			opts = append(opts, inference.WithTemperature(*req.Temperature))
		}

		result, err := model.Generate(context.Background(), req.Prompt, opts...)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(generateResponse{Text: result})
	}
}
