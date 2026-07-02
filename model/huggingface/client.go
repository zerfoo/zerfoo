package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

const defaultBaseURL = "https://huggingface.co"

// Client is a HuggingFace API client.
type Client struct {
	httpClient *http.Client
	baseURL    string
	token      string
}

// NewClient creates a new client. Reads HF_TOKEN from environment.
func NewClient() *Client {
	return &Client{
		httpClient: http.DefaultClient,
		baseURL:    defaultBaseURL,
		token:      os.Getenv("HF_TOKEN"),
	}
}

// ModelInfo holds metadata about a HuggingFace model repository.
type ModelInfo struct {
	ID    string
	Files []FileInfo
}

// FileInfo holds metadata about a single file in a repository.
type FileInfo struct {
	Filename string
	Size     int64
}

// apiResponse matches the JSON shape returned by the HuggingFace API.
type apiResponse struct {
	ID       string       `json:"id"`
	Siblings []apiSibling `json:"siblings"`
}

type apiSibling struct {
	RFilename string `json:"rfilename"`
	Size      int64  `json:"size"`
}

// GetModel fetches metadata for a model repository.
func (c *Client) GetModel(id string) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/api/models/%s", c.baseURL, id)

	req, err := http.NewRequestWithContext(context.Background(), "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("huggingface: create request: %w", err)
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface: unexpected status %d for model %q", resp.StatusCode, id)
	}

	var ar apiResponse
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		return nil, fmt.Errorf("huggingface: decode response: %w", err)
	}

	info := &ModelInfo{ID: ar.ID}
	for _, s := range ar.Siblings {
		info.Files = append(info.Files, FileInfo{
			Filename: s.RFilename,
			Size:     s.Size,
		})
	}
	return info, nil
}

// ListGGUFFiles returns all .gguf files in a repository.
func (c *Client) ListGGUFFiles(id string) ([]FileInfo, error) {
	info, err := c.GetModel(id)
	if err != nil {
		return nil, err
	}

	var gguf []FileInfo
	for _, f := range info.Files {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			gguf = append(gguf, f)
		}
	}
	return gguf, nil
}

// ResolveGGUF finds the best GGUF file for the requested quantization.
// quant examples: "Q4_K_M", "Q8_0", "F16". Case-insensitive.
// Returns the first matching file, preferring exact match then prefix match.
// Returns error if no match found.
func (c *Client) ResolveGGUF(id string, quant string) (*FileInfo, error) {
	files, err := c.ListGGUFFiles(id)
	if err != nil {
		return nil, err
	}

	upper := strings.ToUpper(quant)

	// First pass: exact match (filename contains the quant string as a distinct segment).
	for _, f := range files {
		name := strings.ToUpper(f.Filename)
		if strings.Contains(name, upper+".GGUF") || strings.Contains(name, upper+"-") {
			return &f, nil
		}
	}

	// Second pass: prefix match (quant appears anywhere in filename).
	for _, f := range files {
		name := strings.ToUpper(f.Filename)
		if strings.Contains(name, upper) {
			return &f, nil
		}
	}

	return nil, fmt.Errorf("huggingface: no GGUF file matching quantization %q in %q", quant, id)
}
