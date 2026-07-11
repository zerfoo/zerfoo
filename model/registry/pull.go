package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	defaultHFAPIURL = "https://huggingface.co/api/models"
	defaultHFCDNURL = "https://huggingface.co"
)

// ProgressFunc reports download progress. total may be -1 if unknown.
type ProgressFunc func(downloaded, total int64)

// HFPullOptions configures HuggingFace model downloads.
type HFPullOptions struct {
	// APIURL overrides the HuggingFace API endpoint.
	APIURL string
	// CDNURL overrides the HuggingFace CDN endpoint.
	CDNURL string
	// Token is an optional HuggingFace API token for gated models.
	Token string
	// Quant selects a specific GGUF quantization (e.g. "Q4_K_M", "Q8_0").
	// When set, only the matching GGUF file is downloaded instead of all model files.
	// Default: "Q4_K_M".
	Quant string
	// OnProgress is called during file downloads.
	OnProgress ProgressFunc
	// Client overrides the HTTP client used for downloads.
	Client *http.Client
	// ExpectedHashes optionally pins the expected SHA-256 checksum (lowercase
	// hex) for specific files, keyed by the repository-relative filename
	// (e.g. "model-Q4_K_M.gguf"). When a filename has an entry here, it is
	// verified against this out-of-band value INSTEAD of any hash derived
	// from the download response's own ETag/X-Linked-Etag/Content-Sha256
	// headers -- trusting the serving origin's own headers as the "expected"
	// hash provides no integrity guarantee against a compromised or MITM'd
	// server (HF-1). A mismatch is a hard error. Files not present in this
	// map fall back to the prior trust-on-first-download ETag behavior for
	// backward compatibility.
	ExpectedHashes map[string]string
}

// HFSibling represents a file entry in a HuggingFace model listing.
type HFSibling struct {
	Filename string `json:"rfilename"`
}

// HFModelInfo represents the model metadata returned by the HuggingFace API.
type HFModelInfo struct {
	ID       string      `json:"id"`
	Siblings []HFSibling `json:"siblings"`
}

// NewHFPullFunc creates a PullFunc that downloads models from HuggingFace Hub.
func NewHFPullFunc(opts HFPullOptions) PullFunc {
	if opts.APIURL == "" {
		opts.APIURL = getEnvOr("HUGGINGFACE_API_URL", defaultHFAPIURL)
	}
	if opts.CDNURL == "" {
		opts.CDNURL = getEnvOr("HUGGINGFACE_CDN_URL", defaultHFCDNURL)
	}
	if opts.Token == "" {
		opts.Token = os.Getenv("HF_TOKEN")
	}
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}

	return func(ctx context.Context, modelID string, targetDir string) (*ModelInfo, error) {
		return pullFromHF(ctx, opts, modelID, targetDir)
	}
}

func pullFromHF(ctx context.Context, opts HFPullOptions, modelID string, targetDir string) (*ModelInfo, error) {
	// 1. List model files.
	files, err := listModelFiles(ctx, opts, modelID)
	if err != nil {
		return nil, fmt.Errorf("list files: %w", err)
	}

	// 2. If quant is specified, resolve the matching GGUF file.
	if opts.Quant != "" {
		ggufFile, resolveErr := resolveGGUFByQuant(files, opts.Quant)
		if resolveErr != nil {
			return nil, resolveErr
		}
		size, dlErr := downloadFile(ctx, opts, modelID, ggufFile, targetDir)
		if dlErr != nil {
			return nil, fmt.Errorf("download %s: %w", ggufFile, dlErr)
		}
		return &ModelInfo{
			ID:   modelID,
			Path: targetDir,
			Size: size,
		}, nil
	}

	// 3. Download relevant files (no quant filter).
	var totalSize int64
	for _, f := range files {
		if !shouldDownload(f.Filename) {
			continue
		}
		size, dlErr := downloadFile(ctx, opts, modelID, f.Filename, targetDir)
		if dlErr != nil {
			return nil, fmt.Errorf("download %s: %w", f.Filename, dlErr)
		}
		totalSize += size
	}

	return &ModelInfo{
		ID:   modelID,
		Path: targetDir,
		Size: totalSize,
	}, nil
}

// resolveGGUFByQuant finds the GGUF file matching the requested quantization.
func resolveGGUFByQuant(files []HFSibling, quant string) (string, error) {
	upper := strings.ToUpper(quant)

	// Collect GGUF files.
	var ggufFiles []string
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			ggufFiles = append(ggufFiles, f.Filename)
		}
	}

	if len(ggufFiles) == 0 {
		return "", fmt.Errorf("no GGUF files found in repository")
	}

	// First pass: exact segment match (e.g., "Q4_K_M.gguf" or "Q4_K_M-").
	for _, name := range ggufFiles {
		nameUpper := strings.ToUpper(name)
		if strings.Contains(nameUpper, upper+".GGUF") || strings.Contains(nameUpper, upper+"-") {
			return name, nil
		}
	}

	// Second pass: substring match.
	for _, name := range ggufFiles {
		if strings.Contains(strings.ToUpper(name), upper) {
			return name, nil
		}
	}

	return "", fmt.Errorf("no GGUF file matching quantization %q", quant)
}

// listModelFiles queries the HuggingFace API for model file listing.
func listModelFiles(ctx context.Context, opts HFPullOptions, modelID string) ([]HFSibling, error) {
	url := fmt.Sprintf("%s/%s", opts.APIURL, modelID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	addAuthHeader(req, opts.Token)

	resp, err := opts.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var info HFModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return info.Siblings, nil
}

// downloadFile downloads a single file from HuggingFace CDN using atomic
// writes and optional SHA-256 checksum verification.
func downloadFile(ctx context.Context, opts HFPullOptions, modelID, filename, targetDir string) (int64, error) {
	url := fmt.Sprintf("%s/%s/resolve/main/%s", opts.CDNURL, modelID, filename)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, err
	}
	addAuthHeader(req, opts.Token)

	resp, err := opts.Client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("download returned status %d", resp.StatusCode)
	}

	// Validate filename to prevent path traversal from server-controlled values.
	if strings.Contains(filename, "..") {
		return 0, fmt.Errorf("invalid filename %q: contains path traversal", filename)
	}

	// Create subdirectories if needed (e.g., onnx/model.onnx).
	destPath := filepath.Join(targetDir, filename)
	cleaned := filepath.Clean(destPath)
	targetPrefix := filepath.Clean(targetDir) + string(filepath.Separator)
	if !strings.HasPrefix(cleaned, targetPrefix) && cleaned != filepath.Clean(targetDir) {
		return 0, fmt.Errorf("invalid filename %q: resolves outside target directory", filename)
	}
	if err := os.MkdirAll(filepath.Dir(cleaned), 0o750); err != nil {
		return 0, err
	}

	// Determine the expected SHA-256. An out-of-band pin (opts.ExpectedHashes)
	// takes precedence over anything derived from the response headers, since
	// the headers originate from the same server the content was just
	// fetched from and offer no protection against a compromised or MITM'd
	// origin (HF-1). Absent a pin, fall back to the prior ETag-derived trust
	// behavior for backward compatibility.
	var expectedHash string
	pinned := false
	if h, ok := opts.ExpectedHashes[filename]; ok && h != "" {
		expectedHash = strings.ToLower(h)
		pinned = true
	} else {
		expectedHash = extractSHA256(resp)
	}

	// Atomic write: download to a temp file, then rename on success.
	tmpPath := cleaned + ".tmp"
	f, err := os.Create(tmpPath) //nolint:gosec // path validated above
	if err != nil {
		return 0, err
	}

	// Ensure temp file is cleaned up on any error path.
	success := false
	defer func() {
		f.Close() //nolint:errcheck
		if !success {
			os.Remove(tmpPath) //nolint:errcheck
		}
	}()

	// Set up the reader pipeline: body -> progress -> tee(hash).
	var reader io.Reader = resp.Body
	if opts.OnProgress != nil {
		reader = &progressReader{
			reader:   resp.Body,
			total:    resp.ContentLength,
			callback: opts.OnProgress,
		}
	}

	// Compute SHA-256 while writing via TeeReader.
	hasher := sha256.New()
	reader = io.TeeReader(reader, hasher)

	n, err := io.Copy(f, reader)
	if err != nil {
		return 0, err
	}

	// Verify checksum. A pinned hash is always enforced. Otherwise, verify
	// against the ETag-derived hash if the server provided one.
	gotHash := hex.EncodeToString(hasher.Sum(nil))
	switch {
	case pinned:
		if gotHash != expectedHash {
			return 0, fmt.Errorf("checksum mismatch for %s: expected %s (pinned), got %s", filename, expectedHash, gotHash)
		}
	case expectedHash != "":
		if gotHash != expectedHash {
			return 0, fmt.Errorf("checksum mismatch for %s: expected %s, got %s", filename, expectedHash, gotHash)
		}
	default:
		slog.Warn("no SHA-256 checksum available from server, skipping verification", "file", filename)
	}

	// Atomic rename: temp file -> final path.
	if err := os.Rename(tmpPath, cleaned); err != nil {
		return 0, fmt.Errorf("rename temp file: %w", err)
	}
	success = true

	return n, nil
}

// extractSHA256 extracts a SHA-256 hash from the HTTP response headers.
// It checks X-Linked-Etag, ETag (for hex-encoded SHA-256), and Content-Sha256.
func extractSHA256(resp *http.Response) string {
	// HuggingFace uses X-Linked-Etag or ETag with quoted hex SHA-256.
	for _, header := range []string{"X-Linked-Etag", "ETag", "Content-Sha256"} {
		val := resp.Header.Get(header)
		if val == "" {
			continue
		}
		// Strip quotes from ETag values.
		val = strings.Trim(val, "\"")
		// A valid SHA-256 hex string is exactly 64 characters.
		if len(val) == 64 && isHex(val) {
			return strings.ToLower(val)
		}
	}
	return ""
}

// isHex reports whether s contains only hexadecimal characters.
func isHex(s string) bool {
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

// shouldDownload returns true if the file is relevant for model caching.
func shouldDownload(filename string) bool {
	lower := strings.ToLower(filename)
	switch {
	case strings.HasSuffix(lower, ".gguf"):
		return true
	case strings.HasSuffix(lower, ".onnx"):
		return true
	case strings.Contains(lower, "tokenizer") && (strings.HasSuffix(lower, ".json") || strings.HasSuffix(lower, ".model")):
		return true
	case lower == "config.json" || lower == "generation_config.json":
		return true
	case strings.HasSuffix(lower, ".onnx_data"):
		return true
	default:
		return false
	}
}

// progressReader wraps an io.Reader to report download progress.
type progressReader struct {
	reader     io.Reader
	total      int64
	downloaded int64
	callback   ProgressFunc
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	pr.downloaded += int64(n)
	if pr.callback != nil {
		pr.callback(pr.downloaded, pr.total)
	}
	return n, err
}

func addAuthHeader(req *http.Request, token string) {
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func getEnvOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
