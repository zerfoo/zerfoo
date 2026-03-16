package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// DownloadOption configures the behavior of Download.
type DownloadOption func(*downloadOptions)

type downloadOptions struct {
	progress func(downloaded, total int64)
	sha256   string
}

// WithProgress sets a callback that is invoked periodically with byte counts.
func WithProgress(fn func(downloaded, total int64)) DownloadOption {
	return func(o *downloadOptions) { o.progress = fn }
}

// WithSHA256 sets the expected SHA256 hex digest for post-download verification.
func WithSHA256(hexDigest string) DownloadOption {
	return func(o *downloadOptions) { o.sha256 = hexDigest }
}

// Downloader handles GGUF file downloads with resume support.
type Downloader struct {
	httpClient *http.Client
}

// NewDownloader creates a Downloader using the given HTTP client.
func NewDownloader(httpClient *http.Client) *Downloader {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Downloader{httpClient: httpClient}
}

// Download downloads a file from url to destPath.
// It supports resume via HTTP Range if a partial file exists at destPath.partial.
// The file is written with a .partial suffix during download and renamed on completion.
func (d *Downloader) Download(ctx context.Context, url, destPath string, opts ...DownloadOption) error {
	var o downloadOptions
	for _, fn := range opts {
		fn(&o)
	}

	if err := os.MkdirAll(filepath.Dir(destPath), 0o750); err != nil {
		return fmt.Errorf("huggingface: create directory: %w", err)
	}

	partialPath := destPath + ".partial"

	// Determine resume offset from existing partial file.
	var offset int64
	if info, err := os.Stat(partialPath); err == nil {
		offset = info.Size()
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("huggingface: create request: %w", err)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}

	resp, err := d.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("huggingface: request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignores Range or fresh download — start from scratch.
		offset = 0
	case http.StatusPartialContent:
		// Resume accepted.
	default:
		return fmt.Errorf("huggingface: unexpected status %d", resp.StatusCode)
	}

	// Open file: append if resuming, create if fresh.
	flag := os.O_WRONLY | os.O_CREATE
	if offset > 0 {
		flag |= os.O_APPEND
	} else {
		flag |= os.O_TRUNC
	}
	f, err := os.OpenFile(partialPath, flag, 0o600) //nolint:gosec // path constructed from destPath
	if err != nil {
		return fmt.Errorf("huggingface: open file: %w", err)
	}

	// Total size: for full response use Content-Length, for partial add offset.
	total := resp.ContentLength
	if total > 0 && offset > 0 {
		total += offset
	}

	var reader io.Reader = resp.Body
	if o.progress != nil {
		reader = &progressReader{
			r:          resp.Body,
			downloaded: offset,
			total:      total,
			callback:   o.progress,
		}
	}

	_, copyErr := io.Copy(f, reader)
	closeErr := f.Close()
	if copyErr != nil {
		return fmt.Errorf("huggingface: download: %w", copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("huggingface: close file: %w", closeErr)
	}

	// SHA256 verification.
	if o.sha256 != "" {
		digest, err := fileSHA256(partialPath)
		if err != nil {
			return fmt.Errorf("huggingface: sha256: %w", err)
		}
		if digest != o.sha256 {
			return fmt.Errorf("huggingface: sha256 mismatch: got %s, want %s", digest, o.sha256)
		}
	}

	// Rename to final path.
	if err := os.Rename(partialPath, destPath); err != nil {
		return fmt.Errorf("huggingface: rename: %w", err)
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path) //nolint:gosec // path from controlled cache directory
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// progressReader wraps an io.Reader and reports progress via a callback.
type progressReader struct {
	r          io.Reader
	downloaded int64
	total      int64
	callback   func(downloaded, total int64)
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.r.Read(p)
	pr.downloaded += int64(n)
	if n > 0 {
		pr.callback(pr.downloaded, pr.total)
	}
	return n, err
}

// FormatProgress returns a human-readable progress string suitable for stderr.
func FormatProgress(filename string, downloaded, total int64) string {
	dl := formatBytes(downloaded)
	if total > 0 {
		tl := formatBytes(total)
		pct := float64(downloaded) / float64(total) * 100
		return fmt.Sprintf("\rDownloading %s  %s / %s (%.1f%%)", filename, dl, tl, pct)
	}
	return fmt.Sprintf("\rDownloading %s  %s", filename, dl)
}

func formatBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/float64(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/float64(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(b)/float64(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}
