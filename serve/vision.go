package serve

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/zerfoo/zerfoo/inference/multimodal"
)

// ContentPart represents a single element in a multi-part content array.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds the URL and optional detail level for an image content part.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// UnmarshalJSON handles both string and array content formats for ChatMessage.
// The OpenAI API allows content to be either a plain string or an array of
// content parts (for vision requests with type:"text" and type:"image_url").
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	// Try the simple string-content format first.
	type simple struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	var s simple
	if err := json.Unmarshal(data, &s); err == nil && s.Content != "" {
		m.Role = s.Role
		m.Content = s.Content
		return nil
	}

	// Try the content array format.
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	m.Role = raw.Role

	// If content is a string (possibly empty), use it directly.
	var str string
	if err := json.Unmarshal(raw.Content, &str); err == nil {
		m.Content = str
		return nil
	}

	// Parse as content parts array.
	var parts []ContentPart
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("content must be a string or array of content parts: %w", err)
	}

	var textParts []string
	for _, p := range parts {
		switch p.Type {
		case "text":
			textParts = append(textParts, p.Text)
		case "image_url":
			if p.ImageURL != nil {
				m.ImageURLs = append(m.ImageURLs, *p.ImageURL)
			}
		}
	}
	m.Content = strings.Join(textParts, "")
	return nil
}

// maxImageSize is the maximum allowed size for downloaded images (20 MB).
const maxImageSize = 20 * 1024 * 1024

// fetchImages downloads or decodes images from the given URLs.
// Supports both HTTP(S) URLs and base64 data URIs.
func fetchImages(ctx context.Context, urls []ImageURL) ([][]byte, error) {
	images := make([][]byte, len(urls))
	for i, u := range urls {
		data, err := fetchImageData(ctx, u.URL)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		images[i] = data
	}
	return images, nil
}

// fetchImageData retrieves raw image bytes from a URL or data URI.
func fetchImageData(ctx context.Context, rawURL string) ([]byte, error) {
	if strings.HasPrefix(rawURL, "data:") {
		return decodeDataURI(rawURL)
	}
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return downloadImage(ctx, rawURL)
	}
	return nil, fmt.Errorf("unsupported image URL scheme: %s", rawURL)
}

// decodeDataURI parses a data URI of the form data:<mediatype>;base64,<data>.
func decodeDataURI(uri string) ([]byte, error) {
	idx := strings.Index(uri, ",")
	if idx < 0 {
		return nil, fmt.Errorf("invalid data URI: missing comma separator")
	}
	header := uri[:idx]
	if !strings.Contains(header, ";base64") {
		return nil, fmt.Errorf("unsupported data URI encoding (only base64 is supported)")
	}
	encoded := uri[idx+1:]
	return base64.StdEncoding.DecodeString(encoded)
}

// downloadImage fetches an image from an HTTP(S) URL.
func downloadImage(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download returned status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageSize+1))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	if len(data) > maxImageSize {
		return nil, fmt.Errorf("image exceeds maximum size of %d bytes", maxImageSize)
	}
	return data, nil
}

// detectImageFormat guesses the image format from the first bytes.
func detectImageFormat(data []byte) multimodal.ImageFormat {
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return multimodal.JPEG
	}
	if len(data) >= 4 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G' {
		return multimodal.PNG
	}
	return multimodal.JPEG // default fallback
}
