package serve

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/zerfoo/zerfoo/inference/multimodal"
)

// blockedSSRFHosts contains hostnames that must be blocked to prevent
// Server-Side Request Forgery attacks against cloud metadata services.
var blockedSSRFHosts = map[string]bool{
	"metadata.google.internal": true,
}

// blockedSSRFIPs contains IP addresses that must be blocked to prevent
// SSRF attacks against cloud metadata services (e.g. AWS/GCP/Azure).
var blockedSSRFIPs = map[string]bool{
	"169.254.169.254": true,
}

// maxImageRedirects is the maximum number of HTTP redirects allowed
// when downloading an image.
const maxImageRedirects = 3

// cgnatBlock is the shared address space reserved for carrier-grade NAT
// (RFC 6598). Requests into this range can reach infrastructure that is not
// meant to be internet-reachable, so it is treated the same as RFC 1918
// private space for SSRF purposes.
var cgnatBlock = func() *net.IPNet {
	_, block, err := net.ParseCIDR("100.64.0.0/10")
	if err != nil {
		panic(fmt.Sprintf("invalid CGNAT CIDR: %v", err))
	}
	return block
}()

// isBlockedIP checks whether an IP address should be blocked for SSRF protection.
// It returns a non-nil error if the IP is loopback, private, link-local,
// unspecified, CGNAT (RFC 6598), or a known cloud metadata address.
func isBlockedIP(ip net.IP) error {
	ipStr := ip.String()
	if blockedSSRFIPs[ipStr] {
		return fmt.Errorf("blocked SSRF target: %s", ipStr)
	}
	if ip.IsUnspecified() {
		return fmt.Errorf("blocked SSRF target: unspecified address %s", ipStr)
	}
	if ip.IsLoopback() {
		return fmt.Errorf("blocked SSRF target: loopback address %s", ipStr)
	}
	if ip.IsPrivate() {
		return fmt.Errorf("blocked SSRF target: private address %s", ipStr)
	}
	if ip.IsLinkLocalUnicast() {
		return fmt.Errorf("blocked SSRF target: link-local unicast address %s", ipStr)
	}
	if ip.IsLinkLocalMulticast() {
		return fmt.Errorf("blocked SSRF target: link-local multicast address %s", ipStr)
	}
	if cgnatBlock.Contains(ip) {
		return fmt.Errorf("blocked SSRF target: CGNAT address %s", ipStr)
	}
	return nil
}

// ssrfSafeDialContext returns a DialContext function that validates every
// resolved IP address against the SSRF blocklist before connecting.
// This prevents DNS rebinding attacks by checking the IP at connect time
// rather than in a separate pre-flight validation step.
//
// resolver may be nil, in which case net.DefaultResolver is used.
func ssrfSafeDialContext(resolver *net.Resolver) func(ctx context.Context, network, addr string) (net.Conn, error) {
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, fmt.Errorf("split host/port: %w", err)
		}

		// Block known dangerous hostnames.
		if blockedSSRFHosts[host] {
			return nil, fmt.Errorf("blocked SSRF target: %s", host)
		}

		// Resolve hostname to IPs.
		ips, err := resolver.LookupHost(ctx, host)
		if err != nil {
			return nil, fmt.Errorf("resolve hostname %q: %w", host, err)
		}

		// Check every resolved IP against the blocklist.
		for _, ipStr := range ips {
			ip := net.ParseIP(ipStr)
			if ip == nil {
				continue
			}
			if err := isBlockedIP(ip); err != nil {
				return nil, err
			}
		}

		// All IPs are safe — connect to the first one that works.
		var dialer net.Dialer
		for _, ipStr := range ips {
			conn, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(ipStr, port))
			if dialErr == nil {
				return conn, nil
			}
			err = dialErr
		}
		return nil, fmt.Errorf("dial %s: %w", addr, err)
	}
}

// imageHTTPClient is a dedicated HTTP client for downloading images,
// configured with SSRF-safe connect-time IP validation, a timeout,
// and a redirect limit.
var imageHTTPClient = &http.Client{
	Timeout: 30 * time.Second,
	Transport: &http.Transport{
		DialContext: ssrfSafeDialContext(nil),
	},
	CheckRedirect: func(req *http.Request, via []*http.Request) error {
		if len(via) >= maxImageRedirects {
			return fmt.Errorf("too many redirects (max %d)", maxImageRedirects)
		}
		return nil
	},
}

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

// maxImagesPerRequest caps the number of image_url entries a single
// chat-completion request may reference across all messages. Without this
// cap, a request well within the 10 MB body limit could still enumerate
// thousands of image URLs, each fetched sequentially (SERVE-3b).
const maxImagesPerRequest = 16

// maxTotalImageBytes caps the cumulative decoded/downloaded size of all
// images fetched for a single request, preventing memory exhaustion even
// when every image individually stays under maxImageSize.
const maxTotalImageBytes = 64 * 1024 * 1024

// maxImageFetchWallClock bounds the total wall-clock time spent fetching
// images for a single request, regardless of how many images are fetched
// or how slow individual origins are. It is applied on top of (not instead
// of) the caller's context deadline.
const maxImageFetchWallClock = 60 * time.Second

// imageFetchBudget tracks the remaining decoded-byte budget shared across
// every image fetched for a single request. It is not safe for concurrent
// use; fetchImages consumes it sequentially per request.
type imageFetchBudget struct {
	remaining int64
}

// newImageFetchBudget creates a budget with the given total byte allowance.
func newImageFetchBudget(total int64) *imageFetchBudget {
	return &imageFetchBudget{remaining: total}
}

// fetchImages downloads or decodes images from the given URLs, enforcing a
// shared byte budget and the caller's context (which should carry an overall
// wall-clock deadline for the whole request's image fan-out).
// Supports both HTTP(S) URLs and base64 data URIs.
func fetchImages(ctx context.Context, urls []ImageURL, budget *imageFetchBudget) ([][]byte, error) {
	images := make([][]byte, len(urls))
	for i, u := range urls {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("image %d: fetch deadline exceeded: %w", i, err)
		}
		if budget.remaining <= 0 {
			return nil, fmt.Errorf("image %d: total decoded image bytes exceed request budget of %d bytes", i, maxTotalImageBytes)
		}
		limit := budget.remaining
		if limit > maxImageSize {
			limit = maxImageSize
		}
		data, err := fetchImageData(ctx, u.URL, limit)
		if err != nil {
			return nil, fmt.Errorf("image %d: %w", i, err)
		}
		budget.remaining -= int64(len(data))
		images[i] = data
	}
	return images, nil
}

// fetchImageData retrieves raw image bytes from a URL or data URI, never
// returning more than limit bytes.
func fetchImageData(ctx context.Context, rawURL string, limit int64) ([]byte, error) {
	if strings.HasPrefix(rawURL, "data:") {
		return decodeDataURI(rawURL, limit)
	}
	if strings.HasPrefix(rawURL, "http://") || strings.HasPrefix(rawURL, "https://") {
		return downloadImage(ctx, rawURL, limit)
	}
	return nil, fmt.Errorf("unsupported image URL scheme: %s", rawURL)
}

// decodeDataURI parses a data URI of the form data:<mediatype>;base64,<data>,
// rejecting decoded payloads larger than limit bytes.
func decodeDataURI(uri string, limit int64) ([]byte, error) {
	idx := strings.Index(uri, ",")
	if idx < 0 {
		return nil, fmt.Errorf("invalid data URI: missing comma separator")
	}
	header := uri[:idx]
	if !strings.Contains(header, ";base64") {
		return nil, fmt.Errorf("unsupported data URI encoding (only base64 is supported)")
	}
	encoded := uri[idx+1:]
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("image exceeds remaining budget of %d bytes", limit)
	}
	return data, nil
}

// downloadImage fetches an image from an HTTP(S) URL, never returning more
// than limit bytes.
// SSRF protection is enforced at connect time by ssrfSafeDialContext
// in the HTTP transport, preventing DNS rebinding attacks.
func downloadImage(ctx context.Context, url string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	resp, err := imageHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download returned status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("image exceeds maximum size of %d bytes", limit)
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
