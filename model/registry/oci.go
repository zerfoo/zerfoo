package registry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// maxBlobSize is the maximum allowed size for a blob download (20 GB).
var maxBlobSize = 20 << 30

// Registry is an OCI distribution spec client for pushing and pulling
// GGUF models as OCI artifacts.
type Registry struct {
	url          string
	username     string
	password     string
	client       *http.Client
	insecureHTTP bool
}

// Option configures a Registry.
type Option func(*Registry)

// WithCredentials sets basic auth credentials for the registry.
func WithCredentials(username, password string) Option {
	return func(r *Registry) {
		r.username = username
		r.password = password
	}
}

// WithHTTPClient sets a custom HTTP client for the registry.
func WithHTTPClient(client *http.Client) Option {
	return func(r *Registry) {
		r.client = client
	}
}

// WithInsecureHTTP allows the registry client to talk to a plain http://
// endpoint instead of https://. By default NewRegistry rejects non-https
// registry URLs: an http registry lets a network attacker read
// credentials and blob contents, or tamper with manifests/blobs in
// transit (CWE-319). Only opt in for local development or test servers
// that cannot serve TLS (e.g. httptest.Server).
func WithInsecureHTTP() Option {
	return func(r *Registry) {
		r.insecureHTTP = true
	}
}

// NewRegistry creates a new OCI registry client for the given base URL.
//
// The URL must use the https:// scheme. Plain http:// is rejected unless
// WithInsecureHTTP is passed, since an insecure connection lets a
// network attacker observe credentials and model bytes, or tamper with
// them in transit. Use WithInsecureHTTP() to opt in for localhost
// development or test servers that do not serve TLS.
func NewRegistry(rawURL string, opts ...Option) (*Registry, error) {
	r := &Registry{
		url:    strings.TrimRight(rawURL, "/"),
		client: http.DefaultClient,
	}
	for _, opt := range opts {
		opt(r)
	}

	scheme, _, ok := strings.Cut(r.url, "://")
	if !ok {
		return nil, fmt.Errorf("oci: registry URL %q must include a scheme (https:// or http://)", rawURL)
	}
	switch strings.ToLower(scheme) {
	case "https":
		// Always allowed.
	case "http":
		if !r.insecureHTTP {
			return nil, fmt.Errorf("oci: registry URL %q uses http://; refusing to connect insecurely, use WithInsecureHTTP() to opt in", rawURL)
		}
	default:
		return nil, fmt.Errorf("oci: registry URL %q has unsupported scheme %q; only https:// is allowed (or http:// with WithInsecureHTTP)", rawURL, scheme)
	}

	return r, nil
}

// Reference holds a parsed OCI reference (registry/repo:tag or registry/repo@digest).
type Reference struct {
	Registry   string
	Repository string
	Tag        string
	Digest     string
}

// parseReference parses an OCI reference string.
// Formats: "registry.example.com/repo:tag" or "registry.example.com/repo@sha256:..."
func parseReference(ref string) (Reference, error) {
	r := Reference{}

	// Split off digest first.
	if idx := strings.Index(ref, "@"); idx >= 0 {
		r.Digest = ref[idx+1:]
		ref = ref[:idx]
	}

	// Split off tag.
	if r.Digest == "" {
		if idx := strings.LastIndex(ref, ":"); idx >= 0 {
			// Make sure we're not splitting on a port number by checking
			// if there's a slash after the colon position.
			afterColon := ref[idx+1:]
			if !strings.Contains(afterColon, "/") {
				r.Tag = afterColon
				ref = ref[:idx]
			}
		}
	}

	// What remains is registry/repository.
	if idx := strings.Index(ref, "/"); idx >= 0 {
		r.Registry = ref[:idx]
		r.Repository = ref[idx+1:]
	} else {
		return Reference{}, fmt.Errorf("invalid reference %q: missing repository", ref)
	}

	if r.Repository == "" {
		return Reference{}, fmt.Errorf("invalid reference %q: empty repository", ref)
	}
	if strings.Contains(r.Repository, "..") {
		return Reference{}, fmt.Errorf("invalid reference %q: repository contains path traversal", ref)
	}
	if r.Tag == "" && r.Digest == "" {
		r.Tag = "latest"
	}

	return r, nil
}

// Push uploads a GGUF model file to the registry as an OCI artifact.
func (r *Registry) Push(ctx context.Context, ref string, modelPath string) error {
	parsed, err := parseReference(ref)
	if err != nil {
		return err
	}

	// Read the model file.
	modelData, err := os.ReadFile(modelPath) //nolint:gosec
	if err != nil {
		return fmt.Errorf("read model file: %w", err)
	}

	modelDigest := sha256Digest(modelData)
	modelSize := int64(len(modelData))

	// 1. Upload the model blob.
	if err := r.uploadBlob(ctx, parsed.Repository, modelDigest, modelData); err != nil {
		return fmt.Errorf("upload model blob: %w", err)
	}

	// 2. Create and upload config blob.
	config := ModelConfig{}
	configData, err := json.Marshal(config)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	configDigest := sha256Digest(configData)
	configSize := int64(len(configData))

	if err := r.uploadBlob(ctx, parsed.Repository, configDigest, configData); err != nil {
		return fmt.Errorf("upload config blob: %w", err)
	}

	// 3. Create and upload manifest.
	manifest := Manifest{
		SchemaVersion: 2,
		MediaType:     MediaTypeOCIManifest,
		Config: Descriptor{
			MediaType: MediaTypeModelConfig,
			Digest:    configDigest,
			Size:      configSize,
		},
		Layers: []Descriptor{
			{
				MediaType: MediaTypeGGUF,
				Digest:    modelDigest,
				Size:      modelSize,
			},
		},
	}

	manifestData, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("marshal manifest: %w", err)
	}

	tag := parsed.Tag
	if tag == "" {
		tag = "latest"
	}

	return r.putManifest(ctx, parsed.Repository, tag, manifestData)
}

// Pull downloads a model from the registry to the given destination path.
func (r *Registry) Pull(ctx context.Context, ref string, destPath string) error {
	manifest, err := r.Resolve(ctx, ref)
	if err != nil {
		return fmt.Errorf("resolve manifest: %w", err)
	}

	parsed, err := parseReference(ref)
	if err != nil {
		return err
	}

	// Find the GGUF layer.
	var ggufLayer *Descriptor
	for i := range manifest.Layers {
		if manifest.Layers[i].MediaType == MediaTypeGGUF {
			ggufLayer = &manifest.Layers[i]
			break
		}
	}
	if ggufLayer == nil {
		return fmt.Errorf("no GGUF layer found in manifest")
	}

	// Download the blob.
	data, err := r.getBlob(ctx, parsed.Repository, ggufLayer.Digest)
	if err != nil {
		return fmt.Errorf("download blob: %w", err)
	}

	// Verify the blob content matches the requested digest before writing
	// anything to disk. Without this check a malicious or MITM'd registry
	// could serve arbitrary bytes for any digest.
	if sum := sha256Digest(data); sum != ggufLayer.Digest {
		return fmt.Errorf("oci: blob digest mismatch: got %s want %s", sum, ggufLayer.Digest)
	}

	if err := os.WriteFile(destPath, data, 0o600); err != nil {
		return fmt.Errorf("write model file: %w", err)
	}

	return nil
}

// Tags lists all tags for a repository.
func (r *Registry) Tags(ctx context.Context, repo string) ([]string, error) {
	url := fmt.Sprintf("%s/v2/%s/tags/list", r.url, repo)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	r.setAuth(req)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tags list returned status %d", resp.StatusCode)
	}

	var tagList TagList
	if err := json.NewDecoder(resp.Body).Decode(&tagList); err != nil {
		return nil, fmt.Errorf("decode tags: %w", err)
	}

	return tagList.Tags, nil
}

// Resolve resolves an OCI reference to its manifest.
func (r *Registry) Resolve(ctx context.Context, ref string) (*Manifest, error) {
	parsed, err := parseReference(ref)
	if err != nil {
		return nil, err
	}

	identifier := parsed.Tag
	if parsed.Digest != "" {
		identifier = parsed.Digest
	}
	if identifier == "" {
		identifier = "latest"
	}

	url := fmt.Sprintf("%s/v2/%s/manifests/%s", r.url, parsed.Repository, identifier)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", MediaTypeOCIManifest)
	r.setAuth(req)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("manifest GET returned status %d", resp.StatusCode)
	}

	var m Manifest
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}

	return &m, nil
}

// uploadBlob uploads a blob to the registry using the monolithic POST method.
func (r *Registry) uploadBlob(ctx context.Context, repo, digest string, data []byte) error {
	url := fmt.Sprintf("%s/v2/%s/blobs/uploads/?digest=%s", r.url, repo, digest)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(data)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	r.setAuth(req)

	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close() //nolint:errcheck

	// Accept 201 Created or 202 Accepted.
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		return fmt.Errorf("blob upload returned status %d", resp.StatusCode)
	}

	return nil
}

// putManifest uploads a manifest for the given tag.
func (r *Registry) putManifest(ctx context.Context, repo, tag string, data []byte) error {
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", r.url, repo, tag)

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, strings.NewReader(string(data)))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", MediaTypeOCIManifest)
	r.setAuth(req)

	resp, err := r.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("manifest PUT returned status %d", resp.StatusCode)
	}

	return nil
}

// getBlob downloads a blob by digest.
func (r *Registry) getBlob(ctx context.Context, repo, digest string) ([]byte, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/%s", r.url, repo, digest)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	r.setAuth(req)

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("blob GET returned status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBlobSize)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxBlobSize {
		return nil, fmt.Errorf("blob exceeds maximum size of %d bytes", maxBlobSize)
	}
	return data, nil
}

// setAuth sets basic auth on a request if credentials are configured.
func (r *Registry) setAuth(req *http.Request) {
	if r.username != "" || r.password != "" {
		req.SetBasicAuth(r.username, r.password)
	}
}

// sha256Digest computes the sha256 digest of data in OCI format.
func sha256Digest(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])
}
