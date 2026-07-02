package registry

// OCI distribution spec media types.
const (
	// MediaTypeGGUF is the media type for GGUF model files.
	MediaTypeGGUF = "application/vnd.zerfoo.model.gguf.v1"
	// MediaTypeModelConfig is the media type for model configuration.
	MediaTypeModelConfig = "application/vnd.zerfoo.model.config.v1+json"
	// MediaTypeOCIManifest is the standard OCI image manifest media type.
	MediaTypeOCIManifest = "application/vnd.oci.image.manifest.v1+json"
)

// Descriptor describes an OCI content-addressable blob.
type Descriptor struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

// Manifest represents an OCI image manifest for a GGUF model.
type Manifest struct {
	SchemaVersion int          `json:"schemaVersion"`
	MediaType     string       `json:"mediaType"`
	Config        Descriptor   `json:"config"`
	Layers        []Descriptor `json:"layers"`
}

// ModelConfig holds model metadata stored as the OCI config blob.
type ModelConfig struct {
	Architecture string `json:"architecture,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	Parameters   int64  `json:"parameters,omitempty"`
}

// TagList represents the response from the OCI tags/list endpoint.
type TagList struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
}
