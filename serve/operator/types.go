// Package operator provides a Kubernetes operator for managing ZerfooInferenceService
// custom resources. It reconciles desired inference service state into Kubernetes
// Deployments, Services, and HorizontalPodAutoscalers via a pluggable KubeClient interface.
package operator

import (
	"errors"
	"fmt"
	"time"
)

// Standard errors returned by validation and reconciliation.
var (
	ErrInvalidSpec   = errors.New("operator: invalid spec")
	ErrNotFound      = errors.New("operator: resource not found")
	ErrAlreadyExists = errors.New("operator: resource already exists")
)

// ZerfooInferenceService is the top-level custom resource that declares a
// desired inference service deployment.
type ZerfooInferenceService struct {
	Name      string                      `json:"name"`
	Namespace string                      `json:"namespace"`
	Spec      ZerfooInferenceServiceSpec  `json:"spec"`
	Status    ZerfooInferenceServiceStatus `json:"status"`
}

// ZerfooInferenceServiceSpec describes the desired state of an inference service.
type ZerfooInferenceServiceSpec struct {
	// ModelRef is the model repository reference (e.g. "llama3-8b-q4").
	ModelRef string `json:"modelRef"`

	// Replicas is the desired number of inference pods.
	Replicas int `json:"replicas"`

	// MinReplicas for autoscaling (0 means no autoscaling).
	MinReplicas int `json:"minReplicas,omitempty"`

	// MaxReplicas for autoscaling.
	MaxReplicas int `json:"maxReplicas,omitempty"`

	// Resources specifies compute resource limits.
	Resources ResourceSpec `json:"resources"`

	// Canary optionally configures a canary deployment with traffic splitting.
	Canary *CanarySpec `json:"canary,omitempty"`

	// HealthCheck configures liveness/readiness probes.
	HealthCheck HealthCheckSpec `json:"healthCheck"`
}

// ResourceSpec declares CPU, memory, and GPU resource limits.
type ResourceSpec struct {
	CPU       string `json:"cpu"`       // e.g. "4"
	Memory    string `json:"memory"`    // e.g. "16Gi"
	GPUMemory string `json:"gpuMemory"` // e.g. "24Gi"
}

// CanarySpec configures a canary deployment alongside the primary.
type CanarySpec struct {
	// ModelRef is the canary model reference.
	ModelRef string `json:"modelRef"`

	// Weight is the percentage of traffic routed to the canary (0-100).
	Weight int `json:"weight"`
}

// HealthCheckSpec configures health check probes.
type HealthCheckSpec struct {
	// Path is the HTTP health check endpoint (e.g. "/healthz").
	Path string `json:"path"`

	// Interval between health checks.
	Interval time.Duration `json:"interval"`

	// Timeout for a single health check.
	Timeout time.Duration `json:"timeout"`
}

// ZerfooInferenceServiceStatus represents the observed state.
type ZerfooInferenceServiceStatus struct {
	// Ready indicates whether the service is fully available.
	Ready bool `json:"ready"`

	// Replicas is the current number of running replicas.
	Replicas int `json:"replicas"`

	// Message provides a human-readable status message.
	Message string `json:"message,omitempty"`
}

// Deployment represents a Kubernetes Deployment managed by the operator.
type Deployment struct {
	Name      string       `json:"name"`
	Namespace string       `json:"namespace"`
	Replicas  int          `json:"replicas"`
	ModelRef  string       `json:"modelRef"`
	Resources ResourceSpec `json:"resources"`
	Health    HealthCheckSpec `json:"healthCheck"`
}

// Service represents a Kubernetes Service managed by the operator.
type Service struct {
	Name      string         `json:"name"`
	Namespace string         `json:"namespace"`
	Selector  map[string]string `json:"selector"`
	Weights   []WeightedTarget `json:"weights,omitempty"`
}

// WeightedTarget maps a deployment name to a traffic weight for canary routing.
type WeightedTarget struct {
	DeploymentName string `json:"deploymentName"`
	Weight         int    `json:"weight"`
}

// HPA represents a Kubernetes HorizontalPodAutoscaler.
type HPA struct {
	Name        string `json:"name"`
	Namespace   string `json:"namespace"`
	TargetRef   string `json:"targetRef"`
	MinReplicas int    `json:"minReplicas"`
	MaxReplicas int    `json:"maxReplicas"`
}

// Validate checks the spec for required fields and constraints.
func (s *ZerfooInferenceServiceSpec) Validate() error {
	if s.ModelRef == "" {
		return fmt.Errorf("%w: modelRef is required", ErrInvalidSpec)
	}
	if s.Replicas < 1 {
		return fmt.Errorf("%w: replicas must be >= 1", ErrInvalidSpec)
	}
	if s.MinReplicas > 0 && s.MaxReplicas > 0 && s.MinReplicas > s.MaxReplicas {
		return fmt.Errorf("%w: minReplicas (%d) must be <= maxReplicas (%d)", ErrInvalidSpec, s.MinReplicas, s.MaxReplicas)
	}
	if s.Canary != nil {
		if s.Canary.ModelRef == "" {
			return fmt.Errorf("%w: canary modelRef is required", ErrInvalidSpec)
		}
		if s.Canary.Weight < 0 || s.Canary.Weight > 100 {
			return fmt.Errorf("%w: canary weight must be 0-100, got %d", ErrInvalidSpec, s.Canary.Weight)
		}
	}
	if s.HealthCheck.Path == "" {
		return fmt.Errorf("%w: healthCheck path is required", ErrInvalidSpec)
	}
	if s.HealthCheck.Interval <= 0 {
		return fmt.Errorf("%w: healthCheck interval must be > 0", ErrInvalidSpec)
	}
	if s.HealthCheck.Timeout <= 0 {
		return fmt.Errorf("%w: healthCheck timeout must be > 0", ErrInvalidSpec)
	}
	return nil
}
