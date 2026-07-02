package operator

import (
	"context"
	"errors"
	"testing"
	"time"
)

// mockKubeClient implements KubeClient for testing. Each method records calls
// and returns preconfigured resources or errors.
type mockKubeClient struct {
	deployments map[string]*Deployment
	services    map[string]*Service
	hpas        map[string]*HPA

	createdDeployments []*Deployment
	updatedDeployments []*Deployment
	deletedDeployments []string
	createdServices    []*Service
	updatedServices    []*Service
	createdHPAs        []*HPA
	updatedHPAs        []*HPA
}

func newMockClient() *mockKubeClient {
	return &mockKubeClient{
		deployments: make(map[string]*Deployment),
		services:    make(map[string]*Service),
		hpas:        make(map[string]*HPA),
	}
}

func key(ns, name string) string { return ns + "/" + name }

func (m *mockKubeClient) GetDeployment(_ context.Context, namespace, name string) (*Deployment, error) {
	d, ok := m.deployments[key(namespace, name)]
	if !ok {
		return nil, ErrNotFound
	}
	return d, nil
}

func (m *mockKubeClient) CreateDeployment(_ context.Context, d *Deployment) error {
	m.deployments[key(d.Namespace, d.Name)] = d
	m.createdDeployments = append(m.createdDeployments, d)
	return nil
}

func (m *mockKubeClient) UpdateDeployment(_ context.Context, d *Deployment) error {
	m.deployments[key(d.Namespace, d.Name)] = d
	m.updatedDeployments = append(m.updatedDeployments, d)
	return nil
}

func (m *mockKubeClient) DeleteDeployment(_ context.Context, namespace, name string) error {
	delete(m.deployments, key(namespace, name))
	m.deletedDeployments = append(m.deletedDeployments, key(namespace, name))
	return nil
}

func (m *mockKubeClient) GetService(_ context.Context, namespace, name string) (*Service, error) {
	s, ok := m.services[key(namespace, name)]
	if !ok {
		return nil, ErrNotFound
	}
	return s, nil
}

func (m *mockKubeClient) CreateService(_ context.Context, s *Service) error {
	m.services[key(s.Namespace, s.Name)] = s
	m.createdServices = append(m.createdServices, s)
	return nil
}

func (m *mockKubeClient) UpdateService(_ context.Context, s *Service) error {
	m.services[key(s.Namespace, s.Name)] = s
	m.updatedServices = append(m.updatedServices, s)
	return nil
}

func (m *mockKubeClient) GetHPA(_ context.Context, namespace, name string) (*HPA, error) {
	h, ok := m.hpas[key(namespace, name)]
	if !ok {
		return nil, ErrNotFound
	}
	return h, nil
}

func (m *mockKubeClient) CreateHPA(_ context.Context, h *HPA) error {
	m.hpas[key(h.Namespace, h.Name)] = h
	m.createdHPAs = append(m.createdHPAs, h)
	return nil
}

func (m *mockKubeClient) UpdateHPA(_ context.Context, h *HPA) error {
	m.hpas[key(h.Namespace, h.Name)] = h
	m.updatedHPAs = append(m.updatedHPAs, h)
	return nil
}

func validSpec() ZerfooInferenceServiceSpec {
	return ZerfooInferenceServiceSpec{
		ModelRef: "llama3-8b-q4",
		Replicas: 2,
		Resources: ResourceSpec{
			CPU:       "4",
			Memory:    "16Gi",
			GPUMemory: "24Gi",
		},
		HealthCheck: HealthCheckSpec{
			Path:     "/healthz",
			Interval: 10 * time.Second,
			Timeout:  5 * time.Second,
		},
	}
}

func validService() *ZerfooInferenceService {
	return &ZerfooInferenceService{
		Name:      "my-llama",
		Namespace: "default",
		Spec:      validSpec(),
	}
}

func TestReconciler_CreateDeployment(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	if len(client.createdDeployments) != 1 {
		t.Fatalf("expected 1 created deployment, got %d", len(client.createdDeployments))
	}

	d := client.createdDeployments[0]
	if d.Name != "my-llama-primary" {
		t.Errorf("deployment name = %q, want %q", d.Name, "my-llama-primary")
	}
	if d.Namespace != "default" {
		t.Errorf("deployment namespace = %q, want %q", d.Namespace, "default")
	}
	if d.Replicas != 2 {
		t.Errorf("deployment replicas = %d, want %d", d.Replicas, 2)
	}
	if d.ModelRef != "llama3-8b-q4" {
		t.Errorf("deployment modelRef = %q, want %q", d.ModelRef, "llama3-8b-q4")
	}
	if d.Resources.GPUMemory != "24Gi" {
		t.Errorf("deployment gpuMemory = %q, want %q", d.Resources.GPUMemory, "24Gi")
	}

	// Service should also be created.
	if len(client.createdServices) != 1 {
		t.Fatalf("expected 1 created service, got %d", len(client.createdServices))
	}
	if client.createdServices[0].Name != "my-llama-svc" {
		t.Errorf("service name = %q, want %q", client.createdServices[0].Name, "my-llama-svc")
	}
}

func TestReconciler_Scale(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()

	// First reconcile creates the deployment.
	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("initial Reconcile() error: %v", err)
	}

	// Scale up to 5 replicas.
	svc.Spec.Replicas = 5
	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("scale Reconcile() error: %v", err)
	}

	if len(client.updatedDeployments) != 1 {
		t.Fatalf("expected 1 updated deployment, got %d", len(client.updatedDeployments))
	}
	if client.updatedDeployments[0].Replicas != 5 {
		t.Errorf("updated replicas = %d, want %d", client.updatedDeployments[0].Replicas, 5)
	}
}

func TestReconciler_Canary(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	svc.Spec.Canary = &CanarySpec{
		ModelRef: "llama3-8b-q8",
		Weight:   20,
	}

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	// Should create primary + canary deployments.
	if len(client.createdDeployments) != 2 {
		t.Fatalf("expected 2 created deployments, got %d", len(client.createdDeployments))
	}

	var primary, canary *Deployment
	for _, d := range client.createdDeployments {
		switch d.Name {
		case "my-llama-primary":
			primary = d
		case "my-llama-canary":
			canary = d
		}
	}

	if primary == nil {
		t.Fatal("primary deployment not created")
	}
	if canary == nil {
		t.Fatal("canary deployment not created")
	}
	if canary.ModelRef != "llama3-8b-q8" {
		t.Errorf("canary modelRef = %q, want %q", canary.ModelRef, "llama3-8b-q8")
	}

	// Service should have traffic weights.
	if len(client.createdServices) != 1 {
		t.Fatalf("expected 1 service, got %d", len(client.createdServices))
	}
	s := client.createdServices[0]
	if len(s.Weights) != 2 {
		t.Fatalf("expected 2 weighted targets, got %d", len(s.Weights))
	}
	for _, w := range s.Weights {
		switch w.DeploymentName {
		case "my-llama-primary":
			if w.Weight != 80 {
				t.Errorf("primary weight = %d, want 80", w.Weight)
			}
		case "my-llama-canary":
			if w.Weight != 20 {
				t.Errorf("canary weight = %d, want 20", w.Weight)
			}
		default:
			t.Errorf("unexpected weighted target: %q", w.DeploymentName)
		}
	}
}

func TestReconciler_HealthCheck(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	svc.Spec.HealthCheck = HealthCheckSpec{
		Path:     "/ready",
		Interval: 30 * time.Second,
		Timeout:  10 * time.Second,
	}

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	if len(client.createdDeployments) != 1 {
		t.Fatalf("expected 1 deployment, got %d", len(client.createdDeployments))
	}

	d := client.createdDeployments[0]
	if d.Health.Path != "/ready" {
		t.Errorf("health path = %q, want %q", d.Health.Path, "/ready")
	}
	if d.Health.Interval != 30*time.Second {
		t.Errorf("health interval = %v, want %v", d.Health.Interval, 30*time.Second)
	}
	if d.Health.Timeout != 10*time.Second {
		t.Errorf("health timeout = %v, want %v", d.Health.Timeout, 10*time.Second)
	}
}

func TestCRDSpec_Validation(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*ZerfooInferenceServiceSpec)
		wantErr bool
	}{
		{
			name:    "valid spec",
			modify:  func(s *ZerfooInferenceServiceSpec) {},
			wantErr: false,
		},
		{
			name:    "empty modelRef",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.ModelRef = "" },
			wantErr: true,
		},
		{
			name:    "zero replicas",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.Replicas = 0 },
			wantErr: true,
		},
		{
			name:    "negative replicas",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.Replicas = -1 },
			wantErr: true,
		},
		{
			name: "minReplicas > maxReplicas",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.MinReplicas = 10
				s.MaxReplicas = 5
			},
			wantErr: true,
		},
		{
			name: "canary empty modelRef",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.Canary = &CanarySpec{ModelRef: "", Weight: 10}
			},
			wantErr: true,
		},
		{
			name: "canary weight over 100",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.Canary = &CanarySpec{ModelRef: "m2", Weight: 101}
			},
			wantErr: true,
		},
		{
			name: "canary weight negative",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.Canary = &CanarySpec{ModelRef: "m2", Weight: -1}
			},
			wantErr: true,
		},
		{
			name:    "empty health path",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.HealthCheck.Path = "" },
			wantErr: true,
		},
		{
			name:    "zero health interval",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.HealthCheck.Interval = 0 },
			wantErr: true,
		},
		{
			name:    "zero health timeout",
			modify:  func(s *ZerfooInferenceServiceSpec) { s.HealthCheck.Timeout = 0 },
			wantErr: true,
		},
		{
			name: "valid canary",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.Canary = &CanarySpec{ModelRef: "m2", Weight: 50}
			},
			wantErr: false,
		},
		{
			name: "valid autoscaling",
			modify: func(s *ZerfooInferenceServiceSpec) {
				s.MinReplicas = 2
				s.MaxReplicas = 10
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := validSpec()
			tt.modify(&spec)
			err := spec.Validate()
			if tt.wantErr && err == nil {
				t.Error("Validate() = nil, want error")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
			if tt.wantErr && err != nil && !errors.Is(err, ErrInvalidSpec) {
				t.Errorf("error should wrap ErrInvalidSpec, got %v", err)
			}
		})
	}
}

func TestReconciler_Delete(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	svc.Spec.Canary = &CanarySpec{ModelRef: "llama3-8b-q8", Weight: 20}

	// Create resources first.
	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	// Delete all resources.
	if err := rec.Delete(context.Background(), svc); err != nil {
		t.Fatalf("Delete() error: %v", err)
	}

	if len(client.deletedDeployments) != 2 {
		t.Fatalf("expected 2 deleted deployments, got %d", len(client.deletedDeployments))
	}

	// Verify both primary and canary were deleted.
	deleted := make(map[string]bool)
	for _, d := range client.deletedDeployments {
		deleted[d] = true
	}
	if !deleted["default/my-llama-primary"] {
		t.Error("primary deployment not deleted")
	}
	if !deleted["default/my-llama-canary"] {
		t.Error("canary deployment not deleted")
	}
}

func TestReconciler_Autoscaling(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	svc.Spec.MinReplicas = 2
	svc.Spec.MaxReplicas = 10

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	if len(client.createdHPAs) != 1 {
		t.Fatalf("expected 1 HPA, got %d", len(client.createdHPAs))
	}

	h := client.createdHPAs[0]
	if h.Name != "my-llama-hpa" {
		t.Errorf("HPA name = %q, want %q", h.Name, "my-llama-hpa")
	}
	if h.MinReplicas != 2 {
		t.Errorf("HPA minReplicas = %d, want %d", h.MinReplicas, 2)
	}
	if h.MaxReplicas != 10 {
		t.Errorf("HPA maxReplicas = %d, want %d", h.MaxReplicas, 10)
	}
	if h.TargetRef != "my-llama-primary" {
		t.Errorf("HPA targetRef = %q, want %q", h.TargetRef, "my-llama-primary")
	}
}

func TestReconciler_NoAutoscalingWithoutConfig(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	// MinReplicas and MaxReplicas are zero — no HPA should be created.

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	if len(client.createdHPAs) != 0 {
		t.Errorf("expected 0 HPAs, got %d", len(client.createdHPAs))
	}
}

func TestReconciler_UpdateExistingDeployment(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()

	// Pre-populate with an existing deployment that has different replicas.
	client.deployments[key("default", "my-llama-primary")] = &Deployment{
		Name:      "my-llama-primary",
		Namespace: "default",
		Replicas:  1,
		ModelRef:  "llama3-8b-q4",
		Resources: svc.Spec.Resources,
	}

	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() error: %v", err)
	}

	// Should update, not create.
	if len(client.createdDeployments) != 0 {
		t.Errorf("expected 0 created deployments, got %d", len(client.createdDeployments))
	}
	if len(client.updatedDeployments) != 1 {
		t.Fatalf("expected 1 updated deployment, got %d", len(client.updatedDeployments))
	}
	if client.updatedDeployments[0].Replicas != 2 {
		t.Errorf("updated replicas = %d, want %d", client.updatedDeployments[0].Replicas, 2)
	}
}

func TestReconciler_RemoveCanary(t *testing.T) {
	client := newMockClient()
	rec := NewReconciler(client)
	svc := validService()
	svc.Spec.Canary = &CanarySpec{ModelRef: "llama3-8b-q8", Weight: 20}

	// Create with canary.
	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() with canary error: %v", err)
	}
	if len(client.createdDeployments) != 2 {
		t.Fatalf("expected 2 deployments, got %d", len(client.createdDeployments))
	}

	// Remove canary.
	svc.Spec.Canary = nil
	client.createdDeployments = nil // reset tracking
	if err := rec.Reconcile(context.Background(), svc); err != nil {
		t.Fatalf("Reconcile() without canary error: %v", err)
	}

	// Canary deployment should be deleted.
	found := false
	for _, d := range client.deletedDeployments {
		if d == "default/my-llama-canary" {
			found = true
		}
	}
	if !found {
		t.Error("canary deployment was not deleted when canary spec removed")
	}
}
