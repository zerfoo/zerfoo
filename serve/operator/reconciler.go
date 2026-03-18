package operator

import (
	"context"
	"fmt"
)

// KubeClient abstracts Kubernetes API operations needed by the reconciler.
// Implementations may wrap a real Kubernetes client or a mock for testing.
type KubeClient interface {
	GetDeployment(ctx context.Context, namespace, name string) (*Deployment, error)
	CreateDeployment(ctx context.Context, d *Deployment) error
	UpdateDeployment(ctx context.Context, d *Deployment) error
	DeleteDeployment(ctx context.Context, namespace, name string) error

	GetService(ctx context.Context, namespace, name string) (*Service, error)
	CreateService(ctx context.Context, s *Service) error
	UpdateService(ctx context.Context, s *Service) error

	GetHPA(ctx context.Context, namespace, name string) (*HPA, error)
	CreateHPA(ctx context.Context, h *HPA) error
	UpdateHPA(ctx context.Context, h *HPA) error
}

// Reconciler compares the desired ZerfooInferenceService spec against the
// current cluster state and applies the necessary changes.
type Reconciler struct {
	client KubeClient
}

// NewReconciler creates a Reconciler backed by the given KubeClient.
func NewReconciler(client KubeClient) *Reconciler {
	return &Reconciler{client: client}
}

// Reconcile drives the cluster toward the desired state described by svc.
// It creates, updates, or deletes Deployments, Services, and HPAs as needed.
func (r *Reconciler) Reconcile(ctx context.Context, svc *ZerfooInferenceService) error {
	if err := svc.Spec.Validate(); err != nil {
		return err
	}

	if err := r.reconcileDeployment(ctx, svc); err != nil {
		return fmt.Errorf("reconcile deployment: %w", err)
	}

	if err := r.reconcileCanary(ctx, svc); err != nil {
		return fmt.Errorf("reconcile canary: %w", err)
	}

	if err := r.reconcileService(ctx, svc); err != nil {
		return fmt.Errorf("reconcile service: %w", err)
	}

	if err := r.reconcileHPA(ctx, svc); err != nil {
		return fmt.Errorf("reconcile hpa: %w", err)
	}

	return nil
}

// Delete removes all resources associated with the given service.
func (r *Reconciler) Delete(ctx context.Context, svc *ZerfooInferenceService) error {
	primaryName := deploymentName(svc.Name)
	canaryName := canaryDeploymentName(svc.Name)

	// Delete canary deployment if it exists.
	if _, err := r.client.GetDeployment(ctx, svc.Namespace, canaryName); err == nil {
		if err := r.client.DeleteDeployment(ctx, svc.Namespace, canaryName); err != nil {
			return fmt.Errorf("delete canary deployment: %w", err)
		}
	}

	// Delete primary deployment.
	if err := r.client.DeleteDeployment(ctx, svc.Namespace, primaryName); err != nil {
		return fmt.Errorf("delete deployment: %w", err)
	}

	return nil
}

func (r *Reconciler) reconcileDeployment(ctx context.Context, svc *ZerfooInferenceService) error {
	name := deploymentName(svc.Name)
	desired := &Deployment{
		Name:      name,
		Namespace: svc.Namespace,
		Replicas:  svc.Spec.Replicas,
		ModelRef:  svc.Spec.ModelRef,
		Resources: svc.Spec.Resources,
		Health:    svc.Spec.HealthCheck,
	}

	existing, err := r.client.GetDeployment(ctx, svc.Namespace, name)
	if err != nil {
		// Not found — create.
		return r.client.CreateDeployment(ctx, desired)
	}

	// Update if spec has drifted.
	if needsUpdate(existing, desired) {
		return r.client.UpdateDeployment(ctx, desired)
	}
	return nil
}

func (r *Reconciler) reconcileCanary(ctx context.Context, svc *ZerfooInferenceService) error {
	name := canaryDeploymentName(svc.Name)

	if svc.Spec.Canary == nil {
		// No canary desired — delete if exists.
		if _, err := r.client.GetDeployment(ctx, svc.Namespace, name); err == nil {
			return r.client.DeleteDeployment(ctx, svc.Namespace, name)
		}
		return nil
	}

	desired := &Deployment{
		Name:      name,
		Namespace: svc.Namespace,
		Replicas:  svc.Spec.Replicas, // canary mirrors primary replica count
		ModelRef:  svc.Spec.Canary.ModelRef,
		Resources: svc.Spec.Resources,
		Health:    svc.Spec.HealthCheck,
	}

	existing, err := r.client.GetDeployment(ctx, svc.Namespace, name)
	if err != nil {
		return r.client.CreateDeployment(ctx, desired)
	}

	if needsUpdate(existing, desired) {
		return r.client.UpdateDeployment(ctx, desired)
	}
	return nil
}

func (r *Reconciler) reconcileService(ctx context.Context, svc *ZerfooInferenceService) error {
	name := serviceName(svc.Name)
	desired := &Service{
		Name:      name,
		Namespace: svc.Namespace,
		Selector:  map[string]string{"app": svc.Name},
	}

	if svc.Spec.Canary != nil {
		desired.Weights = []WeightedTarget{
			{DeploymentName: deploymentName(svc.Name), Weight: 100 - svc.Spec.Canary.Weight},
			{DeploymentName: canaryDeploymentName(svc.Name), Weight: svc.Spec.Canary.Weight},
		}
	}

	_, err := r.client.GetService(ctx, svc.Namespace, name)
	if err != nil {
		return r.client.CreateService(ctx, desired)
	}
	return r.client.UpdateService(ctx, desired)
}

func (r *Reconciler) reconcileHPA(ctx context.Context, svc *ZerfooInferenceService) error {
	if svc.Spec.MinReplicas <= 0 || svc.Spec.MaxReplicas <= 0 {
		return nil // autoscaling not configured
	}

	name := hpaName(svc.Name)
	desired := &HPA{
		Name:        name,
		Namespace:   svc.Namespace,
		TargetRef:   deploymentName(svc.Name),
		MinReplicas: svc.Spec.MinReplicas,
		MaxReplicas: svc.Spec.MaxReplicas,
	}

	existing, err := r.client.GetHPA(ctx, svc.Namespace, name)
	if err != nil {
		return r.client.CreateHPA(ctx, desired)
	}

	if existing.MinReplicas != desired.MinReplicas || existing.MaxReplicas != desired.MaxReplicas {
		return r.client.UpdateHPA(ctx, desired)
	}
	return nil
}

// needsUpdate returns true if the existing deployment differs from desired.
func needsUpdate(existing, desired *Deployment) bool {
	if existing.Replicas != desired.Replicas {
		return true
	}
	if existing.ModelRef != desired.ModelRef {
		return true
	}
	if existing.Resources != desired.Resources {
		return true
	}
	return false
}

func deploymentName(svcName string) string       { return svcName + "-primary" }
func canaryDeploymentName(svcName string) string  { return svcName + "-canary" }
func serviceName(svcName string) string           { return svcName + "-svc" }
func hpaName(svcName string) string               { return svcName + "-hpa" }
