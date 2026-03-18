output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate (base64)"
  value       = google_container_cluster.primary.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${var.cluster_name} --zone ${var.zone} --project ${var.project_id}"
}

output "model_bucket_url" {
  description = "GCS bucket URL for model artifacts"
  value       = "gs://${google_storage_bucket.model_artifacts.name}"
}

output "api_gateway_url" {
  description = "Cloud Run API gateway URL"
  value       = google_cloud_run_v2_service.api_gateway.uri
}

output "model_reader_service_account" {
  description = "Email of the model reader service account for Workload Identity"
  value       = google_service_account.model_reader.email
}
