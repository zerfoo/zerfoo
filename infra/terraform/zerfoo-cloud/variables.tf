variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the GKE cluster"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "zerfoo-inference"
}

variable "gpu_type" {
  description = "GPU accelerator type (nvidia-tesla-t4, nvidia-tesla-a100, nvidia-l4)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

variable "gpu_node_count" {
  description = "Number of nodes in the GPU node pool"
  type        = number
  default     = 1
}

variable "gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "n1-standard-8"
}

variable "cpu_node_count" {
  description = "Number of nodes in the default CPU node pool"
  type        = number
  default     = 2
}

variable "cpu_machine_type" {
  description = "Machine type for CPU nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "model_bucket_name" {
  description = "Name of the GCS bucket for model artifacts"
  type        = string
  default     = "zerfoo-model-artifacts"
}

variable "api_gateway_image" {
  description = "Container image for the Cloud Run API gateway"
  type        = string
  default     = "gcr.io/zerfoo/api-gateway:latest"
}

variable "api_gateway_cpu" {
  description = "CPU limit for the Cloud Run API gateway"
  type        = string
  default     = "2"
}

variable "api_gateway_memory" {
  description = "Memory limit for the Cloud Run API gateway"
  type        = string
  default     = "1Gi"
}

variable "environment" {
  description = "Environment name (e.g., prod, staging)"
  type        = string
  default     = "prod"
}
