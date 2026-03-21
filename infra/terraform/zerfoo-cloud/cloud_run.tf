resource "google_cloud_run_v2_service" "api_gateway" {
  name     = "zerfoo-api-gateway"
  location = var.region

  template {
    containers {
      image = var.api_gateway_image

      resources {
        limits = {
          cpu    = var.api_gateway_cpu
          memory = var.api_gateway_memory
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "ZERFOO_CLUSTER_ENDPOINT"
        value = google_container_cluster.primary.endpoint
      }

      env {
        name  = "ZERFOO_MODEL_BUCKET"
        value = google_storage_bucket.model_artifacts.name
      }

      env {
        name  = "ZERFOO_ENVIRONMENT"
        value = var.environment
      }

      startup_probe {
        http_get {
          path = "/healthz"
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/healthz"
        }
        period_seconds = 30
      }
    }

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    labels = local.labels
  }

  labels = local.labels
}

# Dedicated service account for API invocation. Public access (allUsers) was
# removed to enforce authentication and prevent unauthenticated internet
# traffic from reaching the Cloud Run service (T106.25).
resource "google_service_account" "api_invoker" {
  account_id   = "zerfoo-api-invoker"
  display_name = "Zerfoo API Gateway Invoker"
  description  = "Service account authorized to invoke the zerfoo-api-gateway Cloud Run service"
  project      = var.project_id
}

resource "google_cloud_run_v2_service_iam_member" "api_invoker_access" {
  project  = google_cloud_run_v2_service.api_gateway.project
  location = google_cloud_run_v2_service.api_gateway.location
  name     = google_cloud_run_v2_service.api_gateway.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.api_invoker.email}"
}
