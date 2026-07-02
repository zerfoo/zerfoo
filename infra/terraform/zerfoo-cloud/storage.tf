resource "google_storage_bucket" "model_artifacts" {
  name     = "${var.model_bucket_name}-${var.project_id}"
  location = var.region

  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }

  labels = local.labels
}

# Grant the GKE nodes access to the model bucket via Workload Identity.
resource "google_service_account" "model_reader" {
  account_id   = "zerfoo-model-reader"
  display_name = "Zerfoo Model Reader"
  description  = "Service account for GKE pods to read model artifacts from GCS"
}

resource "google_storage_bucket_iam_member" "model_reader_access" {
  bucket = google_storage_bucket.model_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.model_reader.email}"
}

resource "google_service_account_iam_member" "workload_identity_binding" {
  service_account_id = google_service_account.model_reader.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[zerfoo/model-reader]"
}
