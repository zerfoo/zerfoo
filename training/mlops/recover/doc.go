// Experimental — this package is not yet wired into the main framework.
//
// Package recover provides an automatic retrain pipeline for models that have
// degraded in production. It imports the monitor package for drift detection
// and orchestrates rollback, retraining, validation, and redeployment.
package recover
