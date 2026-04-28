// Experimental — this package is not yet wired into the main framework.
//
// Package provenance provides cryptographic model lifecycle tracking.
//
// A Tracker maintains a hash chain of model lifecycle events — training runs,
// datasets, hyperparameters, evaluations, and predictions. Each event is
// assigned a SHA-256 hash that incorporates the hash of its parent event(s),
// forming a directed acyclic graph (DAG) that can be traversed to trace any
// prediction back to the training data and configuration that produced it.
//
// The hash chain provides tamper-evident audit trails: if any event in the
// chain is modified, all downstream hashes become invalid.
//
//	t := provenance.NewTracker()
//
//	trainHash, _ := t.RecordTraining(provenance.TrainingRecord{
//	    RunID:           "run-001",
//	    ModelName:       "llama-3-8b",
//	    ModelVersion:    "v1.0.0",
//	    Hyperparameters: map[string]string{"lr": "3e-4", "epochs": "3"},
//	})
//
//	dataHash, _ := t.RecordDataset(provenance.DatasetRecord{
//	    Name:     "wiki-en",
//	    Version:  "2024-03",
//	    ParentID: trainHash,
//	})
//
//	evalHash, _ := t.RecordEvaluation(provenance.EvaluationRecord{
//	    ParentID: trainHash,
//	    Metrics:  map[string]float64{"loss": 0.42, "accuracy": 0.91},
//	})
//
//	// Trace from evaluation back to training data.
//	events, _ := t.Trace(evalHash)
//
//	// Verify hash chain integrity.
//	ok, _ := t.Verify(evalHash)
//
// (Stability: alpha)
package provenance
