package provenance

import (
	"strings"
	"testing"
)

func TestProvenance_HashChain(t *testing.T) {
	tr := NewTracker()

	// Record a training run (root event, no parent).
	trainHash, err := tr.RecordTraining(TrainingRecord{
		RunID:        "run-001",
		ModelName:    "llama-3-8b",
		ModelVersion: "v1.0.0",
		Hyperparameters: map[string]string{
			"lr":     "3e-4",
			"epochs": "3",
		},
	})
	if err != nil {
		t.Fatalf("RecordTraining: %v", err)
	}
	if trainHash == "" {
		t.Fatal("expected non-empty training hash")
	}
	if len(trainHash) != 64 {
		t.Fatalf("expected 64-char hex SHA-256 hash, got %d chars", len(trainHash))
	}

	// Record a dataset linked to the training run.
	dataHash, err := tr.RecordDataset(DatasetRecord{
		Name:     "wiki-en",
		Version:  "2024-03",
		Checksum: "abc123",
		NumRows:  1000000,
		ParentID: trainHash,
	})
	if err != nil {
		t.Fatalf("RecordDataset: %v", err)
	}
	if dataHash == trainHash {
		t.Fatal("dataset hash should differ from training hash")
	}

	// Record an evaluation linked to the training run.
	evalHash, err := tr.RecordEvaluation(EvaluationRecord{
		ParentID: trainHash,
		Metrics:  map[string]float64{"loss": 0.42, "accuracy": 0.91},
		Dataset:  "wiki-en-test",
		Split:    "test",
	})
	if err != nil {
		t.Fatalf("RecordEvaluation: %v", err)
	}
	if evalHash == trainHash || evalHash == dataHash {
		t.Fatal("evaluation hash should be unique")
	}

	// Verify integrity from evaluation back to root.
	ok, err := tr.Verify(evalHash)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if !ok {
		t.Fatal("expected hash chain to verify successfully")
	}

	// Verify integrity from dataset back to root.
	ok, err = tr.Verify(dataHash)
	if err != nil {
		t.Fatalf("Verify: %v", err)
	}
	if !ok {
		t.Fatal("expected hash chain to verify from dataset")
	}

	// Verify the root event itself.
	ok, err = tr.Verify(trainHash)
	if err != nil {
		t.Fatalf("Verify root: %v", err)
	}
	if !ok {
		t.Fatal("expected root event to verify")
	}

	// Chain a second evaluation to the first evaluation (deeper chain).
	eval2Hash, err := tr.RecordEvaluation(EvaluationRecord{
		ParentID: evalHash,
		Metrics:  map[string]float64{"f1": 0.88},
		Dataset:  "wiki-en-val",
		Split:    "validation",
	})
	if err != nil {
		t.Fatalf("RecordEvaluation (chained): %v", err)
	}

	ok, err = tr.Verify(eval2Hash)
	if err != nil {
		t.Fatalf("Verify chained: %v", err)
	}
	if !ok {
		t.Fatal("expected chained hash to verify")
	}

	// Tamper detection: corrupt a stored event hash.
	tr.mu.Lock()
	ev := tr.events[trainHash]
	ev.Hash = "0000000000000000000000000000000000000000000000000000000000000000"
	tr.events[trainHash] = ev
	tr.mu.Unlock()

	ok, err = tr.Verify(trainHash)
	if err != nil {
		t.Fatalf("Verify tampered: %v", err)
	}
	if ok {
		t.Fatal("expected tampered event to fail verification")
	}
}

func TestProvenance_DAGTraversal(t *testing.T) {
	tr := NewTracker()

	// Build a chain: training -> dataset -> evaluation.
	trainHash, err := tr.RecordTraining(TrainingRecord{
		RunID:        "run-dag",
		ModelName:    "gemma-3-1b",
		ModelVersion: "v2.0.0",
	})
	if err != nil {
		t.Fatalf("RecordTraining: %v", err)
	}

	dataHash, err := tr.RecordDataset(DatasetRecord{
		Name:     "openwebtext",
		Version:  "v1",
		ParentID: trainHash,
	})
	if err != nil {
		t.Fatalf("RecordDataset: %v", err)
	}

	evalHash, err := tr.RecordEvaluation(EvaluationRecord{
		ParentID: dataHash,
		Metrics:  map[string]float64{"perplexity": 12.5},
	})
	if err != nil {
		t.Fatalf("RecordEvaluation: %v", err)
	}

	// Trace from evaluation back to training.
	events, err := tr.Trace(evalHash)
	if err != nil {
		t.Fatalf("Trace: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("expected 3 events in trace, got %d", len(events))
	}

	// First event should be the evaluation (leaf).
	if events[0].Type != EventEvaluation {
		t.Errorf("expected first event to be evaluation, got %s", events[0].Type)
	}
	if events[0].Hash != evalHash {
		t.Error("first event hash mismatch")
	}

	// Second should be the dataset.
	if events[1].Type != EventDataset {
		t.Errorf("expected second event to be dataset, got %s", events[1].Type)
	}
	if events[1].Hash != dataHash {
		t.Error("second event hash mismatch")
	}

	// Third should be the training (root).
	if events[2].Type != EventTraining {
		t.Errorf("expected third event to be training, got %s", events[2].Type)
	}
	if events[2].Hash != trainHash {
		t.Error("third event hash mismatch")
	}

	// Root event should have no parent.
	if events[2].ParentID != "" {
		t.Error("root event should have empty ParentID")
	}

	// Trace from the root should return only the root.
	rootEvents, err := tr.Trace(trainHash)
	if err != nil {
		t.Fatalf("Trace root: %v", err)
	}
	if len(rootEvents) != 1 {
		t.Fatalf("expected 1 event in root trace, got %d", len(rootEvents))
	}

	// Trace a nonexistent hash should error.
	_, err = tr.Trace("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent hash")
	}

	// Error: reference a nonexistent parent.
	_, err = tr.RecordDataset(DatasetRecord{
		Name:     "bad",
		ParentID: "nonexistent-parent",
	})
	if err == nil {
		t.Fatal("expected error for nonexistent parent")
	}
	if !strings.Contains(err.Error(), "not found") {
		t.Errorf("expected 'not found' in error, got: %s", err)
	}
}

func TestProvenance_ValidationErrors(t *testing.T) {
	tr := NewTracker()

	// Training requires RunID.
	_, err := tr.RecordTraining(TrainingRecord{ModelName: "m"})
	if err == nil {
		t.Fatal("expected error for missing RunID")
	}

	// Training requires ModelName.
	_, err = tr.RecordTraining(TrainingRecord{RunID: "r"})
	if err == nil {
		t.Fatal("expected error for missing ModelName")
	}

	// Dataset requires Name.
	_, err = tr.RecordDataset(DatasetRecord{})
	if err == nil {
		t.Fatal("expected error for missing dataset Name")
	}

	// Verify nonexistent hash.
	_, err = tr.Verify("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent hash in Verify")
	}
}
