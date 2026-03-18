package online

import (
	"path/filepath"
	"testing"
	"time"
)

func TestAuditLogWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	al, err := NewAuditLog(path)
	if err != nil {
		t.Fatal(err)
	}
	defer al.Close()

	events := []AuditEvent{
		{Timestamp: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), EventType: EventTrigger, Details: map[string]any{"drift": 0.15}, Outcome: "fired"},
		{Timestamp: time.Date(2026, 1, 1, 1, 0, 0, 0, time.UTC), EventType: EventUpdate, Details: map[string]any{"lr": 1e-4}, Outcome: "applied"},
		{Timestamp: time.Date(2026, 1, 1, 2, 0, 0, 0, time.UTC), EventType: EventRollback, Outcome: "reverted"},
	}
	for _, ev := range events {
		if err := al.Log(ev); err != nil {
			t.Fatal(err)
		}
	}

	got, err := al.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("expected 3 events, got %d", len(got))
	}
	if got[0].EventType != EventTrigger {
		t.Errorf("event 0: want type %q, got %q", EventTrigger, got[0].EventType)
	}
	if got[1].Outcome != "applied" {
		t.Errorf("event 1: want outcome %q, got %q", "applied", got[1].Outcome)
	}
	if got[2].EventType != EventRollback {
		t.Errorf("event 2: want type %q, got %q", EventRollback, got[2].EventType)
	}
}

func TestAuditLogRead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	al, err := NewAuditLog(path)
	if err != nil {
		t.Fatal(err)
	}
	defer al.Close()

	allTypes := []string{EventTrigger, EventUpdate, EventRollback, EventValidation}
	ts := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)

	for i, et := range allTypes {
		ev := AuditEvent{
			Timestamp: ts.Add(time.Duration(i) * time.Hour),
			EventType: et,
			Details:   map[string]any{"index": float64(i)},
			Outcome:   "ok",
		}
		if err := al.Log(ev); err != nil {
			t.Fatal(err)
		}
	}

	got, err := al.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(allTypes) {
		t.Fatalf("expected %d events, got %d", len(allTypes), len(got))
	}
	for i, et := range allTypes {
		if got[i].EventType != et {
			t.Errorf("event %d: want type %q, got %q", i, et, got[i].EventType)
		}
		if got[i].Outcome != "ok" {
			t.Errorf("event %d: want outcome %q, got %q", i, "ok", got[i].Outcome)
		}
		idx, ok := got[i].Details["index"].(float64)
		if !ok || idx != float64(i) {
			t.Errorf("event %d: want details index %d, got %v", i, i, got[i].Details["index"])
		}
		if !got[i].Timestamp.Equal(ts.Add(time.Duration(i) * time.Hour)) {
			t.Errorf("event %d: timestamp mismatch", i)
		}
	}
}

func TestAuditLogPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")

	// Write events and close.
	al, err := NewAuditLog(path)
	if err != nil {
		t.Fatal(err)
	}
	ev1 := AuditEvent{
		Timestamp: time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC),
		EventType: EventValidation,
		Details:   map[string]any{"metric": "accuracy"},
		Outcome:   "passed",
	}
	ev2 := AuditEvent{
		Timestamp: time.Date(2026, 6, 1, 1, 0, 0, 0, time.UTC),
		EventType: EventUpdate,
		Outcome:   "applied",
	}
	if err := al.Log(ev1); err != nil {
		t.Fatal(err)
	}
	if err := al.Log(ev2); err != nil {
		t.Fatal(err)
	}
	if err := al.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen and verify persistence.
	al2, err := NewAuditLog(path)
	if err != nil {
		t.Fatal(err)
	}
	defer al2.Close()

	got, err := al2.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 events after reopen, got %d", len(got))
	}
	if got[0].EventType != EventValidation {
		t.Errorf("event 0: want type %q, got %q", EventValidation, got[0].EventType)
	}
	if got[0].Outcome != "passed" {
		t.Errorf("event 0: want outcome %q, got %q", "passed", got[0].Outcome)
	}
	if got[1].EventType != EventUpdate {
		t.Errorf("event 1: want type %q, got %q", EventUpdate, got[1].EventType)
	}
}
