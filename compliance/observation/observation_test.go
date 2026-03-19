package observation

import (
	"testing"
	"time"
)

func TestPeriodLifecycle(t *testing.T) {
	p := NewPeriod()

	if got := p.Phase(); got != PhaseNotStarted {
		t.Fatalf("expected phase %s, got %s", PhaseNotStarted, got)
	}

	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)
	if err := p.Start(start, 0); err != nil {
		t.Fatalf("Start: %v", err)
	}

	if got := p.Phase(); got != PhaseActive {
		t.Fatalf("expected phase %s, got %s", PhaseActive, got)
	}

	// Minimum duration should be enforced.
	expectedEnd := start.Add(MinimumDuration)
	if got := p.EndDate(); !got.Equal(expectedEnd) {
		t.Fatalf("expected end date %s, got %s", expectedEnd, got)
	}

	// Cannot start twice.
	if err := p.Start(start, 0); err == nil {
		t.Fatal("expected error starting already-started period")
	}

	// Cannot complete before minimum duration.
	tooEarly := start.Add(30 * 24 * time.Hour)
	if err := p.Complete(tooEarly); err == nil {
		t.Fatal("expected error completing before minimum duration")
	}

	// Complete after minimum duration.
	afterMin := start.Add(MinimumDuration + 24*time.Hour)
	if err := p.Complete(afterMin); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if got := p.Phase(); got != PhaseCompleted {
		t.Fatalf("expected phase %s, got %s", PhaseCompleted, got)
	}
}

func TestPeriodCustomDuration(t *testing.T) {
	p := NewPeriod()
	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)
	customDuration := 365 * 24 * time.Hour // 1 year

	if err := p.Start(start, customDuration); err != nil {
		t.Fatalf("Start: %v", err)
	}

	expectedEnd := start.Add(customDuration)
	if got := p.EndDate(); !got.Equal(expectedEnd) {
		t.Fatalf("expected end date %s, got %s", expectedEnd, got)
	}
}

func TestPeriodMilestones(t *testing.T) {
	p := NewPeriod()
	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)

	// Cannot add milestone before starting.
	err := p.AddMilestone(Milestone{Name: "test"})
	if err == nil {
		t.Fatal("expected error adding milestone to not-started period")
	}

	if err := p.Start(start, 0); err != nil {
		t.Fatalf("Start: %v", err)
	}

	m := Milestone{
		Name:        "midpoint-review",
		Description: "3-month midpoint review",
		DueDate:     start.Add(90 * 24 * time.Hour),
	}
	if err := p.AddMilestone(m); err != nil {
		t.Fatalf("AddMilestone: %v", err)
	}

	milestones := p.Milestones()
	if len(milestones) != 1 {
		t.Fatalf("expected 1 milestone, got %d", len(milestones))
	}

	completedAt := start.Add(88 * 24 * time.Hour)
	if err := p.CompleteMilestone("midpoint-review", completedAt); err != nil {
		t.Fatalf("CompleteMilestone: %v", err)
	}

	milestones = p.Milestones()
	if !milestones[0].Completed {
		t.Fatal("expected milestone to be completed")
	}

	// Unknown milestone.
	if err := p.CompleteMilestone("nonexistent", completedAt); err == nil {
		t.Fatal("expected error completing nonexistent milestone")
	}
}

func TestPeriodRemainingDays(t *testing.T) {
	p := NewPeriod()
	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)

	if got := p.RemainingDays(start); got != 0 {
		t.Fatalf("expected 0 remaining days for not-started, got %d", got)
	}

	if err := p.Start(start, 0); err != nil {
		t.Fatalf("Start: %v", err)
	}

	now := start.Add(10 * 24 * time.Hour)
	remaining := p.RemainingDays(now)
	expected := int(MinimumDuration/(24*time.Hour)) - 10
	if remaining != expected {
		t.Fatalf("expected %d remaining days, got %d", expected, remaining)
	}
}

func TestMonitorRunDue(t *testing.T) {
	m := NewMonitor()

	calls := 0
	check := ScheduledCheck{
		ControlID: "AC-001",
		Name:      "access-control-check",
		Interval:  24 * time.Hour,
		Check: func() (bool, string) {
			calls++
			return true, "access controls verified"
		},
	}

	if err := m.Register(check); err != nil {
		t.Fatalf("Register: %v", err)
	}

	now := time.Date(2028, 7, 1, 12, 0, 0, 0, time.UTC)

	// First run should execute.
	results := m.RunDue(now)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if !results[0].Passed {
		t.Fatal("expected check to pass")
	}

	// Same time, should not execute again.
	results = m.RunDue(now)
	if len(results) != 0 {
		t.Fatalf("expected 0 results (not due), got %d", len(results))
	}

	// After interval, should execute again.
	later := now.Add(25 * time.Hour)
	results = m.RunDue(later)
	if len(results) != 1 {
		t.Fatalf("expected 1 result after interval, got %d", len(results))
	}

	if calls != 2 {
		t.Fatalf("expected 2 total calls, got %d", calls)
	}

	// All results should be recorded.
	all := m.Results()
	if len(all) != 2 {
		t.Fatalf("expected 2 total results, got %d", len(all))
	}
}

func TestMonitorRunAll(t *testing.T) {
	m := NewMonitor()

	for _, id := range []string{"AC-001", "AC-002"} {
		cid := id
		err := m.Register(ScheduledCheck{
			ControlID: cid,
			Name:      cid + "-check",
			Interval:  24 * time.Hour,
			Check:     func() (bool, string) { return true, "ok" },
		})
		if err != nil {
			t.Fatalf("Register: %v", err)
		}
	}

	now := time.Date(2028, 7, 1, 12, 0, 0, 0, time.UTC)
	results := m.RunAll(now)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestMonitorRegisterValidation(t *testing.T) {
	m := NewMonitor()

	if err := m.Register(ScheduledCheck{ControlID: "", Interval: time.Hour, Check: func() (bool, string) { return true, "" }}); err == nil {
		t.Fatal("expected error for empty ControlID")
	}

	if err := m.Register(ScheduledCheck{ControlID: "AC-001", Interval: time.Hour}); err == nil {
		t.Fatal("expected error for nil Check")
	}

	if err := m.Register(ScheduledCheck{ControlID: "AC-001", Interval: 0, Check: func() (bool, string) { return true, "" }}); err == nil {
		t.Fatal("expected error for zero interval")
	}
}

func TestEvidenceAccumulator(t *testing.T) {
	a := NewEvidenceAccumulator()

	e1 := EvidenceItem{
		ID:          "ev-001",
		ControlID:   "AC-001",
		Description: "access log export",
		Collector:   "automated",
		Content:     []byte("log data"),
		Tags:        []string{"access", "log"},
	}
	if err := a.Add(e1); err != nil {
		t.Fatalf("Add: %v", err)
	}

	e2 := EvidenceItem{
		ID:          "ev-002",
		ControlID:   "AC-001",
		Description: "config snapshot",
		Collector:   "automated",
		Content:     []byte("config data"),
	}
	if err := a.Add(e2); err != nil {
		t.Fatalf("Add: %v", err)
	}

	e3 := EvidenceItem{
		ID:          "ev-003",
		ControlID:   "ENC-001",
		Description: "encryption verification",
		Collector:   "automated",
		Content:     []byte("encryption ok"),
	}
	if err := a.Add(e3); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if got := a.Count(); got != 3 {
		t.Fatalf("expected 3 items, got %d", got)
	}

	if got := a.CountForControl("AC-001"); got != 2 {
		t.Fatalf("expected 2 items for AC-001, got %d", got)
	}

	items := a.ForControl("AC-001")
	if len(items) != 2 {
		t.Fatalf("expected 2 items for AC-001, got %d", len(items))
	}

	// Hash should be computed.
	if items[0].Hash == "" {
		t.Fatal("expected hash to be computed")
	}

	ids := a.ControlIDs()
	if len(ids) != 2 {
		t.Fatalf("expected 2 control IDs, got %d", len(ids))
	}
}

func TestEvidenceAccumulatorValidation(t *testing.T) {
	a := NewEvidenceAccumulator()

	if err := a.Add(EvidenceItem{ControlID: "AC-001"}); err == nil {
		t.Fatal("expected error for empty ID")
	}
	if err := a.Add(EvidenceItem{ID: "ev-001"}); err == nil {
		t.Fatal("expected error for empty ControlID")
	}
}

func TestDeviationTracker(t *testing.T) {
	dt := NewDeviationTracker()

	d := Deviation{
		ID:          "dev-001",
		ControlID:   "AC-001",
		Severity:    SeverityHigh,
		Description: "unauthorized access detected",
		RootCause:   "misconfigured ACL",
	}
	if err := dt.Record(d); err != nil {
		t.Fatalf("Record: %v", err)
	}

	if got := dt.Count(); got != 1 {
		t.Fatalf("expected 1 deviation, got %d", got)
	}
	if got := dt.OpenCount(); got != 1 {
		t.Fatalf("expected 1 open deviation, got %d", got)
	}

	// Resolve the deviation.
	resolvedAt := time.Now()
	if err := dt.Resolve("dev-001", DeviationRemediated, "ACL fixed", "admin", resolvedAt); err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	if got := dt.OpenCount(); got != 0 {
		t.Fatalf("expected 0 open deviations, got %d", got)
	}

	all := dt.All()
	if all[0].Status != DeviationRemediated {
		t.Fatalf("expected status %s, got %s", DeviationRemediated, all[0].Status)
	}

	// ForControl.
	forCtrl := dt.ForControl("AC-001")
	if len(forCtrl) != 1 {
		t.Fatalf("expected 1 deviation for AC-001, got %d", len(forCtrl))
	}

	// Open filter.
	open := dt.Open()
	if len(open) != 0 {
		t.Fatalf("expected 0 open deviations, got %d", len(open))
	}
}

func TestDeviationTrackerValidation(t *testing.T) {
	dt := NewDeviationTracker()

	if err := dt.Record(Deviation{ControlID: "AC-001"}); err == nil {
		t.Fatal("expected error for empty ID")
	}
	if err := dt.Record(Deviation{ID: "dev-001"}); err == nil {
		t.Fatal("expected error for empty ControlID")
	}
	if err := dt.Resolve("nonexistent", DeviationRemediated, "", "", time.Now()); err == nil {
		t.Fatal("expected error resolving nonexistent deviation")
	}
}

func TestGenerateReport(t *testing.T) {
	period := NewPeriod()
	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)
	if err := period.Start(start, 0); err != nil {
		t.Fatalf("Start: %v", err)
	}

	if err := period.AddMilestone(Milestone{
		Name:    "midpoint",
		DueDate: start.Add(90 * 24 * time.Hour),
	}); err != nil {
		t.Fatalf("AddMilestone: %v", err)
	}

	monitor := NewMonitor()
	passCheck := ScheduledCheck{
		ControlID: "AC-001",
		Name:      "access-check",
		Interval:  24 * time.Hour,
		Check:     func() (bool, string) { return true, "pass" },
	}
	failCheck := ScheduledCheck{
		ControlID: "AC-002",
		Name:      "encryption-check",
		Interval:  24 * time.Hour,
		Check:     func() (bool, string) { return false, "encryption config missing" },
	}
	if err := monitor.Register(passCheck); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if err := monitor.Register(failCheck); err != nil {
		t.Fatalf("Register: %v", err)
	}

	checkTime := start.Add(24 * time.Hour)
	monitor.RunAll(checkTime)

	evidence := NewEvidenceAccumulator()
	if err := evidence.Add(EvidenceItem{
		ID:          "ev-001",
		ControlID:   "AC-001",
		Description: "access log",
		Content:     []byte("log"),
		CollectedAt: checkTime,
	}); err != nil {
		t.Fatalf("Add evidence: %v", err)
	}

	deviations := NewDeviationTracker()
	if err := deviations.Record(Deviation{
		ID:          "dev-001",
		ControlID:   "AC-002",
		Severity:    SeverityMedium,
		Description: "encryption not enabled",
		DetectedAt:  checkTime,
	}); err != nil {
		t.Fatalf("Record deviation: %v", err)
	}

	report := GenerateReport(period, monitor, evidence, deviations)

	if report.Phase != PhaseActive {
		t.Fatalf("expected active phase, got %s", report.Phase)
	}
	if report.TotalChecks != 2 {
		t.Fatalf("expected 2 total checks, got %d", report.TotalChecks)
	}
	if report.TotalPassed != 1 {
		t.Fatalf("expected 1 passed, got %d", report.TotalPassed)
	}
	if report.TotalFailed != 1 {
		t.Fatalf("expected 1 failed, got %d", report.TotalFailed)
	}
	if report.TotalDeviations != 1 {
		t.Fatalf("expected 1 deviation, got %d", report.TotalDeviations)
	}
	if report.OpenDeviations != 1 {
		t.Fatalf("expected 1 open deviation, got %d", report.OpenDeviations)
	}
	if report.TotalEvidence != 1 {
		t.Fatalf("expected 1 evidence item, got %d", report.TotalEvidence)
	}
	if report.OverallEffectiveness != 0.5 {
		t.Fatalf("expected 50%% effectiveness, got %.1f%%", report.OverallEffectiveness*100)
	}

	// Verify String() doesn't panic and contains key info.
	s := report.String()
	if s == "" {
		t.Fatal("expected non-empty report string")
	}
	if len(report.Controls) != 2 {
		t.Fatalf("expected 2 controls in report, got %d", len(report.Controls))
	}
}

func TestReportStringFormat(t *testing.T) {
	period := NewPeriod()
	start := time.Date(2028, 7, 1, 0, 0, 0, 0, time.UTC)
	if err := period.Start(start, 0); err != nil {
		t.Fatalf("Start: %v", err)
	}

	report := GenerateReport(period, NewMonitor(), NewEvidenceAccumulator(), NewDeviationTracker())
	s := report.String()

	if s == "" {
		t.Fatal("expected non-empty report string")
	}
}
