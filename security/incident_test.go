package security

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestIncidentResponderReport(t *testing.T) {
	var called atomic.Int32
	hook := func(_ context.Context, inc Incident) error {
		called.Add(1)
		if inc.Severity != IncidentHigh {
			t.Errorf("expected high severity, got %s", inc.Severity)
		}
		return nil
	}

	ir := NewIncidentResponder([]AlertHook{hook})
	err := ir.Report(context.Background(), Incident{
		Severity: IncidentHigh,
		Source:   "auth",
		Message:  "invalid API key",
		ClientIP: "1.2.3.4",
	})
	if err != nil {
		t.Fatal(err)
	}
	if called.Load() != 1 {
		t.Fatalf("expected hook called once, got %d", called.Load())
	}
}

func TestIncidentResponderAutoLockout(t *testing.T) {
	filter := NewIPFilter(nil, nil)
	ir := NewIncidentResponder(nil, WithLockout(3, time.Minute, filter))

	ctx := context.Background()
	ip := "10.0.0.99"

	for i := 0; i < 3; i++ {
		ir.Report(ctx, Incident{ClientIP: ip, Severity: IncidentMedium})
	}

	if !ir.IsLockedOut(ip) {
		t.Fatal("IP should be locked out after 3 incidents")
	}
	if filter.Allowed(ip) {
		t.Fatal("IP should be denied in filter")
	}
}

func TestIncidentResponderLockoutExpiry(t *testing.T) {
	filter := NewIPFilter(nil, nil)
	ir := NewIncidentResponder(nil, WithLockout(1, time.Millisecond, filter))

	ctx := context.Background()
	ip := "10.0.0.50"
	ir.Report(ctx, Incident{ClientIP: ip, Severity: IncidentLow})

	time.Sleep(5 * time.Millisecond)

	if ir.IsLockedOut(ip) {
		t.Fatal("lockout should have expired")
	}
}

func TestIncidentResponderResetLockout(t *testing.T) {
	filter := NewIPFilter(nil, nil)
	ir := NewIncidentResponder(nil, WithLockout(1, time.Hour, filter))

	ctx := context.Background()
	ip := "10.0.0.77"
	ir.Report(ctx, Incident{ClientIP: ip, Severity: IncidentCritical})

	if !ir.IsLockedOut(ip) {
		t.Fatal("should be locked out")
	}

	ir.ResetLockout(ip)
	if ir.IsLockedOut(ip) {
		t.Fatal("should not be locked out after reset")
	}
	if !filter.Allowed(ip) {
		t.Fatal("IP should be allowed after reset")
	}
}

func TestIncidentResponderNoLockoutConfig(t *testing.T) {
	ir := NewIncidentResponder(nil)
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		ir.Report(ctx, Incident{ClientIP: "1.1.1.1"})
	}
	if ir.IsLockedOut("1.1.1.1") {
		t.Fatal("should not lock out without lockout config")
	}
}
