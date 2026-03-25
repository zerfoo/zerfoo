package support

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestWebhookDispatcher_SSRFBlocksLoopback(t *testing.T) {
	d := NewWebhookDispatcher()
	d.Register(WebhookTarget{
		Name: "loopback",
		URL:  "http://127.0.0.1:9999/hook",
	})

	event := WebhookEvent{
		Type:      EventTicketCreated,
		Timestamp: time.Now(),
		Payload:   map[string]string{"id": "1"},
	}

	errs := d.Dispatch(context.Background(), event)
	if len(errs) == 0 {
		t.Fatal("expected error for loopback address, got none")
	}
	for _, err := range errs {
		if !strings.Contains(err.Error(), "blocked SSRF target") {
			t.Errorf("expected SSRF block error, got: %v", err)
		}
	}
}

func TestWebhookDispatcher_SSRFBlocksMetadata(t *testing.T) {
	d := NewWebhookDispatcher()
	d.Register(WebhookTarget{
		Name: "metadata",
		URL:  "http://169.254.169.254/latest/meta-data/",
	})

	event := WebhookEvent{
		Type:      EventTicketCreated,
		Timestamp: time.Now(),
		Payload:   map[string]string{"id": "1"},
	}

	errs := d.Dispatch(context.Background(), event)
	if len(errs) == 0 {
		t.Fatal("expected error for metadata address, got none")
	}
	for _, err := range errs {
		if !strings.Contains(err.Error(), "blocked SSRF target") {
			t.Errorf("expected SSRF block error, got: %v", err)
		}
	}
}

func TestWebhookDispatcher_SSRFAllowsExternal(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	// Use a plain HTTP client (no SSRF protection) for this test because
	// httptest.NewServer binds to 127.0.0.1 which is intentionally blocked
	// by SSRF protection. This test validates delivery mechanics, not SSRF.
	d := &WebhookDispatcher{
		client: &http.Client{Timeout: 10 * time.Second},
	}
	d.Register(WebhookTarget{
		Name: "test",
		URL:  srv.URL + "/hook",
	})

	event := WebhookEvent{
		Type:      EventTicketCreated,
		Timestamp: time.Now(),
		Payload:   map[string]string{"id": "1"},
	}

	errs := d.Dispatch(context.Background(), event)
	if len(errs) != 0 {
		t.Fatalf("expected no errors, got: %v", errs)
	}
	if !called {
		t.Fatal("expected webhook handler to be called")
	}
}

func TestWebhookDispatcher_SSRFBlocksPrivateIPs(t *testing.T) {
	tests := []struct {
		name string
		ip   string
	}{
		{"10.x private", "10.0.0.1"},
		{"172.16.x private", "172.16.0.1"},
		{"192.168.x private", "192.168.1.1"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			d := NewWebhookDispatcher()
			d.Register(WebhookTarget{
				Name: "private",
				URL:  "http://" + tc.ip + ":9999/hook",
			})

			event := WebhookEvent{
				Type:      EventTicketCreated,
				Timestamp: time.Now(),
				Payload:   map[string]string{"id": "1"},
			}

			errs := d.Dispatch(context.Background(), event)
			if len(errs) == 0 {
				t.Fatalf("expected error for private address %s, got none", tc.ip)
			}
			for _, err := range errs {
				if !strings.Contains(err.Error(), "blocked SSRF target") {
					t.Errorf("expected SSRF block error, got: %v", err)
				}
			}
		})
	}
}

func TestIsBlockedWebhookIP(t *testing.T) {
	tests := []struct {
		ip      string
		blocked bool
	}{
		{"127.0.0.1", true},
		{"::1", true},
		{"10.0.0.1", true},
		{"172.16.0.1", true},
		{"192.168.1.1", true},
		{"169.254.169.254", true},
		{"169.254.1.1", true},  // link-local
		{"8.8.8.8", false},
		{"1.1.1.1", false},
	}

	for _, tc := range tests {
		t.Run(tc.ip, func(t *testing.T) {
			ip := net.ParseIP(tc.ip)
			if ip == nil {
				t.Fatalf("failed to parse IP: %s", tc.ip)
			}
			err := isBlockedWebhookIP(ip)
			if tc.blocked && err == nil {
				t.Errorf("expected %s to be blocked", tc.ip)
			}
			if !tc.blocked && err != nil {
				t.Errorf("expected %s to be allowed, got: %v", tc.ip, err)
			}
		})
	}
}
