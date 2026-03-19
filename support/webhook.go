package support

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// EventType identifies what happened to a ticket.
type EventType string

const (
	EventTicketCreated    EventType = "ticket.created"
	EventTicketTriaged    EventType = "ticket.triaged"
	EventTicketResolved   EventType = "ticket.resolved"
	EventTicketClosed     EventType = "ticket.closed"
	EventCommentAdded     EventType = "ticket.comment_added"
	EventSLABreach        EventType = "sla.breach"
)

// WebhookEvent is the payload sent to registered webhook endpoints.
type WebhookEvent struct {
	Type      EventType   `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Payload   interface{} `json:"payload"`
}

// WebhookTarget is a registered endpoint for receiving notifications.
type WebhookTarget struct {
	Name   string   // e.g. "slack", "pagerduty"
	URL    string
	Events []EventType // empty means all events
}

// shouldFire reports whether this target wants the given event type.
func (wt *WebhookTarget) shouldFire(e EventType) bool {
	if len(wt.Events) == 0 {
		return true
	}
	for _, ev := range wt.Events {
		if ev == e {
			return true
		}
	}
	return false
}

// WebhookDispatcher sends events to registered webhook targets.
type WebhookDispatcher struct {
	mu      sync.RWMutex
	targets []WebhookTarget
	client  *http.Client
}

// NewWebhookDispatcher creates a dispatcher with a default HTTP client.
func NewWebhookDispatcher() *WebhookDispatcher {
	return &WebhookDispatcher{
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

// Register adds a webhook target.
func (d *WebhookDispatcher) Register(target WebhookTarget) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.targets = append(d.targets, target)
}

// Targets returns the list of registered webhook targets.
func (d *WebhookDispatcher) Targets() []WebhookTarget {
	d.mu.RLock()
	defer d.mu.RUnlock()
	result := make([]WebhookTarget, len(d.targets))
	copy(result, d.targets)
	return result
}

// Dispatch sends an event to all matching targets. It returns errors for
// any failed deliveries but does not stop on first failure.
func (d *WebhookDispatcher) Dispatch(ctx context.Context, event WebhookEvent) []error {
	d.mu.RLock()
	targets := make([]WebhookTarget, len(d.targets))
	copy(targets, d.targets)
	d.mu.RUnlock()

	body, err := json.Marshal(event)
	if err != nil {
		return []error{fmt.Errorf("marshal webhook event: %w", err)}
	}

	var errs []error
	for _, t := range targets {
		if !t.shouldFire(event.Type) {
			continue
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.URL, bytes.NewReader(body))
		if err != nil {
			errs = append(errs, fmt.Errorf("webhook %s: %w", t.Name, err))
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := d.client.Do(req)
		if err != nil {
			errs = append(errs, fmt.Errorf("webhook %s: %w", t.Name, err))
			continue
		}
		resp.Body.Close()
		if resp.StatusCode >= 400 {
			errs = append(errs, fmt.Errorf("webhook %s: HTTP %d", t.Name, resp.StatusCode))
		}
	}
	return errs
}
