package support

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
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

// blockedWebhookHosts contains hostnames blocked to prevent SSRF attacks
// against cloud metadata services.
var blockedWebhookHosts = map[string]bool{
	"metadata.google.internal": true,
}

// blockedWebhookIPs contains IP addresses blocked to prevent SSRF attacks
// against cloud metadata services (e.g. AWS/GCP/Azure).
var blockedWebhookIPs = map[string]bool{
	"169.254.169.254": true,
}

// isBlockedWebhookIP checks whether an IP address should be blocked for
// SSRF protection. It blocks loopback, private, link-local, and known
// cloud metadata addresses.
func isBlockedWebhookIP(ip net.IP) error {
	ipStr := ip.String()
	if blockedWebhookIPs[ipStr] {
		return fmt.Errorf("blocked SSRF target: %s", ipStr)
	}
	if ip.IsLoopback() {
		return fmt.Errorf("blocked SSRF target: loopback address %s", ipStr)
	}
	if ip.IsPrivate() {
		return fmt.Errorf("blocked SSRF target: private address %s", ipStr)
	}
	if ip.IsLinkLocalUnicast() {
		return fmt.Errorf("blocked SSRF target: link-local unicast address %s", ipStr)
	}
	if ip.IsLinkLocalMulticast() {
		return fmt.Errorf("blocked SSRF target: link-local multicast address %s", ipStr)
	}
	return nil
}

// webhookSSRFDialContext returns a DialContext function that validates every
// resolved IP address against the SSRF blocklist before connecting. This
// prevents DNS rebinding attacks by checking the IP at connect time rather
// than in a separate pre-flight validation step.
func webhookSSRFDialContext(resolver *net.Resolver) func(ctx context.Context, network, addr string) (net.Conn, error) {
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(addr)
		if err != nil {
			return nil, fmt.Errorf("split host/port: %w", err)
		}

		// Block known dangerous hostnames.
		if blockedWebhookHosts[host] {
			return nil, fmt.Errorf("blocked SSRF target: %s", host)
		}

		// Resolve hostname to IPs.
		ips, err := resolver.LookupHost(ctx, host)
		if err != nil {
			return nil, fmt.Errorf("resolve hostname %q: %w", host, err)
		}

		// Check every resolved IP against the blocklist.
		for _, ipStr := range ips {
			ip := net.ParseIP(ipStr)
			if ip == nil {
				continue
			}
			if err := isBlockedWebhookIP(ip); err != nil {
				return nil, err
			}
		}

		// All IPs are safe — connect to the first one that works.
		var dialer net.Dialer
		for _, ipStr := range ips {
			conn, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(ipStr, port))
			if dialErr == nil {
				return conn, nil
			}
			err = dialErr
		}
		return nil, fmt.Errorf("dial %s: %w", addr, err)
	}
}

// WebhookDispatcher sends events to registered webhook targets.
type WebhookDispatcher struct {
	mu      sync.RWMutex
	targets []WebhookTarget
	client  *http.Client
}

// NewWebhookDispatcher creates a dispatcher with an SSRF-safe HTTP client
// that blocks connections to loopback, private, link-local, and cloud
// metadata IP addresses.
func NewWebhookDispatcher() *WebhookDispatcher {
	return &WebhookDispatcher{
		client: &http.Client{
			Timeout: 10 * time.Second,
			Transport: &http.Transport{
				DialContext: webhookSSRFDialContext(nil),
			},
		},
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
