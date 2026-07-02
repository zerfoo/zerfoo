package azure

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	// maxWebhookBodySize is the maximum allowed webhook request body size (1 MB).
	maxWebhookBodySize = 1 << 20

	// maxWebhookAge is the maximum age of a webhook timestamp before it is rejected.
	maxWebhookAge = 5 * time.Minute

	// deduplicationTTL is how long operation IDs are retained for replay detection.
	deduplicationTTL = 10 * time.Minute
)

// WebhookAction represents the type of subscription lifecycle event.
type WebhookAction string

const (
	ActionChangePlan     WebhookAction = "ChangePlan"
	ActionChangeQuantity WebhookAction = "ChangeQuantity"
	ActionSuspend        WebhookAction = "Suspend"
	ActionReinstate      WebhookAction = "Reinstate"
	ActionUnsubscribe    WebhookAction = "Unsubscribe"
	ActionRenew          WebhookAction = "Renew"
)

// WebhookStatus represents the status of the operation in the webhook payload.
type WebhookStatus string

const (
	WebhookStatusInProgress WebhookStatus = "InProgress"
	WebhookStatusSuccess    WebhookStatus = "Success"
	WebhookStatusFailure    WebhookStatus = "Failure"
)

// WebhookPayload represents the JSON body of an Azure Marketplace webhook notification.
type WebhookPayload struct {
	ID             string        `json:"id"`
	ActivityID     string        `json:"activityId"`
	SubscriptionID string        `json:"subscriptionId"`
	OfferID        string        `json:"offerId"`
	PublisherID    string        `json:"publisherId"`
	PlanID         string        `json:"planId"`
	Quantity       int           `json:"quantity,omitempty"`
	Action         WebhookAction `json:"action"`
	Timestamp      time.Time     `json:"timeStamp"`
	Status         WebhookStatus `json:"status"`
	OperationID    string        `json:"operationId"`
}

// WebhookHandler handles Azure Marketplace webhook notifications for
// subscription lifecycle events.
type WebhookHandler struct {
	// Secret is the shared secret for HMAC-SHA256 signature validation.
	// Must be non-empty; the handler returns 500 if not configured.
	Secret string

	// Manager handles subscription state transitions.
	Manager *SubscriptionManager

	// OnEvent is an optional callback invoked after processing each webhook event.
	// It can be used for logging, metrics, or custom business logic.
	OnEvent func(payload WebhookPayload, err error)

	// Now returns the current time. If nil, time.Now is used.
	// Exposed for testing timestamp validation.
	Now func() time.Time

	// seen tracks recently processed operation IDs for deduplication.
	seen sync.Map
}

// NewWebhookHandler creates a WebhookHandler with the given secret and subscription manager.
func NewWebhookHandler(secret string, manager *SubscriptionManager) *WebhookHandler {
	return &WebhookHandler{
		Secret:  secret,
		Manager: manager,
	}
}

func (h *WebhookHandler) now() time.Time {
	if h.Now != nil {
		return h.Now()
	}
	return time.Now()
}

// ServeHTTP implements http.Handler for Azure Marketplace webhook notifications.
func (h *WebhookHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.Secret == "" {
		http.Error(w, "webhook secret not configured", http.StatusInternalServerError)
		return
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxWebhookBodySize))
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}

	sig := r.Header.Get("x-ms-signature")
	if !h.verifySignature(body, sig) {
		http.Error(w, "invalid signature", http.StatusUnauthorized)
		return
	}

	var payload WebhookPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(w, "invalid payload", http.StatusBadRequest)
		return
	}

	// Reject webhooks with timestamps older than maxWebhookAge.
	if !payload.Timestamp.IsZero() && h.now().Sub(payload.Timestamp) > maxWebhookAge {
		http.Error(w, "webhook timestamp expired", http.StatusBadRequest)
		return
	}

	// Deduplicate by operation ID.
	if payload.OperationID != "" {
		if _, loaded := h.seen.LoadOrStore(payload.OperationID, h.now()); loaded {
			http.Error(w, "duplicate operation", http.StatusConflict)
			return
		}
		// Schedule cleanup after TTL.
		go func(opID string) {
			time.Sleep(deduplicationTTL)
			h.seen.Delete(opID)
		}(payload.OperationID)
	}

	ctx := r.Context()
	processErr := h.processEvent(ctx, payload)

	if h.OnEvent != nil {
		h.OnEvent(payload, processErr)
	}

	if processErr != nil {
		http.Error(w, "webhook processing failed", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func (h *WebhookHandler) processEvent(ctx context.Context, payload WebhookPayload) error {
	switch payload.Action {
	case ActionSuspend:
		return h.Manager.Suspend(ctx, payload.SubscriptionID)

	case ActionReinstate:
		return h.Manager.Reinstate(ctx, payload.SubscriptionID)

	case ActionUnsubscribe:
		return h.Manager.Unsubscribe(ctx, payload.SubscriptionID)

	case ActionChangePlan:
		_, err := h.Manager.ChangePlan(ctx, payload.SubscriptionID, payload.PlanID)
		return err

	case ActionChangeQuantity:
		_, err := h.Manager.ChangeQuantity(ctx, payload.SubscriptionID, payload.Quantity)
		return err

	case ActionRenew:
		// Renewal events are informational; no state change required.
		return nil

	default:
		return fmt.Errorf("unknown webhook action: %s", payload.Action)
	}
}

func (h *WebhookHandler) verifySignature(body []byte, signature string) bool {
	if signature == "" {
		return false
	}

	// Azure sends the signature as a hex-encoded HMAC-SHA256.
	sig, err := hex.DecodeString(strings.TrimPrefix(signature, "sha256="))
	if err != nil {
		return false
	}

	mac := hmac.New(sha256.New, []byte(h.Secret))
	mac.Write(body)
	expected := mac.Sum(nil)

	return hmac.Equal(sig, expected)
}
