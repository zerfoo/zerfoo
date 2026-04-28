package azure

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/zerfoo/zerfoo/marketplace"
)

// MeteringAPI abstracts Azure Marketplace Metering Service API calls.
type MeteringAPI interface {
	// PostUsageEvent submits a single usage event for a metered dimension.
	PostUsageEvent(ctx context.Context, event UsageEvent) (*UsageEventResult, error)

	// PostBatchUsageEvent submits a batch of usage events.
	PostBatchUsageEvent(ctx context.Context, events []UsageEvent) (*BatchUsageResult, error)
}

// UsageEvent represents a single metering event for Azure Marketplace.
type UsageEvent struct {
	ResourceID    string    `json:"resourceId"`
	Quantity      float64   `json:"quantity"`
	Dimension     string    `json:"dimension"`
	EffectiveTime time.Time `json:"effectiveStartTime"`
	PlanID        string    `json:"planId"`
}

// UsageEventResult contains the result of posting a single usage event.
type UsageEventResult struct {
	UsageEventID  string         `json:"usageEventId"`
	Status        MeteringStatus `json:"status"`
	Quantity      float64        `json:"quantity"`
	Dimension     string         `json:"dimension"`
	EffectiveTime time.Time      `json:"effectiveStartTime"`
	PlanID        string         `json:"planId"`
	MessageTime   time.Time      `json:"messageTime"`
}

// MeteringStatus represents the status of a metering event.
type MeteringStatus string

const (
	MeteringStatusAccepted         MeteringStatus = "Accepted"
	MeteringStatusDuplicate        MeteringStatus = "Duplicate"
	MeteringStatusExpired          MeteringStatus = "Expired"
	MeteringStatusResourceNotFound MeteringStatus = "ResourceNotFound"
)

// BatchUsageResult contains the result of posting a batch of usage events.
type BatchUsageResult struct {
	Results []UsageEventResult `json:"result"`
	Count   int                `json:"count"`
}

// DimensionTokens is the metering dimension name for token-based billing.
const DimensionTokens = "tokens_1m"

// MeteringClient implements MeteringAPI by calling the Azure Marketplace
// Metering Service REST endpoints.
type MeteringClient struct {
	// Endpoint is the Azure Marketplace Metering Service endpoint URL.
	// Default: https://marketplaceapi.microsoft.com/api
	Endpoint string

	// APIVersion is the API version query parameter.
	// Default: 2018-08-31
	APIVersion string

	// TokenProvider obtains bearer tokens for authentication.
	TokenProvider TokenProvider

	// HTTPClient is the HTTP client used for API calls.
	HTTPClient *http.Client

	// Retry configures exponential backoff for metering calls.
	// Zero value disables retry.
	Retry marketplace.RetryConfig
}

// NewMeteringClient creates a new MeteringClient with the given endpoint and token provider.
func NewMeteringClient(endpoint string, tokenProvider TokenProvider) *MeteringClient {
	return &MeteringClient{
		Endpoint:      endpoint,
		APIVersion:    "2018-08-31",
		TokenProvider: tokenProvider,
		HTTPClient:    &http.Client{Timeout: 30 * time.Second},
		Retry:         marketplace.DefaultRetryConfig(),
	}
}

// PostUsageEvent submits a single usage event.
func (c *MeteringClient) PostUsageEvent(ctx context.Context, event UsageEvent) (*UsageEventResult, error) {
	data, err := json.Marshal(event)
	if err != nil {
		return nil, fmt.Errorf("marshal usage event: %w", err)
	}

	var resp []byte
	err = marketplace.RetryFunc(ctx, c.Retry, "azure.PostUsageEvent", func() error {
		var reqErr error
		resp, reqErr = c.doRequest(ctx, http.MethodPost, "/usageEvent", data)
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("post usage event: %w", err)
	}

	var out UsageEventResult
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal usage event response: %w", err)
	}
	return &out, nil
}

// PostBatchUsageEvent submits a batch of usage events.
func (c *MeteringClient) PostBatchUsageEvent(ctx context.Context, events []UsageEvent) (*BatchUsageResult, error) {
	payload := struct {
		Request []UsageEvent `json:"request"`
	}{Request: events}

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal batch usage events: %w", err)
	}

	var resp []byte
	err = marketplace.RetryFunc(ctx, c.Retry, "azure.PostBatchUsageEvent", func() error {
		var reqErr error
		resp, reqErr = c.doRequest(ctx, http.MethodPost, "/batchUsageEvent", data)
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("post batch usage events: %w", err)
	}

	var out BatchUsageResult
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal batch usage response: %w", err)
	}
	return &out, nil
}

func (c *MeteringClient) doRequest(ctx context.Context, method, path string, body []byte) ([]byte, error) {
	url := c.Endpoint + path + "?api-version=" + c.APIVersion

	var bodyReader io.Reader
	if body != nil {
		bodyReader = &bytesReader{data: body}
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	if c.TokenProvider != nil {
		token, err := c.TokenProvider.Token(ctx)
		if err != nil {
			return nil, fmt.Errorf("obtain token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// TokenBillingTracker accumulates token usage per subscription and flushes
// metering records to Azure Marketplace in batches.
type TokenBillingTracker struct {
	planID   string
	metering MeteringAPI

	mu    sync.Mutex
	usage map[string]*tokenAccumulator
}

type tokenAccumulator struct {
	inputTokens  int64
	outputTokens int64
}

// NewTokenBillingTracker creates a new TokenBillingTracker for the given plan.
func NewTokenBillingTracker(planID string, metering MeteringAPI) *TokenBillingTracker {
	return &TokenBillingTracker{
		planID:   planID,
		metering: metering,
		usage:    make(map[string]*tokenAccumulator),
	}
}

// RecordUsage adds token usage for a subscription resource.
func (t *TokenBillingTracker) RecordUsage(resourceID string, inputTokens, outputTokens int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	acc, ok := t.usage[resourceID]
	if !ok {
		acc = &tokenAccumulator{}
		t.usage[resourceID] = acc
	}
	acc.inputTokens += inputTokens
	acc.outputTokens += outputTokens
}

// Flush sends accumulated usage to Azure Marketplace via PostBatchUsageEvent
// and resets the accumulators. Usage is billed per 1M tokens.
func (t *TokenBillingTracker) Flush(ctx context.Context) (*BatchUsageResult, error) {
	t.mu.Lock()
	events := make([]UsageEvent, 0, len(t.usage))
	now := time.Now()

	for resourceID, acc := range t.usage {
		totalTokens := acc.inputTokens + acc.outputTokens
		// Bill per 1M tokens, rounding up.
		quantity := float64((totalTokens + 999_999) / 1_000_000)
		if quantity <= 0 {
			continue
		}
		events = append(events, UsageEvent{
			ResourceID:    resourceID,
			Quantity:      quantity,
			Dimension:     DimensionTokens,
			EffectiveTime: now,
			PlanID:        t.planID,
		})
	}

	// Reset accumulators.
	t.usage = make(map[string]*tokenAccumulator)
	t.mu.Unlock()

	if len(events) == 0 {
		return &BatchUsageResult{}, nil
	}

	out, err := t.metering.PostBatchUsageEvent(ctx, events)
	if err != nil {
		return nil, fmt.Errorf("flush token billing: %w", err)
	}

	return out, nil
}

// Snapshot returns the current accumulated usage per subscription without
// resetting the accumulators.
func (t *TokenBillingTracker) Snapshot() []TokenUsageRecord {
	t.mu.Lock()
	defer t.mu.Unlock()

	out := make([]TokenUsageRecord, 0, len(t.usage))
	now := time.Now()
	for resourceID, acc := range t.usage {
		out = append(out, TokenUsageRecord{
			ResourceID:   resourceID,
			InputTokens:  acc.inputTokens,
			OutputTokens: acc.outputTokens,
			TotalTokens:  acc.inputTokens + acc.outputTokens,
			Timestamp:    now,
		})
	}
	return out
}

// TokenUsageRecord tracks token usage for a subscription within a billing period.
type TokenUsageRecord struct {
	ResourceID   string    `json:"resourceId"`
	InputTokens  int64     `json:"inputTokens"`
	OutputTokens int64     `json:"outputTokens"`
	TotalTokens  int64     `json:"totalTokens"`
	Timestamp    time.Time `json:"timestamp"`
}
