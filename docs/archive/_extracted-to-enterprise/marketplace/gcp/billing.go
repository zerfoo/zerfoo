package gcp

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DimensionTokens is the metering metric name for token-based billing.
const DimensionTokens = "zerfoo/tokens_1m"

// TokenUsageRecord tracks token usage for a consumer within a billing period.
type TokenUsageRecord struct {
	ConsumerID   string    `json:"consumerId"`
	InputTokens  int64     `json:"inputTokens"`
	OutputTokens int64     `json:"outputTokens"`
	TotalTokens  int64     `json:"totalTokens"`
	Timestamp    time.Time `json:"timestamp"`
}

// TokenBillingTracker accumulates token usage per consumer and flushes
// metering records to GCP via the Service Control API.
type TokenBillingTracker struct {
	serviceName string
	metering    ServiceControlAPI

	mu    sync.Mutex
	usage map[string]*tokenAccumulator
}

type tokenAccumulator struct {
	inputTokens  int64
	outputTokens int64
}

// NewTokenBillingTracker creates a new TokenBillingTracker for the given service.
func NewTokenBillingTracker(serviceName string, metering ServiceControlAPI) *TokenBillingTracker {
	return &TokenBillingTracker{
		serviceName: serviceName,
		metering:    metering,
		usage:       make(map[string]*tokenAccumulator),
	}
}

// RecordUsage adds token usage for a consumer.
func (t *TokenBillingTracker) RecordUsage(consumerID string, inputTokens, outputTokens int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	acc, ok := t.usage[consumerID]
	if !ok {
		acc = &tokenAccumulator{}
		t.usage[consumerID] = acc
	}
	acc.inputTokens += inputTokens
	acc.outputTokens += outputTokens
}

// Flush sends accumulated usage to GCP via the Service Control API
// and resets the accumulators. Usage is billed per 1M tokens.
func (t *TokenBillingTracker) Flush(ctx context.Context) (int, error) {
	t.mu.Lock()
	ops := make([]Operation, 0, len(t.usage))
	now := time.Now()

	for consumerID, acc := range t.usage {
		totalTokens := acc.inputTokens + acc.outputTokens
		// Bill per 1M tokens, rounding up.
		quantity := (totalTokens + 999_999) / 1_000_000
		if quantity <= 0 {
			continue
		}
		ops = append(ops, Operation{
			OperationID:   fmt.Sprintf("zerfoo-%s-%d", consumerID, now.UnixNano()),
			OperationName: "zerfoo.usage.report",
			ConsumerID:    consumerID,
			StartTime:     now,
			EndTime:       now,
			MetricValues: []MetricValueSet{
				{
					MetricName: DimensionTokens,
					MetricValues: []MetricValue{
						{Int64Value: &quantity},
					},
				},
			},
		})
	}

	// Reset accumulators.
	t.usage = make(map[string]*tokenAccumulator)
	t.mu.Unlock()

	if len(ops) == 0 {
		return 0, nil
	}

	if err := t.metering.Report(ctx, t.serviceName, ops); err != nil {
		return 0, fmt.Errorf("flush token billing: %w", err)
	}

	return len(ops), nil
}

// Snapshot returns the current accumulated usage per consumer without
// resetting the accumulators.
func (t *TokenBillingTracker) Snapshot() []TokenUsageRecord {
	t.mu.Lock()
	defer t.mu.Unlock()

	out := make([]TokenUsageRecord, 0, len(t.usage))
	now := time.Now()
	for consumerID, acc := range t.usage {
		out = append(out, TokenUsageRecord{
			ConsumerID:   consumerID,
			InputTokens:  acc.inputTokens,
			OutputTokens: acc.outputTokens,
			TotalTokens:  acc.inputTokens + acc.outputTokens,
			Timestamp:    now,
		})
	}
	return out
}
