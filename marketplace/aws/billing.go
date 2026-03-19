package aws

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// DimensionTokens is the metering dimension name for token-based billing.
const DimensionTokens = "tokens_1m"

// TokenUsageRecord tracks token usage for a customer within a billing period.
type TokenUsageRecord struct {
	CustomerIdentifier string    `json:"customerIdentifier"`
	InputTokens        int64     `json:"inputTokens"`
	OutputTokens       int64     `json:"outputTokens"`
	TotalTokens        int64     `json:"totalTokens"`
	Timestamp          time.Time `json:"timestamp"`
}

// TokenBillingTracker accumulates token usage per customer and flushes
// metering records to AWS Marketplace in batches.
type TokenBillingTracker struct {
	productCode string
	metering    MeteringAPI

	mu    sync.Mutex
	usage map[string]*tokenAccumulator
}

type tokenAccumulator struct {
	inputTokens  int64
	outputTokens int64
}

// NewTokenBillingTracker creates a new TokenBillingTracker for the given product.
func NewTokenBillingTracker(productCode string, metering MeteringAPI) *TokenBillingTracker {
	return &TokenBillingTracker{
		productCode: productCode,
		metering:    metering,
		usage:       make(map[string]*tokenAccumulator),
	}
}

// RecordUsage adds token usage for a customer.
func (t *TokenBillingTracker) RecordUsage(customerIdentifier string, inputTokens, outputTokens int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	acc, ok := t.usage[customerIdentifier]
	if !ok {
		acc = &tokenAccumulator{}
		t.usage[customerIdentifier] = acc
	}
	acc.inputTokens += inputTokens
	acc.outputTokens += outputTokens
}

// Flush sends accumulated usage to AWS Marketplace via BatchMeterUsage
// and resets the accumulators. Usage is billed per 1M tokens.
func (t *TokenBillingTracker) Flush(ctx context.Context) (*BatchMeterUsageOutput, error) {
	t.mu.Lock()
	records := make([]UsageRecord, 0, len(t.usage))
	now := time.Now()

	for customerID, acc := range t.usage {
		totalTokens := acc.inputTokens + acc.outputTokens
		// Bill per 1M tokens, rounding up.
		quantity := int((totalTokens + 999_999) / 1_000_000)
		if quantity <= 0 {
			continue
		}
		records = append(records, UsageRecord{
			CustomerIdentifier: customerID,
			Dimension:          DimensionTokens,
			Quantity:           quantity,
			Timestamp:          now,
		})
	}

	// Reset accumulators.
	t.usage = make(map[string]*tokenAccumulator)
	t.mu.Unlock()

	if len(records) == 0 {
		return &BatchMeterUsageOutput{}, nil
	}

	out, err := t.metering.BatchMeterUsage(ctx, &BatchMeterUsageInput{
		ProductCode:  t.productCode,
		UsageRecords: records,
	})
	if err != nil {
		return nil, fmt.Errorf("flush token billing: %w", err)
	}

	return out, nil
}

// Snapshot returns the current accumulated usage per customer without
// resetting the accumulators.
func (t *TokenBillingTracker) Snapshot() []TokenUsageRecord {
	t.mu.Lock()
	defer t.mu.Unlock()

	out := make([]TokenUsageRecord, 0, len(t.usage))
	now := time.Now()
	for customerID, acc := range t.usage {
		out = append(out, TokenUsageRecord{
			CustomerIdentifier: customerID,
			InputTokens:        acc.inputTokens,
			OutputTokens:       acc.outputTokens,
			TotalTokens:        acc.inputTokens + acc.outputTokens,
			Timestamp:          now,
		})
	}
	return out
}
