// Package aws provides AWS Marketplace Metering API integration for Zerfoo.
// It uses an interface-based client so no AWS SDK dependency is required at compile time.
package aws

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Dimension names for AWS Marketplace metering.
const (
	DimensionInferenceRequests = "inference-requests"
	DimensionTokensProcessed   = "tokens-processed"
	DimensionGPUHours          = "gpu-hours"
)

// UsageRecord represents a single metering record sent to AWS Marketplace.
type UsageRecord struct {
	// Timestamp is the time the usage occurred (must be within the past hour).
	Timestamp time.Time `json:"timestamp"`
	// CustomerIdentifier is the AWS Marketplace customer identifier.
	CustomerIdentifier string `json:"customerIdentifier"`
	// Dimension is one of the defined metering dimensions.
	Dimension string `json:"dimension"`
	// Quantity is the amount of usage for the given dimension.
	Quantity int64 `json:"quantity"`
}

// MeteringPayload is the request body sent to the Marketplace Metering API.
type MeteringPayload struct {
	// ProductCode identifies the product in AWS Marketplace.
	ProductCode string `json:"productCode"`
	// UsageRecords contains one or more usage records to report.
	UsageRecords []UsageRecord `json:"usageRecords"`
}

// MeteringResult holds the response from a BatchMeterUsage call.
type MeteringResult struct {
	Results []struct {
		UsageRecord UsageRecord `json:"usageRecord"`
		Status      string      `json:"status"` // "Success", "CustomerNotSubscribed", "DuplicateRecord"
	} `json:"results"`
	UnprocessedRecords []UsageRecord `json:"unprocessedRecords"`
}

// MeteringClient defines the interface for AWS Marketplace metering operations.
// Implementing this interface without the AWS SDK allows injecting mocks in tests.
type MeteringClient interface {
	// BatchMeterUsage sends usage records to the AWS Marketplace Metering Service.
	BatchMeterUsage(ctx context.Context, payload MeteringPayload) (*MeteringResult, error)
}

// HTTPMeteringClient is a lightweight MeteringClient that calls the Marketplace
// Metering endpoint directly over HTTP. Replace with an SDK-based client in
// production by implementing the MeteringClient interface with the AWS SDK.
type HTTPMeteringClient struct {
	// Endpoint is the AWS Marketplace Metering Service endpoint URL.
	Endpoint string
	// ProductCode is the AWS Marketplace product code.
	ProductCode string
	// HTTPClient is the underlying HTTP client. Defaults to http.DefaultClient.
	HTTPClient *http.Client
}

// NewHTTPMeteringClient creates a new HTTPMeteringClient with sensible defaults.
func NewHTTPMeteringClient(endpoint, productCode string) *HTTPMeteringClient {
	return &HTTPMeteringClient{
		Endpoint:    endpoint,
		ProductCode: productCode,
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// BatchMeterUsage implements MeteringClient.
func (c *HTTPMeteringClient) BatchMeterUsage(ctx context.Context, payload MeteringPayload) (*MeteringResult, error) {
	payload.ProductCode = c.ProductCode

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("metering: marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint+"/BatchMeterUsage", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("metering: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("metering: http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metering: unexpected status %d", resp.StatusCode)
	}

	var result MeteringResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("metering: decode response: %w", err)
	}

	return &result, nil
}

// Meter aggregates usage and reports it via a MeteringClient.
type Meter struct {
	client             MeteringClient
	productCode        string
	customerIdentifier string
}

// NewMeter creates a Meter backed by the given MeteringClient.
func NewMeter(client MeteringClient, productCode, customerIdentifier string) *Meter {
	return &Meter{
		client:             client,
		productCode:        productCode,
		customerIdentifier: customerIdentifier,
	}
}

// RecordInferenceRequests reports a count of inference requests to Marketplace.
func (m *Meter) RecordInferenceRequests(ctx context.Context, count int64) error {
	return m.record(ctx, DimensionInferenceRequests, count)
}

// RecordTokensProcessed reports a count of tokens processed to Marketplace.
func (m *Meter) RecordTokensProcessed(ctx context.Context, count int64) error {
	return m.record(ctx, DimensionTokensProcessed, count)
}

// RecordGPUHours reports GPU hours consumed (in fractional seconds stored as
// hundredths of an hour, rounded to integer) to Marketplace.
func (m *Meter) RecordGPUHours(ctx context.Context, hours int64) error {
	return m.record(ctx, DimensionGPUHours, hours)
}

func (m *Meter) record(ctx context.Context, dimension string, quantity int64) error {
	payload := MeteringPayload{
		ProductCode: m.productCode,
		UsageRecords: []UsageRecord{
			{
				Timestamp:          time.Now().UTC(),
				CustomerIdentifier: m.customerIdentifier,
				Dimension:          dimension,
				Quantity:           quantity,
			},
		},
	}
	result, err := m.client.BatchMeterUsage(ctx, payload)
	if err != nil {
		return err
	}
	if len(result.UnprocessedRecords) > 0 {
		return fmt.Errorf("metering: %d records unprocessed", len(result.UnprocessedRecords))
	}
	return nil
}

// ValidateDimension returns an error if name is not a known metering dimension.
func ValidateDimension(name string) error {
	switch name {
	case DimensionInferenceRequests, DimensionTokensProcessed, DimensionGPUHours:
		return nil
	default:
		return fmt.Errorf("metering: unknown dimension %q", name)
	}
}
