// Package aws provides AWS Marketplace integration for Zerfoo Cloud,
// including metering, subscription lifecycle management, entitlement
// verification, and token-based billing.
package aws

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

// MeteringAPI abstracts AWS Marketplace Metering Service API calls.
type MeteringAPI interface {
	// ResolveCustomer resolves a registration token to a customer identifier
	// and product code. Called when a customer subscribes via AWS Marketplace.
	ResolveCustomer(ctx context.Context, registrationToken string) (*ResolveCustomerOutput, error)

	// BatchMeterUsage submits a batch of usage records for metered billing.
	BatchMeterUsage(ctx context.Context, input *BatchMeterUsageInput) (*BatchMeterUsageOutput, error)

	// MeterUsage submits a single usage record for metered billing.
	MeterUsage(ctx context.Context, input *MeterUsageInput) (*MeterUsageOutput, error)
}

// ResolveCustomerOutput contains the result of resolving a registration token.
type ResolveCustomerOutput struct {
	CustomerIdentifier string `json:"customerIdentifier"`
	ProductCode        string `json:"productCode"`
}

// UsageRecord represents a single metering record for AWS Marketplace.
type UsageRecord struct {
	CustomerIdentifier string    `json:"customerIdentifier"`
	Dimension          string    `json:"dimension"`
	Quantity           int       `json:"quantity"`
	Timestamp          time.Time `json:"timestamp"`
}

// BatchMeterUsageInput is the input for BatchMeterUsage.
type BatchMeterUsageInput struct {
	ProductCode  string        `json:"productCode"`
	UsageRecords []UsageRecord `json:"usageRecords"`
}

// UsageRecordResult contains the result of processing a single usage record.
type UsageRecordResult struct {
	UsageRecord    UsageRecord `json:"usageRecord"`
	MeteringStatus string      `json:"meteringStatus"`
}

// BatchMeterUsageOutput is the output of BatchMeterUsage.
type BatchMeterUsageOutput struct {
	Results            []UsageRecordResult `json:"results"`
	UnprocessedRecords []UsageRecord       `json:"unprocessedRecords"`
}

// MeterUsageInput is the input for MeterUsage.
type MeterUsageInput struct {
	ProductCode      string            `json:"productCode"`
	Dimension        string            `json:"dimension"`
	Quantity         int               `json:"quantity"`
	Timestamp        time.Time         `json:"timestamp"`
	UsageAllocations []UsageAllocation `json:"usageAllocations,omitempty"`
}

// UsageAllocation allows tagging usage with allocation metadata.
type UsageAllocation struct {
	AllocatedUsageQuantity int               `json:"allocatedUsageQuantity"`
	Tags                   map[string]string `json:"tags,omitempty"`
}

// MeterUsageOutput is the output of MeterUsage.
type MeterUsageOutput struct {
	MeteringRecordID string `json:"meteringRecordId"`
}

// MeteringClient implements MeteringAPI by calling the AWS Marketplace
// Metering Service REST endpoints. It signs requests using the provided
// Signer interface.
type MeteringClient struct {
	// Endpoint is the AWS Marketplace Metering Service endpoint URL.
	Endpoint string
	// Signer signs HTTP requests with AWS credentials.
	Signer RequestSigner
	// HTTPClient is the HTTP client used for API calls.
	HTTPClient *http.Client
	// Retry configures exponential backoff for metering calls.
	// Zero value disables retry.
	Retry marketplace.RetryConfig

	mu sync.Mutex
}

// RequestSigner signs an HTTP request with AWS credentials.
type RequestSigner interface {
	Sign(req *http.Request) error
}

// NewMeteringClient creates a new MeteringClient with the given endpoint and signer.
func NewMeteringClient(endpoint string, signer RequestSigner) *MeteringClient {
	return &MeteringClient{
		Endpoint:   endpoint,
		Signer:     signer,
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
		Retry:      marketplace.DefaultRetryConfig(),
	}
}

// ResolveCustomer resolves a registration token to a customer identifier.
func (c *MeteringClient) ResolveCustomer(ctx context.Context, registrationToken string) (*ResolveCustomerOutput, error) {
	body := map[string]string{"RegistrationToken": registrationToken}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal resolve customer request: %w", err)
	}

	resp, err := c.doRequest(ctx, "AWSMPMeteringService.ResolveCustomer", data)
	if err != nil {
		return nil, fmt.Errorf("resolve customer: %w", err)
	}

	var out ResolveCustomerOutput
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal resolve customer response: %w", err)
	}
	return &out, nil
}

// BatchMeterUsage submits a batch of usage records.
func (c *MeteringClient) BatchMeterUsage(ctx context.Context, input *BatchMeterUsageInput) (*BatchMeterUsageOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal batch meter usage request: %w", err)
	}

	var resp []byte
	err = marketplace.RetryFunc(ctx, c.Retry, "aws.BatchMeterUsage", func() error {
		var reqErr error
		resp, reqErr = c.doRequest(ctx, "AWSMPMeteringService.BatchMeterUsage", data)
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("batch meter usage: %w", err)
	}

	var out BatchMeterUsageOutput
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal batch meter usage response: %w", err)
	}
	return &out, nil
}

// MeterUsage submits a single usage record.
func (c *MeteringClient) MeterUsage(ctx context.Context, input *MeterUsageInput) (*MeterUsageOutput, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("marshal meter usage request: %w", err)
	}

	var resp []byte
	err = marketplace.RetryFunc(ctx, c.Retry, "aws.MeterUsage", func() error {
		var reqErr error
		resp, reqErr = c.doRequest(ctx, "AWSMPMeteringService.MeterUsage", data)
		return reqErr
	})
	if err != nil {
		return nil, fmt.Errorf("meter usage: %w", err)
	}

	var out MeterUsageOutput
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal meter usage response: %w", err)
	}
	return &out, nil
}

func (c *MeteringClient) doRequest(ctx context.Context, target string, body []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, io.NopCloser(newBytesReader(body)))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-amz-json-1.1")
	req.Header.Set("X-Amz-Target", target)

	if c.Signer != nil {
		if err := c.Signer.Sign(req); err != nil {
			return nil, fmt.Errorf("sign request: %w", err)
		}
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

// newBytesReader creates an io.Reader from a byte slice. This avoids
// importing bytes just for bytes.NewReader.
type bytesReader struct {
	data []byte
	pos  int
}

func newBytesReader(data []byte) *bytesReader {
	return &bytesReader{data: data}
}

func (r *bytesReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}
