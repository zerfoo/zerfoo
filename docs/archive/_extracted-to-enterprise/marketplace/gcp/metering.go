package gcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/zerfoo/zerfoo/marketplace"
)

// ServiceControlAPI abstracts GCP Service Control API calls for usage reporting.
type ServiceControlAPI interface {
	// Report submits usage operations to the Service Control API.
	Report(ctx context.Context, serviceName string, ops []Operation) error
}

// Operation represents a single usage operation for the Service Control API.
type Operation struct {
	OperationID   string            `json:"operationId"`
	OperationName string            `json:"operationName"`
	ConsumerID    string            `json:"consumerId"`
	StartTime     time.Time         `json:"startTime"`
	EndTime       time.Time         `json:"endTime"`
	MetricValues  []MetricValueSet  `json:"metricValueSets"`
	Labels        map[string]string `json:"labels,omitempty"`
}

// MetricValueSet represents a set of metric values for a single metric.
type MetricValueSet struct {
	MetricName   string        `json:"metricName"`
	MetricValues []MetricValue `json:"metricValues"`
}

// MetricValue represents a single metric measurement.
type MetricValue struct {
	Int64Value *int64            `json:"int64Value,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
}

// ReportRequest is the request body for the Service Control Report API.
type ReportRequest struct {
	Operations []Operation `json:"operations"`
}

// ReportResponse is the response from the Service Control Report API.
type ReportResponse struct {
	ReportErrors []ReportError `json:"reportErrors,omitempty"`
}

// ReportError describes an error for a specific operation in a report request.
type ReportError struct {
	OperationID string `json:"operationId"`
	Status      Status `json:"status"`
}

// Status represents a gRPC-style status in JSON responses.
type Status struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// ServiceControlClient implements ServiceControlAPI by calling the GCP
// Service Control v1 REST API.
type ServiceControlClient struct {
	// Endpoint is the Service Control API base URL.
	Endpoint string
	// TokenSource provides access tokens for API authentication.
	TokenSource TokenSource
	// HTTPClient is the HTTP client used for API calls.
	HTTPClient *http.Client
	// Retry configures exponential backoff for metering calls.
	// Zero value disables retry.
	Retry marketplace.RetryConfig
}

const defaultServiceControlEndpoint = "https://servicecontrol.googleapis.com/v1"

// NewServiceControlClient creates a new ServiceControlClient with the given token source.
func NewServiceControlClient(tokenSource TokenSource) *ServiceControlClient {
	return &ServiceControlClient{
		Endpoint:    defaultServiceControlEndpoint,
		TokenSource: tokenSource,
		HTTPClient:  &http.Client{Timeout: 30 * time.Second},
		Retry:       marketplace.DefaultRetryConfig(),
	}
}

// Report submits usage operations to the Service Control API.
func (c *ServiceControlClient) Report(ctx context.Context, serviceName string, ops []Operation) error {
	reqBody := ReportRequest{Operations: ops}
	data, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal report request: %w", err)
	}

	return marketplace.RetryFunc(ctx, c.Retry, "gcp.Report", func() error {
		return c.doReport(ctx, serviceName, data)
	})
}

func (c *ServiceControlClient) doReport(ctx context.Context, serviceName string, data []byte) error {
	path := "/services/" + serviceName + ":report"

	var bodyReader io.Reader = &bytesReader{data: data}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	if c.TokenSource != nil {
		token, err := c.TokenSource.Token(ctx)
		if err != nil {
			return fmt.Errorf("get access token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response body: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var reportResp ReportResponse
	if err := json.Unmarshal(respBody, &reportResp); err != nil {
		return fmt.Errorf("unmarshal report response: %w", err)
	}

	if len(reportResp.ReportErrors) > 0 {
		return fmt.Errorf("report errors: %d operations failed, first error: %s",
			len(reportResp.ReportErrors), reportResp.ReportErrors[0].Status.Message)
	}

	return nil
}
