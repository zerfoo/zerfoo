package aws

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// mockMeteringClient is an in-memory MeteringClient for unit tests.
type mockMeteringClient struct {
	calls  []MeteringPayload
	result *MeteringResult
	err    error
}

func (m *mockMeteringClient) BatchMeterUsage(_ context.Context, payload MeteringPayload) (*MeteringResult, error) {
	m.calls = append(m.calls, payload)
	if m.err != nil {
		return nil, m.err
	}
	if m.result != nil {
		return m.result, nil
	}
	// Default: mark every record as success.
	result := &MeteringResult{}
	for _, r := range payload.UsageRecords {
		result.Results = append(result.Results, struct {
			UsageRecord UsageRecord `json:"usageRecord"`
			Status      string      `json:"status"`
		}{UsageRecord: r, Status: "Success"})
	}
	return result, nil
}

// TestAWSMeteringPayload verifies that Meter constructs the correct JSON payload
// and that the MeteringPayload round-trips through JSON faithfully.
func TestAWSMeteringPayload(t *testing.T) {
	ts := time.Date(2026, 3, 18, 12, 0, 0, 0, time.UTC)

	payload := MeteringPayload{
		ProductCode: "prod-abc123",
		UsageRecords: []UsageRecord{
			{
				Timestamp:          ts,
				CustomerIdentifier: "cust-xyz",
				Dimension:          DimensionInferenceRequests,
				Quantity:           42,
			},
			{
				Timestamp:          ts,
				CustomerIdentifier: "cust-xyz",
				Dimension:          DimensionTokensProcessed,
				Quantity:           8192,
			},
		},
	}

	// Round-trip through JSON.
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	var got MeteringPayload
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}

	if got.ProductCode != payload.ProductCode {
		t.Errorf("product code: got %q, want %q", got.ProductCode, payload.ProductCode)
	}
	if len(got.UsageRecords) != len(payload.UsageRecords) {
		t.Fatalf("usage records len: got %d, want %d", len(got.UsageRecords), len(payload.UsageRecords))
	}

	for i, want := range payload.UsageRecords {
		r := got.UsageRecords[i]
		if r.CustomerIdentifier != want.CustomerIdentifier {
			t.Errorf("[%d] customerIdentifier: got %q, want %q", i, r.CustomerIdentifier, want.CustomerIdentifier)
		}
		if r.Dimension != want.Dimension {
			t.Errorf("[%d] dimension: got %q, want %q", i, r.Dimension, want.Dimension)
		}
		if r.Quantity != want.Quantity {
			t.Errorf("[%d] quantity: got %d, want %d", i, r.Quantity, want.Quantity)
		}
		if !r.Timestamp.Equal(want.Timestamp) {
			t.Errorf("[%d] timestamp: got %v, want %v", i, r.Timestamp, want.Timestamp)
		}
	}
}

// TestMeteringDimensions verifies the three required metering dimensions are
// defined, pass validation, and that unknown dimensions are rejected.
func TestMeteringDimensions(t *testing.T) {
	validDimensions := []string{
		DimensionInferenceRequests,
		DimensionTokensProcessed,
		DimensionGPUHours,
	}

	for _, dim := range validDimensions {
		if err := ValidateDimension(dim); err != nil {
			t.Errorf("dimension %q should be valid, got error: %v", dim, err)
		}
	}

	invalidDimensions := []string{"", "cpu-hours", "INFERENCE-REQUESTS", "unknown"}
	for _, dim := range invalidDimensions {
		if err := ValidateDimension(dim); err == nil {
			t.Errorf("dimension %q should be invalid, got nil error", dim)
		}
	}
}

// TestMeterRecordInferenceRequests verifies end-to-end metering through the mock.
func TestMeterRecordInferenceRequests(t *testing.T) {
	mock := &mockMeteringClient{}
	meter := NewMeter(mock, "prod-abc", "cust-xyz")

	if err := meter.RecordInferenceRequests(context.Background(), 10); err != nil {
		t.Fatalf("RecordInferenceRequests: %v", err)
	}

	if len(mock.calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(mock.calls))
	}
	call := mock.calls[0]
	if call.ProductCode != "prod-abc" {
		t.Errorf("product code: got %q, want %q", call.ProductCode, "prod-abc")
	}
	if len(call.UsageRecords) != 1 {
		t.Fatalf("expected 1 usage record, got %d", len(call.UsageRecords))
	}
	r := call.UsageRecords[0]
	if r.Dimension != DimensionInferenceRequests {
		t.Errorf("dimension: got %q, want %q", r.Dimension, DimensionInferenceRequests)
	}
	if r.Quantity != 10 {
		t.Errorf("quantity: got %d, want 10", r.Quantity)
	}
	if r.CustomerIdentifier != "cust-xyz" {
		t.Errorf("customerIdentifier: got %q, want %q", r.CustomerIdentifier, "cust-xyz")
	}
}

// TestMeterRecordTokensProcessed verifies tokens-processed dimension reporting.
func TestMeterRecordTokensProcessed(t *testing.T) {
	mock := &mockMeteringClient{}
	meter := NewMeter(mock, "prod-abc", "cust-xyz")

	if err := meter.RecordTokensProcessed(context.Background(), 4096); err != nil {
		t.Fatalf("RecordTokensProcessed: %v", err)
	}

	r := mock.calls[0].UsageRecords[0]
	if r.Dimension != DimensionTokensProcessed {
		t.Errorf("dimension: got %q, want %q", r.Dimension, DimensionTokensProcessed)
	}
	if r.Quantity != 4096 {
		t.Errorf("quantity: got %d, want 4096", r.Quantity)
	}
}

// TestMeterRecordGPUHours verifies gpu-hours dimension reporting.
func TestMeterRecordGPUHours(t *testing.T) {
	mock := &mockMeteringClient{}
	meter := NewMeter(mock, "prod-abc", "cust-xyz")

	if err := meter.RecordGPUHours(context.Background(), 2); err != nil {
		t.Fatalf("RecordGPUHours: %v", err)
	}

	r := mock.calls[0].UsageRecords[0]
	if r.Dimension != DimensionGPUHours {
		t.Errorf("dimension: got %q, want %q", r.Dimension, DimensionGPUHours)
	}
	if r.Quantity != 2 {
		t.Errorf("quantity: got %d, want 2", r.Quantity)
	}
}

// TestHTTPMeteringClientBatchMeterUsage verifies the HTTP client sends the
// correct request and parses a successful response.
func TestHTTPMeteringClientBatchMeterUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/BatchMeterUsage" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}

		var payload MeteringPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}

		result := MeteringResult{}
		for _, rec := range payload.UsageRecords {
			result.Results = append(result.Results, struct {
				UsageRecord UsageRecord `json:"usageRecord"`
				Status      string      `json:"status"`
			}{UsageRecord: rec, Status: "Success"})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(result)
	}))
	defer srv.Close()

	client := NewHTTPMeteringClient(srv.URL, "prod-test")
	payload := MeteringPayload{
		UsageRecords: []UsageRecord{
			{
				Timestamp:          time.Now().UTC(),
				CustomerIdentifier: "cust-001",
				Dimension:          DimensionGPUHours,
				Quantity:           1,
			},
		},
	}

	result, err := client.BatchMeterUsage(context.Background(), payload)
	if err != nil {
		t.Fatalf("BatchMeterUsage: %v", err)
	}
	if len(result.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(result.Results))
	}
	if result.Results[0].Status != "Success" {
		t.Errorf("status: got %q, want %q", result.Results[0].Status, "Success")
	}
}
