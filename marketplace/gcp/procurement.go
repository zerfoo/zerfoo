// Package gcp provides GCP Marketplace integration for Zerfoo Cloud,
// including Cloud Commerce Partner Procurement API integration, SaaS
// entitlement management, Service Control API usage metering, and
// token-based billing.
package gcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// ProcurementAPI abstracts GCP Cloud Commerce Partner Procurement API calls.
type ProcurementAPI interface {
	// GetAccount retrieves a procurement account by name.
	GetAccount(ctx context.Context, name string) (*Account, error)

	// ListAccounts lists procurement accounts for the provider.
	ListAccounts(ctx context.Context, parent string) ([]Account, error)

	// GetEntitlement retrieves an entitlement by name.
	GetEntitlement(ctx context.Context, name string) (*ProcurementEntitlement, error)

	// ListEntitlements lists entitlements for the provider.
	ListEntitlements(ctx context.Context, parent string) ([]ProcurementEntitlement, error)

	// ApproveEntitlement approves a pending entitlement.
	ApproveEntitlement(ctx context.Context, name string) error

	// RejectEntitlement rejects a pending entitlement with the given reason.
	RejectEntitlement(ctx context.Context, name, reason string) error

	// SuspendEntitlement suspends an active entitlement with the given reason.
	SuspendEntitlement(ctx context.Context, name, reason string) error

	// ReinstateEntitlement reinstates a suspended entitlement.
	ReinstateEntitlement(ctx context.Context, name string) error
}

// Account represents a GCP Marketplace procurement account.
type Account struct {
	Name       string       `json:"name"`
	Provider   string       `json:"provider"`
	State      AccountState `json:"state"`
	CreateTime time.Time    `json:"createTime"`
	UpdateTime time.Time    `json:"updateTime"`
}

// AccountState represents the state of a procurement account.
type AccountState string

const (
	AccountActive              AccountState = "ACCOUNT_ACTIVE"
	AccountPendingVerification AccountState = "ACCOUNT_PENDING_VERIFICATION"
)

// ProcurementEntitlement represents a GCP Marketplace entitlement from the
// Partner Procurement API.
type ProcurementEntitlement struct {
	Name         string           `json:"name"`
	Account      string           `json:"account"`
	Provider     string           `json:"provider"`
	Product      string           `json:"product"`
	Plan         string           `json:"plan"`
	State        EntitlementState `json:"state"`
	NewPendingPlan string         `json:"newPendingPlan,omitempty"`
	CreateTime   time.Time        `json:"createTime"`
	UpdateTime   time.Time        `json:"updateTime"`
}

// EntitlementState represents the lifecycle state of an entitlement.
type EntitlementState string

const (
	EntitlementPending            EntitlementState = "ENTITLEMENT_PENDING"
	EntitlementActive             EntitlementState = "ENTITLEMENT_ACTIVE"
	EntitlementSuspended          EntitlementState = "ENTITLEMENT_SUSPENDED"
	EntitlementCancellationPending EntitlementState = "ENTITLEMENT_CANCELLATION_PENDING"
	EntitlementCancelled          EntitlementState = "ENTITLEMENT_CANCELLED"
)

// TokenSource provides OAuth2 access tokens for authenticating GCP API requests.
type TokenSource interface {
	Token(ctx context.Context) (string, error)
}

// ProcurementClient implements ProcurementAPI by calling the GCP Cloud Commerce
// Partner Procurement REST API.
type ProcurementClient struct {
	// Endpoint is the Partner Procurement API base URL.
	Endpoint string
	// TokenSource provides access tokens for API authentication.
	TokenSource TokenSource
	// HTTPClient is the HTTP client used for API calls.
	HTTPClient *http.Client
}

const defaultProcurementEndpoint = "https://cloudcommerceprocurement.googleapis.com/v1"

// NewProcurementClient creates a new ProcurementClient with the given token source.
func NewProcurementClient(tokenSource TokenSource) *ProcurementClient {
	return &ProcurementClient{
		Endpoint:    defaultProcurementEndpoint,
		TokenSource: tokenSource,
		HTTPClient:  &http.Client{Timeout: 30 * time.Second},
	}
}

// GetAccount retrieves a procurement account by name.
func (c *ProcurementClient) GetAccount(ctx context.Context, name string) (*Account, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/"+name, nil)
	if err != nil {
		return nil, fmt.Errorf("get account: %w", err)
	}

	var acct Account
	if err := json.Unmarshal(resp, &acct); err != nil {
		return nil, fmt.Errorf("unmarshal account: %w", err)
	}
	return &acct, nil
}

// ListAccounts lists procurement accounts for the provider.
func (c *ProcurementClient) ListAccounts(ctx context.Context, parent string) ([]Account, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/"+parent+"/accounts", nil)
	if err != nil {
		return nil, fmt.Errorf("list accounts: %w", err)
	}

	var result struct {
		Accounts []Account `json:"accounts"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("unmarshal accounts: %w", err)
	}
	return result.Accounts, nil
}

// GetEntitlement retrieves an entitlement by name.
func (c *ProcurementClient) GetEntitlement(ctx context.Context, name string) (*ProcurementEntitlement, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/"+name, nil)
	if err != nil {
		return nil, fmt.Errorf("get entitlement: %w", err)
	}

	var ent ProcurementEntitlement
	if err := json.Unmarshal(resp, &ent); err != nil {
		return nil, fmt.Errorf("unmarshal entitlement: %w", err)
	}
	return &ent, nil
}

// ListEntitlements lists entitlements for the provider.
func (c *ProcurementClient) ListEntitlements(ctx context.Context, parent string) ([]ProcurementEntitlement, error) {
	resp, err := c.doRequest(ctx, http.MethodGet, "/"+parent+"/entitlements", nil)
	if err != nil {
		return nil, fmt.Errorf("list entitlements: %w", err)
	}

	var result struct {
		Entitlements []ProcurementEntitlement `json:"entitlements"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("unmarshal entitlements: %w", err)
	}
	return result.Entitlements, nil
}

// ApproveEntitlement approves a pending entitlement.
func (c *ProcurementClient) ApproveEntitlement(ctx context.Context, name string) error {
	_, err := c.doRequest(ctx, http.MethodPost, "/"+name+":approve", nil)
	if err != nil {
		return fmt.Errorf("approve entitlement: %w", err)
	}
	return nil
}

// RejectEntitlement rejects a pending entitlement with the given reason.
func (c *ProcurementClient) RejectEntitlement(ctx context.Context, name, reason string) error {
	body := map[string]string{"reason": reason}
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal reject request: %w", err)
	}
	_, err = c.doRequest(ctx, http.MethodPost, "/"+name+":reject", data)
	if err != nil {
		return fmt.Errorf("reject entitlement: %w", err)
	}
	return nil
}

// SuspendEntitlement suspends an active entitlement with the given reason.
func (c *ProcurementClient) SuspendEntitlement(ctx context.Context, name, reason string) error {
	body := map[string]string{"reason": reason}
	data, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal suspend request: %w", err)
	}
	_, err = c.doRequest(ctx, http.MethodPost, "/"+name+":suspend", data)
	if err != nil {
		return fmt.Errorf("suspend entitlement: %w", err)
	}
	return nil
}

// ReinstateEntitlement reinstates a suspended entitlement.
func (c *ProcurementClient) ReinstateEntitlement(ctx context.Context, name string) error {
	_, err := c.doRequest(ctx, http.MethodPost, "/"+name+":reinstate", nil)
	if err != nil {
		return fmt.Errorf("reinstate entitlement: %w", err)
	}
	return nil
}

func (c *ProcurementClient) doRequest(ctx context.Context, method, path string, body []byte) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		bodyReader = &bytesReader{data: body}
	}

	req, err := http.NewRequestWithContext(ctx, method, c.Endpoint+path, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	if c.TokenSource != nil {
		token, err := c.TokenSource.Token(ctx)
		if err != nil {
			return nil, fmt.Errorf("get access token: %w", err)
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

// bytesReader is a minimal io.Reader over a byte slice.
type bytesReader struct {
	data []byte
	pos  int
}

func (r *bytesReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}
