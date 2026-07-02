// Package azure provides Azure Marketplace integration for Zerfoo Cloud,
// including SaaS Fulfillment API v2, Marketplace Metering Service,
// subscription lifecycle management, and webhook handling.
package azure

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// FulfillmentAPI abstracts Azure Marketplace SaaS Fulfillment API v2 calls.
type FulfillmentAPI interface {
	// Resolve resolves a marketplace purchase identification token to a subscription.
	Resolve(ctx context.Context, marketplaceToken string) (*ResolvedSubscription, error)

	// Activate activates a SaaS subscription after the publisher provisions the resource.
	Activate(ctx context.Context, subscriptionID string, plan PlanDetails) error

	// GetSubscription retrieves the details of a SaaS subscription.
	GetSubscription(ctx context.Context, subscriptionID string) (*SaaSSubscription, error)

	// UpdateSubscription updates the plan or seat quantity for a subscription.
	UpdateSubscription(ctx context.Context, subscriptionID string, update SubscriptionUpdate) (*OperationLocation, error)

	// Suspend marks a subscription as suspended (e.g. payment failure).
	Suspend(ctx context.Context, subscriptionID string) error

	// Delete cancels and deletes a SaaS subscription.
	Delete(ctx context.Context, subscriptionID string) error

	// ListSubscriptions returns all SaaS subscriptions for the publisher.
	ListSubscriptions(ctx context.Context) ([]SaaSSubscription, error)
}

// ResolvedSubscription is the result of resolving a marketplace token.
type ResolvedSubscription struct {
	ID                string `json:"id"`
	SubscriptionName  string `json:"subscriptionName"`
	OfferID           string `json:"offerId"`
	PlanID            string `json:"planId"`
	Quantity          int    `json:"quantity,omitempty"`
	PublisherID       string `json:"publisherId"`
	BeneficiaryTenant string `json:"beneficiary.tenantId"`
}

// SaaSSubscription represents an Azure Marketplace SaaS subscription.
type SaaSSubscription struct {
	ID               string             `json:"id"`
	SubscriptionName string             `json:"name"`
	OfferID          string             `json:"offerId"`
	PlanID           string             `json:"planId"`
	Quantity         int                `json:"quantity,omitempty"`
	Status           SaaSStatus         `json:"saasSubscriptionStatus"`
	Beneficiary      SaaSBeneficiary    `json:"beneficiary"`
	Purchaser        SaaSPurchaser      `json:"purchaser"`
	Term             SaaSTerm           `json:"term"`
	Created          time.Time          `json:"created"`
}

// SaaSBeneficiary identifies the Azure AD tenant and user benefiting from the subscription.
type SaaSBeneficiary struct {
	TenantID string `json:"tenantId"`
	ObjectID string `json:"objectId"`
	Email    string `json:"emailId"`
}

// SaaSPurchaser identifies the Azure AD tenant and user that purchased the subscription.
type SaaSPurchaser struct {
	TenantID string `json:"tenantId"`
	ObjectID string `json:"objectId"`
	Email    string `json:"emailId"`
}

// SaaSTerm describes the subscription billing term.
type SaaSTerm struct {
	StartDate time.Time `json:"startDate"`
	EndDate   time.Time `json:"endDate"`
	TermUnit  string    `json:"termUnit"` // "P1M" or "P1Y"
}

// SaaSStatus represents the lifecycle state of an Azure SaaS subscription.
type SaaSStatus string

const (
	SaaSStatusPendingConfiguration SaaSStatus = "PendingFulfillmentStart"
	SaaSStatusSubscribed           SaaSStatus = "Subscribed"
	SaaSStatusSuspended            SaaSStatus = "Suspended"
	SaaSStatusUnsubscribed         SaaSStatus = "Unsubscribed"
)

// PlanDetails identifies the plan to activate.
type PlanDetails struct {
	PlanID   string `json:"planId"`
	Quantity int    `json:"quantity,omitempty"`
}

// SubscriptionUpdate contains fields for updating a subscription.
type SubscriptionUpdate struct {
	PlanID   string `json:"planId,omitempty"`
	Quantity int    `json:"quantity,omitempty"`
}

// OperationLocation is returned by async operations, containing a URL to poll for status.
type OperationLocation struct {
	Location string `json:"location"`
}

// TokenProvider obtains Azure AD bearer tokens for API authentication.
type TokenProvider interface {
	// Token returns a valid bearer token for the Azure Marketplace API.
	Token(ctx context.Context) (string, error)
}

// FulfillmentClient implements FulfillmentAPI by calling the Azure Marketplace
// SaaS Fulfillment API v2 REST endpoints.
type FulfillmentClient struct {
	// Endpoint is the base URL for the SaaS Fulfillment API.
	// Default: https://marketplaceapi.microsoft.com/api
	Endpoint string

	// APIVersion is the API version query parameter.
	// Default: 2018-08-31
	APIVersion string

	// TokenProvider obtains bearer tokens for API authentication.
	TokenProvider TokenProvider

	// HTTPClient is the HTTP client used for API calls.
	HTTPClient *http.Client
}

// NewFulfillmentClient creates a new FulfillmentClient with the given endpoint and token provider.
func NewFulfillmentClient(endpoint string, tokenProvider TokenProvider) *FulfillmentClient {
	return &FulfillmentClient{
		Endpoint:      endpoint,
		APIVersion:    "2018-08-31",
		TokenProvider: tokenProvider,
		HTTPClient:    &http.Client{Timeout: 30 * time.Second},
	}
}

// Resolve resolves a marketplace purchase token to a subscription.
func (c *FulfillmentClient) Resolve(ctx context.Context, marketplaceToken string) (*ResolvedSubscription, error) {
	req, err := c.newRequest(ctx, http.MethodPost, "/saas/subscriptions/resolve")
	if err != nil {
		return nil, fmt.Errorf("create resolve request: %w", err)
	}
	req.Header.Set("x-ms-marketplace-token", marketplaceToken)

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, fmt.Errorf("resolve subscription: %w", err)
	}

	var out ResolvedSubscription
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal resolve response: %w", err)
	}
	return &out, nil
}

// Activate activates a SaaS subscription.
func (c *FulfillmentClient) Activate(ctx context.Context, subscriptionID string, plan PlanDetails) error {
	body, err := json.Marshal(plan)
	if err != nil {
		return fmt.Errorf("marshal activate request: %w", err)
	}

	req, err := c.newRequestWithBody(ctx, http.MethodPost, "/saas/subscriptions/"+subscriptionID+"/activate", body)
	if err != nil {
		return fmt.Errorf("create activate request: %w", err)
	}

	if _, err := c.doRequest(req); err != nil {
		return fmt.Errorf("activate subscription: %w", err)
	}
	return nil
}

// GetSubscription retrieves a SaaS subscription by ID.
func (c *FulfillmentClient) GetSubscription(ctx context.Context, subscriptionID string) (*SaaSSubscription, error) {
	req, err := c.newRequest(ctx, http.MethodGet, "/saas/subscriptions/"+subscriptionID)
	if err != nil {
		return nil, fmt.Errorf("create get subscription request: %w", err)
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, fmt.Errorf("get subscription: %w", err)
	}

	var out SaaSSubscription
	if err := json.Unmarshal(resp, &out); err != nil {
		return nil, fmt.Errorf("unmarshal subscription response: %w", err)
	}
	return &out, nil
}

// UpdateSubscription updates the plan or seat quantity for a subscription.
func (c *FulfillmentClient) UpdateSubscription(ctx context.Context, subscriptionID string, update SubscriptionUpdate) (*OperationLocation, error) {
	body, err := json.Marshal(update)
	if err != nil {
		return nil, fmt.Errorf("marshal update request: %w", err)
	}

	req, err := c.newRequestWithBody(ctx, http.MethodPatch, "/saas/subscriptions/"+subscriptionID, body)
	if err != nil {
		return nil, fmt.Errorf("create update request: %w", err)
	}

	respBody, err := c.doRequestWithHeaders(req)
	if err != nil {
		return nil, fmt.Errorf("update subscription: %w", err)
	}

	return &OperationLocation{Location: respBody.header.Get("Operation-Location")}, nil
}

// Suspend marks a subscription as suspended.
func (c *FulfillmentClient) Suspend(ctx context.Context, subscriptionID string) error {
	req, err := c.newRequest(ctx, http.MethodPost, "/saas/subscriptions/"+subscriptionID+"/suspend")
	if err != nil {
		return fmt.Errorf("create suspend request: %w", err)
	}

	if _, err := c.doRequest(req); err != nil {
		return fmt.Errorf("suspend subscription: %w", err)
	}
	return nil
}

// Delete cancels and deletes a SaaS subscription.
func (c *FulfillmentClient) Delete(ctx context.Context, subscriptionID string) error {
	req, err := c.newRequest(ctx, http.MethodDelete, "/saas/subscriptions/"+subscriptionID)
	if err != nil {
		return fmt.Errorf("create delete request: %w", err)
	}

	if _, err := c.doRequest(req); err != nil {
		return fmt.Errorf("delete subscription: %w", err)
	}
	return nil
}

// ListSubscriptions returns all SaaS subscriptions.
func (c *FulfillmentClient) ListSubscriptions(ctx context.Context) ([]SaaSSubscription, error) {
	req, err := c.newRequest(ctx, http.MethodGet, "/saas/subscriptions")
	if err != nil {
		return nil, fmt.Errorf("create list subscriptions request: %w", err)
	}

	resp, err := c.doRequest(req)
	if err != nil {
		return nil, fmt.Errorf("list subscriptions: %w", err)
	}

	var result struct {
		Subscriptions []SaaSSubscription `json:"subscriptions"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("unmarshal subscriptions response: %w", err)
	}
	return result.Subscriptions, nil
}

type responseWithHeaders struct {
	body   []byte
	header http.Header
}

func (c *FulfillmentClient) newRequest(ctx context.Context, method, path string) (*http.Request, error) {
	return c.newRequestWithBody(ctx, method, path, nil)
}

func (c *FulfillmentClient) newRequestWithBody(ctx context.Context, method, path string, body []byte) (*http.Request, error) {
	url := c.Endpoint + path + "?api-version=" + c.APIVersion

	var bodyReader io.Reader
	if body != nil {
		bodyReader = &bytesReader{data: body}
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	if c.TokenProvider != nil {
		token, err := c.TokenProvider.Token(ctx)
		if err != nil {
			return nil, fmt.Errorf("obtain token: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
	}

	return req, nil
}

func (c *FulfillmentClient) doRequest(req *http.Request) ([]byte, error) {
	resp, err := c.doRequestWithHeaders(req)
	if err != nil {
		return nil, err
	}
	return resp.body, nil
}

func (c *FulfillmentClient) doRequestWithHeaders(req *http.Request) (*responseWithHeaders, error) {
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

	return &responseWithHeaders{body: respBody, header: resp.Header}, nil
}

// bytesReader is a minimal io.Reader backed by a byte slice.
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
