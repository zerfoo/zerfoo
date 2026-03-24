package support

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
)

// contextKey is an unexported type for context keys in this package.
type contextKey int

const tenantKey contextKey = 0

// TenantFromContext returns the authenticated tenant ID from the request context.
func TenantFromContext(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(tenantKey).(string)
	return id, ok
}

// AuthFunc validates a Bearer token and returns the associated tenant ID.
// If the token is invalid, it returns an empty string and false.
type AuthFunc func(token string) (tenantID string, ok bool)

// API provides HTTP handlers for the customer support portal.
type API struct {
	Store    *Store
	Router   *Router
	SLA      *SLATracker
	Webhooks *WebhookDispatcher
	Auth     AuthFunc
}

// maxBodySize is the maximum allowed request body size (1 MiB).
const maxBodySize = 1 << 20

// RegisterRoutes registers all ticket API endpoints on the given mux.
func (a *API) RegisterRoutes(mux *http.ServeMux) {
	auth := a.authMiddleware
	mux.Handle("POST /support/tickets", auth(http.HandlerFunc(a.CreateTicket)))
	mux.Handle("GET /support/tickets", auth(http.HandlerFunc(a.ListTickets)))
	mux.Handle("GET /support/tickets/{id}", auth(http.HandlerFunc(a.GetTicket)))
	mux.Handle("POST /support/tickets/{id}/comments", auth(http.HandlerFunc(a.AddComment)))
	mux.Handle("POST /support/tickets/{id}/close", auth(http.HandlerFunc(a.CloseTicket)))
}

// authMiddleware wraps a handler with Bearer token authentication.
func (a *API) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if a.Auth == nil {
			next.ServeHTTP(w, r)
			return
		}
		header := r.Header.Get("Authorization")
		if !strings.HasPrefix(header, "Bearer ") {
			http.Error(w, `{"error":"missing or invalid authorization header"}`, http.StatusUnauthorized)
			return
		}
		token := strings.TrimPrefix(header, "Bearer ")
		tenantID, ok := a.Auth(token)
		if !ok || tenantID == "" {
			http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
			return
		}
		ctx := context.WithValue(r.Context(), tenantKey, tenantID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// requireTenant checks that the authenticated tenant matches the given customer ID.
// Returns true if access is allowed; writes a 403 response and returns false otherwise.
func requireTenant(w http.ResponseWriter, r *http.Request, customerID string) bool {
	tenantID, ok := TenantFromContext(r.Context())
	if !ok {
		// No auth configured — allow.
		return true
	}
	if tenantID != customerID {
		http.Error(w, `{"error":"access denied"}`, http.StatusForbidden)
		return false
	}
	return true
}

// CreateTicketRequest is the JSON body for creating a ticket.
type CreateTicketRequest struct {
	CustomerID string   `json:"customer_id"`
	Subject    string   `json:"subject"`
	Body       string   `json:"body"`
	Priority   Priority `json:"priority"`
}

// CreateTicket handles POST /support/tickets.
func (a *API) CreateTicket(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)

	var req CreateTicketRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			http.Error(w, `{"error":"request body too large"}`, http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	if req.CustomerID == "" || req.Subject == "" {
		http.Error(w, `{"error":"customer_id and subject are required"}`, http.StatusBadRequest)
		return
	}

	if !requireTenant(w, r, req.CustomerID) {
		return
	}

	ticket := a.Store.Create(req.CustomerID, req.Subject, req.Body, req.Priority)

	// Route the ticket if a router is configured.
	if a.Router != nil {
		_ = a.Router.Route(ticket, a.Store)
	}

	// Fire webhook.
	if a.Webhooks != nil {
		a.Webhooks.Dispatch(r.Context(), WebhookEvent{
			Type:      EventTicketCreated,
			Timestamp: time.Now().UTC(),
			Payload:   ticket,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(ticket)
}

// ListTickets handles GET /support/tickets?customer_id=...
func (a *API) ListTickets(w http.ResponseWriter, r *http.Request) {
	customerID := r.URL.Query().Get("customer_id")
	if customerID == "" {
		http.Error(w, `{"error":"customer_id query parameter is required"}`, http.StatusBadRequest)
		return
	}

	if !requireTenant(w, r, customerID) {
		return
	}

	tickets := a.Store.ListByCustomer(customerID)
	if tickets == nil {
		tickets = []*Ticket{}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tickets)
}

// GetTicket handles GET /support/tickets/{id}.
func (a *API) GetTicket(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		// Fallback for older Go versions: parse from path.
		parts := strings.Split(r.URL.Path, "/")
		if len(parts) >= 4 {
			id = parts[3]
		}
	}
	ticket, ok := a.Store.Get(id)
	if !ok {
		http.Error(w, `{"error":"ticket not found"}`, http.StatusNotFound)
		return
	}

	if !requireTenant(w, r, ticket.CustomerID) {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ticket)
}

// AddCommentRequest is the JSON body for adding a comment.
type AddCommentRequest struct {
	Author string `json:"author"`
	Body   string `json:"body"`
}

// AddComment handles POST /support/tickets/{id}/comments.
func (a *API) AddComment(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)

	id := r.PathValue("id")
	if id == "" {
		parts := strings.Split(r.URL.Path, "/")
		if len(parts) >= 4 {
			id = parts[3]
		}
	}

	ticket, ok := a.Store.Get(id)
	if !ok {
		http.Error(w, `{"error":"ticket not found"}`, http.StatusNotFound)
		return
	}

	if !requireTenant(w, r, ticket.CustomerID) {
		return
	}

	var req AddCommentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			http.Error(w, `{"error":"request body too large"}`, http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	if req.Author == "" || req.Body == "" {
		http.Error(w, `{"error":"author and body are required"}`, http.StatusBadRequest)
		return
	}

	comment, err := a.Store.AddComment(id, req.Author, req.Body)
	if err != nil {
		http.Error(w, `{"error":"ticket not found"}`, http.StatusNotFound)
		return
	}

	if a.Webhooks != nil {
		a.Webhooks.Dispatch(r.Context(), WebhookEvent{
			Type:      EventCommentAdded,
			Timestamp: time.Now().UTC(),
			Payload:   comment,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(comment)
}

// CloseTicket handles POST /support/tickets/{id}/close.
func (a *API) CloseTicket(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)

	id := r.PathValue("id")
	if id == "" {
		parts := strings.Split(r.URL.Path, "/")
		if len(parts) >= 4 {
			id = parts[3]
		}
	}

	ticket, ok := a.Store.Get(id)
	if !ok {
		http.Error(w, `{"error":"ticket not found"}`, http.StatusNotFound)
		return
	}

	if !requireTenant(w, r, ticket.CustomerID) {
		return
	}

	now := time.Now().UTC()
	if err := ticket.Transition(StatusClosed, now); err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	if a.Webhooks != nil {
		a.Webhooks.Dispatch(r.Context(), WebhookEvent{
			Type:      EventTicketClosed,
			Timestamp: now,
			Payload:   ticket,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ticket)
}
