package support

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"
)

// API provides HTTP handlers for the customer support portal.
type API struct {
	Store      *Store
	Router     *Router
	SLA        *SLATracker
	Webhooks   *WebhookDispatcher
}

// RegisterRoutes registers all ticket API endpoints on the given mux.
func (a *API) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /support/tickets", a.CreateTicket)
	mux.HandleFunc("GET /support/tickets", a.ListTickets)
	mux.HandleFunc("GET /support/tickets/{id}", a.GetTicket)
	mux.HandleFunc("POST /support/tickets/{id}/comments", a.AddComment)
	mux.HandleFunc("POST /support/tickets/{id}/close", a.CloseTicket)
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
	var req CreateTicketRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	if req.CustomerID == "" || req.Subject == "" {
		http.Error(w, `{"error":"customer_id and subject are required"}`, http.StatusBadRequest)
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
	id := r.PathValue("id")
	if id == "" {
		parts := strings.Split(r.URL.Path, "/")
		if len(parts) >= 4 {
			id = parts[3]
		}
	}

	var req AddCommentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
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
