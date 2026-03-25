package support

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- Ticket lifecycle tests ---

func TestTicketLifecycle(t *testing.T) {
	store := NewStore()
	ticket := store.Create("cust-1", "Cannot deploy", "Deployment fails on v2.3", P1High)

	if ticket.Status != StatusOpen {
		t.Fatalf("expected open, got %s", ticket.Status)
	}
	if ticket.ID == "" {
		t.Fatal("expected non-empty ticket ID")
	}
	if ticket.CustomerID != "cust-1" {
		t.Fatalf("expected cust-1, got %s", ticket.CustomerID)
	}

	now := time.Now().UTC()

	// open -> triaged
	if err := ticket.Transition(StatusTriaged, now); err != nil {
		t.Fatalf("transition to triaged: %v", err)
	}
	// triaged -> in_progress
	if err := ticket.Transition(StatusInProgress, now); err != nil {
		t.Fatalf("transition to in_progress: %v", err)
	}
	// in_progress -> resolved
	if err := ticket.Transition(StatusResolved, now); err != nil {
		t.Fatalf("transition to resolved: %v", err)
	}
	if ticket.ResolvedAt.IsZero() {
		t.Fatal("expected non-zero ResolvedAt")
	}
	// resolved -> closed
	if err := ticket.Transition(StatusClosed, now); err != nil {
		t.Fatalf("transition to closed: %v", err)
	}
	if ticket.ClosedAt.IsZero() {
		t.Fatal("expected non-zero ClosedAt")
	}
}

func TestInvalidTransition(t *testing.T) {
	store := NewStore()
	ticket := store.Create("cust-1", "Test", "", P2Medium)

	// open -> in_progress is invalid (must go through triaged)
	if err := ticket.Transition(StatusInProgress, time.Now()); err == nil {
		t.Fatal("expected error for invalid transition")
	}
}

func TestClosedTicketCannotTransition(t *testing.T) {
	store := NewStore()
	ticket := store.Create("cust-1", "Test", "", P3Low)
	now := time.Now()
	ticket.Transition(StatusClosed, now)
	if err := ticket.Transition(StatusOpen, now); err == nil {
		t.Fatal("expected error transitioning from closed")
	}
}

// --- Store tests ---

func TestStoreListByCustomer(t *testing.T) {
	store := NewStore()
	store.Create("cust-1", "Ticket A", "", P2Medium)
	store.Create("cust-2", "Ticket B", "", P3Low)
	store.Create("cust-1", "Ticket C", "", P1High)

	tickets := store.ListByCustomer("cust-1")
	if len(tickets) != 2 {
		t.Fatalf("expected 2 tickets, got %d", len(tickets))
	}
}

func TestListByCustomerOrderNewestFirst(t *testing.T) {
	store := NewStore()

	// Create 6 tickets for the same customer with staggered times.
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	subjects := []string{"Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"}
	for i, subj := range subjects {
		store.mu.Lock()
		store.nextID++
		ts := base.Add(time.Duration(i) * time.Hour)
		tk := &Ticket{
			ID:         fmt.Sprintf("TKT-%06d", store.nextID),
			CustomerID: "cust-sort",
			Subject:    subj,
			Priority:   P2Medium,
			Status:     StatusOpen,
			CreatedAt:  ts,
			UpdatedAt:  ts,
		}
		store.tickets[tk.ID] = tk
		store.mu.Unlock()
	}

	// Also update the 2nd ticket ("Bravo") to have the newest UpdatedAt,
	// so we can verify sorting uses UpdatedAt, not CreatedAt.
	for _, tk := range store.tickets {
		if tk.Subject == "Bravo" {
			tk.UpdatedAt = base.Add(100 * time.Hour)
			break
		}
	}

	// Add a ticket for a different customer to ensure filtering works.
	store.Create("cust-other", "Unrelated", "", P3Low)

	tickets := store.ListByCustomer("cust-sort")
	if len(tickets) != 6 {
		t.Fatalf("expected 6 tickets, got %d", len(tickets))
	}

	// Verify descending UpdatedAt order.
	for i := 1; i < len(tickets); i++ {
		if tickets[i].UpdatedAt.After(tickets[i-1].UpdatedAt) {
			t.Fatalf("ticket at index %d (UpdatedAt=%v) is newer than ticket at index %d (UpdatedAt=%v)",
				i, tickets[i].UpdatedAt, i-1, tickets[i-1].UpdatedAt)
		}
	}

	// The first ticket should be "Bravo" since we gave it the latest UpdatedAt.
	if tickets[0].Subject != "Bravo" {
		t.Fatalf("expected first ticket to be Bravo (newest UpdatedAt), got %s", tickets[0].Subject)
	}
}

func TestStoreAddComment(t *testing.T) {
	store := NewStore()
	ticket := store.Create("cust-1", "Test", "", P2Medium)

	c, err := store.AddComment(ticket.ID, "agent", "Working on it")
	if err != nil {
		t.Fatalf("add comment: %v", err)
	}
	if c.Author != "agent" {
		t.Fatalf("expected author 'agent', got %s", c.Author)
	}
	if len(ticket.Comments) != 1 {
		t.Fatalf("expected 1 comment, got %d", len(ticket.Comments))
	}

	// Comment on non-existent ticket.
	_, err = store.AddComment("TKT-999999", "agent", "test")
	if err == nil {
		t.Fatal("expected error for missing ticket")
	}
}

// --- Router tests ---

func TestRouterAssignment(t *testing.T) {
	router := NewRouter("general-support")
	router.AddRule(P0Critical, "on-call-sre")
	router.AddRule(P1High, "senior-engineers")

	store := NewStore()
	ticket := store.Create("cust-1", "Outage", "Production down", P0Critical)

	if err := router.Route(ticket, store); err != nil {
		t.Fatalf("route: %v", err)
	}
	if ticket.AssignedTo != "on-call-sre" {
		t.Fatalf("expected on-call-sre, got %s", ticket.AssignedTo)
	}
	if ticket.Status != StatusTriaged {
		t.Fatalf("expected triaged, got %s", ticket.Status)
	}

	// Fallback for P3.
	ticket2 := store.Create("cust-2", "Question", "How to configure?", P3Low)
	if err := router.Route(ticket2, store); err != nil {
		t.Fatalf("route fallback: %v", err)
	}
	if ticket2.AssignedTo != "general-support" {
		t.Fatalf("expected general-support, got %s", ticket2.AssignedTo)
	}
}

func TestRouterNoFallback(t *testing.T) {
	router := NewRouter("")
	store := NewStore()
	ticket := store.Create("cust-1", "Test", "", P3Low)
	if err := router.Route(ticket, store); err == nil {
		t.Fatal("expected error with no routing rule and no fallback")
	}
}

// --- SLA tests ---

func TestSLAResponseBreach(t *testing.T) {
	tracker := NewSLATracker(DefaultSLAPolicies())

	var breaches []Breach
	tracker.OnBreach(func(b Breach) {
		breaches = append(breaches, b)
	})

	store := NewStore()
	ticket := store.Create("cust-1", "Outage", "", P0Critical)

	// Check before response deadline — no breach.
	soon := ticket.CreatedAt.Add(10 * time.Minute)
	b := tracker.Check(ticket, soon)
	if len(b) != 0 {
		t.Fatalf("expected no breaches, got %d", len(b))
	}

	// Check after response deadline — breach.
	late := ticket.CreatedAt.Add(20 * time.Minute)
	b = tracker.Check(ticket, late)
	if len(b) == 0 {
		t.Fatal("expected response breach")
	}
	found := false
	for _, br := range b {
		if br.Type == BreachResponse {
			found = true
		}
	}
	if !found {
		t.Fatal("expected response breach type")
	}
	if len(breaches) == 0 {
		t.Fatal("expected breach handler to have been called")
	}
}

func TestSLAResolutionBreach(t *testing.T) {
	tracker := NewSLATracker(DefaultSLAPolicies())
	store := NewStore()
	ticket := store.Create("cust-1", "Slow API", "", P2Medium)

	// Triage it so response SLA is met.
	ticket.Transition(StatusTriaged, ticket.CreatedAt.Add(1*time.Hour))

	// Check well after resolution deadline (72h).
	late := ticket.CreatedAt.Add(80 * time.Hour)
	b := tracker.Check(ticket, late)
	found := false
	for _, br := range b {
		if br.Type == BreachResolution {
			found = true
		}
	}
	if !found {
		t.Fatal("expected resolution breach")
	}
}

func TestSLANoBreachWhenResolved(t *testing.T) {
	tracker := NewSLATracker(DefaultSLAPolicies())
	store := NewStore()
	ticket := store.Create("cust-1", "Bug", "", P1High)

	now := ticket.CreatedAt.Add(30 * time.Minute)
	ticket.Transition(StatusTriaged, now)
	ticket.Transition(StatusInProgress, now)
	ticket.Transition(StatusResolved, now)

	// Even well past deadline, no breach since resolved.
	late := ticket.CreatedAt.Add(48 * time.Hour)
	b := tracker.Check(ticket, late)
	if len(b) != 0 {
		t.Fatalf("expected no breaches for resolved ticket, got %d", len(b))
	}
}

func TestSLACheckAll(t *testing.T) {
	tracker := NewSLATracker(DefaultSLAPolicies())
	store := NewStore()
	store.Create("cust-1", "T1", "", P0Critical)
	store.Create("cust-2", "T2", "", P1High)

	// Way past all deadlines.
	late := time.Now().Add(200 * time.Hour)
	breaches := tracker.CheckAll(store, late)
	if len(breaches) < 2 {
		t.Fatalf("expected at least 2 breaches, got %d", len(breaches))
	}
}

// --- Webhook tests ---

func TestWebhookDispatch(t *testing.T) {
	var received []WebhookEvent
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var event WebhookEvent
		json.NewDecoder(r.Body).Decode(&event)
		received = append(received, event)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Use a plain HTTP client (no SSRF protection) since httptest
	// servers bind to 127.0.0.1 which would be blocked.
	dispatcher := &WebhookDispatcher{
		client: &http.Client{Timeout: 10 * time.Second},
	}
	dispatcher.Register(WebhookTarget{
		Name: "test-hook",
		URL:  server.URL,
	})

	event := WebhookEvent{
		Type:      EventTicketCreated,
		Timestamp: time.Now().UTC(),
		Payload:   map[string]string{"id": "TKT-000001"},
	}
	errs := dispatcher.Dispatch(t.Context(), event)
	if len(errs) != 0 {
		t.Fatalf("dispatch errors: %v", errs)
	}
	if len(received) != 1 {
		t.Fatalf("expected 1 event, got %d", len(received))
	}
	if received[0].Type != EventTicketCreated {
		t.Fatalf("expected ticket.created, got %s", received[0].Type)
	}
}

func TestWebhookEventFilter(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Use a plain HTTP client (no SSRF protection) since httptest
	// servers bind to 127.0.0.1 which would be blocked.
	dispatcher := &WebhookDispatcher{
		client: &http.Client{Timeout: 10 * time.Second},
	}
	dispatcher.Register(WebhookTarget{
		Name:   "sla-only",
		URL:    server.URL,
		Events: []EventType{EventSLABreach},
	})

	// Send a ticket.created event — should not fire.
	event := WebhookEvent{Type: EventTicketCreated, Timestamp: time.Now().UTC()}
	dispatcher.Dispatch(t.Context(), event)
	if called {
		t.Fatal("expected webhook not to fire for filtered event")
	}
}

// --- API handler tests ---

func newTestAPI() *API {
	store := NewStore()
	router := NewRouter("general-support")
	router.AddRule(P0Critical, "on-call-sre")
	sla := NewSLATracker(DefaultSLAPolicies())
	return &API{
		Store:    store,
		Router:   router,
		SLA:      sla,
		Webhooks: NewWebhookDispatcher(),
	}
}

func TestAPICreateTicket(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	body := `{"customer_id":"cust-1","subject":"Help","body":"Need help","priority":1}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}

	var ticket Ticket
	json.NewDecoder(rr.Body).Decode(&ticket)
	if ticket.ID == "" {
		t.Fatal("expected ticket ID")
	}
	if ticket.AssignedTo == "" {
		t.Fatal("expected ticket to be assigned via router")
	}
}

func TestAPICreateTicketValidation(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	body := `{"customer_id":"","subject":""}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rr.Code)
	}
}

func TestAPIListTickets(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create two tickets.
	for _, subj := range []string{"A", "B"} {
		body := `{"customer_id":"cust-1","subject":"` + subj + `","priority":2}`
		req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)
	}

	req := httptest.NewRequest("GET", "/support/tickets?customer_id=cust-1", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	var tickets []Ticket
	json.NewDecoder(rr.Body).Decode(&tickets)
	if len(tickets) != 2 {
		t.Fatalf("expected 2 tickets, got %d", len(tickets))
	}
}

func TestAPIGetTicket(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create a ticket.
	body := `{"customer_id":"cust-1","subject":"Test","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	var created Ticket
	json.NewDecoder(rr.Body).Decode(&created)

	// Get it.
	req = httptest.NewRequest("GET", "/support/tickets/"+created.ID, nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
}

func TestAPIGetTicketNotFound(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/support/tickets/TKT-999999", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rr.Code)
	}
}

func TestAPIAddComment(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create ticket.
	body := `{"customer_id":"cust-1","subject":"Test","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	var created Ticket
	json.NewDecoder(rr.Body).Decode(&created)

	// Add comment.
	cbody := `{"author":"agent","body":"Looking into it"}`
	req = httptest.NewRequest("POST", "/support/tickets/"+created.ID+"/comments", strings.NewReader(cbody))
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPICloseTicket(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create ticket.
	body := `{"customer_id":"cust-1","subject":"Test","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	var created Ticket
	json.NewDecoder(rr.Body).Decode(&created)

	// Close it (triaged -> closed is valid).
	req = httptest.NewRequest("POST", "/support/tickets/"+created.ID+"/close", nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var closed Ticket
	json.NewDecoder(rr.Body).Decode(&closed)
	if closed.Status != StatusClosed {
		t.Fatalf("expected closed, got %s", closed.Status)
	}
}

func TestAPICloseTicketConflictJSONEscaping(t *testing.T) {
	api := newTestAPI()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create and close a ticket so that a second close triggers a conflict error.
	body := `{"customer_id":"cust-1","subject":"Test","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	var created Ticket
	json.NewDecoder(rr.Body).Decode(&created)

	// First close succeeds.
	req = httptest.NewRequest("POST", "/support/tickets/"+created.ID+"/close", nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 on first close, got %d: %s", rr.Code, rr.Body.String())
	}

	// Second close should return 409 Conflict with a properly encoded JSON body.
	req = httptest.NewRequest("POST", "/support/tickets/"+created.ID+"/close", nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d", rr.Code)
	}

	// The response must be valid JSON with the error message properly escaped.
	var errResp map[string]string
	if err := json.NewDecoder(rr.Body).Decode(&errResp); err != nil {
		t.Fatalf("response is not valid JSON: %v; body: %s", err, rr.Body.String())
	}
	if errResp["error"] == "" {
		t.Fatal("expected non-empty error field in JSON response")
	}

	// Verify Content-Type header.
	ct := rr.Header().Get("Content-Type")
	if !strings.Contains(ct, "application/json") {
		t.Fatalf("expected application/json content type, got %s", ct)
	}
}

// --- Auth middleware tests ---

// testAuth returns an AuthFunc that maps "token-<tenantID>" to tenantID.
func testAuth(token string) (string, bool) {
	if strings.HasPrefix(token, "token-") {
		return strings.TrimPrefix(token, "token-"), true
	}
	return "", false
}

func newTestAPIWithAuth() *API {
	api := newTestAPI()
	api.Auth = testAuth
	return api
}

func TestAPIUnauthenticatedReturns401(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// No Authorization header at all.
	req := httptest.NewRequest("GET", "/support/tickets?customer_id=cust-1", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPIInvalidTokenReturns401(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/support/tickets?customer_id=cust-1", nil)
	req.Header.Set("Authorization", "Bearer bad-token")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPICrossTenantListReturns403(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Authenticated as cust-1, but requesting cust-2's tickets.
	req := httptest.NewRequest("GET", "/support/tickets?customer_id=cust-2", nil)
	req.Header.Set("Authorization", "Bearer token-cust-1")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPICrossTenantGetReturns403(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create a ticket as cust-1.
	body := `{"customer_id":"cust-1","subject":"My ticket","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer token-cust-1")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("setup: expected 201, got %d", rr.Code)
	}
	var created Ticket
	json.NewDecoder(rr.Body).Decode(&created)

	// Try to get it as cust-2.
	req = httptest.NewRequest("GET", "/support/tickets/"+created.ID, nil)
	req.Header.Set("Authorization", "Bearer token-cust-2")
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPICrossTenantCreateReturns403(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Authenticated as cust-1, but creating ticket for cust-2.
	body := `{"customer_id":"cust-2","subject":"Sneaky","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer token-cust-1")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPIOversizedBodyReturns413(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create valid JSON that exceeds 1 MiB. The body field contains a large string.
	bigValue := strings.Repeat("a", (1<<20)+1)
	bigBody := `{"customer_id":"cust-1","subject":"Test","body":"` + bigValue + `","priority":2}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(bigBody))
	req.Header.Set("Authorization", "Bearer token-cust-1")
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPIAuthenticatedRequestSucceeds(t *testing.T) {
	api := newTestAPIWithAuth()
	mux := http.NewServeMux()
	api.RegisterRoutes(mux)

	// Create ticket as cust-1 with valid auth.
	body := `{"customer_id":"cust-1","subject":"Help","body":"Need help","priority":1}`
	req := httptest.NewRequest("POST", "/support/tickets", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer token-cust-1")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", rr.Code, rr.Body.String())
	}

	// List own tickets.
	req = httptest.NewRequest("GET", "/support/tickets?customer_id=cust-1", nil)
	req.Header.Set("Authorization", "Bearer token-cust-1")
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAPIPriorityString(t *testing.T) {
	tests := []struct {
		p    Priority
		want string
	}{
		{P0Critical, "P0-Critical"},
		{P1High, "P1-High"},
		{P2Medium, "P2-Medium"},
		{P3Low, "P3-Low"},
		{Priority(99), "P99-Unknown"},
	}
	for _, tt := range tests {
		if got := tt.p.String(); got != tt.want {
			t.Errorf("Priority(%d).String() = %s, want %s", tt.p, got, tt.want)
		}
	}
}

// --- Bbolt backend tests ---

func TestBboltStoreBackend(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "tickets.db")
	backend, err := NewBboltStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("NewBboltStoreBackend: %v", err)
	}
	defer backend.Close()

	store := NewStore(WithStoreBackend(backend))

	// Create a ticket.
	ticket := store.Create("cust-1", "Cannot deploy", "Deployment fails on v2.3", P1High)
	if ticket.ID == "" {
		t.Fatal("expected non-empty ticket ID")
	}
	if ticket.Status != StatusOpen {
		t.Fatalf("expected open, got %s", ticket.Status)
	}

	// Get the ticket back.
	got, ok := store.Get(ticket.ID)
	if !ok {
		t.Fatal("expected to find ticket")
	}
	if got.ID != ticket.ID {
		t.Fatalf("expected ID %s, got %s", ticket.ID, got.ID)
	}
	if got.CustomerID != "cust-1" {
		t.Fatalf("expected cust-1, got %s", got.CustomerID)
	}
	if got.Subject != "Cannot deploy" {
		t.Fatalf("expected subject 'Cannot deploy', got %s", got.Subject)
	}

	// List by customer.
	store.Create("cust-2", "Other issue", "", P3Low)
	store.Create("cust-1", "Second ticket", "", P2Medium)

	list := store.ListByCustomer("cust-1")
	if len(list) != 2 {
		t.Fatalf("expected 2 tickets for cust-1, got %d", len(list))
	}

	// Close the ticket (transition open -> closed).
	now := time.Now().UTC()
	if err := ticket.Transition(StatusClosed, now); err != nil {
		t.Fatalf("transition to closed: %v", err)
	}
	// Persist the updated ticket.
	if err := backend.Save(ticket); err != nil {
		t.Fatalf("save after close: %v", err)
	}

	// Read it back and verify closed status round-trips.
	got2, ok := store.Get(ticket.ID)
	if !ok {
		t.Fatal("expected to find closed ticket")
	}
	if got2.Status != StatusClosed {
		t.Fatalf("expected closed, got %s", got2.Status)
	}
	if got2.ClosedAt.IsZero() {
		t.Fatal("expected non-zero ClosedAt after close")
	}

	// Delete.
	if err := backend.Delete(ticket.ID); err != nil {
		t.Fatalf("delete: %v", err)
	}
	_, ok = store.Get(ticket.ID)
	if ok {
		t.Fatal("expected ticket to be deleted")
	}

	// Verify round-trip by reopening the database.
	backend.Close()

	backend2, err := NewBboltStoreBackend(dbPath)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer backend2.Close()

	// The remaining two tickets should still be there.
	all := backend2.All()
	if len(all) != 2 {
		t.Fatalf("expected 2 tickets after reopen, got %d", len(all))
	}
}

