// Package support implements an enterprise support ticketing system with
// priority routing, SLA tracking, and webhook notifications.
package support

import (
	"fmt"
	"sync"
	"time"
)

// Status represents the lifecycle state of a ticket.
type Status string

const (
	StatusOpen       Status = "open"
	StatusTriaged    Status = "triaged"
	StatusInProgress Status = "in_progress"
	StatusResolved   Status = "resolved"
	StatusClosed     Status = "closed"
)

// validTransitions defines allowed state transitions.
var validTransitions = map[Status][]Status{
	StatusOpen:       {StatusTriaged, StatusClosed},
	StatusTriaged:    {StatusInProgress, StatusClosed},
	StatusInProgress: {StatusResolved, StatusClosed},
	StatusResolved:   {StatusClosed, StatusInProgress},
	StatusClosed:     {},
}

// ValidTransition reports whether transitioning from one status to another is allowed.
func ValidTransition(from, to Status) bool {
	for _, s := range validTransitions[from] {
		if s == to {
			return true
		}
	}
	return false
}

// Priority represents ticket urgency.
type Priority int

const (
	P0Critical Priority = iota
	P1High
	P2Medium
	P3Low
)

func (p Priority) String() string {
	switch p {
	case P0Critical:
		return "P0-Critical"
	case P1High:
		return "P1-High"
	case P2Medium:
		return "P2-Medium"
	case P3Low:
		return "P3-Low"
	default:
		return fmt.Sprintf("P%d-Unknown", p)
	}
}

// Comment is a message attached to a ticket.
type Comment struct {
	ID        string    `json:"id"`
	Author    string    `json:"author"`
	Body      string    `json:"body"`
	CreatedAt time.Time `json:"created_at"`
}

// Ticket is the core data model for a support ticket.
type Ticket struct {
	ID         string    `json:"id"`
	CustomerID string    `json:"customer_id"`
	Subject    string    `json:"subject"`
	Body       string    `json:"body"`
	Priority   Priority  `json:"priority"`
	Status     Status    `json:"status"`
	AssignedTo string    `json:"assigned_to,omitempty"`
	Comments   []Comment `json:"comments,omitempty"`
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
	ResolvedAt time.Time `json:"resolved_at,omitempty"`
	ClosedAt   time.Time `json:"closed_at,omitempty"`
}

// Transition moves the ticket to a new status, returning an error if the
// transition is invalid.
func (t *Ticket) Transition(to Status, now time.Time) error {
	if !ValidTransition(t.Status, to) {
		return fmt.Errorf("invalid transition from %s to %s", t.Status, to)
	}
	t.Status = to
	t.UpdatedAt = now
	switch to {
	case StatusResolved:
		t.ResolvedAt = now
	case StatusClosed:
		t.ClosedAt = now
	}
	return nil
}

// Store is a thread-safe in-memory ticket store.
type Store struct {
	mu      sync.RWMutex
	tickets map[string]*Ticket
	nextID  int
}

// NewStore creates an empty ticket store.
func NewStore() *Store {
	return &Store{tickets: make(map[string]*Ticket)}
}

// Create adds a new ticket and returns it.
func (s *Store) Create(customerID, subject, body string, priority Priority) *Ticket {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	now := time.Now().UTC()
	t := &Ticket{
		ID:         fmt.Sprintf("TKT-%06d", s.nextID),
		CustomerID: customerID,
		Subject:    subject,
		Body:       body,
		Priority:   priority,
		Status:     StatusOpen,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	s.tickets[t.ID] = t
	return t
}

// Get returns a ticket by ID.
func (s *Store) Get(id string) (*Ticket, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	t, ok := s.tickets[id]
	return t, ok
}

// ListByCustomer returns all tickets for a customer, ordered by creation time (newest first).
func (s *Store) ListByCustomer(customerID string) []*Ticket {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var result []*Ticket
	for _, t := range s.tickets {
		if t.CustomerID == customerID {
			result = append(result, t)
		}
	}
	// Sort newest first.
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		if result[i].CreatedAt.Before(result[j].CreatedAt) {
			result[i], result[j] = result[j], result[i]
		}
	}
	return result
}

// All returns every ticket in the store.
func (s *Store) All() []*Ticket {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]*Ticket, 0, len(s.tickets))
	for _, t := range s.tickets {
		result = append(result, t)
	}
	return result
}

// AddComment appends a comment to the ticket.
func (s *Store) AddComment(ticketID, author, body string) (*Comment, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tickets[ticketID]
	if !ok {
		return nil, fmt.Errorf("ticket %s not found", ticketID)
	}
	c := Comment{
		ID:        fmt.Sprintf("%s-C%d", ticketID, len(t.Comments)+1),
		Author:    author,
		Body:      body,
		CreatedAt: time.Now().UTC(),
	}
	t.Comments = append(t.Comments, c)
	t.UpdatedAt = c.CreatedAt
	return &c, nil
}
