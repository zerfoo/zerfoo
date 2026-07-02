package support

import "fmt"

// RoutingRule maps a priority to an assignee (team or individual).
type RoutingRule struct {
	Priority Priority
	Assignee string
}

// Router assigns tickets to teams or individuals based on priority.
type Router struct {
	rules    map[Priority]string
	fallback string
}

// NewRouter creates a router with the given fallback assignee used when no
// priority-specific rule matches.
func NewRouter(fallback string) *Router {
	return &Router{
		rules:    make(map[Priority]string),
		fallback: fallback,
	}
}

// AddRule registers a routing rule for a specific priority.
func (r *Router) AddRule(priority Priority, assignee string) {
	r.rules[priority] = assignee
}

// Assign determines the assignee for a ticket based on its priority and
// updates the ticket's AssignedTo field. It also transitions the ticket to
// triaged status.
func (r *Router) Assign(t *Ticket) error {
	assignee, ok := r.rules[t.Priority]
	if !ok {
		assignee = r.fallback
	}
	if assignee == "" {
		return fmt.Errorf("no routing rule for priority %s and no fallback configured", t.Priority)
	}
	t.AssignedTo = assignee
	return nil
}

// Route assigns and triages the ticket in one step.
func (r *Router) Route(t *Ticket, store *Store) error {
	if err := r.Assign(t); err != nil {
		return err
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	return t.Transition(StatusTriaged, t.UpdatedAt)
}
