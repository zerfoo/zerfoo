package federated

import "errors"

// ClientID uniquely identifies a federated learning participant.
type ClientID string

// ModelUpdate holds the result of a client's local training round.
type ModelUpdate struct {
	ClientID ClientID
	Weights  []float64
	NSamples int
	Metrics  map[string]float64
}

// AggregatedModel holds the result of aggregating client updates.
type AggregatedModel struct {
	Weights       []float64
	Round         int
	NParticipants int
}

// Strategy defines the federated aggregation and client selection policy.
type Strategy interface {
	// Aggregate combines multiple client updates into a single model.
	Aggregate(updates []ModelUpdate) (*AggregatedModel, error)
	// SelectClients chooses which clients participate in the given round.
	SelectClients(round int, available []ClientID) []ClientID
}

// Client represents a federated learning participant.
type Client interface {
	// Train performs local training starting from globalWeights and returns
	// the resulting model update.
	Train(globalWeights []float64) (*ModelUpdate, error)
	// ID returns the client's unique identifier.
	ID() ClientID
}

// CoordinatorConfig holds configuration for a federated learning coordinator.
type CoordinatorConfig struct {
	// MinClients is the minimum number of clients required to run a round.
	MinClients int
	// MaxRounds is the maximum number of federated rounds to execute.
	MaxRounds int
	// ConvergenceThreshold stops training when the aggregated loss delta
	// falls below this value. Zero disables convergence checking.
	ConvergenceThreshold float64
}

// RoundResult holds the outcome of a single federated round.
type RoundResult struct {
	Model   *AggregatedModel
	Updates []ModelUpdate
}

// Coordinator manages federated learning rounds using a pluggable strategy.
type Coordinator struct {
	strategy      Strategy
	config        CoordinatorConfig
	round         int
	globalWeights []float64
}

// NewCoordinator creates a coordinator with the given strategy and config.
func NewCoordinator(strategy Strategy, config CoordinatorConfig) *Coordinator {
	return &Coordinator{
		strategy: strategy,
		config:   config,
	}
}

// RunRound executes a single federated learning round. It selects clients,
// distributes global weights (empty on the first round), collects updates,
// and aggregates them via the strategy.
func (c *Coordinator) RunRound(clients []Client) (*RoundResult, error) {
	if len(clients) == 0 {
		return nil, errors.New("federated: no clients provided")
	}

	// Build available client IDs.
	available := make([]ClientID, len(clients))
	clientByID := make(map[ClientID]Client, len(clients))
	for i, cl := range clients {
		id := cl.ID()
		available[i] = id
		clientByID[id] = cl
	}

	// Select participants for this round.
	selected := c.strategy.SelectClients(c.round, available)
	if len(selected) < c.config.MinClients {
		return nil, errors.New("federated: not enough clients selected")
	}

	// Collect updates from selected clients.
	var updates []ModelUpdate
	for _, id := range selected {
		cl, ok := clientByID[id]
		if !ok {
			continue
		}
		update, err := cl.Train(c.globalWeights)
		if err != nil {
			return nil, err
		}
		updates = append(updates, *update)
	}

	if len(updates) == 0 {
		return nil, errors.New("federated: no updates received")
	}

	// Aggregate updates.
	c.round++
	agg, err := c.strategy.Aggregate(updates)
	if err != nil {
		return nil, err
	}
	agg.Round = c.round
	c.globalWeights = agg.Weights

	return &RoundResult{
		Model:   agg,
		Updates: updates,
	}, nil
}

// Round returns the current round number.
func (c *Coordinator) Round() int {
	return c.round
}
