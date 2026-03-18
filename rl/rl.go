package rl

// State represents an observation from the environment.
type State = []float64

// Action represents a decision made by an agent.
type Action = []float64

// Experience holds a single transition tuple for replay.
type Experience struct {
	State     State
	Action    Action
	Reward    float64
	NextState State
	Done      bool
}

// Environment defines the RL environment contract.
type Environment interface {
	// Reset initialises the environment and returns the starting state.
	Reset() State
	// Step advances the environment by one time step.
	// It returns the next state, the scalar reward, a done flag, and any error.
	Step(action Action) (next State, reward float64, done bool, err error)
}

// Agent defines the RL agent contract.
type Agent interface {
	// Act selects an action given the current state.
	Act(state State) Action
	// Learn updates the agent's parameters from a batch of experiences.
	Learn(batch []Experience) error
}
