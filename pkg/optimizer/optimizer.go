package optimizer

// Optimizer defines the interface for optimization algorithms that update model parameters.
type Optimizer interface {
    // Step updates the given parameters based on stored gradients (e.g., one iteration of SGD or Adam).
    Step(params interface{})
}

// SGD is a simple stochastic gradient descent optimizer.
type SGD struct {
    LearningRate float64
}

// Step updates the parameters by subtracting the scaled gradients (stub implementation).
func (opt *SGD) Step(params interface{}) {
    // TODO: loop through params and apply SGD update: param.Value -= LearningRate * param.Grad
}
