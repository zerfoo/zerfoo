package layers

// Layer is the interface that all neural network layers must implement (e.g., Dense, Conv).
// It defines the basic operations for forward and backward propagation:contentReference[oaicite:23]{index=23}.
type Layer interface {
    Forward(x interface{}) interface{}
    Backward(grads interface{}) interface{}
    // Update adjusts the layer's parameters based on gradients using an optimizer.
    Update(opt optimizer.Optimizer)
}

// Dense is a stub for a fully-connected layer.
type Dense struct {
    // Fields: weights, bias, etc.
}

// Forward computes the output of the Dense layer for the given input.
func (l *Dense) Forward(x interface{}) interface{} {
    // TODO: implement forward pass (e.g., x*W + b)
    return nil
}

// Backward computes the gradient of the Dense layer with respect to its input.
func (l *Dense) Backward(grads interface{}) interface{} {
    // TODO: implement backward pass (propagate gradients to previous layer)
    return nil
}

// Update updates the Dense layer's parameters using the provided optimizer.
func (l *Dense) Update(opt optimizer.Optimizer) {
    // TODO: use optimizer to update weights and bias based on stored gradients
}
