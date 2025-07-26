package model

// Model defines the interface for a complete model, composed of layers and supporting training and inference.
type Model interface {
    // Forward performs a forward pass and returns the output predictions.
    Forward(input interface{}) interface{}
    // Train runs one training iteration on the given input and target, updating model parameters.
    Train(input interface{}, target interface{}) error
}

// Sequential is a simple model that applies a sequence of layers.
type Sequential struct {
    Layers []interface{}  // slice of layers (to be replaced with specific Layer interface type)
}

// Forward runs the input through all layers sequentially.
func (m *Sequential) Forward(input interface{}) interface{} {
    output := input
    for _, layer := range m.Layers {
        // Assume layer has a Forward method (type assertion for example)
        if l, ok := layer.(interface{ Forward(interface{}) interface{} }); ok {
            output = l.Forward(output)
        }
    }
    return output
}

// Train performs a single training step (forward + backward) on the given data.
func (m *Sequential) Train(input interface{}, target interface{}) error {
    // TODO: implement training step (forward pass, compute loss, backward pass, update parameters)
    return nil
}
