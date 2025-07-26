package params

// Param represents a trainable model parameter (e.g., weights or biases).
type Param struct {
    Name  string
    Value interface{}  // placeholder for tensor value
    Grad  interface{}  // placeholder for gradient tensor
}

// Store manages parameters and their gradients (akin to a ParameterStore).
// TODO: Link each parameter with its gradient and provide centralized update utilities:contentReference[oaicite:14]{index=14}.
type Store struct {
    params map[string]*Param
}

// Register adds a new parameter to the store.
func (s *Store) Register(p *Param) {
    if s.params == nil {
        s.params = make(map[string]*Param)
    }
    s.params[p.Name] = p
}

// Get retrieves a parameter by name.
func (s *Store) Get(name string) *Param {
    return s.params[name]
}
