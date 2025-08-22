package graph

import "errors"

// ErrInvalidInputCount is returned when the number of inputs to a node is incorrect.
var ErrInvalidInputCount = errors.New("invalid number of inputs")
