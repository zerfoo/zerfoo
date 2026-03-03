package graph

import (
	"context"
	"sync"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGraph_ConcurrentForward verifies that concurrent Forward calls do not
// race on the internal memo map. Run with -race to detect data races.
func TestGraph_ConcurrentForward(t *testing.T) {
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	b := NewBuilder[int](engine)

	in := b.Input([]int{2})
	node := &mockNode{name: "passthrough"}
	b.AddNode(node, in)

	g, err := b.Build(node)
	if err != nil {
		t.Fatal(err)
	}

	const goroutines = 8

	var wg sync.WaitGroup
	wg.Add(goroutines)

	errs := make(chan error, goroutines)

	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			data := []int{id, id + 1}
			inp, tErr := tensor.New[int]([]int{2}, data)
			if tErr != nil {
				errs <- tErr
				return
			}
			out, fErr := g.Forward(context.Background(), inp)
			if fErr != nil {
				errs <- fErr
				return
			}
			if out == nil {
				t.Errorf("goroutine %d: output is nil", id)
			}
		}(i)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent Forward error: %v", err)
	}
}
