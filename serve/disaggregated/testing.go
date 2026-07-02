package disaggregated

import (
	"context"
	"fmt"
	"time"
)

// NewTestGateway creates a Gateway with mock clients (no gRPC connections).
// This is intended for benchmarks and integration tests that operate outside
// the disaggregated package.
func NewTestGateway(prefillClients []PrefillClient, decodeClients []DecodeClient) *Gateway {
	g := &Gateway{
		healthy:        make(map[string]bool),
		healthInterval: time.Second,
		maxBackoff:     30 * time.Second,
	}

	for i, pc := range prefillClients {
		addr := fmt.Sprintf("prefill-%d", i)
		g.prefillWorkers = append(g.prefillWorkers, &workerEntry{
			addr:    addr,
			prefill: pc,
		})
		g.healthy[addr] = true
	}

	for i, dc := range decodeClients {
		addr := fmt.Sprintf("decode-%d", i)
		g.decodeWorkers = append(g.decodeWorkers, &workerEntry{
			addr:   addr,
			decode: dc,
		})
		g.healthy[addr] = true
	}

	_, cancel := context.WithCancel(context.Background())
	g.cancel = cancel

	return g
}
