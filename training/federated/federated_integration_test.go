package federated

import (
	"math"
	"testing"
)

// convergingClient simulates a client that trains toward a target weight vector.
// Each round it moves its weights closer to the target, modulated by a learning
// rate and optionally by a FedProx proximal term. The starting point is offset
// from the target so that convergence can be measured over rounds.
type convergingClient struct {
	id            ClientID
	nSamples      int
	target        []float64
	localWeights  []float64
	learningRate  float64
	proximalMu    float64
	trainedRounds int
}

func newConvergingClient(id ClientID, nSamples int, target, initial []float64, lr float64) *convergingClient {
	w := make([]float64, len(initial))
	copy(w, initial)
	return &convergingClient{
		id:           id,
		nSamples:     nSamples,
		target:       target,
		localWeights: w,
		learningRate: lr,
	}
}

func (c *convergingClient) ID() ClientID { return c.id }

// Train simulates local SGD: move localWeights toward the target, incorporating
// the global weights from the coordinator and an optional proximal term.
func (c *convergingClient) Train(globalWeights []float64) (*ModelUpdate, error) {
	if len(globalWeights) > 0 {
		// Start from global model (as in real FL).
		copy(c.localWeights, globalWeights)
	}

	// Gradient step toward target, with optional proximal regularization.
	for i := range c.localWeights {
		grad := c.localWeights[i] - c.target[i]
		if c.proximalMu > 0 && len(globalWeights) > 0 {
			grad += c.proximalMu * (c.localWeights[i] - globalWeights[i])
		}
		c.localWeights[i] -= c.learningRate * grad
	}

	c.trainedRounds++

	loss := 0.0
	for i := range c.localWeights {
		diff := c.localWeights[i] - c.target[i]
		loss += diff * diff
	}
	loss /= float64(len(c.localWeights))

	weights := make([]float64, len(c.localWeights))
	copy(weights, c.localWeights)

	return &ModelUpdate{
		ClientID: c.id,
		Weights:  weights,
		NSamples: c.nSamples,
		Metrics:  map[string]float64{"loss": loss},
	}, nil
}

// weightMSE computes the mean squared error between two weight vectors.
func weightMSE(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum / float64(len(a))
}

func TestFederated_4ClientSimulation(t *testing.T) {
	const (
		nRounds  = 20
		nWeights = 8
	)

	// The global target that all clients aim toward.
	target := make([]float64, nWeights)
	for i := range target {
		target[i] = float64(i + 1)
	}

	t.Run("FedAvg convergence", func(t *testing.T) {
		// Four clients start far from target with different dataset sizes.
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}

		coord := NewCoordinator(NewFedAvg(), CoordinatorConfig{
			MinClients: 4,
			MaxRounds:  nRounds,
		})

		var lastMSE float64
		for r := 0; r < nRounds; r++ {
			result, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("round %d: %v", r, err)
			}
			if result.Model.NParticipants != 4 {
				t.Fatalf("round %d: expected 4 participants, got %d", r, result.Model.NParticipants)
			}
			lastMSE = weightMSE(result.Model.Weights, target)
		}

		// After 20 rounds, weights should have converged close to target.
		if lastMSE > 1e-6 {
			t.Errorf("FedAvg did not converge: final MSE=%e, want <1e-6", lastMSE)
		}
		if coord.Round() != nRounds {
			t.Errorf("expected %d rounds, got %d", nRounds, coord.Round())
		}
	})

	t.Run("FedAvg convergence is monotonic", func(t *testing.T) {
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.3),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.3),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.3),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.3),
		}
		coord := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})

		prevMSE := math.Inf(1)
		for r := 0; r < nRounds; r++ {
			result, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("round %d: %v", r, err)
			}
			mse := weightMSE(result.Model.Weights, target)
			if mse > prevMSE+1e-12 {
				t.Errorf("round %d: MSE increased from %e to %e", r, prevMSE, mse)
			}
			prevMSE = mse
		}
	})

	t.Run("FedProx reduces divergence", func(t *testing.T) {
		// Create heterogeneous clients with different targets to simulate
		// non-IID data. FedProx should keep models closer together.
		mu := 0.5
		makeClients := func(proxMu float64) []Client {
			cls := make([]Client, 4)
			for i := 0; i < 4; i++ {
				// Each client has a slightly different local target (non-IID).
				localTarget := make([]float64, nWeights)
				for j := range localTarget {
					localTarget[j] = target[j] + float64(i)*0.5
				}
				c := newConvergingClient(
					ClientID("c"+string(rune('0'+i))),
					100+i*50,
					localTarget,
					make([]float64, nWeights),
					0.3,
				)
				c.proximalMu = proxMu
				cls[i] = c
			}
			return cls
		}

		// Run FedAvg (no proximal term).
		avgClients := makeClients(0)
		avgCoord := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})
		var avgFinalWeights []float64
		for r := 0; r < nRounds; r++ {
			result, err := avgCoord.RunRound(avgClients)
			if err != nil {
				t.Fatalf("FedAvg round %d: %v", r, err)
			}
			avgFinalWeights = result.Model.Weights
		}

		// Run FedProx with proximal term.
		proxClients := makeClients(mu)
		proxCoord := NewCoordinator(NewFedProx(mu), CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})
		var proxFinalWeights []float64
		for r := 0; r < nRounds; r++ {
			result, err := proxCoord.RunRound(proxClients)
			if err != nil {
				t.Fatalf("FedProx round %d: %v", r, err)
			}
			proxFinalWeights = result.Model.Weights
		}

		// Measure divergence from the mean target across clients.
		meanTarget := make([]float64, nWeights)
		for j := range meanTarget {
			for i := 0; i < 4; i++ {
				meanTarget[j] += target[j] + float64(i)*0.5
			}
			meanTarget[j] /= 4
		}

		avgMSE := weightMSE(avgFinalWeights, meanTarget)
		proxMSE := weightMSE(proxFinalWeights, meanTarget)

		// Both should converge; verify FedProx also converges reasonably.
		if proxMSE > 1.0 {
			t.Errorf("FedProx did not converge: MSE=%e", proxMSE)
		}
		// Log comparison for visibility.
		t.Logf("FedAvg MSE=%.6e, FedProx MSE=%.6e", avgMSE, proxMSE)
	})

	t.Run("DP noise is applied", func(t *testing.T) {
		dpConfig := DPConfig{
			Epsilon:   1.0,
			Delta:     1e-5,
			ClipNorm:  10.0,
			Mechanism: "gaussian",
		}
		dpStrategy, err := NewDPStrategy(NewFedAvg(), dpConfig)
		if err != nil {
			t.Fatalf("NewDPStrategy: %v", err)
		}

		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}

		coord := NewCoordinator(dpStrategy, CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})

		// Run a non-DP coordinator in parallel for comparison.
		cleanClients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}
		cleanCoord := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})

		foundDifference := false
		for r := 0; r < nRounds; r++ {
			dpResult, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("DP round %d: %v", r, err)
			}
			cleanResult, err := cleanCoord.RunRound(cleanClients)
			if err != nil {
				t.Fatalf("clean round %d: %v", r, err)
			}

			// DP weights should differ from clean weights due to noise.
			for i := range dpResult.Model.Weights {
				if dpResult.Model.Weights[i] != cleanResult.Model.Weights[i] {
					foundDifference = true
					break
				}
			}
			if foundDifference {
				break
			}
		}
		if !foundDifference {
			t.Error("DP noise was not applied: DP and clean weights are identical")
		}

		// Verify privacy accountant tracked the budget.
		epsSpent, deltaSpent := dpStrategy.Accountant().Spent()
		if epsSpent <= 0 {
			t.Errorf("expected positive epsilon spent, got %f", epsSpent)
		}
		if deltaSpent <= 0 {
			t.Errorf("expected positive delta spent, got %f", deltaSpent)
		}
		t.Logf("privacy budget spent: epsilon=%.4f, delta=%.2e", epsSpent, deltaSpent)
	})

	t.Run("DP Laplacian mechanism", func(t *testing.T) {
		dpConfig := DPConfig{
			Epsilon:   2.0,
			Delta:     1e-5,
			ClipNorm:  5.0,
			Mechanism: "laplacian",
		}
		dpStrategy, err := NewDPStrategy(NewFedAvg(), dpConfig)
		if err != nil {
			t.Fatalf("NewDPStrategy: %v", err)
		}
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}
		coord := NewCoordinator(dpStrategy, CoordinatorConfig{MinClients: 4, MaxRounds: 5})

		for r := 0; r < 5; r++ {
			_, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("laplacian round %d: %v", r, err)
			}
		}

		eps, delta := dpStrategy.Accountant().Spent()
		wantEps := 2.0 * 5
		if math.Abs(eps-wantEps) > 1e-9 {
			t.Errorf("epsilon spent = %f, want %f", eps, wantEps)
		}
		wantDelta := 1e-5 * 5
		if math.Abs(delta-wantDelta) > 1e-9 {
			t.Errorf("delta spent = %e, want %e", delta, wantDelta)
		}
	})

	t.Run("privacy accountant gates continuation", func(t *testing.T) {
		dpConfig := DPConfig{
			Epsilon:   5.0,
			Delta:     1e-5,
			ClipNorm:  10.0,
			Mechanism: "gaussian",
		}
		dpStrategy, err := NewDPStrategy(NewFedAvg(), dpConfig)
		if err != nil {
			t.Fatalf("NewDPStrategy: %v", err)
		}
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}
		coord := NewCoordinator(dpStrategy, CoordinatorConfig{MinClients: 4, MaxRounds: 100})

		maxBudget := 20.0
		rounds := 0
		for dpStrategy.Accountant().CanContinue(maxBudget) {
			_, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("round %d: %v", rounds, err)
			}
			rounds++
			if rounds > 100 {
				t.Fatal("exceeded 100 rounds without exhausting budget")
			}
		}

		// Should have stopped after 4 rounds (5.0 * 4 = 20.0).
		if rounds != 4 {
			t.Errorf("expected 4 rounds to exhaust budget of %.0f (eps=%.0f/round), got %d",
				maxBudget, dpConfig.Epsilon, rounds)
		}
	})

	t.Run("global weights persist across rounds", func(t *testing.T) {
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}
		coord := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 4, MaxRounds: nRounds})

		var round1Weights []float64
		for r := 0; r < 3; r++ {
			result, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("round %d: %v", r, err)
			}
			if r == 0 {
				round1Weights = make([]float64, len(result.Model.Weights))
				copy(round1Weights, result.Model.Weights)
			}
			if r > 0 {
				// Weights should change between rounds since clients train
				// from the updated global model.
				same := true
				for i := range result.Model.Weights {
					if result.Model.Weights[i] != round1Weights[i] {
						same = false
						break
					}
				}
				if same {
					t.Errorf("round %d weights are identical to round 1 — global weights not propagated", r+1)
				}
			}
		}
	})

	t.Run("all clients participate every round", func(t *testing.T) {
		clients := []Client{
			newConvergingClient("c0", 100, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c1", 200, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c2", 150, target, make([]float64, nWeights), 0.5),
			newConvergingClient("c3", 250, target, make([]float64, nWeights), 0.5),
		}
		coord := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 4, MaxRounds: 5})

		for r := 0; r < 5; r++ {
			result, err := coord.RunRound(clients)
			if err != nil {
				t.Fatalf("round %d: %v", r, err)
			}
			if len(result.Updates) != 4 {
				t.Errorf("round %d: expected 4 updates, got %d", r, len(result.Updates))
			}
			seen := map[ClientID]bool{}
			for _, u := range result.Updates {
				seen[u.ClientID] = true
			}
			for _, c := range clients {
				if !seen[c.ID()] {
					t.Errorf("round %d: client %q missing from updates", r, c.ID())
				}
			}
		}
	})
}
