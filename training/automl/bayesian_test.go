package automl

import (
	"math"
	"testing"
)

func TestSuggest(t *testing.T) {
	params := []HParam{
		{Name: "lr", Min: 0.001, Max: 1.0},
		{Name: "dropout", Min: 0.0, Max: 0.5},
		{Name: "layers", Min: 1, Max: 10},
	}
	opt := NewBayesianOptimizer(params, 42)

	for i := 0; i < 10; i++ {
		id, suggested := opt.Suggest()
		if id != i {
			t.Errorf("expected trial ID %d, got %d", i, id)
		}
		for _, hp := range params {
			v, ok := suggested[hp.Name]
			if !ok {
				t.Errorf("missing param %s in trial %d", hp.Name, id)
				continue
			}
			if v < hp.Min || v > hp.Max {
				t.Errorf("param %s=%f out of bounds [%f, %f] in trial %d",
					hp.Name, v, hp.Min, hp.Max, id)
			}
		}
		// Report so EI kicks in after exploration phase.
		_ = opt.Report(id, float64(i)*0.1)
	}
}

func TestReport(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	opt := NewBayesianOptimizer(params, 1)

	id0, _ := opt.Suggest()
	id1, _ := opt.Suggest()

	if err := opt.Report(id0, 0.5); err != nil {
		t.Fatal(err)
	}
	if err := opt.Report(id1, 0.9); err != nil {
		t.Fatal(err)
	}

	best, ok := opt.BestTrial()
	if !ok {
		t.Fatal("expected best trial")
	}
	if best.ID != id1 {
		t.Errorf("expected best trial ID %d, got %d", id1, best.ID)
	}
	if best.Score != 0.9 {
		t.Errorf("expected best score 0.9, got %f", best.Score)
	}

	// Double report should error.
	if err := opt.Report(id0, 0.3); err == nil {
		t.Error("expected error on double report")
	}

	// Unknown trial should error.
	if err := opt.Report(999, 0.1); err == nil {
		t.Error("expected error on unknown trial")
	}
}

func TestBestTrial(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	opt := NewBayesianOptimizer(params, 7)

	// No trials yet.
	_, ok := opt.BestTrial()
	if ok {
		t.Error("expected ok=false when no trials completed")
	}

	// Suggest but don't report.
	opt.Suggest()
	_, ok = opt.BestTrial()
	if ok {
		t.Error("expected ok=false when no trials reported")
	}

	// Report and check.
	id1, _ := opt.Suggest()
	_ = opt.Report(id1, 0.7)
	best, ok := opt.BestTrial()
	if !ok {
		t.Fatal("expected ok=true")
	}
	if best.Score != 0.7 {
		t.Errorf("expected score 0.7, got %f", best.Score)
	}
}

func TestLogSpaceParam(t *testing.T) {
	params := []HParam{
		{Name: "lr", Min: 1e-6, Max: 1.0, IsLog: true},
	}
	opt := NewBayesianOptimizer(params, 99)

	var values []float64
	for i := 0; i < 50; i++ {
		id, suggested := opt.Suggest()
		v := suggested["lr"]
		if v < 1e-6 || v > 1.0 {
			t.Errorf("param lr=%e out of bounds", v)
		}
		values = append(values, v)
		_ = opt.Report(id, -v) // arbitrary score
	}

	// Check that values span several orders of magnitude.
	// With log-space sampling over [1e-6, 1.0], we expect values
	// in both small (< 1e-3) and large (> 1e-2) ranges.
	var hasSmall, hasLarge bool
	for _, v := range values {
		if v < 1e-3 {
			hasSmall = true
		}
		if v > 1e-2 {
			hasLarge = true
		}
	}
	if !hasSmall {
		t.Error("log-space sampling produced no values < 1e-3; expected multi-magnitude coverage")
	}
	if !hasLarge {
		t.Error("log-space sampling produced no values > 1e-2; expected multi-magnitude coverage")
	}
}

func TestExploration(t *testing.T) {
	params := []HParam{
		{Name: "x", Min: 0, Max: 1},
		{Name: "y", Min: 0, Max: 100},
	}
	opt := NewBayesianOptimizer(params, 123)

	// Run 20 trials and check that params cover the search space.
	var xs, ys []float64
	for i := 0; i < 20; i++ {
		id, suggested := opt.Suggest()
		xs = append(xs, suggested["x"])
		ys = append(ys, suggested["y"])
		// Score based on a simple quadratic so EI has something to work with.
		score := -(math.Pow(suggested["x"]-0.5, 2) + math.Pow((suggested["y"]-50)/100, 2))
		_ = opt.Report(id, score)
	}

	// Check that values are not all identical (diversity).
	if allSame(xs) {
		t.Error("all x values identical; expected diverse exploration")
	}
	if allSame(ys) {
		t.Error("all y values identical; expected diverse exploration")
	}

	// Check that values cover a reasonable portion of the space.
	// We split each dimension into quartiles and require at least 2 quartiles hit.
	xQuartiles := countQuartiles(xs, 0, 1)
	yQuartiles := countQuartiles(ys, 0, 100)
	if xQuartiles < 2 {
		t.Errorf("x values cover only %d quartile(s); expected at least 2", xQuartiles)
	}
	if yQuartiles < 2 {
		t.Errorf("y values cover only %d quartile(s); expected at least 2", yQuartiles)
	}
}

func TestTrials(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	opt := NewBayesianOptimizer(params, 5)

	id0, _ := opt.Suggest()
	id1, _ := opt.Suggest()
	_ = opt.Report(id0, 1.0)

	trials := opt.Trials()
	if len(trials) != 2 {
		t.Fatalf("expected 2 trials, got %d", len(trials))
	}
	if trials[0].ID != id0 || !trials[0].Done {
		t.Error("trial 0 should be done")
	}
	if trials[1].ID != id1 || trials[1].Done {
		t.Error("trial 1 should not be done")
	}

	// Verify returned slice is a copy.
	trials[0].Score = 999
	original := opt.Trials()
	if original[0].Score == 999 {
		t.Error("Trials() should return a copy")
	}
}

func allSame(vals []float64) bool {
	if len(vals) == 0 {
		return true
	}
	for _, v := range vals[1:] {
		if v != vals[0] {
			return false
		}
	}
	return true
}

func countQuartiles(vals []float64, min, max float64) int {
	span := max - min
	hit := [4]bool{}
	for _, v := range vals {
		q := int((v - min) / span * 4)
		if q >= 4 {
			q = 3
		}
		if q < 0 {
			q = 0
		}
		hit[q] = true
	}
	count := 0
	for _, h := range hit {
		if h {
			count++
		}
	}
	return count
}
