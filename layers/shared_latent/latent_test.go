package shared_latent

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

// cosine returns the cosine similarity between two vectors.
func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// euclidean returns the Euclidean distance between two vectors.
func euclidean(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// TestLatentSpace_SharedRepresentation verifies that similar inputs from
// different models map to nearby points in the shared latent space.
func TestLatentSpace_SharedRepresentation(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	latentDim := 4
	ls := NewLatentSpace(latentDim, engine)

	// Two models with different input dimensions.
	ls.Register("modelA", 3)
	ls.Register("modelB", 5)

	// Generate aligned training data where corresponding samples encode the
	// same underlying concept. Model A gets 3D features, Model B gets 5D
	// features, but paired samples share a latent structure.
	nSamples := 200
	dataA := make([][]float64, nSamples)
	dataB := make([][]float64, nSamples)

	for i := 0; i < nSamples; i++ {
		// Shared latent signal.
		s := float64(i) / float64(nSamples)

		dataA[i] = []float64{s, s * 0.5, 1.0 - s}
		dataB[i] = []float64{s * 0.8, s * 0.3, 1.0 - s*0.9, s * 0.2, 0.5 - s*0.5}
	}

	err := ls.TrainProjections(ctx, map[string][][]float64{
		"modelA": dataA,
		"modelB": dataB,
	}, ProjectionConfig{
		LearningRate:    0.005,
		NEpochs:         300,
		AlignmentWeight: 2.0,
	})
	if err != nil {
		t.Fatalf("TrainProjections: %v", err)
	}

	// Test: similar inputs from different models should map to nearby
	// latent representations.
	idxLow := 10
	idxHigh := 190

	zA_low, err := ls.Project(ctx, "modelA", dataA[idxLow])
	if err != nil {
		t.Fatal(err)
	}
	zB_low, err := ls.Project(ctx, "modelB", dataB[idxLow])
	if err != nil {
		t.Fatal(err)
	}
	zA_high, err := ls.Project(ctx, "modelA", dataA[idxHigh])
	if err != nil {
		t.Fatal(err)
	}
	zB_high, err := ls.Project(ctx, "modelB", dataB[idxHigh])
	if err != nil {
		t.Fatal(err)
	}

	// Corresponding pairs (same index) should be closer than non-corresponding.
	distSame := euclidean(zA_low, zB_low)
	distDiff := euclidean(zA_low, zB_high)

	t.Logf("distance(A_low, B_low) = %.4f (same concept)", distSame)
	t.Logf("distance(A_low, B_high) = %.4f (different concept)", distDiff)

	if distSame >= distDiff {
		t.Errorf("aligned samples should be closer: same=%.4f >= diff=%.4f", distSame, distDiff)
	}

	// Also check that high-index pairs are aligned.
	distSameHigh := euclidean(zA_high, zB_high)
	distCross := euclidean(zA_high, zB_low)

	t.Logf("distance(A_high, B_high) = %.4f (same concept)", distSameHigh)
	t.Logf("distance(A_high, B_low) = %.4f (different concept)", distCross)

	if distSameHigh >= distCross {
		t.Errorf("aligned high samples should be closer: same=%.4f >= diff=%.4f", distSameHigh, distCross)
	}
}

// TestLatentSpace_TransferBenefit verifies that model B improves after
// training with model A's shared representations.
func TestLatentSpace_TransferBenefit(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	latentDim := 4
	ls := NewLatentSpace(latentDim, engine)

	ls.Register("modelA", 3)
	ls.Register("modelB", 5)

	// Training data with shared latent structure.
	nTrain := 200
	dataA := make([][]float64, nTrain)
	dataB := make([][]float64, nTrain)

	for i := 0; i < nTrain; i++ {
		s := float64(i) / float64(nTrain)
		dataA[i] = []float64{s, s * 0.5, 1.0 - s}
		dataB[i] = []float64{s * 0.8, s * 0.3, 1.0 - s*0.9, s * 0.2, 0.5 - s*0.5}
	}

	// Train projections.
	err := ls.TrainProjections(ctx, map[string][][]float64{
		"modelA": dataA,
		"modelB": dataB,
	}, ProjectionConfig{
		LearningRate:    0.005,
		NEpochs:         300,
		AlignmentWeight: 2.0,
	})
	if err != nil {
		t.Fatalf("TrainProjections: %v", err)
	}

	// Transfer test: Take model A's representation of a sample, project
	// into shared space, then retrieve in model B's space. Compare to
	// model B's actual features. If transfer works, the reconstructed
	// features should approximate the original.
	nTest := 20
	transferErrors := make([]float64, nTest)
	randomErrors := make([]float64, nTest)

	for i := 0; i < nTest; i++ {
		idx := 50 + i*5 // spread across training range

		// Transfer path: A features -> shared -> B features
		zA, err := ls.Project(ctx, "modelA", dataA[idx])
		if err != nil {
			t.Fatal(err)
		}
		bTransferred, err := ls.Retrieve(ctx, "modelB", zA)
		if err != nil {
			t.Fatal(err)
		}

		// Random baseline: retrieve from random latent vector.
		randomZ := make([]float64, latentDim)
		for d := 0; d < latentDim; d++ {
			randomZ[d] = ls.rng.NormFloat64()
		}
		bRandom, err := ls.Retrieve(ctx, "modelB", randomZ)
		if err != nil {
			t.Fatal(err)
		}

		// MSE between transferred/random and actual B features.
		actual := dataB[idx]
		var errTransfer, errRandom float64
		for d := range actual {
			dt := bTransferred[d] - actual[d]
			dr := bRandom[d] - actual[d]
			errTransfer += dt * dt
			errRandom += dr * dr
		}
		transferErrors[i] = errTransfer / float64(len(actual))
		randomErrors[i] = errRandom / float64(len(actual))
	}

	// Average errors.
	var avgTransfer, avgRandom float64
	for i := 0; i < nTest; i++ {
		avgTransfer += transferErrors[i]
		avgRandom += randomErrors[i]
	}
	avgTransfer /= float64(nTest)
	avgRandom /= float64(nTest)

	t.Logf("average transfer MSE: %.6f", avgTransfer)
	t.Logf("average random MSE:   %.6f", avgRandom)

	if avgTransfer >= avgRandom {
		t.Errorf("transfer should beat random: transfer=%.6f >= random=%.6f", avgTransfer, avgRandom)
	}
}

func TestLatentSpace_RegisterAndProject(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ls := NewLatentSpace(3, engine)
	ls.Register("m1", 4)

	features := []float64{1, 2, 3, 4}
	z, err := ls.Project(ctx, "m1", features)
	if err != nil {
		t.Fatal(err)
	}

	if len(z) != 3 {
		t.Fatalf("expected latent dim 3, got %d", len(z))
	}

	// Retrieve back.
	out, err := ls.Retrieve(ctx, "m1", z)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("expected output dim 4, got %d", len(out))
	}
}

func TestLatentSpace_TrainProjections_Errors(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ls := NewLatentSpace(3, engine)
	ls.Register("a", 2)

	// Too few models.
	err := ls.TrainProjections(ctx, map[string][][]float64{
		"a": {{1, 2}},
	}, ProjectionConfig{})
	if err == nil {
		t.Fatal("expected error for single model")
	}

	// Unregistered model.
	err = ls.TrainProjections(ctx, map[string][][]float64{
		"a": {{1, 2}},
		"b": {{3, 4}},
	}, ProjectionConfig{})
	if err == nil {
		t.Fatal("expected error for unregistered model")
	}

	// Mismatched sample counts.
	ls.Register("b", 2)
	err = ls.TrainProjections(ctx, map[string][][]float64{
		"a": {{1, 2}, {3, 4}},
		"b": {{5, 6}},
	}, ProjectionConfig{})
	if err == nil {
		t.Fatal("expected error for mismatched sample counts")
	}

	// No samples.
	err = ls.TrainProjections(ctx, map[string][][]float64{
		"a": {},
		"b": {},
	}, ProjectionConfig{})
	if err == nil {
		t.Fatal("expected error for empty data")
	}
}

func TestLatentSpace_CosineAlignment(t *testing.T) {
	ctx := context.Background()
	engine := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	ls := NewLatentSpace(4, engine)
	ls.Register("x", 3)
	ls.Register("y", 3)

	n := 100
	dx := make([][]float64, n)
	dy := make([][]float64, n)
	for i := 0; i < n; i++ {
		s := float64(i) / float64(n)
		dx[i] = []float64{s, 1 - s, s * s}
		dy[i] = []float64{s * 0.9, 1 - s*0.95, s * s * 1.1}
	}

	if err := ls.TrainProjections(ctx, map[string][][]float64{
		"x": dx, "y": dy,
	}, ProjectionConfig{
		LearningRate:    0.005,
		NEpochs:         200,
		AlignmentWeight: 2.0,
	}); err != nil {
		t.Fatal(err)
	}

	// Paired latent vectors should have high cosine similarity.
	idx := 50
	zx, err := ls.Project(ctx, "x", dx[idx])
	if err != nil {
		t.Fatal(err)
	}
	zy, err := ls.Project(ctx, "y", dy[idx])
	if err != nil {
		t.Fatal(err)
	}
	sim := cosine(zx, zy)

	t.Logf("cosine similarity for paired sample: %.4f", sim)
	if sim < 0.5 {
		t.Errorf("expected cosine similarity > 0.5 for aligned pair, got %.4f", sim)
	}
}
