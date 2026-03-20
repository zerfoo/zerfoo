// Command fine-tuning demonstrates parameter-efficient fine-tuning using LoRA
// (Low-Rank Adaptation) on a tabular model. The workflow covers the full
// lifecycle: pre-train a base model, apply LoRA adapters, fine-tune on a
// domain-specific dataset, and predict with the adapted model.
//
// No external files or GPU are required -- the example generates synthetic data
// inline and runs entirely on CPU.
//
// Usage:
//
//	go build -o fine-tuning ./examples/fine-tuning/
//	./fine-tuning
package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"

	"github.com/zerfoo/zerfoo/tabular"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// -----------------------------------------------------------------------
	// Step 1: Set up the compute engine
	// -----------------------------------------------------------------------
	// All tensor arithmetic in Zerfoo flows through compute.Engine[T]. Using the
	// CPU engine means this example works everywhere -- swap in a CUDA engine for
	// GPU acceleration with zero code changes.
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	fmt.Println("=== Zerfoo Fine-Tuning Example ===")
	fmt.Println()

	// -----------------------------------------------------------------------
	// Step 2: Generate synthetic pre-training data from multiple "sources"
	// -----------------------------------------------------------------------
	// Pre-training learns universal patterns across many sources. Here we
	// simulate three data sources with a shared 4-feature space and 3 classes
	// (Long=0, Short=1, Flat=2). Each source has a slightly different decision
	// boundary, forcing the model to learn general representations rather than
	// memorising source-specific quirks.
	fmt.Println("Step 1: Generating synthetic pre-training data (3 sources)...")

	allData := make([][][]float64, 3)
	allLabels := make([][]int, 3)
	for src := range 3 {
		data, labels := generateSourceData(200, src)
		allData[src] = data
		allLabels[src] = labels
	}
	fmt.Printf("  Sources: %d, samples per source: %d, features: %d\n",
		len(allData), len(allData[0]), len(allData[0][0]))

	// -----------------------------------------------------------------------
	// Step 3: Pre-train the base model
	// -----------------------------------------------------------------------
	// Pre-training gives us a general-purpose base model. We use a small MLP
	// with two hidden layers (32, 16 units) and train for 30 epochs.
	fmt.Println("Step 2: Pre-training base model...")

	base, err := tabular.PreTrain(allData, allLabels, tabular.PreTrainConfig{
		Epochs:       30,
		BatchSize:    32,
		LearningRate: 0.01,
		WeightDecay:  1e-4,
		HiddenDims:   []int{32, 16},
		Activation:   tabular.ActivationReLU,
	}, engine, ops)
	if err != nil {
		return fmt.Errorf("pre-train: %w", err)
	}
	fmt.Println("  Base model pre-trained.")

	// Evaluate the base model on domain-specific data BEFORE fine-tuning.
	// This establishes a baseline to show how much LoRA adaptation helps.
	domainData, domainLabels := generateDomainData(100)
	baseAcc := evaluate(base.Model, domainData, domainLabels)
	fmt.Printf("  Base model accuracy on domain data: %.1f%%\n", baseAcc*100)

	// -----------------------------------------------------------------------
	// Step 4: Apply LoRA adapters and fine-tune
	// -----------------------------------------------------------------------
	// LoRA freezes the base model weights and injects small trainable matrices
	// (A and B) into each hidden layer. Only these matrices are updated during
	// fine-tuning. With rank=4, the number of trainable parameters is a tiny
	// fraction of the total model size -- making LoRA ideal for adapting a
	// pre-trained model to a new domain with limited data.
	fmt.Println("Step 3: Fine-tuning with LoRA (rank=4)...")

	adapter, err := tabular.FineTuneLoRA(base, domainData, domainLabels, tabular.LoRAConfig{
		Rank:         4,
		Alpha:        8.0, // scale = alpha/rank = 2.0
		Epochs:       50,
		BatchSize:    16,
		LearningRate: 0.001, // Lower LR for fine-tuning to avoid catastrophic forgetting.
		WeightDecay:  1e-4,
	}, engine, ops)
	if err != nil {
		return fmt.Errorf("lora fine-tune: %w", err)
	}
	fmt.Println("  LoRA fine-tuning complete.")

	// -----------------------------------------------------------------------
	// Step 5: Merge the adapter into the base model
	// -----------------------------------------------------------------------
	// MergeAdapter folds the LoRA matrices back into the base weights:
	//   W_merged = W_base + (alpha/rank) * A @ B
	// The resulting model has the same architecture as the original (no LoRA
	// overhead at inference time) but incorporates the domain adaptation.
	fmt.Println("Step 4: Merging LoRA adapter into base model...")

	merged, err := tabular.MergeAdapter(base, adapter, engine)
	if err != nil {
		return fmt.Errorf("merge adapter: %w", err)
	}

	mergedAcc := evaluate(merged, domainData, domainLabels)
	fmt.Printf("  Merged model accuracy on domain data: %.1f%%\n", mergedAcc*100)
	fmt.Printf("  Improvement: %+.1f percentage points\n", (mergedAcc-baseAcc)*100)

	// -----------------------------------------------------------------------
	// Step 6: Save and reload the fine-tuned model
	// -----------------------------------------------------------------------
	// The merged model is a standard tabular.Model, so we can save and load it
	// with the regular Save/Load functions. No special adapter handling needed.
	fmt.Println("Step 5: Saving and reloading fine-tuned model...")

	tmpDir, err := os.MkdirTemp("", "zerfoo-finetune-*")
	if err != nil {
		return fmt.Errorf("create temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	modelPath := filepath.Join(tmpDir, "finetuned.ztab")
	if err := tabular.Save(merged, modelPath); err != nil {
		return fmt.Errorf("save: %w", err)
	}
	fmt.Printf("  Saved to %s\n", modelPath)

	loaded, err := tabular.Load(modelPath, engine, ops)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}

	loadedAcc := evaluate(loaded, domainData, domainLabels)
	fmt.Printf("  Loaded model accuracy: %.1f%%\n", loadedAcc*100)

	// -----------------------------------------------------------------------
	// Step 7: Run predictions on new samples
	// -----------------------------------------------------------------------
	fmt.Println("Step 6: Running predictions on new samples...")
	fmt.Println()

	testSamples := [][]float64{
		{2.0, 0.5, -0.3, 1.0},
		{-1.5, 0.2, 0.8, -0.5},
		{0.1, 0.1, 0.1, 0.1},
	}
	for i, sample := range testSamples {
		dir, conf, err := loaded.Predict(sample)
		if err != nil {
			return fmt.Errorf("predict sample %d: %w", i, err)
		}
		fmt.Printf("  Sample %d: features=%v -> %s (confidence=%.2f)\n",
			i+1, sample, dir, conf)
	}

	fmt.Println()
	fmt.Println("Done. The fine-tuned model is ready for production use.")
	return nil
}

// generateSourceData creates synthetic training data for one source.
// Each source applies a slightly different linear decision boundary (shifted
// by srcOffset) so the pre-trained model must learn general feature
// relationships rather than source-specific rules.
func generateSourceData(n int, srcIdx int) ([][]float64, []int) {
	data := make([][]float64, n)
	labels := make([]int, n)
	offset := float64(srcIdx) * 0.3 // small shift per source

	for i := range n {
		f := [4]float64{
			rand.NormFloat64(),
			rand.NormFloat64(),
			rand.NormFloat64(),
			rand.NormFloat64(),
		}
		// Decision boundary: weighted sum with source-specific offset.
		score := 0.5*f[0] + 0.3*f[1] - 0.2*f[2] + 0.4*f[3] + offset
		switch {
		case score > 0.5:
			labels[i] = 0 // Long
		case score < -0.5:
			labels[i] = 1 // Short
		default:
			labels[i] = 2 // Flat
		}
		data[i] = f[:]
	}
	return data, labels
}

// generateDomainData creates a domain-specific dataset with a different
// decision boundary than the pre-training sources. The difference forces
// fine-tuning to adapt the model's representations.
func generateDomainData(n int) ([][]float64, []int) {
	data := make([][]float64, n)
	labels := make([]int, n)

	for i := range n {
		f := [4]float64{
			rand.NormFloat64(),
			rand.NormFloat64(),
			rand.NormFloat64(),
			rand.NormFloat64(),
		}
		// Domain-specific boundary: different weight pattern.
		score := -0.3*f[0] + 0.6*f[1] + 0.5*f[2] - 0.1*f[3]
		switch {
		case score > 0.4:
			labels[i] = 0
		case score < -0.4:
			labels[i] = 1
		default:
			labels[i] = 2
		}
		data[i] = f[:]
	}
	return data, labels
}

// evaluate computes classification accuracy on a dataset.
func evaluate(model *tabular.Model, data [][]float64, labels []int) float64 {
	correct := 0
	for i, sample := range data {
		dir, _, err := model.Predict(sample)
		if err != nil {
			continue
		}
		if int(dir) == labels[i] {
			correct++
		}
	}
	if len(data) == 0 {
		return 0
	}
	return math.Round(float64(correct)/float64(len(data))*1000) / 1000
}
