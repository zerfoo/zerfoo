# Fine-Tuning with LoRA

This example demonstrates parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) on a tabular model. It covers the complete lifecycle: pre-training a base model, applying LoRA adapters, fine-tuning on domain-specific data, and saving/loading the adapted model.

No external files, datasets, or GPU are required -- everything runs on CPU with synthetic data.

## Prerequisites

- Go 1.25 or later

## Build and Run

```bash
go build -o fine-tuning ./examples/fine-tuning/
./fine-tuning
```

Or run directly:

```bash
go run ./examples/fine-tuning/
```

## What It Does

The example walks through six steps:

### Step 1: Generate synthetic pre-training data

Three simulated data sources are created, each with 200 samples of 4 features and 3 classes (Long, Short, Flat). Each source uses a slightly different decision boundary, forcing the model to learn general patterns.

### Step 2: Pre-train the base model

A small MLP (32 -> 16 hidden units) is trained on all three sources using `tabular.PreTrain`. This produces a `BaseModel` with general-purpose feature representations.

### Step 3: Fine-tune with LoRA

`tabular.FineTuneLoRA` freezes the base model weights and injects low-rank adapter matrices (rank=4) into each hidden layer. Only these small matrices are trained -- the total number of trainable parameters is a fraction of the full model. A lower learning rate (0.001 vs 0.01) prevents catastrophic forgetting.

### Step 4: Merge the adapter

`tabular.MergeAdapter` folds the LoRA matrices back into the base weights: `W_merged = W_base + (alpha/rank) * A @ B`. The result is a standard `tabular.Model` with no LoRA overhead at inference time.

### Step 5: Save and reload

The merged model is saved in ZTAB format with `tabular.Save` and reloaded with `tabular.Load`, confirming the fine-tuned weights persist correctly.

### Step 6: Run predictions

New samples are classified using `model.Predict`, which returns a direction (Long/Short/Flat) and confidence score.

## Expected Output

```
=== Zerfoo Fine-Tuning Example ===

Step 1: Generating synthetic pre-training data (3 sources)...
  Sources: 3, samples per source: 200, features: 4
Step 2: Pre-training base model...
  Base model pre-trained.
  Base model accuracy on domain data: 45.0%
Step 3: Fine-tuning with LoRA (rank=4)...
  LoRA fine-tuning complete.
Step 4: Merging LoRA adapter into base model...
  Merged model accuracy on domain data: 72.0%
  Improvement: +27.0 percentage points
Step 5: Saving and reloading fine-tuned model...
  Saved to /tmp/zerfoo-finetune-XXXX/finetuned.ztab
  Loaded model accuracy: 72.0%
Step 6: Running predictions on new samples...

  Sample 1: features=[2 0.5 -0.3 1] -> Long (confidence=0.65)
  Sample 2: features=[-1.5 0.2 0.8 -0.5] -> Short (confidence=0.58)
  Sample 3: features=[0.1 0.1 0.1 0.1] -> Flat (confidence=0.42)

Done. The fine-tuned model is ready for production use.
```

Exact numbers will vary across runs due to random initialization and data generation.

## Key APIs Used

| Package | Function | Purpose |
|---------|----------|---------|
| `ztensor/compute` | `NewCPUEngine` | Create the CPU compute engine |
| `ztensor/numeric` | `Float32Ops{}` | Float32 arithmetic operations |
| `tabular` | `PreTrain` | Pre-train a base model on multi-source data |
| `tabular` | `FineTuneLoRA` | Apply LoRA adapters and fine-tune |
| `tabular` | `MergeAdapter` | Fold adapters back into base weights |
| `tabular` | `Save` / `Load` | Persist and restore models in ZTAB format |
| `tabular` | `Model.Predict` | Run inference on new samples |

## Further Reading

- [training/lora/](../../training/lora/) -- Low-level LoRA layer implementation for graph-based models
- [tabular/](../../tabular/) -- Tabular model training, LoRA, and ensemble APIs
