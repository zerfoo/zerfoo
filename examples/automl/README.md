# AutoML Hyperparameter Search

Demonstrates automated hyperparameter optimization using Bayesian search with early stopping via the `training/automl` package.

## What this example shows

1. **Search space definition** -- defining hyperparameters with ranges and log-scale options
2. **Bayesian optimization** -- using `automl.NewBayesianOptimizer` as the search strategy
3. **Coordinator loop** -- running `automl.NewCoordinator` with a worker, max trials, and early stopping
4. **Result reporting** -- extracting the best trial configuration

## Build

```bash
go build -o automl ./examples/automl/
```

## Run

```bash
./automl
```

## Expected output

```
=== AutoML Hyperparameter Search Example ===

Search space:
  lr: [0.0001, 0.1000] (log scale)
  hidden_dim: [32.0000, 256.0000]
  num_layers: [1.0000, 4.0000]
  dropout: [0.0000, 0.5000]

Max trials: 20, Early stop patience: 5

--- Running trials ---
  Trial 1: lr=0.0234 hidden=187 layers=3 dropout=0.42 -> score=0.7043
  Trial 2: lr=0.0012 hidden=95 layers=2 dropout=0.15 -> score=0.8921
  ...

=== Search Complete ===
Best trial: #N
Best score: 0.XXXX
Best hyperparameters:
  lr = ...
  hidden_dim = ...
  num_layers = ...
  dropout = ...

Total evaluations: N
```

The Bayesian optimizer starts with random exploration, then uses Expected Improvement to focus on promising regions of the search space. Early stopping halts the search if no improvement is found for 5 consecutive trials.

## Key APIs

| Type | Package | Purpose |
|------|---------|---------|
| `automl.HParam` | `training/automl/` | Hyperparameter definition (name, min, max, log scale) |
| `automl.BayesianOptimizer` | `training/automl/` | Bayesian search strategy with Expected Improvement |
| `automl.Coordinator` | `training/automl/` | Orchestrates trial dispatch and early stopping |
| `automl.Worker` | `training/automl/` | Interface for trial evaluation (implement `RunTrial`) |

## Production usage

Replace `syntheticWorker` with a worker that:

1. Builds a model from the suggested hyperparameters
2. Trains for a few epochs on your dataset
3. Returns the validation metric as `automl.Metric{Score: ...}`

For architecture search across tabular/time-series models, use `automl.AutoML()` which searches over MLP, FT-Transformer, TabNet, SAINT, TabResNet, TFT, N-BEATS, and PatchTST simultaneously.
