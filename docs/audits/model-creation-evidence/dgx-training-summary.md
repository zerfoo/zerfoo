# DGX classifier training validation — 2026-09-09

Tested source: `3273fa881b1a9a9fd8d7ce0665e4976641b2be9a`.
Spark ran `scripts/model-creation-gpu-inpod.sh` in a fresh Go 1.26.8 Linux
ARM64 container with one GPU and a 2 GiB container memory limit. The pod
completed successfully. [Captured log](dgx-training.txt).

`TestClassifierTrainingGPU`: 1 test executed and passed, 0 skipped;
6 recipe/seed runs completed 1,200 optimizer steps each. Every run checked
GPU-backed forward logits, frozen quality thresholds, bundle export and CPU
reload accuracy. The package completed in 3.728 seconds.

| Recipe | Seed | Test accuracy | Macro-F1 | Cross entropy |
| --- | ---: | ---: | ---: | ---: |
| Linear | 17 | 29/30 | 0.966583 | 0.106215 |
| Linear | 42 | 29/30 | 0.966583 | 0.132283 |
| Linear | 91 | 29/30 | 0.966583 | 0.112548 |
| MLP [16] | 17 | 29/30 | 0.966583 | 0.204932 |
| MLP [16] | 42 | 28/30 | 0.933333 | 0.252160 |
| MLP [16] | 91 | 28/30 | 0.933333 | 0.240817 |

All six CPU-loaded bundles had the same aggregate accuracy as their GPU
evaluation. This check does not establish probability-level CPU/GPU parity or
that each individual prediction agrees, and it does not claim every optimizer
operation stays on device. Artifacts were created in test temporary directories;
the log retains their content hashes, not durable copies of the bundles.

This targeted gate does not cover HTTP serving, optimizer checkpoint/recovery,
independent PyTorch parity, large datasets, or text-model parity. Those remain
separate qualification work.
