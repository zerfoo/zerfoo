// Experimental — this package is not yet wired into the main framework.
//
// Package meta provides meta-learning algorithms for few-shot adaptation. (Stability: alpha)
//
// The primary algorithm is MAML (Model-Agnostic Meta-Learning), which learns
// initialization weights that can be rapidly adapted to new tasks with only a
// few gradient steps. MAML operates with two nested optimization loops:
//
//   - Inner loop: task-specific adaptation via gradient descent on a small
//     support set (few-shot learning).
//   - Outer loop: meta-update that optimizes the initial weights so that
//     inner-loop adaptation generalizes well across a distribution of tasks.
//
// Usage:
//
//	config := meta.MAMLConfig{
//	    InnerLR:         0.01,
//	    OuterLR:         0.001,
//	    InnerSteps:      5,
//	    NTasksPerBatch:  4,
//	}
//	maml := meta.NewMAML(config)
//	err := maml.MetaTrain(tasks, config)
//	adapted := maml.Adapt(newTask, 5)
package meta
