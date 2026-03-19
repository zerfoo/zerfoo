// Package federated provides federated learning interfaces and a FedAvg
// baseline implementation. (Stability: alpha)
//
// Federated learning enables training across distributed clients without
// centralising raw data. The [Coordinator] orchestrates training rounds:
// it selects participating clients, distributes global weights, collects
// local [ModelUpdate] results, and aggregates them via a pluggable [Strategy].
//
// # Strategy Interface
//
// [Strategy] defines how client updates are aggregated and which clients
// participate in each round. The package ships [FedAvg], which computes a
// weighted average of model updates proportional to each client's dataset
// size.
//
// # Coordinator
//
// [NewCoordinator] creates a coordinator from a [Strategy] and a
// [CoordinatorConfig]. Call [Coordinator.RunRound] to execute a single
// federated learning round end-to-end:
//
//	coord := federated.NewCoordinator(federated.NewFedAvg(), federated.CoordinatorConfig{
//	    MinClients: 3,
//	    MaxRounds:  100,
//	})
//	result, err := coord.RunRound(clients)
//
// # Client Interface
//
// [Client] represents a federated participant that performs local training
// and reports a [ModelUpdate] back to the coordinator.
package federated
