// Experimental — this package is not yet wired into the main framework.
//
// Package causal provides causal structure learning from observational data.
//
// The primary entry point is DiscoverGraph, which implements the PC algorithm
// to recover a directed acyclic graph (DAG) from data. The algorithm proceeds
// in three phases:
//
//  1. Skeleton discovery — start with a fully connected undirected graph and
//     remove edges between conditionally independent variables.
//  2. V-structure orientation — orient colliders (X -> Z <- Y) where X and Y
//     are non-adjacent but both adjacent to Z.
//  3. Edge orientation — apply Meek's rules to orient remaining undirected
//     edges without creating new v-structures or cycles.
//
// Conditional independence is tested via partial correlation with a
// Fisher z-transform significance test.
//
// (Stability: alpha)
package causal
