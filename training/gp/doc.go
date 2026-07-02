// Experimental — this package is not yet wired into the main framework.
//
// Package gp provides tree-based genetic programming. (Stability: alpha)
//
// Genetic programming evolves expression trees composed of primitive functions
// (Add, Mul, Sin, etc.) and terminals (variables, constants) to approximate
// a target behaviour specified by a fitness function.
//
// The Evolve function runs a generational evolutionary loop with tournament
// selection, subtree crossover, and subtree mutation to search the space of
// programs for the best-fit individual.
package gp
