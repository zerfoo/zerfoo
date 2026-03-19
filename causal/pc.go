package causal

// sepSet stores the conditioning set that rendered two nodes independent.
// sepSets[i][j] is the set Z such that X_i ⊥ X_j | Z, or nil if no such set
// was found.
type sepSet = map[[2]int][]int

// discoverSkeleton implements phase 1 of the PC algorithm. It starts with a
// fully connected undirected graph and removes edges between conditionally
// independent pairs.
func discoverSkeleton(nVars, nSamples int, corr [][]float64, alpha float64, maxCond int) ([][]bool, sepSet) {
	// Initialise fully connected undirected adjacency.
	adj := make([][]bool, nVars)
	for i := range adj {
		adj[i] = make([]bool, nVars)
		for j := 0; j < nVars; j++ {
			if i != j {
				adj[i][j] = true
			}
		}
	}

	seps := make(sepSet)

	// Increase conditioning set size from 0 up to maxCond (or nVars-2).
	limit := nVars - 2
	if maxCond > 0 && maxCond < limit {
		limit = maxCond
	}

	for condSize := 0; condSize <= limit; condSize++ {
		// Track which edges to remove (avoid modifying adj during iteration).
		type removal struct {
			i, j int
			z    []int
		}
		var removals []removal

		for i := 0; i < nVars; i++ {
			for j := i + 1; j < nVars; j++ {
				if !adj[i][j] {
					continue
				}
				// Get neighbours of i excluding j.
				neighbours := adjacentExcluding(adj, nVars, i, j)
				if len(neighbours) < condSize {
					continue
				}
				// Test all subsets of size condSize.
				found := false
				forEachSubset(neighbours, condSize, func(z []int) bool {
					r := partialCorrelation(corr, i, j, z)
					if fisherZTest(r, nSamples, len(z), alpha) {
						zCopy := make([]int, len(z))
						copy(zCopy, z)
						removals = append(removals, removal{i, j, zCopy})
						found = true
						return false // stop iterating subsets
					}
					return true // continue
				})
				if found {
					continue
				}
			}
		}

		for _, rm := range removals {
			adj[rm.i][rm.j] = false
			adj[rm.j][rm.i] = false
			key := [2]int{rm.i, rm.j}
			if rm.i > rm.j {
				key = [2]int{rm.j, rm.i}
			}
			seps[key] = rm.z
		}
	}

	return adj, seps
}

// orientVStructures implements phase 2 of the PC algorithm. For every triple
// X - Z - Y where X and Y are not adjacent, if Z is not in sepSet(X,Y) then
// orient X -> Z <- Y (a v-structure / collider).
func orientVStructures(nVars int, skeleton [][]bool, seps sepSet) [][]bool {
	// directed[i][j] means there is a directed edge i -> j.
	directed := make([][]bool, nVars)
	for i := range directed {
		directed[i] = make([]bool, nVars)
	}

	for z := 0; z < nVars; z++ {
		// Find pairs (x, y) both adjacent to z but not to each other.
		neighbours := make([]int, 0)
		for n := 0; n < nVars; n++ {
			if skeleton[z][n] {
				neighbours = append(neighbours, n)
			}
		}
		for ni := 0; ni < len(neighbours); ni++ {
			for nj := ni + 1; nj < len(neighbours); nj++ {
				x, y := neighbours[ni], neighbours[nj]
				if skeleton[x][y] {
					continue // x and y are adjacent, not a v-structure
				}
				// Check if z is in sepSet(x, y).
				key := [2]int{x, y}
				if x > y {
					key = [2]int{y, x}
				}
				sep := seps[key]
				if !intSliceContains(sep, z) {
					// Orient x -> z <- y.
					directed[x][z] = true
					directed[y][z] = true
				}
			}
		}
	}

	return directed
}

// applyMeekRules orients remaining undirected edges in the skeleton using
// Meek's four rules, iterating until no more edges can be oriented.
func applyMeekRules(nVars int, skeleton, directed [][]bool) {
	changed := true
	for changed {
		changed = false

		for i := 0; i < nVars; i++ {
			for j := 0; j < nVars; j++ {
				if !skeleton[i][j] || directed[i][j] || directed[j][i] {
					continue // not an undirected edge
				}
				// Rule 1: If k -> i - j and k not adjacent to j, orient i -> j.
				if meekRule1(nVars, skeleton, directed, i, j) {
					directed[i][j] = true
					changed = true
					continue
				}
				// Rule 2: If i -> k -> j, orient i -> j.
				if meekRule2(nVars, skeleton, directed, i, j) {
					directed[i][j] = true
					changed = true
					continue
				}
				// Rule 3: If k -> j and l -> j, k - i - l, k not adj l,
				// orient i -> j.
				if meekRule3(nVars, skeleton, directed, i, j) {
					directed[i][j] = true
					changed = true
					continue
				}
			}
		}
	}
}

// meekRule1: exists k such that k -> i, k not adj j => orient i -> j.
func meekRule1(nVars int, skeleton, directed [][]bool, i, j int) bool {
	for k := 0; k < nVars; k++ {
		if k == i || k == j {
			continue
		}
		if directed[k][i] && !skeleton[k][j] {
			return true
		}
	}
	return false
}

// meekRule2: exists k such that i -> k -> j => orient i -> j.
func meekRule2(nVars int, _ [][]bool, directed [][]bool, i, j int) bool {
	for k := 0; k < nVars; k++ {
		if k == i || k == j {
			continue
		}
		if directed[i][k] && directed[k][j] {
			return true
		}
	}
	return false
}

// meekRule3: exist distinct k, l both adj to i and j, k not adj l, k -> j and
// l -> j => orient i -> j.
func meekRule3(nVars int, skeleton, directed [][]bool, i, j int) bool {
	for k := 0; k < nVars; k++ {
		if k == i || k == j {
			continue
		}
		if !skeleton[k][i] || !directed[k][j] {
			continue
		}
		for l := k + 1; l < nVars; l++ {
			if l == i || l == j {
				continue
			}
			if !skeleton[l][i] || !directed[l][j] {
				continue
			}
			if !skeleton[k][l] {
				return true
			}
		}
	}
	return false
}

// adjacentExcluding returns all nodes adjacent to node in the skeleton,
// excluding the given node.
func adjacentExcluding(adj [][]bool, nVars, node, exclude int) []int {
	var result []int
	for k := 0; k < nVars; k++ {
		if k != exclude && adj[node][k] {
			result = append(result, k)
		}
	}
	return result
}

// forEachSubset calls fn with every subset of the given slice that has exactly
// size elements. fn returns true to continue, false to stop early.
func forEachSubset(items []int, size int, fn func([]int) bool) {
	if size == 0 {
		fn(nil)
		return
	}
	if size > len(items) {
		return
	}
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i
	}
	subset := make([]int, size)

	for {
		for i, idx := range indices {
			subset[i] = items[idx]
		}
		if !fn(subset) {
			return
		}

		// Advance to next combination.
		i := size - 1
		for i >= 0 && indices[i] == len(items)-size+i {
			i--
		}
		if i < 0 {
			return
		}
		indices[i]++
		for j := i + 1; j < size; j++ {
			indices[j] = indices[j-1] + 1
		}
	}
}

// intSliceContains returns true if the slice contains the value.
func intSliceContains(s []int, v int) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
