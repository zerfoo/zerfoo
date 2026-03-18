// Command bench_prefix simulates a multi-turn chat workload to measure
// prefix cache hit rate and TTFT reduction. Each simulated user shares
// a common system prompt and appends unique per-turn history tokens.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"time"

	"github.com/zerfoo/zerfoo/generate"
)

func main() {
	users := flag.Int("users", 10, "number of simulated users")
	turns := flag.Int("turns", 5, "chat turns per user")
	sysTokens := flag.Int("sys-prompt-tokens", 256, "system prompt length in tokens")
	histTokens := flag.Int("hist-tokens-per-turn", 32, "unique history tokens added per turn")
	blockSize := flag.Int("block-size", 16, "tokens per KV block")
	flag.Parse()

	stats := RunSimulation(Config{
		Users:            *users,
		Turns:            *turns,
		SysPromptTokens:  *sysTokens,
		HistTokensPerTurn: *histTokens,
		BlockSize:        *blockSize,
	})

	fmt.Println("=== Prefix Cache Hit Rate Benchmark ===")
	fmt.Printf("Users:              %d\n", *users)
	fmt.Printf("Turns per user:     %d\n", *turns)
	fmt.Printf("System prompt:      %d tokens\n", *sysTokens)
	fmt.Printf("History per turn:   %d tokens\n", *histTokens)
	fmt.Printf("Block size:         %d tokens\n", *blockSize)
	fmt.Println()
	fmt.Printf("Total requests:     %d\n", stats.TotalRequests)
	fmt.Printf("Cache hits:         %d\n", stats.Hits)
	fmt.Printf("Cache misses:       %d\n", stats.Misses)
	fmt.Printf("Hit rate:           %.1f%%\n", stats.HitRate*100)
	fmt.Printf("Avg tokens matched: %.1f\n", stats.AvgTokensMatched)
	fmt.Printf("Avg tokens total:   %.1f\n", stats.AvgTokensTotal)
	fmt.Printf("TTFT reduction:     %.1f%%\n", stats.TTFTReduction*100)
	fmt.Printf("Elapsed:            %s\n", stats.Elapsed.Round(time.Microsecond))
}

// Config controls the simulation parameters.
type Config struct {
	Users             int
	Turns             int
	SysPromptTokens   int
	HistTokensPerTurn int
	BlockSize         int
}

// Stats holds the results of a simulation run.
type Stats struct {
	TotalRequests    int
	Hits             int
	Misses           int
	HitRate          float64
	AvgTokensMatched float64
	AvgTokensTotal   float64
	TTFTReduction    float64
	Elapsed          time.Duration
}

// RunSimulation executes the multi-turn chat simulation and returns statistics.
func RunSimulation(cfg Config) Stats {
	const (
		numLayers = 2
		headDim   = 64
		poolMB    = 512
	)

	pool, err := generate.NewBlockPool[float32](numLayers, cfg.BlockSize, headDim, poolMB)
	if err != nil {
		panic(fmt.Sprintf("NewBlockPool: %v", err))
	}

	// Cache capacity: enough to hold all unique prefix blocks across users.
	// Each user's longest prefix = sysPromptTokens + turns*histTokensPerTurn.
	maxPrefixLen := cfg.SysPromptTokens + cfg.Users*cfg.Turns*cfg.HistTokensPerTurn
	cacheCapacity := maxPrefixLen * 2 // headroom for radix tree splits
	pc := generate.NewPrefixCache[float32](cacheCapacity, pool)

	// Generate a shared system prompt (deterministic tokens).
	sysPrompt := make([]int32, cfg.SysPromptTokens)
	for i := range sysPrompt {
		sysPrompt[i] = int32(1000 + i)
	}

	start := time.Now()

	var totalRequests, hits, misses int
	var totalTokensMatched, totalTokensInRequest int

	rng := rand.New(rand.NewPCG(42, 0))

	for u := 0; u < cfg.Users; u++ {
		// Each user accumulates history across turns.
		var userHistory []int32

		for t := 0; t < cfg.Turns; t++ {
			// Build the full prefix: system prompt + user history so far.
			turnTokens := make([]int32, cfg.HistTokensPerTurn)
			for i := range turnTokens {
				turnTokens[i] = int32(rng.IntN(30000) + 5000)
			}
			userHistory = append(userHistory, turnTokens...)

			fullPrefix := make([]int32, 0, len(sysPrompt)+len(userHistory))
			fullPrefix = append(fullPrefix, sysPrompt...)
			fullPrefix = append(fullPrefix, userHistory...)

			totalRequests++
			totalTokensInRequest += len(fullPrefix)

			// Try cache match.
			matched, matchedLen := pc.Match(fullPrefix)
			if matchedLen > 0 {
				hits++
				totalTokensMatched += matchedLen
				// Return matched blocks to pool.
				for _, b := range matched {
					pool.Free(b)
				}
			} else {
				misses++
			}

			// After processing, insert the full prefix into cache.
			// RadixTree maps one block per token ID.
			blocks := make([]*generate.Block[float32], 0, len(fullPrefix))
			for range fullPrefix {
				b, allocErr := pool.Alloc()
				if allocErr != nil {
					break
				}
				b.Used = 1
				blocks = append(blocks, b)
			}

			pc.Insert(fullPrefix, blocks)

			// Return blocks to pool (cache keeps its own copy).
			for _, b := range blocks {
				pool.Free(b)
			}
		}
	}

	elapsed := time.Since(start)

	var hitRate, avgMatched, avgTotal, ttftReduction float64
	if totalRequests > 0 {
		hitRate = float64(hits) / float64(totalRequests)
		avgTotal = float64(totalTokensInRequest) / float64(totalRequests)
	}
	if hits > 0 {
		avgMatched = float64(totalTokensMatched) / float64(hits)
	}
	// TTFT reduction = fraction of total prefill tokens saved by cache hits.
	if totalTokensInRequest > 0 {
		ttftReduction = float64(totalTokensMatched) / float64(totalTokensInRequest)
	}

	return Stats{
		TotalRequests:    totalRequests,
		Hits:             hits,
		Misses:           misses,
		HitRate:          hitRate,
		AvgTokensMatched: avgMatched,
		AvgTokensTotal:   avgTotal,
		TTFTReduction:    ttftReduction,
		Elapsed:          elapsed,
	}
}
