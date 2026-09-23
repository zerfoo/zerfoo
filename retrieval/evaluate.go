package retrieval

import (
	"context"
	"errors"
	"math"
)

// Query is a labeled retrieval request. Relevant contains all known relevant
// document IDs. An empty set represents a no-result request.
type Query struct {
	Text     string
	Relevant []string
}

// Metrics summarizes ranking over a fixed set of queries. Recall and
// completeness exclude no-result queries; FalsePositiveRate uses only them.
type Metrics struct {
	Recall            float64
	NDCG              float64
	MRR               float64
	Completeness      float64
	FalsePositiveRate float64
	PositiveQueries   int
	NoResultQueries   int
}

// Evaluate runs a fixed-depth retrieval benchmark. Callers must ensure that
// labels and candidate documents use the intended held-out split.
func Evaluate(ctx context.Context, index *Index, queries []Query, at int) (Metrics, error) {
	if index == nil || at <= 0 {
		return Metrics{}, errors.New("index and positive cutoff required")
	}
	var m Metrics
	for _, query := range queries {
		if err := ctx.Err(); err != nil {
			return Metrics{}, err
		}
		results, err := index.Search(ctx, query.Text, at)
		if err != nil {
			return Metrics{}, err
		}
		gold := make(map[string]bool, len(query.Relevant))
		for _, id := range query.Relevant {
			gold[id] = true
		}
		if len(gold) == 0 {
			m.NoResultQueries++
			if len(results) > 0 {
				m.FalsePositiveRate++
			}
			continue
		}
		m.PositiveQueries++
		hits := 0
		var dcg float64
		for rank, result := range results {
			if !gold[result.Document.ID] {
				continue
			}
			hits++
			dcg += 1 / math.Log2(float64(rank+2))
			if mrr := 1 / float64(rank+1); mrr > 0 && hits == 1 {
				m.MRR += mrr
			}
		}
		m.Recall += float64(hits) / float64(len(gold))
		if hits == len(gold) {
			m.Completeness++
		}
		idealHits := len(gold)
		if idealHits > at {
			idealHits = at
		}
		var ideal float64
		for rank := 0; rank < idealHits; rank++ {
			ideal += 1 / math.Log2(float64(rank+2))
		}
		m.NDCG += dcg / ideal
	}
	if m.PositiveQueries > 0 {
		n := float64(m.PositiveQueries)
		m.Recall /= n
		m.NDCG /= n
		m.MRR /= n
		m.Completeness /= n
	}
	if m.NoResultQueries > 0 {
		m.FalsePositiveRate /= float64(m.NoResultQueries)
	}
	return m, nil
}
