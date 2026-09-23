package retrieval

import (
	"context"
	"math"
	"strings"
	"testing"
)

func TestEvaluateRankingAndNoResult(t *testing.T) {
	idx, err := NewIndex(context.Background(), []Document{
		{ID: "a", Text: "go parser"}, {ID: "b", Text: "go formatter"}, {ID: "c", Text: "swift views"},
	}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	m, err := Evaluate(context.Background(), idx, []Query{
		{Text: "go", Relevant: []string{"a", "b"}},
		{Text: "missing", Relevant: nil},
		{Text: "go", Relevant: []string{"c"}},
	}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if m.PositiveQueries != 2 || m.NoResultQueries != 1 || math.Abs(m.Recall-0.5) > 1e-9 || math.Abs(m.Completeness-0.5) > 1e-9 || m.FalsePositiveRate != 0 {
		t.Fatalf("metrics: %+v", m)
	}
	if m.NDCG <= 0 || m.NDCG >= 1 || m.MRR != 0.5 {
		t.Fatalf("ranking: %+v", m)
	}
}

func TestEvaluateRejectsUnknownRelevantID(t *testing.T) {
	idx, err := NewIndex(context.Background(), []Document{{ID: "known", Text: "known"}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = Evaluate(context.Background(), idx, []Query{{Text: "known", Relevant: []string{"known", "typo"}}}, 2)
	if err == nil || !strings.Contains(err.Error(), "typo") {
		t.Fatalf("unknown relevant ID: %v", err)
	}
}
