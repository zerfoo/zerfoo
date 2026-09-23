// Package retrieval provides an in-process search index for text records.
// Applications control the record format, encoder, and any downstream action.
package retrieval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode"
)

// Document is a text record that can be discovered by ID. Text is the short
// search representation; Body is returned on lookup but is not indexed.
type Document struct {
	ID   string
	Text string
	Body string
}

// Embedder maps text to vectors. An application can supply a pretrained
// encoder or a future Zerfoo-native retrieval model.
type Embedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}

// Reranker scores a shortlist against the original query. Higher is better.
type Reranker interface {
	Rerank(context.Context, string, []Document) ([]float32, error)
}

type Options struct {
	Embedder           Embedder
	Reranker           Reranker
	CandidateLimit     int     // candidates passed to reranking; defaults to 40
	MinDenseSimilarity float32 // dense-only candidates below this are excluded
}

type Result struct {
	Document    Document
	Score       float64 // reciprocal-rank fusion score, or reranker score
	LexicalRank int     // zero if absent
	DenseRank   int     // zero if absent
}

type posting struct{ doc, frequency int }

// Index is immutable after construction and safe for concurrent search if its
// optional Embedder and Reranker are safe for concurrent calls.
type Index struct {
	docs      []Document
	byID      map[string]int
	terms     map[string][]posting
	lengths   []int
	avgLength float64
	vectors   [][]float32
	dimension int
	opts      Options
}

// NewIndex builds an in-memory BM25 index and, if configured, dense vectors.
func NewIndex(ctx context.Context, docs []Document, opts Options) (*Index, error) {
	if opts.CandidateLimit < 0 || math.IsNaN(float64(opts.MinDenseSimilarity)) || opts.MinDenseSimilarity < -1 || opts.MinDenseSimilarity > 1 {
		return nil, errors.New("invalid retrieval options")
	}
	if opts.CandidateLimit == 0 {
		opts.CandidateLimit = 40
	}
	idx := &Index{docs: append([]Document(nil), docs...), byID: make(map[string]int, len(docs)), terms: make(map[string][]posting), lengths: make([]int, len(docs)), opts: opts}
	texts := make([]string, len(docs))
	for i, d := range docs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if strings.TrimSpace(d.ID) == "" {
			return nil, fmt.Errorf("document %d has empty ID", i)
		}
		if _, exists := idx.byID[d.ID]; exists {
			return nil, fmt.Errorf("duplicate document ID %q", d.ID)
		}
		idx.byID[d.ID] = i
		texts[i] = d.Text
		counts := make(map[string]int)
		for _, term := range tokenize(d.Text) {
			counts[term]++
			idx.lengths[i]++
		}
		for term, frequency := range counts {
			idx.terms[term] = append(idx.terms[term], posting{i, frequency})
		}
		idx.avgLength += float64(idx.lengths[i])
	}
	if len(docs) > 0 {
		idx.avgLength /= float64(len(docs))
	}
	if opts.Embedder != nil && len(docs) > 0 {
		vectors, err := opts.Embedder.Embed(ctx, texts)
		if err != nil {
			return nil, fmt.Errorf("embed documents: %w", err)
		}
		if len(vectors) != len(docs) {
			return nil, fmt.Errorf("embed documents: got %d vectors, want %d", len(vectors), len(docs))
		}
		idx.vectors = make([][]float32, len(vectors))
		for i, vector := range vectors {
			if i == 0 {
				idx.dimension = len(vector)
			}
			if idx.dimension == 0 || len(vector) != idx.dimension {
				return nil, fmt.Errorf("document %d: invalid vector dimension", i)
			}
			normalized, err := normalize(vector)
			if err != nil {
				return nil, fmt.Errorf("document %d: %w", i, err)
			}
			idx.vectors[i] = normalized
		}
	}
	return idx, nil
}

func (idx *Index) Get(id string) (Document, bool) {
	i, ok := idx.byID[id]
	if !ok {
		return Document{}, false
	}
	return idx.docs[i], true
}

// Search returns at most limit results. A lexical-only search returns no
// results when no query terms match. Dense-only results obey MinDenseSimilarity.
func (idx *Index) Search(ctx context.Context, query string, limit int) ([]Result, error) {
	if limit < 0 {
		return nil, errors.New("negative result limit")
	}
	if limit == 0 || len(idx.docs) == 0 || strings.TrimSpace(query) == "" {
		return nil, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lexical := make([]float64, len(idx.docs))
	seen := make(map[string]bool)
	for _, term := range tokenize(query) {
		if seen[term] {
			continue
		}
		seen[term] = true
		list := idx.terms[term]
		if len(list) == 0 {
			continue
		}
		idf := math.Log(1 + (float64(len(idx.docs)-len(list))+0.5)/(float64(len(list))+0.5))
		for _, p := range list {
			length := float64(idx.lengths[p.doc])
			denominator := float64(p.frequency) + 1.2*(0.25+0.75*length/idx.avgLength)
			lexical[p.doc] += idf * float64(p.frequency) * 2.2 / denominator
		}
	}
	lexOrder := make([]int, 0)
	for i, score := range lexical {
		if score > 0 {
			lexOrder = append(lexOrder, i)
		}
	}
	sort.Slice(lexOrder, func(a, b int) bool {
		if lexical[lexOrder[a]] != lexical[lexOrder[b]] {
			return lexical[lexOrder[a]] > lexical[lexOrder[b]]
		}
		return idx.docs[lexOrder[a]].ID < idx.docs[lexOrder[b]].ID
	})

	denseOrder := make([]int, 0)
	denseScores := make([]float64, len(idx.docs))
	if idx.opts.Embedder != nil {
		vectors, err := idx.opts.Embedder.Embed(ctx, []string{query})
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		if len(vectors) != 1 || len(vectors[0]) != idx.dimension {
			return nil, errors.New("query embedding has invalid dimension")
		}
		q, err := normalize(vectors[0])
		if err != nil {
			return nil, fmt.Errorf("query embedding: %w", err)
		}
		for i, v := range idx.vectors {
			if i&255 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			var dot float64
			for j, x := range v {
				dot += float64(x) * float64(q[j])
			}
			denseScores[i] = dot
			if dot > float64(idx.opts.MinDenseSimilarity) {
				denseOrder = append(denseOrder, i)
			}
		}
		sort.Slice(denseOrder, func(a, b int) bool {
			if denseScores[denseOrder[a]] != denseScores[denseOrder[b]] {
				return denseScores[denseOrder[a]] > denseScores[denseOrder[b]]
			}
			return idx.docs[denseOrder[a]].ID < idx.docs[denseOrder[b]].ID
		})
	}
	width := limit
	if width < idx.opts.CandidateLimit {
		width = idx.opts.CandidateLimit
	}
	ranked := make(map[int]*Result)
	add := func(order []int, dense bool) {
		for rank, i := range order {
			if rank >= width {
				break
			}
			r := ranked[i]
			if r == nil {
				d := idx.docs[i]
				d.Body = ""
				r = &Result{Document: d}
				ranked[i] = r
			}
			r.Score += 1 / float64(60+rank+1)
			if dense {
				r.DenseRank = rank + 1
			} else {
				r.LexicalRank = rank + 1
			}
		}
	}
	add(lexOrder, false)
	add(denseOrder, true)
	results := make([]Result, 0, len(ranked))
	for _, r := range ranked {
		results = append(results, *r)
	}
	sortResults(results)
	if idx.opts.Reranker != nil && len(results) > 0 {
		if len(results) > width {
			results = results[:width]
		}
		candidates := make([]Document, len(results))
		for i := range results {
			candidates[i] = idx.docs[idx.byID[results[i].Document.ID]]
		}
		scores, err := idx.opts.Reranker.Rerank(ctx, query, candidates)
		if err != nil {
			return nil, fmt.Errorf("rerank: %w", err)
		}
		if len(scores) != len(results) {
			return nil, errors.New("reranker returned wrong score count")
		}
		for i, score := range scores {
			if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
				return nil, errors.New("reranker returned non-finite score")
			}
			results[i].Score = float64(score)
		}
		sortResults(results)
	}
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func sortResults(results []Result) {
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].Document.ID < results[j].Document.ID
	})
}

func normalize(vector []float32) ([]float32, error) {
	var norm float64
	for _, x := range vector {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return nil, errors.New("non-finite embedding")
		}
		norm += float64(x) * float64(x)
	}
	if norm == 0 || math.IsInf(norm, 0) {
		return nil, errors.New("zero or overflowing embedding")
	}
	out := make([]float32, len(vector))
	for i, x := range vector {
		out[i] = float32(float64(x) / math.Sqrt(norm))
	}
	return out, nil
}

func tokenize(s string) []string {
	var words []string
	var word []rune
	flush := func() {
		if len(word) > 0 {
			words = append(words, string(word))
			word = word[:0]
		}
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			word = append(word, unicode.ToLower(r))
		} else {
			flush()
		}
	}
	flush()
	return words
}
