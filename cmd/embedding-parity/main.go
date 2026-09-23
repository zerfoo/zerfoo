// embedding-parity compares a Zerfoo contextual GGUF encoder with frozen
// reference token IDs, vectors, and query/document rankings.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"slices"
	"sort"
	"time"

	"github.com/zerfoo/zerfoo/inference"
)

type reference struct {
	Model     string `json:"model"`
	MaxLength int    `json:"max_length"`
	Rows      []struct {
		Role     string    `json:"role"`
		Text     string    `json:"text"`
		TokenIDs []int     `json:"token_ids"`
		Vector   []float64 `json:"vector"`
	} `json:"rows"`
	ReferenceRanks [][]int `json:"reference_ranks"`
}

type rowReport struct {
	Role       string  `json:"role"`
	Tokens     int     `json:"tokens"`
	TokenMatch bool    `json:"token_match"`
	Cosine     float64 `json:"cosine"`
	MaxAbs     float64 `json:"max_abs"`
	Millis     int64   `json:"millis"`
}

func main() {
	base := flag.String("base", "", "base embedding GGUF")
	adapter := flag.String("adapter", "", "optional GGUF LoRA adapter")
	fixture := flag.String("reference", "", "reference vectors JSON")
	device := flag.String("device", "cpu", "compute device")
	minCosine := flag.Float64("min-cosine", 0.999, "minimum vector cosine")
	maxAbs := flag.Float64("max-abs", 0.005, "maximum per-component absolute error")
	flag.Parse()
	if *base == "" || *fixture == "" {
		fatal(fmt.Errorf("-base and -reference are required"))
	}
	raw, err := os.ReadFile(*fixture)
	if err != nil {
		fatal(err)
	}
	var ref reference
	if err := json.Unmarshal(raw, &ref); err != nil {
		fatal(err)
	}
	if ref.MaxLength <= 0 || len(ref.Rows) == 0 {
		fatal(fmt.Errorf("reference has no rows or invalid maximum length"))
	}
	var model *inference.EmbeddingModel
	start := time.Now()
	if *adapter == "" {
		model, err = inference.LoadEmbeddingFile(*base, inference.WithDevice(*device), inference.WithMaxSeqLen(ref.MaxLength))
	} else {
		model, err = inference.LoadEmbeddingFileWithAdapter(*base, *adapter, inference.WithDevice(*device), inference.WithMaxSeqLen(ref.MaxLength))
	}
	if err != nil {
		fatal(err)
	}
	defer func() { _ = model.Close() }()
	loadMillis := time.Since(start).Milliseconds()
	var loadedMemory runtime.MemStats
	runtime.ReadMemStats(&loadedMemory)
	var rows []rowReport
	var goQueries, goDocs [][]float32
	var refQueries, refDocs [][]float64
	passed := true
	for _, item := range ref.Rows {
		ids, err := model.TokenIDs(item.Text)
		if err != nil {
			fatal(err)
		}
		started := time.Now()
		vec, err := model.EmbedIDs(context.Background(), ids)
		if err != nil {
			fatal(err)
		}
		if len(vec) != len(item.Vector) {
			fatal(fmt.Errorf("embedding dimension %d, reference %d", len(vec), len(item.Vector)))
		}
		var dot, goNorm, refNorm, rowMax float64
		for i, x := range vec {
			y := item.Vector[i]
			dot += float64(x) * y
			goNorm += float64(x) * float64(x)
			refNorm += y * y
			rowMax = math.Max(rowMax, math.Abs(float64(x)-y))
		}
		cosine := dot / math.Sqrt(goNorm*refNorm)
		matched := slices.Equal(ids, item.TokenIDs)
		rows = append(rows, rowReport{Role: item.Role, Tokens: len(ids), TokenMatch: matched, Cosine: cosine, MaxAbs: rowMax, Millis: time.Since(started).Milliseconds()})
		if !matched || cosine < *minCosine || rowMax > *maxAbs {
			passed = false
		}
		switch item.Role {
		case "query":
			goQueries = append(goQueries, vec)
			refQueries = append(refQueries, item.Vector)
		case "document":
			goDocs = append(goDocs, vec)
			refDocs = append(refDocs, item.Vector)
		}
	}
	if len(goQueries) == 0 || len(goDocs) == 0 {
		fatal(fmt.Errorf("reference requires at least one query and document"))
	}
	goRanks := ranks(goQueries, goDocs)
	refRanks := ranks64(refQueries, refDocs)
	top1Matches := 0
	for i := range goRanks {
		if len(refRanks[i]) > 0 && len(goRanks[i]) > 0 && goRanks[i][0] == refRanks[i][0] {
			top1Matches++
		} else {
			passed = false
		}
	}
	if len(ref.ReferenceRanks) > 0 && !slices.EqualFunc(refRanks, ref.ReferenceRanks, func(a, b []int) bool { return slices.Equal(a, b) }) {
		fatal(fmt.Errorf("stored reference ranks disagree with reference vectors"))
	}
	baseHash, err := digest(*base)
	if err != nil {
		fatal(err)
	}
	adapterHash := ""
	if *adapter != "" {
		adapterHash, err = digest(*adapter)
		if err != nil {
			fatal(err)
		}
	}
	result := map[string]any{
		"passed": passed, "model": ref.Model, "device": *device,
		"base_sha256": baseHash, "adapter_sha256": adapterHash,
		"reference_sha256": hex.EncodeToString(hashBytes(raw)),
		"load_millis":      loadMillis, "go_heap_alloc_bytes_after_load": loadedMemory.Alloc,
		"go_heap_sys_bytes_after_load": loadedMemory.Sys, "rows": rows,
		"go_ranks": goRanks, "reference_ranks": refRanks,
		"top1_matches": top1Matches, "query_count": len(goQueries),
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		fatal(err)
	}
	fmt.Println(string(encoded))
	if !passed {
		os.Exit(1)
	}
}

func ranks(queries, docs [][]float32) [][]int {
	out := make([][]int, len(queries))
	for i, q := range queries {
		scores := make([]float64, len(docs))
		for j, d := range docs {
			for k, v := range d {
				scores[j] += float64(q[k]) * float64(v)
			}
		}
		out[i] = order(scores)
	}
	return out
}

func ranks64(queries, docs [][]float64) [][]int {
	out := make([][]int, len(queries))
	for i, q := range queries {
		scores := make([]float64, len(docs))
		for j, d := range docs {
			for k, v := range d {
				scores[j] += q[k] * v
			}
		}
		out[i] = order(scores)
	}
	return out
}

func order(scores []float64) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		if scores[indices[i]] != scores[indices[j]] {
			return scores[indices[i]] > scores[indices[j]]
		}
		return indices[i] < indices[j]
	})
	return indices
}

func hashBytes(b []byte) []byte {
	h := sha256.Sum256(b)
	return h[:]
}

func digest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
