package testutil

import (
	"bufio"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
)

func LoadPrompts(path string, n, seed int) []string {
	var out []string
	f, err := os.Open(path)
	if err == nil {
		defer func() { _ = f.Close() }()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			s := strings.TrimSpace(sc.Text())
			if s != "" {
				out = append(out, s)
			}
		}
	}
	if len(out) == 0 {
		r := rand.New(rand.NewPCG(uint64(seed), uint64(seed*2)))
		for i := 0; i < n; i++ {
			out = append(out, randSentence(r))
		}
	}
	if n > 0 && len(out) > n {
		out = out[:n]
	}
	return out
}

func randSentence(r *rand.Rand) string {
	words := []string{"alpha", "beta", "gamma", "delta", "stock", "crypto", "market", "trend", "signal", "risk", "matrix", "tensor"}
	k := 3 + r.IntN(9)
	var b strings.Builder
	for i := 0; i < k; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[r.IntN(len(words))])
	}
	return b.String()
}

func MeanRelativeError(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		num += math.Abs(ai - bi)
		den += math.Abs(bi) + 1e-12
	}
	return num / den
}

func TopKAgreement(a, b []float32, k int) float64 {
	top := func(x []float32, k int) map[int]struct{} {
		type p struct {
			i int
			v float32
		}
		n := len(x)
		if k > n {
			k = n
		}
		idx := make([]p, n)
		for i := range x {
			idx[i] = p{i: i, v: x[i]}
		}
		// partial selection sort for simplicity
		for i := 0; i < k; i++ {
			m := i
			for j := i + 1; j < n; j++ {
				if idx[j].v > idx[m].v {
					m = j
				}
			}
			idx[i], idx[m] = idx[m], idx[i]
		}
		set := make(map[int]struct{}, k)
		for i := 0; i < k; i++ {
			set[idx[i].i] = struct{}{}
		}
		return set
	}
	A := top(a, k)
	B := top(b, k)
	var hit int
	for i := range A {
		if _, ok := B[i]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(A))
}

func RelError(a, b []float64) float64 {
	var num, den float64
	for i := range a {
		num += math.Abs(a[i] - b[i])
		den += math.Abs(b[i]) + 1e-12
	}
	return num / den
}

func Require[T any](t *testing.T, v T, ok bool, msg string) T {
	if !ok {
		t.Fatal(msg)
	}
	return v
}
