package workerpool

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func TestPool(t *testing.T) {
	tests := []struct {
		name    string
		workers int
		tasks   int
	}{
		{"1 worker, 1 task", 1, 1},
		{"1 worker, 100 tasks", 1, 100},
		{"4 workers, 100 tasks", 4, 100},
		{"20 workers, 10000 tasks", 20, 10000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := New(tt.workers)
			defer p.Close()

			var counter atomic.Int64
			tasks := make([]func(), tt.tasks)
			for i := range tasks {
				tasks[i] = func() {
					counter.Add(1)
				}
			}
			p.Submit(tasks)
			if got := counter.Load(); got != int64(tt.tasks) {
				t.Errorf("executed %d tasks, want %d", got, tt.tasks)
			}
		})
	}
}

func TestPoolCorrectResults(t *testing.T) {
	p := New(4)
	defer p.Close()

	const n = 1000
	results := make([]int, n)
	tasks := make([]func(), n)
	for i := range n {
		tasks[i] = func() {
			results[i] = i * i
		}
	}
	p.Submit(tasks)
	for i := range n {
		if results[i] != i*i {
			t.Fatalf("results[%d] = %d, want %d", i, results[i], i*i)
		}
	}
}

func TestPoolEmptySubmit(t *testing.T) {
	p := New(4)
	defer p.Close()
	p.Submit(nil)
	p.Submit([]func(){})
}

func TestPoolCloseMultiple(t *testing.T) {
	p := New(4)
	p.Close()
	p.Close() // should not panic
}

func TestPoolCloseConcurrent(t *testing.T) {
	p := New(4)
	p.Submit([]func(){func() {}})

	var wg sync.WaitGroup
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.Close() // must not panic or race
		}()
	}
	wg.Wait()
}

func TestPoolNoGoroutineLeak(t *testing.T) {
	before := runtime.NumGoroutine()
	p := New(20)
	p.Submit([]func(){func() {}})
	p.Close()
	runtime.Gosched()
	after := runtime.NumGoroutine()
	// Allow some slack for GC/runtime goroutines, but pool goroutines must be gone.
	if after > before+5 {
		t.Errorf("goroutine leak: before=%d, after=%d", before, after)
	}
}

func BenchmarkPoolSubmit(b *testing.B) {
	p := New(runtime.NumCPU())
	defer p.Close()

	tasks := make([]func(), 20)
	for i := range tasks {
		tasks[i] = func() {
			// Simulate minimal work (like a GEMV chunk)
			x := 0
			for j := range 100 {
				x += j
			}
			_ = x
		}
	}
	b.ResetTimer()
	for b.Loop() {
		p.Submit(tasks)
	}
}

func BenchmarkGoroutineSpawn(b *testing.B) {
	tasks := make([]func(), 20)
	for i := range tasks {
		tasks[i] = func() {
			x := 0
			for j := range 100 {
				x += j
			}
			_ = x
		}
	}
	b.ResetTimer()
	for b.Loop() {
		done := make(chan struct{})
		var counter atomic.Int64
		n := int64(len(tasks))
		for _, task := range tasks {
			go func() {
				task()
				if counter.Add(1) == n {
					close(done)
				}
			}()
		}
		<-done
	}
}
