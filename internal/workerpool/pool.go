// Package workerpool provides a persistent pool of goroutines that process
// submitted tasks. It replaces per-call goroutine spawning in GEMV and
// element-wise operations.
package workerpool

import "sync"

// Pool is a fixed-size pool of long-lived worker goroutines.
type Pool struct {
	tasks chan func()
	wg    sync.WaitGroup
	once  sync.Once
}

// New creates a pool with n worker goroutines that block on a shared task channel.
func New(n int) *Pool {
	p := &Pool{
		tasks: make(chan func(), n*4),
	}
	p.wg.Add(n)
	for range n {
		go func() {
			defer p.wg.Done()
			for task := range p.tasks {
				task()
			}
		}()
	}
	return p
}

// Submit sends all tasks to the pool and blocks until every task completes.
func (p *Pool) Submit(tasks []func()) {
	if len(tasks) == 0 {
		return
	}
	var done sync.WaitGroup
	done.Add(len(tasks))
	for _, task := range tasks {
		p.tasks <- func() {
			task()
			done.Done()
		}
	}
	done.Wait()
}

// Close shuts down the pool. It is safe to call concurrently and multiple times.
func (p *Pool) Close() {
	p.once.Do(func() {
		close(p.tasks)
		p.wg.Wait()
	})
}
