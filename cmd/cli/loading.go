package cli

import (
	"fmt"
	"io"
	"sync"
	"time"
)

// loadingIndicator displays a progress indicator while a model is loading.
type loadingIndicator struct {
	out  io.Writer
	done chan struct{}
	wg   sync.WaitGroup
}

// startLoading begins a loading indicator on the given writer.
// Call the returned indicator's stop method when loading completes.
func startLoading(out io.Writer) *loadingIndicator {
	li := &loadingIndicator{
		out:  out,
		done: make(chan struct{}),
	}
	li.wg.Add(1)
	go li.run()
	return li
}

func (li *loadingIndicator) run() {
	defer li.wg.Done()
	start := time.Now()
	tty := isTTY(li.out)

	if !tty {
		_, _ = fmt.Fprint(li.out, "Loading model...\n")
		<-li.done
		return
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	frames := []string{"|", "/", "-", "\\"}
	i := 0

	for {
		elapsed := time.Since(start).Truncate(100 * time.Millisecond)
		_, _ = fmt.Fprintf(li.out, "\rLoading model... %s %.1fs", frames[i%len(frames)], elapsed.Seconds())
		i++

		select {
		case <-li.done:
			_, _ = fmt.Fprintf(li.out, "\rLoading model... done (%.1fs)\n", time.Since(start).Seconds())
			return
		case <-ticker.C:
		}
	}
}

// stop stops the loading indicator and waits for it to finish writing.
func (li *loadingIndicator) stop() {
	close(li.done)
	li.wg.Wait()
}
