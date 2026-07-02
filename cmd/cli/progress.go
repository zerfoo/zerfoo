package cli

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// isTTY returns true if the writer is connected to a terminal.
func isTTY(w io.Writer) bool {
	if f, ok := w.(*os.File); ok {
		fi, err := f.Stat()
		if err != nil {
			return false
		}
		return fi.Mode()&os.ModeCharDevice != 0
	}
	return false
}

// progressDisplay formats and writes download progress to an io.Writer.
// It throttles output to avoid excessive writes (at most once per 100ms for TTY,
// or every 10% change for non-TTY).
type progressDisplay struct {
	out   io.Writer
	isTTY bool

	mu          sync.Mutex
	lastPct     int
	lastWritten time.Time
}

// newProgressDisplay creates a progress display that writes to out.
// If isTTY is true, it uses carriage return for in-place updates with a progress bar.
// Otherwise, it prints a percentage line every 10%.
func newProgressDisplay(out io.Writer, isTTY bool) *progressDisplay {
	return &progressDisplay{
		out:     out,
		isTTY:   isTTY,
		lastPct: -1,
	}
}

// callback returns a function suitable for use as registry.ProgressFunc.
func (p *progressDisplay) callback(downloaded, total int64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now()

	if total <= 0 {
		// Unknown total: throttle to once per 250ms.
		if now.Sub(p.lastWritten) < 250*time.Millisecond {
			return
		}
		p.lastWritten = now
		fmt.Fprintf(p.out, "Downloaded %s\n", formatBytes(downloaded))
		return
	}

	pct := int(downloaded * 100 / total)
	if pct > 100 {
		pct = 100
	}

	if p.isTTY {
		// Throttle to at most once per 100ms, but always print 100%.
		if pct < 100 && now.Sub(p.lastWritten) < 100*time.Millisecond {
			return
		}
		p.lastWritten = now
		p.lastPct = pct

		bar := renderBar(pct, 30)
		fmt.Fprintf(p.out, "\r%s %3d%% %s/%s", bar, pct, formatBytes(downloaded), formatBytes(total))
		if pct == 100 {
			fmt.Fprintln(p.out)
		}
	} else {
		// Non-TTY: print every 10% boundary.
		bucket := (pct / 10) * 10
		if bucket <= p.lastPct && pct < 100 {
			return
		}
		p.lastPct = bucket
		fmt.Fprintf(p.out, "Downloading: %d%% (%s/%s)\n", pct, formatBytes(downloaded), formatBytes(total))
	}
}

// renderBar produces a progress bar string like "[=====>    ]".
func renderBar(pct, width int) string {
	filled := width * pct / 100
	if filled > width {
		filled = width
	}

	bar := make([]byte, width)
	for i := range bar {
		if i < filled {
			bar[i] = '='
		} else {
			bar[i] = ' '
		}
	}
	if filled > 0 && filled < width {
		bar[filled-1] = '>'
	}
	return "[" + string(bar) + "]"
}

// formatBytes formats a byte count as a human-readable string.
func formatBytes(b int64) string {
	const (
		kb = 1024
		mb = 1024 * kb
		gb = 1024 * mb
	)
	switch {
	case b >= gb:
		return fmt.Sprintf("%.1fGB", float64(b)/float64(gb))
	case b >= mb:
		return fmt.Sprintf("%.1fMB", float64(b)/float64(mb))
	case b >= kb:
		return fmt.Sprintf("%.1fKB", float64(b)/float64(kb))
	default:
		return fmt.Sprintf("%dB", b)
	}
}
