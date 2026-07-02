package ios_test

import (
	"os/exec"
	"testing"
)

// TestIOSDemo_Build verifies that the Go mobile package compiles for
// iOS targets. This test is skipped if gomobile is not installed.
func TestIOSDemo_Build(t *testing.T) {
	// Check that gomobile is available.
	gomobilePath, err := exec.LookPath("gomobile")
	if err != nil {
		t.Skip("gomobile not installed; skipping iOS build test")
	}

	// Verify gomobile can at least parse the mobile package for iOS.
	// Use "gomobile bind -target=ios" in dry-run-like fashion by just
	// building the package without producing output.
	cmd := exec.Command(gomobilePath, "bind",
		"-target", "ios",
		"-o", t.TempDir()+"/Mobile.xcframework",
		"github.com/zerfoo/zerfoo/tests/mobile/",
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("gomobile bind failed: %v\n%s", err, output)
	}
}
