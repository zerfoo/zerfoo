package android_test

import (
	"os/exec"
	"testing"
)

// TestAndroidDemo_Build verifies that the Go mobile package compiles for the
// Android target using gomobile. This test is skipped when gomobile is not
// installed (CI environments without Android NDK).
func TestAndroidDemo_Build(t *testing.T) {
	gomobilePath, err := exec.LookPath("gomobile")
	if err != nil {
		t.Skip("gomobile not found in PATH; skipping Android build test")
	}

	// Verify gomobile can at least parse the mobile package for Android.
	// We use "gomobile bind -target=android" in dry-run fashion by checking
	// that the command starts without immediate flag errors. A full bind
	// requires the Android NDK, so we fall back to GOOS/GOARCH cross-compile
	// if gomobile bind is not feasible.
	cmd := exec.Command(gomobilePath, "bind", "-target=android", "-androidapi=24", "-o", t.TempDir()+"/mobile.aar", "github.com/zerfoo/zerfoo/tests/mobile")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// If gomobile bind fails due to missing NDK, verify the package at
		// least cross-compiles with GOOS=android.
		t.Logf("gomobile bind failed (likely missing NDK): %s\n%s", err, output)
		t.Log("Falling back to GOOS=android cross-compilation check")

		crossCmd := exec.Command("go", "build", "github.com/zerfoo/zerfoo/tests/mobile")
		crossCmd.Env = append(crossCmd.Environ(), "GOOS=android", "GOARCH=arm64", "CGO_ENABLED=0")
		crossOutput, crossErr := crossCmd.CombinedOutput()
		if crossErr != nil {
			t.Fatalf("mobile package does not cross-compile for android/arm64: %s\n%s", crossErr, crossOutput)
		}
		t.Log("mobile package cross-compiles successfully for android/arm64")
		return
	}
	t.Log("gomobile bind succeeded for Android target")
}
