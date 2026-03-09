//go:build !cuda && !(darwin || (linux && arm64))

package cuda

// On unsupported platforms, CUDA is never available.
// These stubs ensure the package compiles everywhere.

func dlopenImpl(_ string, _ int) uintptr { return 0 }
func dlsymImpl(_ uintptr, _ string) uintptr { return 0 }
func dlcloseImpl(_ uintptr) int { return -1 }
func dlerrorImpl() string { return "cuda: unsupported platform" }
func ccall(_ uintptr, _ ...uintptr) uintptr { return 0 }
