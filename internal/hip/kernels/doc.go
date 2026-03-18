// Package kernels provides Go wrappers for custom HIP kernels via purego (Stability: alpha)
// dlopen. Build libhipkernels.so first: cd internal/hip/kernels && make.
// No build tags required; use kernels.Available() to check runtime availability.
// Stability: alpha
package kernels
