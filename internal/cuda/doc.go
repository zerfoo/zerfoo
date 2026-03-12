// Package cuda provides low-level bindings for the CUDA runtime API using
// dlopen/dlsym (no CGo). CUDA availability is detected at runtime; when
// libcudart is not loadable the package functions return descriptive errors.
package cuda
