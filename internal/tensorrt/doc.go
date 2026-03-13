// Package tensorrt provides bindings for the NVIDIA TensorRT inference
// library. The default build uses purego (dlopen/dlsym, no CGo) to load
// libtrt_capi.so at runtime. A CGo variant is available with the
// "cuda,tensorrt" build tags.
package tensorrt
