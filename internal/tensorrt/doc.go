// Package tensorrt provides bindings for the NVIDIA TensorRT inference
// library via purego (dlopen/dlsym, no CGo). It loads libtrt_capi.so at
// runtime. Call Available() to check whether the library was found.
package tensorrt
