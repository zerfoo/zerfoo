//go:build opencl

package kernels

/*
#cgo LDFLAGS: -lOpenCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdlib.h>
#include <string.h>

// compile_program compiles an OpenCL program from source.
static cl_program compile_program(cl_context ctx, cl_device_id dev, const char* src, cl_int* err) {
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, err);
    if (*err != CL_SUCCESS) return NULL;
    *err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    return prog;
}

// run_binary_kernel dispatches a kernel with two input buffers and one output buffer.
static cl_int run_binary_kernel(cl_program prog, const char* name,
    cl_command_queue queue, cl_mem a, cl_mem b, cl_mem c, int n) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &b);
    clSetKernelArg(k, 2, sizeof(cl_mem), &c);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}

// run_unary_kernel dispatches a kernel with one input buffer and one output buffer.
static cl_int run_unary_kernel(cl_program prog, const char* name,
    cl_command_queue queue, cl_mem a, cl_mem c, int n) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &c);
    clSetKernelArg(k, 2, sizeof(int), &n);
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}

// run_scalar_kernel dispatches a kernel with one input buffer, a scalar, and one output buffer.
static cl_int run_scalar_kernel(cl_program prog, const char* name,
    cl_command_queue queue, cl_mem a, float scalar, cl_mem c, int n) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(float), &scalar);
    clSetKernelArg(k, 2, sizeof(cl_mem), &c);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}

// run_fill_kernel dispatches the fill kernel.
static cl_int run_fill_kernel(cl_program prog, const char* name,
    cl_command_queue queue, cl_mem data, float value, int n) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &data);
    clSetKernelArg(k, 1, sizeof(float), &value);
    clSetKernelArg(k, 2, sizeof(int), &n);
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}

// run_reduction_kernel dispatches a reduction kernel (sum_axis, softmax).
static cl_int run_reduction_kernel(cl_program prog, const char* name,
    cl_command_queue queue, cl_mem input, cl_mem output,
    int outer, int inner, int axis_size) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, name, &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &input);
    clSetKernelArg(k, 1, sizeof(cl_mem), &output);
    clSetKernelArg(k, 2, sizeof(int), &outer);
    clSetKernelArg(k, 3, sizeof(int), &inner);
    clSetKernelArg(k, 4, sizeof(int), &axis_size);
    size_t global = (size_t)(outer * inner);
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}

// run_tanh_prime_kernel dispatches tanh_prime with three buffers.
static cl_int run_tanh_prime_kernel(cl_program prog,
    cl_command_queue queue, cl_mem a, cl_mem upstream, cl_mem c, int n) {
    cl_int err;
    cl_kernel k = clCreateKernel(prog, "kernel_tanh_prime", &err);
    if (err != CL_SUCCESS) return err;
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k, 1, sizeof(cl_mem), &upstream);
    clSetKernelArg(k, 2, sizeof(cl_mem), &c);
    clSetKernelArg(k, 3, sizeof(int), &n);
    size_t global = (size_t)n;
    err = clEnqueueNDRangeKernel(queue, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clReleaseKernel(k);
    return err;
}
*/
import "C"

import (
	_ "embed"
	"fmt"
	"unsafe"
)

//go:embed elementwise.cl
var elementwiseCL string

// Program holds a compiled OpenCL program.
type Program struct {
	prog  C.cl_program
	queue C.cl_command_queue
}

// Compile compiles the elementwise kernels for the given context and device.
func Compile(ctx, dev, queue unsafe.Pointer) (*Program, error) {
	cSrc := C.CString(elementwiseCL)
	defer C.free(unsafe.Pointer(cSrc))

	var errCode C.cl_int
	prog := C.compile_program(C.cl_context(ctx), C.cl_device_id(dev), cSrc, &errCode)
	if errCode != C.CL_SUCCESS {
		return nil, fmt.Errorf("compile OpenCL kernels: error %d", errCode)
	}

	return &Program{
		prog:  prog,
		queue: C.cl_command_queue(queue),
	}, nil
}

// Destroy releases the compiled program.
func (p *Program) Destroy() {
	if p.prog != nil {
		C.clReleaseProgram(p.prog)
		p.prog = nil
	}
}

// SetQueue updates the command queue for kernel dispatch.
func (p *Program) SetQueue(queue unsafe.Pointer) {
	p.queue = C.cl_command_queue(queue)
}

func (p *Program) Add(a, b, c unsafe.Pointer, n int) error {
	if err := C.run_binary_kernel(p.prog, C.CString("kernel_add"), p.queue, C.cl_mem(a), C.cl_mem(b), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_add: error %d", err)
	}
	return nil
}

func (p *Program) Sub(a, b, c unsafe.Pointer, n int) error {
	if err := C.run_binary_kernel(p.prog, C.CString("kernel_sub"), p.queue, C.cl_mem(a), C.cl_mem(b), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_sub: error %d", err)
	}
	return nil
}

func (p *Program) Mul(a, b, c unsafe.Pointer, n int) error {
	if err := C.run_binary_kernel(p.prog, C.CString("kernel_mul"), p.queue, C.cl_mem(a), C.cl_mem(b), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_mul: error %d", err)
	}
	return nil
}

func (p *Program) Div(a, b, c unsafe.Pointer, n int) error {
	if err := C.run_binary_kernel(p.prog, C.CString("kernel_div"), p.queue, C.cl_mem(a), C.cl_mem(b), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_div: error %d", err)
	}
	return nil
}

func (p *Program) Pow(base, exp, c unsafe.Pointer, n int) error {
	if err := C.run_binary_kernel(p.prog, C.CString("kernel_pow"), p.queue, C.cl_mem(base), C.cl_mem(exp), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_pow: error %d", err)
	}
	return nil
}

func (p *Program) Exp(a, c unsafe.Pointer, n int) error {
	if err := C.run_unary_kernel(p.prog, C.CString("kernel_exp"), p.queue, C.cl_mem(a), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_exp: error %d", err)
	}
	return nil
}

func (p *Program) Log(a, c unsafe.Pointer, n int) error {
	if err := C.run_unary_kernel(p.prog, C.CString("kernel_log"), p.queue, C.cl_mem(a), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_log: error %d", err)
	}
	return nil
}

func (p *Program) Sqrt(a, c unsafe.Pointer, n int) error {
	if err := C.run_unary_kernel(p.prog, C.CString("kernel_sqrt"), p.queue, C.cl_mem(a), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_sqrt: error %d", err)
	}
	return nil
}

func (p *Program) Rsqrt(a, c unsafe.Pointer, n int) error {
	if err := C.run_unary_kernel(p.prog, C.CString("kernel_rsqrt"), p.queue, C.cl_mem(a), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_rsqrt: error %d", err)
	}
	return nil
}

func (p *Program) Tanh(a, c unsafe.Pointer, n int) error {
	if err := C.run_unary_kernel(p.prog, C.CString("kernel_tanh"), p.queue, C.cl_mem(a), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_tanh: error %d", err)
	}
	return nil
}

func (p *Program) TanhPrime(a, upstream, c unsafe.Pointer, n int) error {
	if err := C.run_tanh_prime_kernel(p.prog, p.queue, C.cl_mem(a), C.cl_mem(upstream), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_tanh_prime: error %d", err)
	}
	return nil
}

func (p *Program) AddScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	if err := C.run_scalar_kernel(p.prog, C.CString("kernel_add_scalar"), p.queue, C.cl_mem(a), C.float(scalar), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_add_scalar: error %d", err)
	}
	return nil
}

func (p *Program) MulScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	if err := C.run_scalar_kernel(p.prog, C.CString("kernel_mul_scalar"), p.queue, C.cl_mem(a), C.float(scalar), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_mul_scalar: error %d", err)
	}
	return nil
}

func (p *Program) DivScalar(a unsafe.Pointer, scalar float32, c unsafe.Pointer, n int) error {
	if err := C.run_scalar_kernel(p.prog, C.CString("kernel_div_scalar"), p.queue, C.cl_mem(a), C.float(scalar), C.cl_mem(c), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_div_scalar: error %d", err)
	}
	return nil
}

func (p *Program) Fill(data unsafe.Pointer, value float32, n int) error {
	if err := C.run_fill_kernel(p.prog, C.CString("kernel_fill"), p.queue, C.cl_mem(data), C.float(value), C.int(n)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_fill: error %d", err)
	}
	return nil
}

func (p *Program) SumAxis(input, output unsafe.Pointer, outer, inner, axisSize int) error {
	if err := C.run_reduction_kernel(p.prog, C.CString("kernel_sum_axis"), p.queue, C.cl_mem(input), C.cl_mem(output), C.int(outer), C.int(inner), C.int(axisSize)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_sum_axis: error %d", err)
	}
	return nil
}

func (p *Program) Softmax(input, output unsafe.Pointer, outer, inner, axisSize int) error {
	if err := C.run_reduction_kernel(p.prog, C.CString("kernel_softmax"), p.queue, C.cl_mem(input), C.cl_mem(output), C.int(outer), C.int(inner), C.int(axisSize)); err != C.CL_SUCCESS {
		return fmt.Errorf("kernel_softmax: error %d", err)
	}
	return nil
}
