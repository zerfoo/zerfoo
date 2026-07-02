// OpenCL kernels for elementwise operations.
// Each kernel operates on float arrays with n elements.

__kernel void kernel_add(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}

__kernel void kernel_sub(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] - b[i];
}

__kernel void kernel_mul(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] * b[i];
}

__kernel void kernel_div(__global const float* a, __global const float* b, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] / b[i];
}

__kernel void kernel_pow(__global const float* base, __global const float* exp, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = pow(base[i], exp[i]);
}

__kernel void kernel_exp(__global const float* a, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = exp(a[i]);
}

__kernel void kernel_log(__global const float* a, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = log(a[i]);
}

__kernel void kernel_sqrt(__global const float* a, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = sqrt(a[i]);
}

__kernel void kernel_rsqrt(__global const float* a, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = rsqrt(a[i]);
}

__kernel void kernel_tanh(__global const float* a, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = tanh(a[i]);
}

__kernel void kernel_tanh_prime(__global const float* a, __global const float* upstream, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) {
        float t = tanh(a[i]);
        c[i] = (1.0f - t * t) * upstream[i];
    }
}

__kernel void kernel_add_scalar(__global const float* a, float scalar, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + scalar;
}

__kernel void kernel_mul_scalar(__global const float* a, float scalar, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] * scalar;
}

__kernel void kernel_div_scalar(__global const float* a, float scalar, __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] / scalar;
}

__kernel void kernel_fill(__global float* data, float value, int n) {
    int i = get_global_id(0);
    if (i < n) data[i] = value;
}

__kernel void kernel_sum_axis(
    __global const float* input,
    __global float* output,
    int outer, int inner, int axis_size
) {
    int idx = get_global_id(0);
    int total = outer * inner;
    if (idx >= total) return;

    int o = idx / inner;
    int i = idx % inner;

    float sum = 0.0f;
    for (int a = 0; a < axis_size; a++) {
        sum += input[(o * axis_size + a) * inner + i];
    }
    output[o * inner + i] = sum;
}

__kernel void kernel_softmax(
    __global const float* input,
    __global float* output,
    int outer, int inner, int axis_size
) {
    int idx = get_global_id(0);
    if (idx >= outer * inner) return;

    int o = idx / inner;
    int i = idx % inner;

    // Find max for numerical stability.
    float max_val = -INFINITY;
    for (int a = 0; a < axis_size; a++) {
        float val = input[(o * axis_size + a) * inner + i];
        if (val > max_val) max_val = val;
    }

    // Compute exp and sum.
    float sum = 0.0f;
    for (int a = 0; a < axis_size; a++) {
        float val = exp(input[(o * axis_size + a) * inner + i] - max_val);
        output[(o * axis_size + a) * inner + i] = val;
        sum += val;
    }

    // Normalize.
    for (int a = 0; a < axis_size; a++) {
        output[(o * axis_size + a) * inner + i] /= sum;
    }
}
