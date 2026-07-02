/* C shim for TensorRT C++ API, enabling CGo access from Go. */
#ifndef TRT_CAPI_H
#define TRT_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles wrapping TensorRT C++ objects. */
typedef void* trt_logger_t;
typedef void* trt_builder_t;
typedef void* trt_network_t;
typedef void* trt_builder_config_t;
typedef void* trt_runtime_t;
typedef void* trt_engine_t;
typedef void* trt_context_t;
typedef void* trt_layer_t;
typedef void* trt_tensor_t;
typedef void* trt_host_memory_t;

/* Severity levels for the logger. */
typedef enum {
    TRT_SEVERITY_INTERNAL_ERROR = 0,
    TRT_SEVERITY_ERROR          = 1,
    TRT_SEVERITY_WARNING        = 2,
    TRT_SEVERITY_INFO           = 3,
    TRT_SEVERITY_VERBOSE        = 4
} trt_severity_t;

/* Data types. */
typedef enum {
    TRT_FLOAT32 = 0,
    TRT_FLOAT16 = 1,
    TRT_INT8    = 2,
    TRT_INT32   = 3
} trt_data_type_t;

/* Activation types. */
typedef enum {
    TRT_ACTIVATION_RELU    = 0,
    TRT_ACTIVATION_SIGMOID = 1,
    TRT_ACTIVATION_TANH    = 2
} trt_activation_type_t;

/* ElementWise operation types. */
typedef enum {
    TRT_ELEMENTWISE_SUM  = 0,
    TRT_ELEMENTWISE_PROD = 1,
    TRT_ELEMENTWISE_MAX  = 2,
    TRT_ELEMENTWISE_MIN  = 3,
    TRT_ELEMENTWISE_SUB  = 4,
    TRT_ELEMENTWISE_DIV  = 5
} trt_elementwise_op_t;

/* MatrixMultiply operation types. */
typedef enum {
    TRT_MATMUL_NONE      = 0,
    TRT_MATMUL_TRANSPOSE = 1
} trt_matrix_op_t;

/* Reduce operation types. */
typedef enum {
    TRT_REDUCE_SUM  = 0,
    TRT_REDUCE_PROD = 1,
    TRT_REDUCE_MAX  = 2,
    TRT_REDUCE_MIN  = 3,
    TRT_REDUCE_AVG  = 4
} trt_reduce_op_t;

/* Builder config flags. */
typedef enum {
    TRT_FLAG_FP16 = 0,
    TRT_FLAG_INT8 = 1
} trt_builder_flag_t;

/* ---- Logger ---- */
trt_logger_t trt_create_logger(trt_severity_t min_severity);
void trt_destroy_logger(trt_logger_t logger);

/* ---- Builder ---- */
trt_builder_t trt_create_builder(trt_logger_t logger);
void trt_destroy_builder(trt_builder_t builder);

/* ---- Network ---- */
/* Creates a network with explicit batch (kEXPLICIT_BATCH flag). */
trt_network_t trt_create_network(trt_builder_t builder);
void trt_destroy_network(trt_network_t network);
int trt_network_num_inputs(trt_network_t network);
int trt_network_num_outputs(trt_network_t network);
int trt_network_num_layers(trt_network_t network);

/* ---- Network: add layers ---- */
trt_tensor_t trt_network_add_input(trt_network_t network, const char* name,
                                   trt_data_type_t dtype, int nb_dims,
                                   const int32_t* dims);
void trt_network_mark_output(trt_network_t network, trt_tensor_t tensor);

trt_layer_t trt_network_add_activation(trt_network_t network,
                                       trt_tensor_t input,
                                       trt_activation_type_t type);

trt_layer_t trt_network_add_elementwise(trt_network_t network,
                                        trt_tensor_t input1,
                                        trt_tensor_t input2,
                                        trt_elementwise_op_t op);

trt_layer_t trt_network_add_matrix_multiply(trt_network_t network,
                                            trt_tensor_t input0,
                                            trt_matrix_op_t op0,
                                            trt_tensor_t input1,
                                            trt_matrix_op_t op1);

trt_layer_t trt_network_add_softmax(trt_network_t network,
                                    trt_tensor_t input, int axis);

trt_layer_t trt_network_add_reduce(trt_network_t network, trt_tensor_t input,
                                   trt_reduce_op_t op, uint32_t reduce_axes,
                                   int keep_dims);

trt_layer_t trt_network_add_constant(trt_network_t network, int nb_dims,
                                     const int32_t* dims,
                                     trt_data_type_t dtype,
                                     const void* weights, int64_t count);

trt_layer_t trt_network_add_shuffle(trt_network_t network,
                                    trt_tensor_t input);

trt_layer_t trt_network_add_convolution_nd(trt_network_t network,
                                           trt_tensor_t input,
                                           int nb_output_maps,
                                           int kernel_nb_dims,
                                           const int32_t* kernel_size,
                                           const void* kernel_weights,
                                           int64_t kernel_count,
                                           const void* bias_weights,
                                           int64_t bias_count);

/* ---- Layer helpers ---- */
trt_tensor_t trt_layer_get_output(trt_layer_t layer, int index);
void trt_layer_set_name(trt_layer_t layer, const char* name);

/* Shuffle layer reshape. */
void trt_shuffle_set_reshape_dims(trt_layer_t shuffle, int nb_dims,
                                  const int32_t* dims);
void trt_shuffle_set_first_transpose(trt_layer_t shuffle, int nb_dims,
                                     const int32_t* perm);

/* ---- BuilderConfig ---- */
trt_builder_config_t trt_create_builder_config(trt_builder_t builder);
void trt_destroy_builder_config(trt_builder_config_t config);
void trt_builder_config_set_memory_pool_limit(trt_builder_config_t config,
                                              size_t workspace_bytes);
void trt_builder_config_set_flag(trt_builder_config_t config,
                                 trt_builder_flag_t flag);

/* ---- Build engine ---- */
/* Builds a serialized engine. Returns a host memory object; caller must free. */
trt_host_memory_t trt_builder_build_serialized_network(trt_builder_t builder,
                                                       trt_network_t network,
                                                       trt_builder_config_t config);
const void* trt_host_memory_data(trt_host_memory_t mem);
size_t trt_host_memory_size(trt_host_memory_t mem);
void trt_destroy_host_memory(trt_host_memory_t mem);

/* ---- Runtime ---- */
trt_runtime_t trt_create_runtime(trt_logger_t logger);
void trt_destroy_runtime(trt_runtime_t runtime);

/* ---- Engine ---- */
trt_engine_t trt_deserialize_engine(trt_runtime_t runtime, const void* data,
                                    size_t size);
void trt_destroy_engine(trt_engine_t engine);
int trt_engine_num_io_tensors(trt_engine_t engine);
const char* trt_engine_get_io_tensor_name(trt_engine_t engine, int index);

/* ---- Optimization Profiles (dynamic shapes) ---- */
typedef void* trt_optimization_profile_t;

/* Creates a new optimization profile from a builder. */
trt_optimization_profile_t trt_create_optimization_profile(trt_builder_t builder);

/* Sets min/opt/max dimensions for a named input tensor in the profile.
 * Returns 1 on success, 0 on failure. */
int trt_profile_set_dimensions(trt_optimization_profile_t profile,
                               const char* input_name,
                               int nb_dims,
                               const int32_t* min_dims,
                               const int32_t* opt_dims,
                               const int32_t* max_dims);

/* Adds an optimization profile to the builder config. Returns profile index. */
int trt_config_add_optimization_profile(trt_builder_config_t config,
                                        trt_optimization_profile_t profile);

/* ---- ExecutionContext ---- */
trt_context_t trt_create_execution_context(trt_engine_t engine);
void trt_destroy_execution_context(trt_context_t context);
int trt_context_set_tensor_address(trt_context_t context, const char* name,
                                   void* data);
int trt_context_enqueue_v3(trt_context_t context, void* stream);

/* Sets the input shape for a named tensor on the execution context.
 * Required for dynamic shapes before calling enqueueV3.
 * Returns 1 on success, 0 on failure. */
int trt_context_set_input_shape(trt_context_t context, const char* name,
                                int nb_dims, const int32_t* dims);

/* Sets the active optimization profile on the execution context.
 * Returns 1 on success, 0 on failure. */
int trt_context_set_optimization_profile(trt_context_t context,
                                         int profile_index);

#ifdef __cplusplus
}
#endif

#endif /* TRT_CAPI_H */
