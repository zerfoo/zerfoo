/* C shim wrapping TensorRT C++ API for CGo access. */

#include "trt_capi.h"
#include <NvInfer.h>
#include <cstdio>
#include <cstring>

using namespace nvinfer1;

/* ---- Simple logger that prints to stderr ---- */

class SimpleLogger : public ILogger {
public:
    Severity minSeverity;
    explicit SimpleLogger(Severity s) : minSeverity(s) {}
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= minSeverity) {
            const char* prefix = "";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: prefix = "[TRT INTERNAL_ERROR] "; break;
                case Severity::kERROR:          prefix = "[TRT ERROR] "; break;
                case Severity::kWARNING:        prefix = "[TRT WARNING] "; break;
                case Severity::kINFO:           prefix = "[TRT INFO] "; break;
                case Severity::kVERBOSE:        prefix = "[TRT VERBOSE] "; break;
            }
            fprintf(stderr, "%s%s\n", prefix, msg);
        }
    }
};

static ILogger::Severity toSeverity(trt_severity_t s) {
    switch (s) {
        case TRT_SEVERITY_INTERNAL_ERROR: return ILogger::Severity::kINTERNAL_ERROR;
        case TRT_SEVERITY_ERROR:          return ILogger::Severity::kERROR;
        case TRT_SEVERITY_WARNING:        return ILogger::Severity::kWARNING;
        case TRT_SEVERITY_INFO:           return ILogger::Severity::kINFO;
        case TRT_SEVERITY_VERBOSE:        return ILogger::Severity::kVERBOSE;
        default:                          return ILogger::Severity::kWARNING;
    }
}

static DataType toDataType(trt_data_type_t dt) {
    switch (dt) {
        case TRT_FLOAT32: return DataType::kFLOAT;
        case TRT_FLOAT16: return DataType::kHALF;
        case TRT_INT8:    return DataType::kINT8;
        case TRT_INT32:   return DataType::kINT32;
        default:          return DataType::kFLOAT;
    }
}

static ActivationType toActivationType(trt_activation_type_t a) {
    switch (a) {
        case TRT_ACTIVATION_RELU:    return ActivationType::kRELU;
        case TRT_ACTIVATION_SIGMOID: return ActivationType::kSIGMOID;
        case TRT_ACTIVATION_TANH:    return ActivationType::kTANH;
        default:                     return ActivationType::kRELU;
    }
}

static ElementWiseOperation toElementWiseOp(trt_elementwise_op_t op) {
    switch (op) {
        case TRT_ELEMENTWISE_SUM:  return ElementWiseOperation::kSUM;
        case TRT_ELEMENTWISE_PROD: return ElementWiseOperation::kPROD;
        case TRT_ELEMENTWISE_MAX:  return ElementWiseOperation::kMAX;
        case TRT_ELEMENTWISE_MIN:  return ElementWiseOperation::kMIN;
        case TRT_ELEMENTWISE_SUB:  return ElementWiseOperation::kSUB;
        case TRT_ELEMENTWISE_DIV:  return ElementWiseOperation::kDIV;
        default:                   return ElementWiseOperation::kSUM;
    }
}

static MatrixOperation toMatrixOp(trt_matrix_op_t op) {
    switch (op) {
        case TRT_MATMUL_NONE:      return MatrixOperation::kNONE;
        case TRT_MATMUL_TRANSPOSE: return MatrixOperation::kTRANSPOSE;
        default:                   return MatrixOperation::kNONE;
    }
}

static ReduceOperation toReduceOp(trt_reduce_op_t op) {
    switch (op) {
        case TRT_REDUCE_SUM:  return ReduceOperation::kSUM;
        case TRT_REDUCE_PROD: return ReduceOperation::kPROD;
        case TRT_REDUCE_MAX:  return ReduceOperation::kMAX;
        case TRT_REDUCE_MIN:  return ReduceOperation::kMIN;
        case TRT_REDUCE_AVG:  return ReduceOperation::kAVG;
        default:              return ReduceOperation::kSUM;
    }
}

static Dims toDims(int nb, const int32_t* d) {
    Dims dims;
    dims.nbDims = nb;
    for (int i = 0; i < nb && i < Dims::MAX_DIMS; i++) {
        dims.d[i] = d[i];
    }
    return dims;
}

static Permutation toPermutation(int nb, const int32_t* perm) {
    Permutation p;
    for (int i = 0; i < nb && i < Dims::MAX_DIMS; i++) {
        p.order[i] = perm[i];
    }
    return p;
}

/* ---- Logger ---- */

trt_logger_t trt_create_logger(trt_severity_t min_severity) {
    return new SimpleLogger(toSeverity(min_severity));
}

void trt_destroy_logger(trt_logger_t logger) {
    delete static_cast<SimpleLogger*>(logger);
}

/* ---- Builder ---- */

trt_builder_t trt_create_builder(trt_logger_t logger) {
    auto* l = static_cast<ILogger*>(static_cast<SimpleLogger*>(logger));
    IBuilder* b = createInferBuilder(*l);
    return static_cast<void*>(b);
}

void trt_destroy_builder(trt_builder_t builder) {
    auto* b = static_cast<IBuilder*>(builder);
    if (b) delete b;
}

/* ---- Network ---- */

trt_network_t trt_create_network(trt_builder_t builder) {
    auto* b = static_cast<IBuilder*>(builder);
    /* TRT 10+: explicit batch is the only mode (kEXPLICIT_BATCH=0). */
    INetworkDefinition* n = b->createNetworkV2(0);
    return static_cast<void*>(n);
}

void trt_destroy_network(trt_network_t network) {
    auto* n = static_cast<INetworkDefinition*>(network);
    if (n) delete n;
}

int trt_network_num_inputs(trt_network_t network) {
    return static_cast<INetworkDefinition*>(network)->getNbInputs();
}

int trt_network_num_outputs(trt_network_t network) {
    return static_cast<INetworkDefinition*>(network)->getNbOutputs();
}

int trt_network_num_layers(trt_network_t network) {
    return static_cast<INetworkDefinition*>(network)->getNbLayers();
}

/* ---- Network: add layers ---- */

trt_tensor_t trt_network_add_input(trt_network_t network, const char* name,
                                   trt_data_type_t dtype, int nb_dims,
                                   const int32_t* dims) {
    auto* n = static_cast<INetworkDefinition*>(network);
    ITensor* t = n->addInput(name, toDataType(dtype), toDims(nb_dims, dims));
    return static_cast<void*>(t);
}

void trt_network_mark_output(trt_network_t network, trt_tensor_t tensor) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(tensor);
    n->markOutput(*t);
}

trt_layer_t trt_network_add_activation(trt_network_t network,
                                       trt_tensor_t input,
                                       trt_activation_type_t type) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(input);
    IActivationLayer* layer = n->addActivation(*t, toActivationType(type));
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_elementwise(trt_network_t network,
                                        trt_tensor_t input1,
                                        trt_tensor_t input2,
                                        trt_elementwise_op_t op) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t1 = static_cast<ITensor*>(input1);
    auto* t2 = static_cast<ITensor*>(input2);
    IElementWiseLayer* layer = n->addElementWise(*t1, *t2, toElementWiseOp(op));
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_matrix_multiply(trt_network_t network,
                                            trt_tensor_t input0,
                                            trt_matrix_op_t op0,
                                            trt_tensor_t input1,
                                            trt_matrix_op_t op1) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t0 = static_cast<ITensor*>(input0);
    auto* t1 = static_cast<ITensor*>(input1);
    IMatrixMultiplyLayer* layer = n->addMatrixMultiply(*t0, toMatrixOp(op0),
                                                       *t1, toMatrixOp(op1));
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_softmax(trt_network_t network,
                                    trt_tensor_t input, int axis) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(input);
    ISoftMaxLayer* layer = n->addSoftMax(*t);
    if (layer) layer->setAxes(1U << axis);
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_reduce(trt_network_t network, trt_tensor_t input,
                                   trt_reduce_op_t op, uint32_t reduce_axes,
                                   int keep_dims) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(input);
    IReduceLayer* layer = n->addReduce(*t, toReduceOp(op), reduce_axes,
                                       keep_dims != 0);
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_constant(trt_network_t network, int nb_dims,
                                     const int32_t* dims,
                                     trt_data_type_t dtype,
                                     const void* weights, int64_t count) {
    auto* n = static_cast<INetworkDefinition*>(network);
    Weights w;
    w.type = toDataType(dtype);
    w.values = weights;
    w.count = count;
    IConstantLayer* layer = n->addConstant(toDims(nb_dims, dims), w);
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_shuffle(trt_network_t network,
                                    trt_tensor_t input) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(input);
    IShuffleLayer* layer = n->addShuffle(*t);
    return static_cast<void*>(layer);
}

trt_layer_t trt_network_add_convolution_nd(trt_network_t network,
                                           trt_tensor_t input,
                                           int nb_output_maps,
                                           int kernel_nb_dims,
                                           const int32_t* kernel_size,
                                           const void* kernel_weights,
                                           int64_t kernel_count,
                                           const void* bias_weights,
                                           int64_t bias_count) {
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* t = static_cast<ITensor*>(input);
    Weights kw;
    kw.type = DataType::kFLOAT;
    kw.values = kernel_weights;
    kw.count = kernel_count;
    Weights bw;
    bw.type = DataType::kFLOAT;
    bw.values = bias_weights;
    bw.count = bias_count;
    IConvolutionLayer* layer = n->addConvolutionNd(
        *t, nb_output_maps, toDims(kernel_nb_dims, kernel_size), kw, bw);
    return static_cast<void*>(layer);
}

/* ---- Layer helpers ---- */

trt_tensor_t trt_layer_get_output(trt_layer_t layer, int index) {
    auto* l = static_cast<ILayer*>(layer);
    ITensor* t = l->getOutput(index);
    return static_cast<void*>(t);
}

void trt_layer_set_name(trt_layer_t layer, const char* name) {
    auto* l = static_cast<ILayer*>(layer);
    l->setName(name);
}

void trt_shuffle_set_reshape_dims(trt_layer_t shuffle, int nb_dims,
                                  const int32_t* dims) {
    auto* s = static_cast<IShuffleLayer*>(shuffle);
    s->setReshapeDimensions(toDims(nb_dims, dims));
}

void trt_shuffle_set_first_transpose(trt_layer_t shuffle, int nb_dims,
                                     const int32_t* perm) {
    auto* s = static_cast<IShuffleLayer*>(shuffle);
    s->setFirstTranspose(toPermutation(nb_dims, perm));
}

/* ---- BuilderConfig ---- */

trt_builder_config_t trt_create_builder_config(trt_builder_t builder) {
    auto* b = static_cast<IBuilder*>(builder);
    IBuilderConfig* cfg = b->createBuilderConfig();
    return static_cast<void*>(cfg);
}

void trt_destroy_builder_config(trt_builder_config_t config) {
    auto* cfg = static_cast<IBuilderConfig*>(config);
    if (cfg) delete cfg;
}

void trt_builder_config_set_memory_pool_limit(trt_builder_config_t config,
                                              size_t workspace_bytes) {
    auto* cfg = static_cast<IBuilderConfig*>(config);
    cfg->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace_bytes);
}

void trt_builder_config_set_flag(trt_builder_config_t config,
                                 trt_builder_flag_t flag) {
    auto* cfg = static_cast<IBuilderConfig*>(config);
    BuilderFlag bf;
    switch (flag) {
        case TRT_FLAG_FP16: bf = BuilderFlag::kFP16; break;
        case TRT_FLAG_INT8: bf = BuilderFlag::kINT8; break;
        default: return;
    }
    cfg->setFlag(bf);
}

/* ---- Build engine ---- */

trt_host_memory_t trt_builder_build_serialized_network(trt_builder_t builder,
                                                       trt_network_t network,
                                                       trt_builder_config_t config) {
    auto* b = static_cast<IBuilder*>(builder);
    auto* n = static_cast<INetworkDefinition*>(network);
    auto* cfg = static_cast<IBuilderConfig*>(config);
    IHostMemory* mem = b->buildSerializedNetwork(*n, *cfg);
    return static_cast<void*>(mem);
}

const void* trt_host_memory_data(trt_host_memory_t mem) {
    auto* m = static_cast<IHostMemory*>(mem);
    return m ? m->data() : nullptr;
}

size_t trt_host_memory_size(trt_host_memory_t mem) {
    auto* m = static_cast<IHostMemory*>(mem);
    return m ? m->size() : 0;
}

void trt_destroy_host_memory(trt_host_memory_t mem) {
    auto* m = static_cast<IHostMemory*>(mem);
    if (m) delete m;
}

/* ---- Runtime ---- */

trt_runtime_t trt_create_runtime(trt_logger_t logger) {
    auto* l = static_cast<ILogger*>(static_cast<SimpleLogger*>(logger));
    IRuntime* rt = createInferRuntime(*l);
    return static_cast<void*>(rt);
}

void trt_destroy_runtime(trt_runtime_t runtime) {
    auto* rt = static_cast<IRuntime*>(runtime);
    if (rt) delete rt;
}

/* ---- Engine ---- */

trt_engine_t trt_deserialize_engine(trt_runtime_t runtime, const void* data,
                                    size_t size) {
    auto* rt = static_cast<IRuntime*>(runtime);
    ICudaEngine* engine = rt->deserializeCudaEngine(data, size);
    return static_cast<void*>(engine);
}

void trt_destroy_engine(trt_engine_t engine) {
    auto* e = static_cast<ICudaEngine*>(engine);
    if (e) delete e;
}

int trt_engine_num_io_tensors(trt_engine_t engine) {
    auto* e = static_cast<ICudaEngine*>(engine);
    return e->getNbIOTensors();
}

const char* trt_engine_get_io_tensor_name(trt_engine_t engine, int index) {
    auto* e = static_cast<ICudaEngine*>(engine);
    return e->getIOTensorName(index);
}

/* ---- ExecutionContext ---- */

trt_context_t trt_create_execution_context(trt_engine_t engine) {
    auto* e = static_cast<ICudaEngine*>(engine);
    IExecutionContext* ctx = e->createExecutionContext();
    return static_cast<void*>(ctx);
}

void trt_destroy_execution_context(trt_context_t context) {
    auto* ctx = static_cast<IExecutionContext*>(context);
    if (ctx) delete ctx;
}

int trt_context_set_tensor_address(trt_context_t context, const char* name,
                                   void* data) {
    auto* ctx = static_cast<IExecutionContext*>(context);
    return ctx->setTensorAddress(name, data) ? 1 : 0;
}

int trt_context_enqueue_v3(trt_context_t context, void* stream) {
    auto* ctx = static_cast<IExecutionContext*>(context);
    return ctx->enqueueV3(static_cast<cudaStream_t>(stream)) ? 1 : 0;
}

/* ---- Optimization Profiles (dynamic shapes) ---- */

trt_optimization_profile_t trt_create_optimization_profile(trt_builder_t builder) {
    auto* b = static_cast<IBuilder*>(builder);
    IOptimizationProfile* profile = b->createOptimizationProfile();
    return static_cast<void*>(profile);
}

int trt_profile_set_dimensions(trt_optimization_profile_t profile,
                               const char* input_name,
                               int nb_dims,
                               const int32_t* min_dims,
                               const int32_t* opt_dims,
                               const int32_t* max_dims) {
    auto* p = static_cast<IOptimizationProfile*>(profile);
    bool ok = p->setDimensions(input_name, OptProfileSelector::kMIN,
                               toDims(nb_dims, min_dims));
    ok = ok && p->setDimensions(input_name, OptProfileSelector::kOPT,
                                toDims(nb_dims, opt_dims));
    ok = ok && p->setDimensions(input_name, OptProfileSelector::kMAX,
                                toDims(nb_dims, max_dims));
    return ok ? 1 : 0;
}

int trt_config_add_optimization_profile(trt_builder_config_t config,
                                        trt_optimization_profile_t profile) {
    auto* cfg = static_cast<IBuilderConfig*>(config);
    auto* p = static_cast<IOptimizationProfile*>(profile);
    return cfg->addOptimizationProfile(p);
}

int trt_context_set_input_shape(trt_context_t context, const char* name,
                                int nb_dims, const int32_t* dims) {
    auto* ctx = static_cast<IExecutionContext*>(context);
    return ctx->setInputShape(name, toDims(nb_dims, dims)) ? 1 : 0;
}

int trt_context_set_optimization_profile(trt_context_t context,
                                         int profile_index) {
    auto* ctx = static_cast<IExecutionContext*>(context);
    return ctx->setOptimizationProfileAsync(profile_index, nullptr) ? 1 : 0;
}
