NVCC ?= nvcc
# Default sm_75 (Turing T4). Override for other GPUs:
#   make CUDA_ARCH=sm_80   # Ampere A100
#   make CUDA_ARCH=sm_89   # Ada Lovelace L4/L40
#   make CUDA_ARCH=sm_121  # Blackwell GB10 (DGX Spark)
CUDA_ARCH ?= sm_75
NVCC_FLAGS ?= -O3 --use_fast_math -arch=$(CUDA_ARCH)

SRCS = elementwise.cu flash_attention.cu fused_add_rmsnorm.cu fused_norm_add.cu fused_qk_norm_rope.cu fused_rope.cu fused_swiglu.cu gemm_int8.cu gemm_int4.cu gemm_q4.cu gemm_q8.cu scaled_softmax.cu transpose.cu gather.cu rmsnorm.cu argmax.cu
OBJS = $(SRCS:.cu=.o)
PIC_OBJS = $(SRCS:.cu=.pic.o)
LIB  = libkernels.a
SO   = libkernels.so

.PHONY: all shared clean

all: $(LIB)

shared: $(SO)

$(LIB): $(OBJS)
	ar rcs $@ $^

$(SO): $(PIC_OBJS)
	$(NVCC) -shared -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.pic.o: %.cu
	$(NVCC) $(NVCC_FLAGS) --compiler-options -fPIC -c $< -o $@

clean:
	rm -f $(OBJS) $(PIC_OBJS) $(LIB) $(SO)
