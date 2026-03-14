NVCC ?= nvcc
# Default sm_75 (Turing T4). Override for other GPUs:
#   make CUDA_ARCH=sm_80   # Ampere A100
#   make CUDA_ARCH=sm_89   # Ada Lovelace L4/L40
#   make CUDA_ARCH=sm_121  # Blackwell GB10 (DGX Spark)
CUDA_ARCH ?= sm_75
NVCC_FLAGS ?= -O3 --use_fast_math -arch=$(CUDA_ARCH)

# Use larger flash attention tile on Blackwell (228KB shared memory per SM).
ifeq ($(CUDA_ARCH),sm_121)
NVCC_FLAGS += -DFLASH_BLOCK_SIZE=64
endif

SRCS = counter.cu dequant_q4k.cu elementwise.cu elementwise_fp16.cu flash_attention.cu fp8_ops.cu fused_add_rmsnorm.cu fused_norm_add.cu fused_qk_norm_rope.cu fused_rope.cu fused_swiglu.cu gemm_int8.cu gemm_int4.cu gemm_q4.cu gemm_q8.cu gemv_q4k.cu offset_memcpy.cu rope_select.cu scaled_softmax.cu sgemv_m1.cu transpose.cu gather.cu rmsnorm.cu argmax.cu
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

# Limit register pressure for kernels that benefit from higher occupancy.
# gemm_q4: 40->32 regs/thread, no spills, occupancy 75%->100% (256-thread blocks).
# transpose: 40->32 regs/thread (transpose_nd), no spills, occupancy 75%->100%.
MAXREG32_SRCS = gemm_q4.cu transpose.cu

# Monitor register pressure for Q4K GEMV after vectorization.
gemv_q4k.o: gemv_q4k.cu
	$(NVCC) $(NVCC_FLAGS) --ptxas-options=-v -c $< -o $@

gemv_q4k.pic.o: gemv_q4k.cu
	$(NVCC) $(NVCC_FLAGS) --ptxas-options=-v --compiler-options -fPIC -c $< -o $@

gemm_q4.o: gemm_q4.cu
	$(NVCC) $(NVCC_FLAGS) --maxrregcount=32 -c $< -o $@

gemm_q4.pic.o: gemm_q4.cu
	$(NVCC) $(NVCC_FLAGS) --maxrregcount=32 --compiler-options -fPIC -c $< -o $@

transpose.o: transpose.cu
	$(NVCC) $(NVCC_FLAGS) --maxrregcount=32 -c $< -o $@

transpose.pic.o: transpose.cu
	$(NVCC) $(NVCC_FLAGS) --maxrregcount=32 --compiler-options -fPIC -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

%.pic.o: %.cu
	$(NVCC) $(NVCC_FLAGS) --compiler-options -fPIC -c $< -o $@

clean:
	rm -f $(OBJS) $(PIC_OBJS) $(LIB) $(SO)
