NVCC ?= nvcc
# Default sm_75 (Turing T4). Override for other GPUs:
#   make CUDA_ARCH=sm_80   # Ampere A100
#   make CUDA_ARCH=sm_89   # Ada Lovelace L4/L40
#   make CUDA_ARCH=sm_121  # Blackwell GB10 (DGX Spark)
CUDA_ARCH ?= sm_75
NVCC_FLAGS ?= -O2 -arch=$(CUDA_ARCH)

SRCS = elementwise.cu flash_attention.cu gemm_int8.cu gemm_int4.cu gemm_q4.cu transpose.cu gather.cu rmsnorm.cu
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
