NVCC ?= nvcc
CUDA_ARCH ?= sm_75
NVCC_FLAGS ?= -O2 -arch=$(CUDA_ARCH)

SRCS = elementwise.cu
OBJS = $(SRCS:.cu=.o)
LIB  = libkernels.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
