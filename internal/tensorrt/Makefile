CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
MULTIARCH := $(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo x86_64-linux-gnu)
CXXFLAGS ?= -O2 -std=c++17 -I/usr/include/$(MULTIARCH) -I$(CUDA_HOME)/include

SRCS = cshim/trt_capi.cpp
OBJS = cshim/trt_capi.o
LIB  = libtrt_capi.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

cshim/trt_capi.o: cshim/trt_capi.cpp cshim/trt_capi.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
