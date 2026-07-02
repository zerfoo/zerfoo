HIPCC ?= hipcc
HIP_ARCH ?= gfx906
HIPCC_FLAGS ?= -O2 --offload-arch=$(HIP_ARCH)

SRCS = elementwise.hip flash_attention.hip
OBJS = $(SRCS:.hip=.o)
LIB  = libhipkernels.a

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.hip
	$(HIPCC) $(HIPCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
