# t4q build (Kaggle image: nvcc 12.8 at /usr/local/cuda, gcc 11). No CMake.
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_75
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fPIC -Xcompiler -ffp-contract=off -Iinclude
BUILD = build

CU_SRCS = src/loader.cu src/repack.cu src/engine.cu src/kernels/gemv_ref.cu src/kernels/misc.cu src/kernels/gdn.cu src/kernels/attn.cu src/tp_engine.cu src/kernels/tp_kernels.cu
CPP_SRCS = src/gguf.cpp src/quant_cpu.cpp src/api.cpp
OBJS = $(patsubst %.cu,$(BUILD)/%.o,$(CU_SRCS)) $(patsubst %.cpp,$(BUILD)/%.o,$(CPP_SRCS))

all: $(BUILD)/libt4q.so

$(BUILD)/%.o: %.cu src/*.h src/kernels/*.h src/kernels/*.cuh include/t4q.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -Xptxas -v -c $< -o $@ 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)

$(BUILD)/%.o: %.cpp src/*.h include/t4q.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -x cu -c $< -o $@

$(BUILD)/libt4q.so: $(OBJS)
	$(NVCC) -shared -arch=$(ARCH) -o $@ $(OBJS) -lcudart

# oracle: links against the sm_75 libllama from kernel output otdoges/t4-qwen38-baseline
LLAMA_LIB ?= /tmp/llb
$(BUILD)/oracle_dump: tools/oracle_dump.cpp
	@mkdir -p $(BUILD)
	g++ -O2 -std=c++17 -Itools/oracle_include -o $@ $< -L$(LLAMA_LIB) -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_LIB)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
