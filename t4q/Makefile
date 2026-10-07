# t4q build (Kaggle image: nvcc 12.8 at /usr/local/cuda, gcc 11). No CMake.
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_75
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fPIC -Xcompiler -ffp-contract=off -Iinclude
BUILD = build

CU_SRCS = src/loader.cu src/repack.cu src/engine.cu src/kernels/gemv_ref.cu src/kernels/misc.cu src/kernels/gdn.cu src/kernels/attn.cu src/tp_engine.cu src/tp_prefill.cu src/tp_spec.cu src/kernels/tp_kernels.cu src/cf_kernels.cu src/cf_loader.cu src/cf_engine.cu
CPP_SRCS = src/gguf.cpp src/quant_cpu.cpp src/api.cpp src/cupti_trace.cpp
OBJS = $(patsubst %.cu,$(BUILD)/%.o,$(CU_SRCS)) $(patsubst %.cpp,$(BUILD)/%.o,$(CPP_SRCS))

all: $(BUILD)/libt4q.so

$(BUILD)/%.o: %.cu src/*.h src/*.cuh src/kernels/*.h src/kernels/*.cuh include/t4q.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -Xptxas -v -c $< -o $@ 2> $@.ptxas.txt || (cat $@.ptxas.txt; false)

$(BUILD)/%.o: %.cpp src/*.h include/t4q.h
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -x cu -c $< -o $@

$(BUILD)/libt4q.so: $(OBJS)
	$(NVCC) -shared -arch=$(ARCH) -o $@ $(OBJS) -lcudart -ldl

# cf-m1: the CYBER-FROST driver, linked against the full library
$(BUILD)/cf_run: tools/cf_run.cu $(BUILD)/libt4q.so src/cf_model.h
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -Isrc -o $@ $< -L$(BUILD) -lt4q -Xlinker -rpath -Xlinker $(abspath $(BUILD))

# oracle: links against the sm_75 libllama from kernel output t4-qwen38-baseline
LLAMA_LIB ?= /tmp/llb
$(BUILD)/oracle_dump: tools/oracle_dump.cpp
	@mkdir -p $(BUILD)
	g++ -O2 -std=c++17 -Itools/oracle_include -o $@ $< -L$(LLAMA_LIB) -lllama -lggml -lggml-base -Wl,-rpath,$(LLAMA_LIB)

# cf-m3 (r19o): the UVA feasibility probe - mapped-read bandwidth vs the staged path
$(BUILD)/uva_probe: tools/uva_probe.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -Isrc -o $@ $<

# cf-m3 (r19s): the CUDA-graph launch-wall probe - the G1/G2 win measured pre-engine
$(BUILD)/graph_probe: tools/graph_probe.cu
	@mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -Isrc -o $@ $<

# cf-m6 r1 (CF_REQUANT.md section 7, gate 1): the iq1_s host round-trip - the packer's blocks
# decoded by the deq32 host sim must be bit-identical to the packer's own reference decode,
# plus the fp16 all-pattern round-trip + the RNE spot checks + the kmap self-check.
# Host-only (T4Q_HOST_SIM), no CUDA: builds and runs on any host with g++.
HOSTCXX ?= g++
$(BUILD)/test_iq1s: tests/test_iq1s.cpp src/requant.h src/iq1s_table.h src/packed.h src/kernels/deq.cuh
	@mkdir -p $(BUILD)
	$(HOSTCXX) -O2 -std=c++17 -ffp-contract=off -o $@ $<

test_iq1s: $(BUILD)/test_iq1s
	$(BUILD)/test_iq1s

# cf-m6 r3 (CF_REQUANT.md section 5): the iq1_s expert-slab packer + its local gate.
# Host-only (g++ -fopenmp): the pack reads the BF16 expert tensors, the gate runs the
# full synthetic round (generate -> pack -> read back -> deq32-vs-ref on every row).
$(BUILD)/cf_requant_pack: tools/cf_requant_pack.cpp src/requant.h src/iq1s_table.h src/packed.h src/kernels/deq.cuh
	@mkdir -p $(BUILD)
	$(HOSTCXX) -O3 -fopenmp -std=c++17 -ffp-contract=off -o $@ $<

pack_gate: $(BUILD)/cf_requant_pack
	$(BUILD)/cf_requant_pack --synthetic --ne 4 --out $(BUILD)/pack_gate_slab.bin

clean:
	rm -rf $(BUILD)

.PHONY: all clean test_iq1s pack_gate
