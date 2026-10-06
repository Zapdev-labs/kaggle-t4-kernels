// Minimal GGUF v3 reader (mmap). Tokenizer arrays are skipped; Python owns tokenization.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

enum GgmlType : uint32_t {
    GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q4_1 = 3, GT_Q5_0 = 6, GT_Q5_1 = 7, GT_Q8_0 = 8, GT_Q2_K = 10,
    GT_Q3_K = 11, GT_Q4_K = 12, GT_Q5_K = 13, GT_Q6_K = 14, GT_IQ4_NL = 20, GT_IQ4_XS = 23, GT_BF16 = 30,
};

struct GgufTensor {
    std::string name;
    uint32_t type = 0;
    int n_dims = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t offset = 0;      // relative to data start
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    size_t row_bytes = 0;     // bytes of one ne0 row
    int64_t nrows() const { return ne[1] * ne[2] * ne[3]; }
};

struct GgufKV {
    uint32_t type = 0;        // gguf value type
    double num = 0;           // numeric scalar
    std::string str;          // string scalar
    std::vector<double> arr;  // numeric array (only when short)
    std::vector<uint64_t> u64;  // exact u64 array entries (only when short and et==10)
    uint64_t arr_n = 0;
};

struct GgufFile {
    std::string path;
    int fd = -1;
    uint8_t* map = nullptr;
    size_t size = 0;
    size_t data_start = 0;
    uint32_t version = 0;
    std::map<std::string, GgufKV> kv;
    std::vector<GgufTensor> tensors;
    std::map<std::string, size_t> index;

    bool open(const std::string& p, std::string& err);
    void close();
    const GgufTensor* find(const std::string& name) const;
    double num(const std::string& key, double def) const;
    const std::vector<uint64_t>& u64_arr(const std::string& key) const;
    ~GgufFile() { close(); }
};

// block geometry: elements per block and bytes per block; returns false for unknown types
bool ggml_block_info(uint32_t type, int& blk_elems, int& blk_bytes);
const char* ggml_type_name(uint32_t type);
