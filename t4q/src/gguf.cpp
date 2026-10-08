#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

bool ggml_block_info(uint32_t type, int& be, int& bb) {
    switch (type) {
        case GT_F32: be = 1; bb = 4; return true;
        case GT_F16: be = 1; bb = 2; return true;
        case GT_BF16: be = 1; bb = 2; return true;
        case GT_Q4_0: be = 32; bb = 18; return true;
        case GT_Q4_1: be = 32; bb = 20; return true;
        case GT_Q5_1: be = 32; bb = 24; return true;
        case GT_Q8_0: be = 32; bb = 34; return true;
        case GT_IQ4_NL: be = 32; bb = 18; return true;
        case GT_Q2_K: be = 256; bb = 84; return true;
        case GT_Q3_K: be = 256; bb = 110; return true;
        case GT_Q4_K: be = 256; bb = 144; return true;
        case GT_Q5_K: be = 256; bb = 176; return true;
        case GT_Q6_K: be = 256; bb = 210; return true;
        case GT_IQ4_XS: be = 256; bb = 136; return true;
        default: return false;
    }
}

const char* ggml_type_name(uint32_t t) {
    switch (t) {
        case GT_F32: return "F32"; case GT_F16: return "F16"; case GT_BF16: return "BF16";
        case GT_Q4_0: return "Q4_0"; case GT_Q4_1: return "Q4_1"; case GT_Q5_0: return "Q5_0";
        case GT_Q5_1: return "Q5_1"; case GT_Q8_0: return "Q8_0"; case GT_IQ4_NL: return "IQ4_NL"; case GT_Q2_K: return "Q2_K"; case GT_Q3_K: return "Q3_K";
        case GT_Q4_K: return "Q4_K"; case GT_Q5_K: return "Q5_K"; case GT_Q6_K: return "Q6_K";
        case GT_IQ4_XS: return "IQ4_XS";
        default: return "?";
    }
}

namespace {
struct Rd {
    const uint8_t* p; const uint8_t* end; bool ok = true;  // invariant: p <= end
    template <class T> T get() {
        T v{};
        if ((uint64_t)(end - p) < sizeof(T)) { ok = false; return v; }
        memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    // bounds-checked skip of n file bytes (n is file-controlled: no p + n wraparound)
    bool skip(uint64_t n) {
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return false; }
        p += n; return true;
    }
    bool skip_str() { return skip(get<uint64_t>()); }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (!skip(n)) return {};
        return std::string((const char*)p - n, (size_t)n);
    }
};
size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}
double read_num(Rd& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>(); case 1: return r.get<int8_t>();
        case 2: return r.get<uint16_t>(); case 3: return r.get<int16_t>();
        case 4: return r.get<uint32_t>(); case 5: return r.get<int32_t>();
        case 6: return r.get<float>(); case 7: return r.get<uint8_t>();
        case 10: return (double)r.get<uint64_t>(); case 11: return (double)r.get<int64_t>();
        case 12: return r.get<double>();
        default: r.ok = false; return 0;
    }
}
}  // namespace

bool GgufFile::open(const std::string& p, std::string& err) {
    path = p;
    close();  // tolerate a second open() on the same object without leaking the first map
    kv.clear(); tensors.clear(); index.clear();
    fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) { err = "open failed: " + p; return false; }
    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size <= 0) { err = "stat failed/empty: " + p; return false; }
    size = (size_t)st.st_size;
    map = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { map = nullptr; err = "mmap failed"; return false; }
    Rd r{map, map + size};
    uint32_t magic = r.get<uint32_t>();
    if (magic != 0x46554747u) { err = "bad magic"; return false; }
    version = r.get<uint32_t>();
    uint64_t n_t = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv && r.ok; i++) {
        std::string key = r.str();
        GgufKV v;
        v.type = r.get<uint32_t>();
        if (v.type == 8) v.str = r.str();
        else if (v.type == 9) {
            uint32_t et = r.get<uint32_t>();
            uint64_t n = r.get<uint64_t>();
            v.arr_n = n;
            if (et == 8) { for (uint64_t j = 0; j < n && r.ok; j++) r.skip_str(); }
            else {
                size_t es = scalar_size(et);
                if (!es) { err = "nested array unsupported"; return false; }
                if (n <= 64) {
                    for (uint64_t j = 0; j < n; j++) {
                        if (et == 10) v.u64.push_back(r.get<uint64_t>());  // exact, cf PLE hash needs u64
                        else v.arr.push_back(read_num(r, et));
                    }
                // skip: division form - es * n itself can overflow u64
                } else if (n > (uint64_t)(r.end - r.p) / es) { r.ok = false; break;
                } else r.p += es * n;
            }
        } else v.num = read_num(r, v.type);
        kv[key] = v;
    }
    const double av = num("general.alignment", 32);  // 0 would SIGFPE below; clamp bad values
    if (!(av >= 1.0) || av > (double)size) { err = "bad general.alignment"; return false; }
    const size_t align = (size_t)av;
    for (uint64_t i = 0; i < n_t && r.ok; i++) {
        GgufTensor t;
        t.name = r.str();
        t.n_dims = (int)r.get<uint32_t>();
        if (t.n_dims < 0 || t.n_dims > 4) { err = "bad n_dims: " + t.name; return false; }
        for (int d = 0; d < t.n_dims && r.ok; d++) t.ne[d] = (int64_t)r.get<uint64_t>();
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        tensors.push_back(t);
    }
    if (!r.ok) { err = "truncated header"; return false; }
    size_t hdr = r.p - map;
    data_start = (hdr + align - 1) / align * align;
    for (size_t i = 0; i < tensors.size(); i++) {
        GgufTensor& t = tensors[i];
        int be, bb;
        if (!ggml_block_info(t.type, be, bb)) { err = "unknown type for " + t.name; return false; }
        for (int d = 0; d < 4; d++)
            if (t.ne[d] < 0) { err = "negative dim in " + t.name; return false; }
        // checked multiply: file-controlled dims must not wrap the byte counts
        uint64_t rb = (uint64_t)t.ne[0] / (uint64_t)be;
        if (rb > UINT64_MAX / (uint64_t)bb) { err = "tensor size overflow: " + t.name; return false; }
        rb *= (uint64_t)bb;
        uint64_t nr = 1;
        for (int d = 1; d < 4; d++) {
            if ((uint64_t)t.ne[d] > UINT64_MAX / nr) { err = "tensor size overflow: " + t.name; return false; }
            nr *= (uint64_t)t.ne[d];
        }
        uint64_t nb = 0;
        if (nr) {
            if (rb > UINT64_MAX / nr) { err = "tensor size overflow: " + t.name; return false; }
            nb = rb * nr;
        }
        t.row_bytes = (size_t)rb;
        t.nbytes = (size_t)nb;
        // offset-sum overflow can mask an out-of-bounds tensor: check each addend
        if (data_start > size || t.offset > size - data_start || nb > size - data_start - t.offset) {
            err = "tensor past EOF: " + t.name; return false;
        }
        t.data = map + data_start + t.offset;
        index[t.name] = i;
    }
    return true;
}

void GgufFile::close() {
    if (map) munmap(map, size);
    if (fd >= 0) ::close(fd);
    map = nullptr; fd = -1;
}

const GgufTensor* GgufFile::find(const std::string& n) const {
    auto it = index.find(n);
    return it == index.end() ? nullptr : &tensors[it->second];
}

double GgufFile::num(const std::string& key, double def) const {
    auto it = kv.find(key);
    if (it == kv.end() || it->second.type == 8 || it->second.type == 9) return def;
    return it->second.num;
}

const std::vector<uint64_t>& GgufFile::u64_arr(const std::string& key) const {
    static const std::vector<uint64_t> empty;
    auto it = kv.find(key);
    if (it == kv.end()) return empty;
    return it->second.u64;
}
