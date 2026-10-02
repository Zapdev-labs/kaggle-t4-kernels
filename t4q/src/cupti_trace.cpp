// Kernel timeline via the CUPTI activity API (dlopen'ed libcupti; works inside CUDA graphs, unlike event nodes).
// trace_begin() enables CONCURRENT_KERNEL records; trace_end() flushes and returns the records sorted per device.
#include "cupti_trace.h"

#include <cxxabi.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if __has_include(<cupti.h>)
#include <cupti.h>
#define T4Q_HAVE_CUPTI 1
#else
#define T4Q_HAVE_CUPTI 0
#endif

namespace trace {

#if T4Q_HAVE_CUPTI
namespace {
std::mutex g_mu;
std::vector<Rec> g_recs;
void* g_lib = nullptr;
decltype(&cuptiActivityEnable) p_enable = nullptr;
decltype(&cuptiActivityDisable) p_disable = nullptr;
decltype(&cuptiActivityRegisterCallbacks) p_reg = nullptr;
decltype(&cuptiActivityFlushAll) p_flush = nullptr;
decltype(&cuptiActivityGetNextRecord) p_next = nullptr;
bool g_registered = false;

std::string short_name(const char* mangled) {
    int st = 0;
    char* d = abi::__cxa_demangle(mangled, nullptr, nullptr, &st);
    std::string s = (st == 0 && d) ? d : mangled;
    free(d);
    for (size_t p; (p = s.find("(anonymous namespace)::")) != std::string::npos;) s.erase(p, 23);
    // drop the parameter list: "void tp::(anonymous namespace)::k_gemv<...>(...)" -> "k_gemv<...>"
    const size_t lt = s.find('<'), lp = s.find('(');
    size_t end = s.size();
    if (lt != std::string::npos && (lp == std::string::npos || lt < lp)) {
        int depth = 0;
        for (size_t i = lt; i < s.size(); i++) {
            if (s[i] == '<') depth++;
            else if (s[i] == '>' && --depth == 0) { end = i + 1; break; }
        }
    } else if (lp != std::string::npos) {
        end = lp;
    }
    s = s.substr(0, end);
    const size_t ns = s.rfind("::", lt == std::string::npos ? std::string::npos : lt);
    if (ns != std::string::npos) s = s.substr(ns + 2);
    if (s.rfind("void ", 0) == 0) s = s.substr(5);
    // compact the template arguments of the GEMV kernels
    const char* subs[][2] = {{"t4q::gemv::", ""}, {"(t4q::gemv::FastFmt)", ""}, {"(tp::ProKind)", ""},
                             {"true", "1"}, {"false", "0"}, {", ", ","}};
    for (auto& sb : subs) {
        size_t p;
        while ((p = s.find(sb[0])) != std::string::npos) s.replace(p, strlen(sb[0]), sb[1]);
    }
    return s;
}

void CUPTIAPI buf_req(uint8_t** buf, size_t* size, size_t* maxrec) {
    *size = 16u << 20;
    *buf = (uint8_t*)aligned_alloc(8, *size);
    *maxrec = 0;
}
void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
    CUpti_Activity* r = nullptr;
    std::vector<Rec> local;
    while (p_next(buf, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            const CUpti_ActivityKernel9* k = (const CUpti_ActivityKernel9*)r;
            Rec x;
            x.dev = (int)k->deviceId;
            x.start = k->start;
            x.end = k->end;
            x.name = k->name ? short_name(k->name) : "?";
            local.push_back(std::move(x));
        }
    }
    free(buf);
    std::lock_guard<std::mutex> lk(g_mu);
    g_recs.insert(g_recs.end(), local.begin(), local.end());
}

bool load(std::string& err) {
    if (g_lib) return true;
    std::vector<std::string> cands;
    if (const char* e = getenv("T4Q_CUPTI")) cands.push_back(e);
    for (const char* c : {"libcupti.so", "libcupti.so.12", "/usr/local/cuda/lib64/libcupti.so",
                          "/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so",
                          "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"})
        cands.push_back(c);
    for (auto& c : cands) {
        g_lib = dlopen(c.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (g_lib) break;
    }
    if (!g_lib) { err = "libcupti not found"; return false; }
    p_enable = (decltype(p_enable))dlsym(g_lib, "cuptiActivityEnable");
    p_disable = (decltype(p_disable))dlsym(g_lib, "cuptiActivityDisable");
    p_reg = (decltype(p_reg))dlsym(g_lib, "cuptiActivityRegisterCallbacks");
    p_flush = (decltype(p_flush))dlsym(g_lib, "cuptiActivityFlushAll");
    p_next = (decltype(p_next))dlsym(g_lib, "cuptiActivityGetNextRecord");
    if (!p_enable || !p_disable || !p_reg || !p_flush || !p_next) { err = "libcupti symbols missing"; g_lib = nullptr; return false; }
    return true;
}
}  // namespace

bool begin(std::string& err) {
    if (!load(err)) return false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_recs.clear();
    }
    if (!g_registered) {
        if (p_reg(buf_req, buf_done) != CUPTI_SUCCESS) { err = "cuptiActivityRegisterCallbacks failed"; return false; }
        g_registered = true;
    }
    const CUptiResult r = p_enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (r != CUPTI_SUCCESS) { err = "cuptiActivityEnable failed: " + std::to_string((int)r); return false; }
    return true;
}

std::vector<Rec> end() {
    p_flush(1);
    p_disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    p_flush(1);
    std::vector<Rec> out;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        out.swap(g_recs);
    }
    std::sort(out.begin(), out.end(), [](const Rec& a, const Rec& b) {
        return a.dev != b.dev ? a.dev < b.dev : a.start < b.start;
    });
    return out;
}
#else
bool begin(std::string& err) { err = "built without cupti.h"; return false; }
std::vector<Rec> end() { return {}; }
#endif

}  // namespace trace
