// CUPTI kernel-activity tracer (graph-mode kernel timelines). See cupti_trace.cpp.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace trace {
struct Rec {
    int dev;
    uint64_t start, end;  // ns, common timebase across devices
    std::string name;     // demangled, compacted
};
bool begin(std::string& err);  // false (with err) when CUPTI is unavailable
std::vector<Rec> end();        // flush, disable; records sorted by (device, start)
}  // namespace trace
