// Minimal NVML via dlopen (no link-time dependency). Used by probe.cu and gemv_bench.cu.
#pragma once
#include <dlfcn.h>
#include <cstdio>

struct NvmlLite {
    typedef int (*fn_init)();
    typedef int (*fn_handle)(unsigned, void**);
    typedef int (*fn_clock)(void*, int, unsigned*);
    typedef int (*fn_reasons)(void*, unsigned long long*);
    typedef int (*fn_power)(void*, unsigned*);
    typedef int (*fn_temp)(void*, int, unsigned*);
    void* lib = nullptr;
    fn_handle handle = nullptr; fn_clock clock = nullptr; fn_reasons reasons = nullptr; fn_power power = nullptr;
    fn_temp temp = nullptr;
    bool ok = false;
    bool init() {
        lib = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_NOW);
        if (!lib) return false;
        fn_init in = (fn_init)dlsym(lib, "nvmlInit_v2");
        handle = (fn_handle)dlsym(lib, "nvmlDeviceGetHandleByIndex_v2");
        clock = (fn_clock)dlsym(lib, "nvmlDeviceGetClockInfo");
        reasons = (fn_reasons)dlsym(lib, "nvmlDeviceGetCurrentClocksEventReasons");
        if (!reasons) reasons = (fn_reasons)dlsym(lib, "nvmlDeviceGetCurrentClocksThrottleReasons");
        power = (fn_power)dlsym(lib, "nvmlDeviceGetPowerUsage");
        temp = (fn_temp)dlsym(lib, "nvmlDeviceGetTemperature");
        ok = in && handle && clock && in() == 0;
        return ok;
    }
    struct Sample { unsigned sm = 0, mem = 0, mw = 0, temp = 0; unsigned long long reasons = 0; };
    Sample sample(unsigned idx) {
        Sample s;
        if (!ok) return s;
        void* h = nullptr;
        if (handle(idx, &h)) return s;
        clock(h, 1 /*SM*/, &s.sm);
        clock(h, 2 /*MEM*/, &s.mem);
        if (power) power(h, &s.mw);
        if (temp) temp(h, 0, &s.temp);
        if (reasons) reasons(h, &s.reasons);
        return s;
    }
};

// 0x1 idle, 0x2 app clocks, 0x4 sw power cap, 0x8 hw slowdown, 0x20 sw thermal, 0x40 hw thermal, 0x80 hw power brake
static inline void nvml_reason_str(unsigned long long r, char* buf, int n) {
    snprintf(buf, n, "%s%s%s%s%s%s%s", r & 1 ? "idle|" : "", r & 2 ? "appclk|" : "", r & 4 ? "swpower|" : "",
             r & 8 ? "hwslow|" : "", r & 0x20 ? "swthermal|" : "", r & 0x40 ? "hwthermal|" : "",
             r & 0x80 ? "powerbrake|" : "");
}
