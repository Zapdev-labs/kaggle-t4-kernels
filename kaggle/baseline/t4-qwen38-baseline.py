"""Baseline: upstream llama.cpp master on Kaggle 2x T4 for Qwen3.8-27B (qwen35 arch).

Steps: env probe -> CUDA microbench (bw / P2P / latency) -> build llama.cpp (sm_75) while models download ->
llama-bench (sm layer/row/tensor, fa on) for UD-Q4_K_XL and Q4_0 -> llama-server decode on a coding prompt,
with and without the MTP draft head (--spec-type draft-mtp -md mtp.gguf). Prints RESULT lines and a final
RESULTS block; everything reusable is copied to /kaggle/working.
No secrets are used or needed (all downloads are public).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

T0 = time.time()
DEADLINE = 2.35 * 3600          # stop optional work after this
OUT = Path("/kaggle/working")
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
W = Path("/tmp/bl")
W.mkdir(parents=True, exist_ok=True)
MD = W / "models"
MD.mkdir(exist_ok=True)
REPO = "unsloth/Qwen3.8-27B-GGUF"
MODELS = {"UD-Q4_K_XL": "Qwen3.8-27B-UD-Q4_K_XL.gguf", "Q4_0": "Qwen3.8-27B-Q4_0.gguf"}
MTP = "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"
RESULTS = {}
BW_CU = r'''// T4 bandwidth / P2P / latency microbench. nvcc -O3 -arch=sm_75 bw.cu -o bw
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("ERR %s: %s (line %d)\n", #x, cudaGetErrorString(e), __LINE__); } } while (0)

__global__ void read_kernel(const float4 *__restrict__ p, size_t n, float *out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;  // defeat DCE
}
__global__ void read_kernel_peer(const float4 *__restrict__ p, size_t n, float *out) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) out[0] = acc;
}
__global__ void copy_kernel(const float4 *__restrict__ a, float4 *__restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}
__global__ void empty_kernel() {}

static float ms_since(cudaEvent_t a, cudaEvent_t b) { float m; cudaEventSynchronize(b); cudaEventElapsedTime(&m, a, b); return m; }

int main() {
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("RES ndev=%d driver_api=%d runtime=%d\n", ndev, drv, rt);
    for (int d = 0; d < ndev; d++) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, d);
        int clk = 0, memclk = 0, busw = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, d);
        cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, d);
        cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, d);
        printf("RES dev%d name=%s sm=%d.%d SMs=%d clk_khz=%d memclk_khz=%d buswidth=%d theo_GBps=%.1f l2=%d smem_per_block_optin=%zu regs_per_sm=%d pci=%02x:%02x\n",
               d, p.name, p.major, p.minor, p.multiProcessorCount, clk, memclk, busw,
               2.0 * memclk * 1e3 * (busw / 8) / 1e9, p.l2CacheSize, p.sharedMemPerBlockOptin, p.regsPerMultiprocessor, p.pciBusID, p.pciDeviceID);
    }
    if (ndev >= 2) {
        int a = 0, b = 0; CK(cudaDeviceCanAccessPeer(&a, 0, 1)); CK(cudaDeviceCanAccessPeer(&b, 1, 0));
        printf("RES can_access_peer_0_1=%d can_access_peer_1_0=%d\n", a, b);
        int perf = -1, atom = -1, sup = -1;
        cudaDeviceGetP2PAttribute(&sup, cudaDevP2PAttrAccessSupported, 0, 1);
        cudaDeviceGetP2PAttribute(&perf, cudaDevP2PAttrPerformanceRank, 0, 1);
        cudaDeviceGetP2PAttribute(&atom, cudaDevP2PAttrNativeAtomicSupported, 0, 1);
        printf("RES p2p_attr access_supported=%d perf_rank=%d native_atomic=%d\n", sup, perf, atom);
    }

    // ---- single GPU bandwidth ----
    const size_t BYTES = (size_t)1 << 30;  // 1 GiB
    CK(cudaSetDevice(0));
    float4 *A, *B; float *out;
    CK(cudaMalloc(&A, BYTES)); CK(cudaMalloc(&B, BYTES)); CK(cudaMalloc(&out, 4));
    CK(cudaMemset(A, 0, BYTES)); CK(cudaMemset(B, 0, BYTES));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int IT = 20;
    CK(cudaMemcpy(B, A, BYTES, cudaMemcpyDeviceToDevice));
    cudaEventRecord(e0);
    for (int i = 0; i < IT; i++) cudaMemcpy(B, A, BYTES, cudaMemcpyDeviceToDevice);
    cudaEventRecord(e1);
    float m = ms_since(e0, e1);
    printf("RES memcpy_d2d_GBps=%.1f (counting read+write, 1GiB x%d)\n", 2.0 * BYTES * IT / (m * 1e-3) / 1e9, IT);

    size_t n4 = BYTES / 16;
    int sms = 40; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int best_grid = 0; double best = 0;
    for (int bpsm = 2; bpsm <= 32; bpsm *= 2) for (int th = 256; th <= 1024; th *= 2) {
        int grid = sms * bpsm;
        read_kernel<<<grid, th>>>(A, n4, out);
        cudaEventRecord(e0);
        for (int i = 0; i < IT; i++) read_kernel<<<grid, th>>>(A, n4, out);
        cudaEventRecord(e1);
        m = ms_since(e0, e1);
        double gbps = (double)BYTES * IT / (m * 1e-3) / 1e9;
        printf("RES read_kernel grid=%d threads=%d GBps=%.1f\n", grid, th, gbps);
        if (gbps > best) { best = gbps; best_grid = grid * 10000 + th; }
    }
    printf("RES read_kernel_best_GBps=%.1f cfg=%d\n", best, best_grid);
    // smaller working sets typical of one weight matrix (e.g. 5120x17408 q4 ~ 45MB)
    for (size_t sz : {(size_t)8 << 20, (size_t)45 << 20, (size_t)128 << 20}) {
        size_t n = sz / 16; int grid = sms * 8;
        read_kernel<<<grid, 512>>>(A, n, out);
        cudaEventRecord(e0);
        for (int i = 0; i < 200; i++) read_kernel<<<grid, 512>>>(A, n, out);
        cudaEventRecord(e1);
        m = ms_since(e0, e1);
        printf("RES read_kernel_size_MB=%zu GBps=%.1f us_per=%.2f\n", sz >> 20, (double)sz * 200 / (m * 1e-3) / 1e9, m * 1e3 / 200);
    }
    cudaEventRecord(e0);
    for (int i = 0; i < IT; i++) copy_kernel<<<sms * 8, 512>>>(A, B, n4);
    cudaEventRecord(e1);
    m = ms_since(e0, e1);
    printf("RES copy_kernel_GBps=%.1f (read+write)\n", 2.0 * BYTES * IT / (m * 1e-3) / 1e9);

    // launch latency
    for (int i = 0; i < 100; i++) empty_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < 10000; i++) empty_kernel<<<1, 32>>>();
    cudaEventRecord(e1);
    m = ms_since(e0, e1);
    printf("RES empty_kernel_back_to_back_us=%.2f\n", m * 1e3 / 10000);
    {
        cudaStream_t s; cudaStreamCreate(&s);
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < 1000; i++) empty_kernel<<<1, 32, 0, s>>>();
        cudaStreamEndCapture(s, &g);
        CK(cudaGraphInstantiate(&ge, g, 0));
        cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
        cudaEventRecord(e0, s);
        for (int r = 0; r < 10; r++) cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        m = ms_since(e0, e1);
        printf("RES empty_kernel_in_graph_us=%.2f\n", m * 1e3 / 10000);
    }
    // launch+sync roundtrip (host)
    {
        float tot = 0; cudaEvent_t h0, h1; cudaEventCreate(&h0); cudaEventCreate(&h1);
        auto t0 = clock();
        (void)t0;
        cudaEventRecord(h0);
        for (int i = 0; i < 2000; i++) { empty_kernel<<<1, 32>>>(); cudaDeviceSynchronize(); }
        cudaEventRecord(h1);
        tot = ms_since(h0, h1);
        printf("RES launch_sync_roundtrip_us=%.2f\n", tot * 1e3 / 2000);
    }

    if (ndev < 2) { printf("RES done_single\n"); return 0; }

    // ---- cross-GPU copies ----
    const size_t SMALL = 10 * 1024;
    float4 *S0, *S1, *L1; void *H;
    CK(cudaSetDevice(1)); CK(cudaMalloc(&S1, SMALL)); CK(cudaMalloc(&L1, (size_t)256 << 20));
    CK(cudaSetDevice(0)); CK(cudaMalloc(&S0, SMALL));
    CK(cudaMallocHost(&H, (size_t)256 << 20));
    auto peer_lat = [&](const char *tag) {
        CK(cudaSetDevice(0));
        for (int i = 0; i < 50; i++) cudaMemcpyPeer(S1, 1, S0, 0, SMALL);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        const int N = 2000;
        for (int i = 0; i < N; i++) { cudaMemcpyPeerAsync(S1, 1, S0, 0, SMALL, 0); cudaStreamSynchronize(0); }
        cudaEventRecord(e1);
        float mm = ms_since(e0, e1);
        printf("RES %s peer_copy_10KB_sync_us=%.2f\n", tag, mm * 1e3 / N);
        cudaEventRecord(e0);
        for (int i = 0; i < N; i++) cudaMemcpyPeerAsync(S1, 1, S0, 0, SMALL, 0);
        cudaEventRecord(e1);
        mm = ms_since(e0, e1);
        printf("RES %s peer_copy_10KB_pipelined_us=%.2f\n", tag, mm * 1e3 / N);
        // large peer bandwidth 256MB, dev0->dev1 (source A on dev0)
        cudaMemcpyPeer(L1, 1, A, 0, (size_t)256 << 20);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) cudaMemcpyPeerAsync(L1, 1, A, 0, (size_t)256 << 20, 0);
        cudaEventRecord(e1);
        mm = ms_since(e0, e1);
        printf("RES %s peer_copy_256MB_GBps=%.2f\n", tag, 5.0 * (256 << 20) / (mm * 1e-3) / 1e9);
    };
    peer_lat("nop2p");
    // explicit host staging (D2H + H2D, pinned), 10KB
    {
        cudaStream_t s0, s1; CK(cudaSetDevice(0)); cudaStreamCreate(&s0); CK(cudaSetDevice(1)); cudaStreamCreate(&s1); CK(cudaSetDevice(0));
        const int N = 2000;
        cudaEventRecord(e0);
        for (int i = 0; i < N; i++) {
            cudaMemcpyAsync(H, S0, SMALL, cudaMemcpyDeviceToHost, s0); cudaStreamSynchronize(s0);
            cudaMemcpyAsync(S1, H, SMALL, cudaMemcpyHostToDevice, s1); cudaStreamSynchronize(s1);
        }
        cudaEventRecord(e1);
        printf("RES host_staged_10KB_us=%.2f\n", ms_since(e0, e1) * 1e3 / N);
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) { cudaMemcpyAsync(H, A, (size_t)256 << 20, cudaMemcpyDeviceToHost, s0); cudaStreamSynchronize(s0); }
        cudaEventRecord(e1);
        printf("RES d2h_pinned_256MB_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(e0, e1) * 1e-3) / 1e9);
        CK(cudaSetDevice(1));
        cudaEvent_t f0, f1; cudaEventCreate(&f0); cudaEventCreate(&f1);
        cudaEventRecord(f0, s1);
        for (int i = 0; i < 5; i++) cudaMemcpyAsync(L1, H, (size_t)256 << 20, cudaMemcpyHostToDevice, s1);
        cudaEventRecord(f1, s1);
        printf("RES h2d_pinned_256MB_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(f0, f1) * 1e-3) / 1e9);
        CK(cudaSetDevice(0));
    }
    int can = 0; cudaDeviceCanAccessPeer(&can, 0, 1);
    if (can) {
        CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
        CK(cudaSetDevice(1)); CK(cudaDeviceEnablePeerAccess(0, 0));
        peer_lat("p2p");
        // dev1 kernel directly reading dev0 memory over PCIe
        CK(cudaSetDevice(1));
        float *o1; cudaMalloc(&o1, 4);
        cudaEvent_t f0, f1; cudaEventCreate(&f0); cudaEventCreate(&f1);
        size_t n = ((size_t)256 << 20) / 16;
        read_kernel_peer<<<40 * 8, 512>>>(A, n, o1);
        cudaEventRecord(f0);
        for (int i = 0; i < 5; i++) read_kernel_peer<<<40 * 8, 512>>>(A, n, o1);
        cudaEventRecord(f1);
        printf("RES p2p_remote_read_kernel_GBps=%.2f\n", 5.0 * (256 << 20) / (ms_since(f0, f1) * 1e-3) / 1e9);
        CK(cudaGetLastError());
    } else {
        printf("RES p2p_unavailable=1\n");
    }
    printf("RES done\n");
    return 0;
}
'''


def el():
    return round(time.time() - T0)


def log(*a):
    print(f"[{el():5d}s]", *a, flush=True)


def result(key, val):
    RESULTS[key] = val
    print("RESULT " + json.dumps({key: val}), flush=True)
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1))


def sh(cmd, timeout=None, env=None, logname=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + \
              ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        out += f"\n<<TIMEOUT after {timeout}s>>"
        rc = -9
    if logname:
        (LOGS / logname).write_text(f"$ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}\nrc={rc} secs={time.time()-t:.1f}\n{out}")
    return rc, out


# ----------------------------------------------------------------------------------------------- env probe
def probe():
    info = {}
    for k, c in {
        "nvidia_smi": "nvidia-smi",
        "smi_query": "nvidia-smi --query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit --format=csv",
        "topo": "nvidia-smi topo -m",
        "p2p_read": "nvidia-smi topo -p2p r",
        "nvcc": "nvcc --version || /usr/local/cuda/bin/nvcc --version",
        "cuda_dirs": "ls -d /usr/local/cuda* ; readlink -f /usr/local/cuda",
        "cmake": "cmake --version | head -1; ninja --version; gcc --version | head -1",
        "cpu": "lscpu | grep -E 'Model name|^CPU\\(s\\)|Flags' | sed -E 's/Flags:(.*)/Flags: (avx512: \\1)/' | grep -oE 'Model name.*|CPU\\(s\\).*|avx512[a-z_]*|avx2|fma|f16c' | sort -u | tr '\\n' ' '",
        "mem": "free -g; df -h / /tmp /kaggle/working",
        "os": "cat /etc/os-release | head -2; uname -r",
    }.items():
        rc, o = sh(c, timeout=60)
        info[k] = o.strip()[-3000:]
        log(f"== {k}\n{o.strip()[-3000:]}")
    py = r'''
import json, importlib
r = {}
try:
    import torch
    r["torch"] = torch.__version__; r["torch_cuda"] = torch.version.cuda
    r["cuda_available"] = torch.cuda.is_available(); r["ngpu"] = torch.cuda.device_count()
    try: r["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e: r["nccl"] = "ERR " + str(e)
    if torch.cuda.device_count() >= 2:
        r["torch_can_access_peer_0_1"] = torch.cuda.can_device_access_peer(0, 1)
        r["torch_can_access_peer_1_0"] = torch.cuda.can_device_access_peer(1, 0)
    r["cudnn"] = torch.backends.cudnn.version()
except Exception as e: r["torch"] = "ERR " + str(e)
for m in ["triton", "cupy", "flash_attn", "vllm", "bitsandbytes", "transformers", "accelerate", "xformers", "numba", "llama_cpp", "pycuda"]:
    try:
        mod = importlib.import_module(m); r[m] = getattr(mod, "__version__", "present")
    except Exception as e: r[m] = None
print("PYJSON" + json.dumps(r))
'''
    rc, o = sh(["python", "-c", py], timeout=300, logname="py_probe.txt")
    m = re.search(r"PYJSON(.*)", o)
    info["python"] = json.loads(m.group(1)) if m else o[-2000:]
    log("python probe", info["python"])
    rc, o = sh("ls /usr/lib/x86_64-linux-gnu | grep -i nccl; find / -name 'libnccl*.so*' -not -path '*/proc/*' 2>/dev/null | head", timeout=120)
    info["nccl_libs"] = o.strip()[-1500:]
    result("env", info)
    return info


def find_nvcc():
    for c in [shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", *sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"))]:
        if c and os.path.exists(c):
            return c
    return None


def microbench(nvcc):
    if not nvcc:
        result("microbench", "nvcc missing")
        return torch_microbench()
    src = W / "bw.cu"
    src.write_text(BW_CU)
    rc, o = sh([nvcc, "-O3", "-arch=sm_75", "-o", str(W / "bw"), str(src)], timeout=600, logname="bw_build.txt")
    if rc:
        result("microbench_build_error", o[-2000:])
        return torch_microbench()
    shutil.copy(src, OUT / "bw.cu")
    rc, o = sh([str(W / "bw")], timeout=900, logname="bw_run.txt")
    log(o)
    res = {}
    for line in o.splitlines():
        if line.startswith("RES "):
            toks = line[4:].split()
            pre = "_".join(x for x in toks if "=" not in x and not x.startswith("("))
            for k, v in re.findall(r"(\S+?)=(\S+)", line[4:]):
                res[(pre + "." if pre else "") + k] = v
        elif line.startswith("ERR"):
            res.setdefault("errors", []).append(line)
    result("microbench", res)
    result("microbench_raw", [l for l in o.splitlines() if l.startswith(("RES", "ERR"))])


def torch_microbench():
    py = r'''
import torch, time, json
r = {}
x = torch.empty(1<<30, dtype=torch.uint8, device="cuda:0"); y = torch.empty_like(x)
torch.cuda.synchronize(); t=time.time()
for _ in range(20): y.copy_(x)
torch.cuda.synchronize(); r["torch_d2d_GBps_rw"] = 2*20*(1<<30)/(time.time()-t)/1e9
s0 = torch.ones(2560, device="cuda:0"); s1 = torch.empty(2560, device="cuda:1")
for _ in range(50): s1.copy_(s0)
torch.cuda.synchronize(); t=time.time()
for _ in range(2000): s1.copy_(s0); torch.cuda.synchronize(1)
r["torch_xgpu_10KB_us"] = (time.time()-t)/2000*1e6
print("PYJSON"+json.dumps(r))
'''
    rc, o = sh(["python", "-c", py], timeout=600, logname="torch_bw.txt")
    m = re.search(r"PYJSON(.*)", o)
    result("torch_microbench", json.loads(m.group(1)) if m else o[-1500:])


# ----------------------------------------------------------------------------------------------- downloads
DL = {}


def downloader():
    env = dict(os.environ, HF_XET_HIGH_PERFORMANCE="1", HF_HUB_ENABLE_HF_TRANSFER="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               HF_HOME=str(W / "hf"))
    env.pop("HF_TOKEN", None)
    if not shutil.which("hf"):
        sh("pip install -q -U 'huggingface_hub[hf_xet]' hf_transfer", timeout=600, logname="pip_hf.txt")
    for name, fn in [("UD-Q4_K_XL", MODELS["UD-Q4_K_XL"]), ("MTP", MTP), ("Q4_0", MODELS["Q4_0"])]:
        t = time.time()
        rc, o = sh(["hf", "download", REPO, fn, "--local-dir", str(MD)], env=env, timeout=3600, logname=f"dl_{name}.txt")
        p = MD / fn
        if rc or not p.exists():
            log(f"hf download failed for {fn}, trying curl")
            rc, o = sh(f"mkdir -p {p.parent} && curl -fL --retry 5 -o {p} https://huggingface.co/{REPO}/resolve/main/{fn}",
                       timeout=3600, logname=f"dl_{name}_curl.txt")
        ok = p.exists() and p.stat().st_size > 1e8
        secs = time.time() - t
        DL[name] = str(p) if ok else None
        result(f"download_{name}", {"ok": ok, "gb": round(p.stat().st_size / 1e9, 2) if ok else 0, "secs": round(secs),
                                    "MBps": round(p.stat().st_size / 1e6 / secs) if ok else 0})
    shutil.rmtree(W / "hf", ignore_errors=True)


def wait_model(name, limit=3600):
    t = time.time()
    while name not in DL and time.time() - t < limit:
        time.sleep(5)
    return DL.get(name)


# ----------------------------------------------------------------------------------------------- build
def build_llama(nvcc):
    src = W / "llama.cpp"
    rc, o = sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}", timeout=600, logname="git_clone.txt")
    rc, commit = sh(f"git -C {src} log -1 --format='%H %cd %s'")
    result("llama_commit", commit.strip())
    if not nvcc:
        return None
    env = dict(os.environ, PATH=f"{Path(nvcc).parent}:" + os.environ["PATH"], CUDACXX=nvcc)
    gen = "-G Ninja" if shutil.which("ninja") else ""
    if not gen:
        sh("pip install -q ninja", timeout=300)
        gen = "-G Ninja" if shutil.which("ninja") else ""
    rc, libs = sh("ls /usr/local/cuda*/lib64/stubs/libcuda.so /usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so "
                  "2>/dev/null; ldconfig -p | grep -oE '/\\S+/libcuda\\.so(\\.1)?$'; find /usr /lib /opt \\( -name 'libcuda.so' -o -name 'libcuda.so.1' \\) 2>/dev/null | head -20",
                  timeout=300)
    cands = [x.strip() for x in libs.splitlines() if x.strip().endswith(("libcuda.so", "libcuda.so.1")) and os.path.exists(x.strip())]
    result("libcuda_candidates", cands)
    drv = f"-DCUDA_cuda_driver_LIBRARY={cands[0]} " if cands else ""
    cfg = (f"cmake -S {src} -B {src}/build {gen} {drv}-DCUDAToolkit_ROOT={Path(nvcc).parent.parent} -DGGML_CCACHE=OFF "
           f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
           f"-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF "
           f"-DBUILD_SHARED_LIBS=ON")
    t = time.time()
    rc, o = sh(cfg, env=env, timeout=900, logname="cmake_config.txt")
    if rc:
        result("build", {"ok": False, "stage": "config", "tail": o[-2000:]})
        return None
    targets = "llama-bench llama-server llama-cli llama-speculative-simple llama-batched-bench"
    rc, o = sh(f"cmake --build {src}/build --config Release -j {os.cpu_count()} --target {targets}", env=env,
               timeout=int(70 * 60), logname="cmake_build.txt")
    secs = round(time.time() - t)
    b = src / "build" / "bin"
    ok = (b / "llama-bench").exists() and (b / "llama-server").exists()
    result("build", {"ok": ok, "secs": secs, "rc": rc, "tail": o[-1500:] if not ok else ""})
    if not ok:
        return None
    # ship binaries for reuse
    dst = OUT / "llama-bin-sm75"
    dst.mkdir(exist_ok=True)
    for f in list(b.glob("llama-*")) + list(b.glob("*.so*")) + list((src / "build").rglob("*.so*")):
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    sh(f"tar -czf {OUT}/llama-bin-sm75.tgz -C {OUT} llama-bin-sm75", timeout=600)
    (dst / "COMMIT.txt").write_text(commit)
    return b


def prebuilt():
    t = time.time()
    try:
        rel = json.loads(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10", timeout=30).read())
        tag = next(r["tag_name"] for r in rel if re.fullmatch(r"b\d+", r["tag_name"]))
    except Exception:
        tag = "b11344"
    d = W / "pre"
    d.mkdir(exist_ok=True)
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}"
    sh(f"curl -fsSL {base}/llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C {d}", timeout=900, logname="prebuilt.txt")
    sh(f"curl -fsSL {base}/cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz -C {d}", timeout=900)
    bench = next(iter(glob.glob(f"{d}/**/llama-bench", recursive=True)), None)
    if not bench:
        result("prebuilt", {"ok": False})
        return None
    b = Path(bench).parent
    for so in glob.glob(f"{d}/**/*.so*", recursive=True):
        if Path(so).parent != b:
            shutil.copy2(so, b / Path(so).name)
    for f in b.iterdir():
        f.chmod(0o755)
    result("prebuilt", {"ok": True, "tag": tag, "secs": round(time.time() - t)})
    return b


# ----------------------------------------------------------------------------------------------- benches
def bench(b, env, model, tag, extra, timeout=900):
    if time.time() - T0 > DEADLINE:
        result(f"bench/{tag}", "skipped (deadline)")
        return None
    cmd = [str(b / "llama-bench"), "-m", model, "-ngl", "99", "-fa", "1", "-o", "json", "-t", str(os.cpu_count())] + extra
    t = time.time()
    rc, o = sh(cmd, env=env, timeout=timeout, logname=f"bench_{tag.replace('/', '_')}.txt")
    rows = []
    m = re.search(r"\[\s*\{.*\}\s*\]", o, re.S)
    if m:
        try:
            for r in json.loads(m.group(0)):
                rows.append({"test": f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}", "depth": r.get("n_depth", 0),
                             "split_mode": r.get("split_mode"), "avg_ts": round(r["avg_ts"], 2), "stddev_ts": round(r["stddev_ts"], 2),
                             "n_ubatch": r.get("n_ubatch"), "type_k": r.get("type_k")})
        except Exception as e:  # noqa: BLE001
            rows = [f"parse error {e}"]
    err = "" if rows else o[-2500:]
    mem = re.findall(r"(CUDA\d) model buffer size = +([\d.]+) MiB", o)
    result(f"bench/{tag}", {"rc": rc, "secs": round(time.time() - t), "rows": rows, "err": err})
    log(f"bench {tag}: {rows if rows else err[-800:]}")
    return rows


CODE_PROMPTS = [
    "Write a complete Python module `lru_cache.py` that implements a thread-safe LRU cache class with get, put, delete, "
    "resize and a `__len__`, using an OrderedDict and a threading.Lock. Include type hints, docstrings, and a full "
    "unittest test suite at the bottom covering eviction order, resizing, and concurrent access.",
    "Write a Python script that parses an Apache access log file, aggregates requests per IP, per status code and per "
    "hour, detects IPs with more than 100 requests per minute, and prints a report. Use argparse, dataclasses, "
    "collections.Counter and re. Include docstrings and example usage.",
]


def gpu_mem():
    rc, o = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    return o.strip().replace("\n", ",")


def serve_test(b, env, model, tag, sm, spec=None, n_max_list=(None,), ctx=16384, timeout_load=900):
    if time.time() - T0 > DEADLINE:
        result(f"serve/{tag}", "skipped (deadline)")
        return
    port = 18080
    args = [str(b / "llama-server"), "-m", model, "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-sm", sm,
            "-fa", "on", "-c", str(ctx), "-np", "1", "-fit", "off", "-t", str(os.cpu_count()), "--no-webui",
            "-cram", "0", "--metrics"]
    if spec:
        args += ["--spec-type", "draft-mtp", "-md", spec, "--spec-draft-n-max", str(max(x for x in n_max_list if x)),
                 "--spec-draft-ngl", "99"]
    logp = LOGS / f"server_{tag.replace('/', '_')}.log"
    srv = subprocess.Popen(args, env=env, stdout=open(logp, "w"), stderr=subprocess.STDOUT)
    t = time.time()
    ok = False
    while srv.poll() is None and time.time() - t < timeout_load:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200:
                ok = True
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    load_s = round(time.time() - t)
    if not ok:
        srv.kill()
        result(f"serve/{tag}", {"ok": False, "load_s": load_s, "log_tail": logp.read_text()[-2500:]})
        return
    out = {"ok": True, "load_s": load_s, "gpu_mem_used_MiB": gpu_mem(), "runs": []}

    def req(prompt, n_max, n=512):
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0.0, "top_k": 1,
                "seed": 1, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
        if n_max is not None:
            body["speculative.n_max"] = n_max
        rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                    headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(rq, timeout=1800).read())
        tm = r.get("timings", {})
        txt = r["choices"][0]["message"].get("content") or ""
        return tm, txt

    try:
        req("Say hi.", n_max_list[0], 16)  # warmup
        for n_max in n_max_list:
            for i, p in enumerate(CODE_PROMPTS):
                tm, txt = req(p, n_max)
                row = {"n_max": n_max, "prompt": i, "prompt_n": tm.get("prompt_n"),
                       "prefill_tok_s": round(tm.get("prompt_per_second", 0), 1), "decode_n": tm.get("predicted_n"),
                       "decode_tok_s": round(tm.get("predicted_per_second", 0), 2),
                       "draft_n": tm.get("draft_n"), "draft_accepted": tm.get("draft_n_accepted")}
                if row["draft_n"]:
                    row["accept_rate"] = round(row["draft_accepted"] / row["draft_n"], 3)
                out["runs"].append(row)
                log(f"serve {tag} {row}")
                if i == 0 and n_max == n_max_list[0]:
                    (LOGS / f"sample_{tag.replace('/', '_')}.txt").write_text(txt)
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    srv.terminate()
    try:
        srv.wait(60)
    except Exception:  # noqa: BLE001
        srv.kill()
    lt = logp.read_text()
    out["mtp_log"] = re.findall(r".*(?:draft-mtp|MTP|mtp).*", lt)[:12]
    out["buffers"] = re.findall(r"(CUDA\d|CUDA_Host) (model|KV|compute|RS) buffer size = +([\d.]+) MiB", lt)[:20]
    result(f"serve/{tag}", out)


# ----------------------------------------------------------------------------------------------- main
def main():
    info = probe()
    nvcc = find_nvcc()
    log("nvcc:", nvcc)
    microbench(nvcc)
    dl = threading.Thread(target=downloader, daemon=True)
    dl.start()
    b = build_llama(nvcc)
    source = "source"
    if b is None:
        b = prebuilt()
        source = "prebuilt"
    result("engine_source", source)
    if b is None:
        result("fatal", "no llama.cpp binaries")
        return
    env = dict(os.environ, LD_LIBRARY_PATH=f"{b}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    rc, o = sh([str(b / "llama-bench"), "--help"], env=env)
    result("bench_help_sm", re.findall(r".*split-mode.*", o))

    m1 = wait_model("UD-Q4_K_XL")
    mtp = wait_model("MTP")
    if m1:
        modes = ["layer", "row", "tensor"]
        ok_modes = []
        for sm in modes:
            rows = bench(b, env, m1, f"Q4_K_XL/{sm}", ["-sm", sm, "-p", "512", "-n", "128", "-r", "3"])
            if rows and isinstance(rows[0], dict):
                ok_modes.append((sm, max(r["avg_ts"] for r in rows if r["test"].startswith("tg"))))
        # depth sensitivity on layer mode
        bench(b, env, m1, "Q4_K_XL/layer_d16k", ["-sm", "layer", "-p", "512", "-n", "128", "-d", "16384", "-r", "2"], timeout=1800)
        # ubatch sensitivity for prefill
        bench(b, env, m1, "Q4_K_XL/layer_ub", ["-sm", "layer", "-p", "2048", "-n", "0", "-ub", "512,1024,2048", "-b", "2048", "-r", "2"])
        best = max(ok_modes, key=lambda x: x[1])[0] if ok_modes else "layer"
        result("best_split_mode_Q4_K_XL", best)
        serve_test(b, env, m1, "Q4_K_XL/layer/nospec", "layer")
        if mtp:
            serve_test(b, env, m1, "Q4_K_XL/layer/mtp", "layer", spec=mtp, n_max_list=(1, 2, 3, 4))
        if best != "layer":
            serve_test(b, env, m1, f"Q4_K_XL/{best}/nospec", best)
            if mtp:
                serve_test(b, env, m1, f"Q4_K_XL/{best}/mtp", best, spec=mtp, n_max_list=(2, 3))
    m2 = wait_model("Q4_0")
    if m2:
        for sm in ["layer", "row", "tensor"]:
            bench(b, env, m2, f"Q4_0/{sm}", ["-sm", sm, "-p", "512", "-n", "128", "-r", "3"])
        if mtp:
            serve_test(b, env, m2, "Q4_0/layer/mtp", "layer", spec=mtp, n_max_list=(2, 3))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        result("fatal", traceback.format_exc()[-3000:])
    result("total_secs", el())
    print("=====RESULTS_BEGIN=====")
    print(json.dumps(RESULTS, indent=1))
    print("=====RESULTS_END=====", flush=True)
