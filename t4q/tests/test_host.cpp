// Host-only test: parse a GGUF and dump the CPU-port dequant of the first rows of a tensor.
// usage: test_host <gguf> <tensor> <nrows> <out.f32>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/gguf.h"
#include "../src/quant_cpu.h"

int main(int argc, char** argv) {
    if (argc < 5) return 2;
    GgufFile f;
    std::string err;
    if (!f.open(argv[1], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const GgufTensor* t = f.find(argv[2]);
    if (!t) { fprintf(stderr, "no tensor\n"); return 1; }
    int nr = atoi(argv[3]);
    std::vector<float> out((size_t)nr * t->ne[0]);
    for (int r = 0; r < nr; r++)
        if (!dequant_row_cpu(t->type, t->data + (size_t)r * t->row_bytes, out.data() + (size_t)r * t->ne[0], t->ne[0])) {
            fprintf(stderr, "unsupported\n");
            return 1;
        }
    FILE* o = fopen(argv[4], "wb");
    fwrite(out.data(), 4, out.size(), o);
    fclose(o);
    printf("%s type=%s ne=[%lld,%lld] data_start=%zu kv=%zu tensors=%zu\n", t->name.c_str(), ggml_type_name(t->type),
           (long long)t->ne[0], (long long)t->ne[1], f.data_start, f.kv.size(), f.tensors.size());
    return 0;
}
