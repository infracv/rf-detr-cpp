// rfdetr_smoke — load an engine, fill all inputs with zeros, run a single sync
// infer, and print per-output stats. Proves the TrtSession plumbing end-to-end.
//
// usage: rfdetr_smoke <engine.plan>
//        rfdetr_smoke <engine.plan> <iters>

#include "rfdetr/core/trt_session.hpp"
#include "rfdetr/version.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {

void print_dims(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) std::fputc(',', stdout);
        std::printf("%ld", static_cast<long>(d.d[i]));
    }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr,
                     "usage: %s <engine.plan> [iters]\n"
                     "  rfdetr v%s\n",
                     argv[0], rfdetr::version());
        return 2;
    }
    const std::filesystem::path engine_path{argv[1]};
    int iters = 1;
    if (argc == 3) {
        iters = std::atoi(argv[2]);
        if (iters <= 0) iters = 1;
    }

    try {
        rfdetr::TrtSession sess(engine_path);
        std::printf("loaded: %s\n", engine_path.c_str());
        std::printf("  inputs:  %zu\n", sess.input_indices().size());
        std::printf("  outputs: %zu\n", sess.output_indices().size());

        for (const auto& b : sess.bindings()) {
            std::printf("  %s %-12s  dtype=%-5s  bytes=%-10zu  shape=",
                        b.is_input ? "IN " : "OUT", b.name.c_str(),
                        rfdetr::TrtSession::dtype_name(b.dtype), b.bytes);
            print_dims(b.shape);
            std::putchar('\n');
        }

        // Sanity-check input shapes are resolved (static engine, or set_input_shape called).
        for (int i : sess.input_indices()) {
            const auto& b = sess.bindings()[i];
            if (b.bytes == 0) {
                std::fprintf(stderr,
                             "error: input '%s' has unresolved shape "
                             "(dynamic engine without optimization profile applied yet)\n",
                             b.name.c_str());
                return 1;
            }
        }

        // Fill all inputs with zeros — single allocation reused per input.
        for (int i : sess.input_indices()) {
            const auto& b = sess.bindings()[i];
            std::vector<std::uint8_t> zeros(b.bytes, 0);
            sess.set_input(b.name, zeros.data(), zeros.size());
        }

        // First infer = warm-up; subsequent ones average for a rough wall-clock read.
        std::printf("running infer x%d...\n", iters);
        sess.infer();  // warm

        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < iters; ++k) {
            for (int i : sess.input_indices()) {
                const auto& b = sess.bindings()[i];
                std::vector<std::uint8_t> zeros(b.bytes, 0);
                sess.set_input(b.name, zeros.data(), zeros.size());
            }
            sess.infer();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms_total =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("infer OK  total=%.3f ms  mean=%.3f ms/iter\n",
                    ms_total, ms_total / iters);

        // Pull outputs and print summary stats.
        for (int i : sess.output_indices()) {
            const auto& b = sess.bindings()[i];
            std::vector<std::uint8_t> buf(b.bytes);
            sess.get_output(b.name, buf.data(), buf.size());

            if (b.dtype == nvinfer1::DataType::kFLOAT) {
                const float* f = reinterpret_cast<const float*>(buf.data());
                const std::size_t n = b.bytes / 4;
                double sum = 0.0;
                double max_abs = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    sum += f[k];
                    const double a = std::fabs(static_cast<double>(f[k]));
                    if (a > max_abs) max_abs = a;
                }
                const double mean = (n > 0) ? sum / static_cast<double>(n) : 0.0;
                std::printf("  out %-12s  fp32  n=%zu  mean=%+.6f  max|x|=%.6f\n",
                            b.name.c_str(), n, mean, max_abs);
            } else if (b.dtype == nvinfer1::DataType::kHALF) {
                // Don't pull in fp16 conversion just for a smoke print — byte-checksum.
                std::uint64_t s = 0;
                for (auto v : buf) s += v;
                std::printf("  out %-12s  fp16  bytes=%zu  byte_sum=%llu\n",
                            b.name.c_str(), b.bytes,
                            static_cast<unsigned long long>(s));
            } else {
                std::uint64_t s = 0;
                for (auto v : buf) s += v;
                std::printf("  out %-12s  %s  bytes=%zu  byte_sum=%llu\n",
                            b.name.c_str(), rfdetr::TrtSession::dtype_name(b.dtype),
                            b.bytes, static_cast<unsigned long long>(s));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
