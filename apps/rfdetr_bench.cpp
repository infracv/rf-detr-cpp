// rfdetr_bench — latency and throughput benchmark for RF-DETR TRT engines.
//
// Measures end-to-end wall time (H2D + infer + D2H + postprocess) and
// GPU-only time (cudaEvent), then writes a JSON report.
//
// Usage:
//   rfdetr_bench --engine ENGINE --image IMAGE \
//                [--device rtx-5070ti] [--batch 1] \
//                [--warmup 50] [--iters 500] \
//                [--report results/out.json]
//
// The JSON schema matches §7.7 of Implementation.md.

#include "rfdetr/core/cuda_check.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/tasks/segmenter.hpp"
#include "rfdetr/version.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::string engine;
    std::string image;
    std::string device{"unknown"};
    int         batch{1};
    int         warmup{50};
    int         iters{500};
    std::string report;       // path for JSON output; empty = stdout only
};

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
const char* next_val(int argc, char** argv, int& i, std::string_view flag) {
    std::string_view a = argv[i];
    if (a.size() > flag.size() + 1 && a[flag.size()] == '=')
        return argv[i] + flag.size() + 1;
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + std::string(flag));
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::fprintf(stderr,
                "usage: %s --engine PATH --image PATH [options]\n"
                "  --device TAG     Label for this device in the JSON (default: unknown)\n"
                "  --batch N        Batch size (default: 1; needs dynamic-batch engine for >1)\n"
                "  --warmup N       Warm-up iterations (default: 50)\n"
                "  --iters N        Measurement iterations (default: 500)\n"
                "  --report PATH    Write JSON report to this path (default: stdout only)\n",
                argv[0]);
            std::exit(0);
        }
        else if (starts_with(arg, "--engine")) a.engine = next_val(argc, argv, i, "--engine");
        else if (starts_with(arg, "--image"))  a.image  = next_val(argc, argv, i, "--image");
        else if (starts_with(arg, "--device")) a.device = next_val(argc, argv, i, "--device");
        else if (starts_with(arg, "--batch"))  a.batch  = std::atoi(next_val(argc, argv, i, "--batch"));
        else if (starts_with(arg, "--warmup")) a.warmup = std::atoi(next_val(argc, argv, i, "--warmup"));
        else if (starts_with(arg, "--iters"))  a.iters  = std::atoi(next_val(argc, argv, i, "--iters"));
        else if (starts_with(arg, "--report")) a.report = next_val(argc, argv, i, "--report");
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.image.empty())  throw std::runtime_error("--image is required");
    return a;
}

// --------------------------------------------------------------------------
// Percentile helpers
// --------------------------------------------------------------------------

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(idx);
    const std::size_t hi = lo + 1 < v.size() ? lo + 1 : lo;
    const double frac    = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// --------------------------------------------------------------------------
// CUDA version string from runtime API
// --------------------------------------------------------------------------

std::string cuda_version_str() {
    int ver = 0;
    cudaRuntimeGetVersion(&ver);
    return std::to_string(ver / 1000) + "." + std::to_string((ver % 1000) / 10);
}

std::string driver_version_str() {
    int ver = 0;
    cudaDriverGetVersion(&ver);
    return std::to_string(ver / 1000) + "." + std::to_string((ver % 1000) / 10);
}

std::string trt_version_str() {
    return std::to_string(NV_TENSORRT_MAJOR) + "." +
           std::to_string(NV_TENSORRT_MINOR) + "." +
           std::to_string(NV_TENSORRT_PATCH);
}

// --------------------------------------------------------------------------
// Generic run loop — works for both detector and segmenter via std::function
// --------------------------------------------------------------------------

struct RunResult {
    std::vector<double> wall_ms;   // end-to-end including postprocess
    std::vector<double> gpu_ms;    // GPU-only (cudaEvent)
};

template <typename InferFn>
RunResult run_loop(InferFn infer_fn, const std::vector<cv::Mat>& frames,
                   int warmup, int iters) {
    cudaEvent_t ev_start, ev_stop;
    RFDETR_CUDA_CHECK(cudaEventCreate(&ev_start));
    RFDETR_CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warm up.
    for (int i = 0; i < warmup; ++i) {
        infer_fn(frames[static_cast<std::size_t>(i) % frames.size()]);
    }

    RunResult r;
    r.wall_ms.reserve(static_cast<std::size_t>(iters));
    r.gpu_ms.reserve(static_cast<std::size_t>(iters));

    for (int i = 0; i < iters; ++i) {
        const auto& frame = frames[static_cast<std::size_t>(i) % frames.size()];

        RFDETR_CUDA_CHECK(cudaEventRecord(ev_start));
        const auto t0 = std::chrono::steady_clock::now();

        infer_fn(frame);

        RFDETR_CUDA_CHECK(cudaEventRecord(ev_stop));
        RFDETR_CUDA_CHECK(cudaEventSynchronize(ev_stop));
        const auto t1 = std::chrono::steady_clock::now();

        float gpu_elapsed = 0.0f;
        RFDETR_CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed, ev_start, ev_stop));

        r.wall_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        r.gpu_ms.push_back(static_cast<double>(gpu_elapsed));
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    return r;
}

// --------------------------------------------------------------------------
// Build and print stats
// --------------------------------------------------------------------------

nlohmann::json stats_json(std::vector<double>& v) {
    return {
        {"mean", mean(v)},
        {"p50",  percentile(v, 50.0)},
        {"p90",  percentile(v, 90.0)},
        {"p99",  percentile(v, 99.0)},
        {"min",  *std::min_element(v.begin(), v.end())},
        {"max",  *std::max_element(v.begin(), v.end())},
    };
}

void print_stats(const char* label, std::vector<double>& v) {
    std::printf("  %-18s mean=%.3f  p50=%.3f  p90=%.3f  p99=%.3f  min=%.3f  max=%.3f  (ms)\n",
                label, mean(v),
                percentile(v, 50.0), percentile(v, 90.0), percentile(v, 99.0),
                *std::min_element(v.begin(), v.end()),
                *std::max_element(v.begin(), v.end()));
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }

    cv::Mat frame = cv::imread(args.image);
    if (frame.empty()) {
        std::fprintf(stderr, "error: cannot read image: %s\n", args.image.c_str());
        return 1;
    }

    // Load meta to decide det vs seg and read precision.
    std::string meta_path = args.engine + ".json";
    rfdetr::EngineMeta meta;
    bool have_meta = false;
    if (std::filesystem::is_regular_file(meta_path)) {
        meta      = rfdetr::EngineMeta::from_json_file(meta_path);
        have_meta = true;
    }

    const bool is_seg = have_meta && meta.has_masks;

    std::printf("[rfdetr_bench] engine    : %s\n", args.engine.c_str());
    std::printf("[rfdetr_bench] variant   : %s\n", have_meta ? meta.variant.c_str() : "?");
    std::printf("[rfdetr_bench] precision : %s\n", have_meta ? meta.precision.c_str() : "?");
    std::printf("[rfdetr_bench] task      : %s\n", is_seg ? "segmentation" : "detection");
    std::printf("[rfdetr_bench] batch     : %d\n", args.batch);
    std::printf("[rfdetr_bench] warmup    : %d  iters: %d\n", args.warmup, args.iters);
    std::printf("[rfdetr_bench] trt       : %s  cuda: %s  driver: %s\n",
                trt_version_str().c_str(), cuda_version_str().c_str(), driver_version_str().c_str());
    std::fflush(stdout);

    // Build a small rotating pool of frames (just copies of the single input image).
    const int pool_size = std::max(1, std::min(8, args.iters));
    std::vector<cv::Mat> pool(static_cast<std::size_t>(pool_size), frame);

    // Batch pool: vectors of batch_size copies.
    std::vector<std::vector<cv::Mat>> batch_pool(
        static_cast<std::size_t>(pool_size),
        std::vector<cv::Mat>(static_cast<std::size_t>(args.batch), frame));

    RunResult result;

    try {
        if (is_seg) {
            rfdetr::SegmenterOptions sopts;
            rfdetr::RFDetrSegmenter seg(args.engine);
            std::printf("[rfdetr_bench] running segmentation benchmark...\n");
            std::fflush(stdout);
            if (args.batch == 1) {
                result = run_loop([&](const cv::Mat& f) { seg.segment(f); },
                                  pool, args.warmup, args.iters);
            } else {
                result = run_loop([&](const cv::Mat&) {
                    seg.segment_batch(batch_pool[0]);
                }, pool, args.warmup, args.iters);
            }
        } else {
            rfdetr::DetectorOptions dopts;
            rfdetr::RFDetrDetector det(args.engine);
            std::printf("[rfdetr_bench] running detection benchmark...\n");
            std::fflush(stdout);
            if (args.batch == 1) {
                result = run_loop([&](const cv::Mat& f) { det.detect(f); },
                                  pool, args.warmup, args.iters);
            } else {
                result = run_loop([&](const cv::Mat&) {
                    det.detect_batch(batch_pool[0]);
                }, pool, args.warmup, args.iters);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const double fps = 1000.0 * args.batch * static_cast<double>(args.iters) /
                       std::accumulate(result.wall_ms.begin(), result.wall_ms.end(), 0.0);

    std::printf("\n[rfdetr_bench] results (%d iters, batch=%d):\n", args.iters, args.batch);
    print_stats("wall_ms (total)", result.wall_ms);
    print_stats("gpu_ms (infer)", result.gpu_ms);
    std::printf("  %-18s %.1f img/s\n", "throughput", fps);

    // Build JSON report.
    nlohmann::json report;
    report["device"]         = args.device;
    report["variant"]        = have_meta ? meta.variant : "unknown";
    report["precision"]      = have_meta ? meta.precision : "unknown";
    report["task"]           = is_seg ? "segmentation" : "detection";
    report["input_hw"]       = have_meta
                                   ? nlohmann::json{meta.input_h, meta.input_w}
                                   : nlohmann::json{frame.rows, frame.cols};
    report["batch"]          = args.batch;
    report["trt_version"]    = trt_version_str();
    report["cuda_version"]   = cuda_version_str();
    report["driver_version"] = driver_version_str();
    report["warmup_iters"]   = args.warmup;
    report["measure_iters"]  = args.iters;
    report["latency_ms"]     = stats_json(result.wall_ms);
    report["gpu_ms"]         = stats_json(result.gpu_ms);
    report["throughput_fps"] = fps;
    // mAP fields left null — filled by benchmarks/scripts/run_accuracy.py
    report["map_50_95"]      = nullptr;
    report["map_50"]         = nullptr;
    report["map_75"]         = nullptr;
    report["mask_map_50_95"] = nullptr;

    const std::string json_str = report.dump(2);
    if (!args.report.empty()) {
        std::filesystem::create_directories(
            std::filesystem::path(args.report).parent_path());
        std::ofstream f(args.report);
        if (!f) {
            std::fprintf(stderr, "warning: cannot write report to %s\n", args.report.c_str());
        } else {
            f << json_str << "\n";
            std::printf("[rfdetr_bench] report written to %s\n", args.report.c_str());
        }
    }

    // Always print JSON to stdout too (easy to pipe or redirect).
    std::printf("\n%s\n", json_str.c_str());
    return 0;
}
