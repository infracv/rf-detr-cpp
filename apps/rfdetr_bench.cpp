// rfdetr_bench — comprehensive benchmarking for RF-DETR detection models.
//
// Usage:
//   rfdetr_bench quick  <engine> <image>
//   rfdetr_bench image  <engine> <image>  [options]
//   rfdetr_bench video  <engine> <video>  [options]
//   rfdetr_bench camera <engine> <cam_id> [options]

#include "cli_helpers.hpp"
#include "rfdetr/core/cuda_check.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/tasks/segmenter.hpp"

#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace colors {
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* RED     = "\033[31m";
    constexpr const char* GREEN   = "\033[32m";
    constexpr const char* YELLOW  = "\033[33m";
    constexpr const char* CYAN    = "\033[36m";
    constexpr const char* BOLD    = "\033[1m";
    constexpr const char* DIM     = "\033[2m";

    inline bool is_tty() {
#ifdef _WIN32
        return false;
#else
        return isatty(fileno(stdout));
#endif
    }
    inline std::string c(const std::string& text, const char* col) {
        return is_tty() ? std::string(col) + text + RESET : text;
    }
}

class ProgressBar {
public:
    ProgressBar(int total, const std::string& prefix = "", int width = 40)
        : total_(total), width_(width), prefix_(prefix) {}

    void update(int n) {
        if (!colors::is_tty()) return;
        float p = static_cast<float>(n) / static_cast<float>(total_);
        int filled = static_cast<int>(p * width_);
        std::cout << "\r" << prefix_ << " [";
        for (int i = 0; i < width_; ++i) {
            if (i < filled)       std::cout << "\xE2\x96\x88";
            else if (i == filled) std::cout << "\xE2\x96\x93";
            else                  std::cout << "\xE2\x96\x91";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(p * 100) << "% "
                  << "(" << n << "/" << total_ << ")" << std::flush;
    }
    void finish() { update(total_); std::cout << "\n"; }

private:
    int total_, width_;
    std::string prefix_;
};

namespace sysmon {

double cpu_usage() {
#ifdef _WIN32
    return 0.0;
#else
    static unsigned long long lu = 0, lul = 0, ls = 0, li = 0;
    std::ifstream f("/proc/stat");
    std::string line;
    if (!std::getline(f, line)) return 0.0;
    unsigned long long u, ul, s, id;
    if (std::sscanf(line.c_str(), "cpu %llu %llu %llu %llu", &u, &ul, &s, &id) != 4) return 0.0;
    if (lu == 0) { lu = u; lul = ul; ls = s; li = id; return 0.0; }
    auto total = (u - lu) + (ul - lul) + (s - ls);
    double pct = static_cast<double>(total);
    total += (id - li);
    double out = total > 0 ? pct / total * 100.0 : 0.0;
    lu = u; lul = ul; ls = s; li = id;
    return out;
#endif
}

struct GpuStats { double util{0}; double mem_mb{0}; };

GpuStats gpu_stats() {
#ifdef _WIN32
    return {};
#else
    FILE* p = popen(
        "nvidia-smi --query-gpu=utilization.gpu,memory.used "
        "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return {};
    char buf[256]; std::string s;
    while (fgets(buf, sizeof buf, p)) s += buf;
    pclose(p);
    GpuStats g;
    std::sscanf(s.c_str(), "%lf, %lf", &g.util, &g.mem_mb);
    return g;
#endif
}

} // namespace sysmon

struct LatencyStats {
    double avg{}, stddev{}, min{}, max{};
    double p50{}, p90{}, p95{}, p99{};

    static LatencyStats compute(std::vector<double>& v) {
        LatencyStats s;
        if (v.empty()) return s;
        std::sort(v.begin(), v.end());
        std::size_t n = v.size();
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        s.avg = sum / n;
        s.min = v.front();
        s.max = v.back();
        double sq = 0;
        for (double x : v) sq += (x - s.avg) * (x - s.avg);
        s.stddev = std::sqrt(sq / n);
        auto pct = [&](double p) {
            double idx = p * (n - 1);
            std::size_t lo = static_cast<std::size_t>(idx);
            std::size_t hi = std::min(lo + 1, n - 1);
            return v[lo] * (1 - (idx - lo)) + v[hi] * (idx - lo);
        };
        s.p50 = pct(0.50); s.p90 = pct(0.90);
        s.p95 = pct(0.95); s.p99 = pct(0.99);
        return s;
    }
};

struct BenchmarkConfig {
    std::string engine;
    std::string device{"unknown"};
    std::string task{"detection"};   // "detection" or "segmentation"
    int         warmup{50};
    int         iters{500};
    int         max_video_frames{2000};
    int         camera_seconds{30};
    float       threshold{0.5f};
    std::string output_dir{"benchmarks/results"};
    bool        json_output{false};
    bool        verbose{true};
};

struct BenchmarkMetrics {
    double load_ms{};
    double warmup_ms{};
    LatencyStats latency;        // wall time (preprocess + infer + postprocess)
    LatencyStats gpu_latency;    // GPU-only (cudaEvent)
    double fps{};
    int    frame_count{};
    double cpu_usage_avg{};
    double gpu_usage_avg{};
    double gpu_memory_mb{};
    int    avg_detections{};
};

static std::string trt_ver() {
    return std::to_string(NV_TENSORRT_MAJOR) + "." +
           std::to_string(NV_TENSORRT_MINOR) + "." +
           std::to_string(NV_TENSORRT_PATCH);
}
static std::string cuda_ver() {
    int v = 0; cudaRuntimeGetVersion(&v);
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}
static std::string driver_ver() {
    int v = 0; cudaDriverGetVersion(&v);
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

template <typename InferFn>
BenchmarkMetrics run_loop(InferFn infer_fn,
                          const std::vector<cv::Mat>& frames,
                          const BenchmarkConfig& cfg) {
    BenchmarkMetrics m;

    cudaEvent_t ev0, ev1;
    RFDETR_CUDA_CHECK(cudaEventCreate(&ev0));
    RFDETR_CUDA_CHECK(cudaEventCreate(&ev1));

    // Warm up
    auto wt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.warmup; ++i)
        infer_fn(frames[static_cast<std::size_t>(i) % frames.size()]);
    m.warmup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wt0).count();

    std::vector<double> wall_ms, gpu_ms;
    wall_ms.reserve(static_cast<std::size_t>(cfg.iters));
    gpu_ms.reserve(static_cast<std::size_t>(cfg.iters));

    std::vector<double> cpu_s, gpu_s;
    double gpu_mem_first = 0;
    int total_det = 0;

    sysmon::cpu_usage(); // init

    ProgressBar bar(cfg.iters, "  Benchmarking");

    for (int i = 0; i < cfg.iters; ++i) {
        const auto& f = frames[static_cast<std::size_t>(i) % frames.size()];

        cpu_s.push_back(sysmon::cpu_usage());
        auto gs = sysmon::gpu_stats();
        gpu_s.push_back(gs.util);
        if (i == 0) gpu_mem_first = gs.mem_mb;

        RFDETR_CUDA_CHECK(cudaEventRecord(ev0));
        auto t0 = std::chrono::steady_clock::now();
        int nd = infer_fn(f);
        RFDETR_CUDA_CHECK(cudaEventRecord(ev1));
        RFDETR_CUDA_CHECK(cudaEventSynchronize(ev1));
        auto t1 = std::chrono::steady_clock::now();

        float gpu_el = 0;
        RFDETR_CUDA_CHECK(cudaEventElapsedTime(&gpu_el, ev0, ev1));

        wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        gpu_ms.push_back(static_cast<double>(gpu_el));
        total_det += nd;

        if (cfg.verbose) bar.update(i + 1);
    }
    if (cfg.verbose) bar.finish();

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    m.latency     = LatencyStats::compute(wall_ms);
    m.gpu_latency = LatencyStats::compute(gpu_ms);
    m.fps         = 1000.0 / m.latency.avg;
    m.frame_count = cfg.iters;
    m.avg_detections = total_det / cfg.iters;
    if (!cpu_s.empty())
        m.cpu_usage_avg = std::accumulate(cpu_s.begin(), cpu_s.end(), 0.0) / cpu_s.size();
    if (!gpu_s.empty())
        m.gpu_usage_avg = std::accumulate(gpu_s.begin(), gpu_s.end(), 0.0) / gpu_s.size();
    m.gpu_memory_mb = gpu_mem_first;

    return m;
}

static void print_metrics(const BenchmarkConfig& cfg, const BenchmarkMetrics& m,
                          const rfdetr::EngineMeta& meta, bool have_meta,
                          const std::string& input_type) {
    using namespace colors;
    std::cout << "\n" << c(std::string(72, '='), CYAN) << "\n";
    std::cout << c("  RF-DETR BENCHMARK RESULTS", BOLD) << "\n";
    std::cout << c(std::string(72, '='), CYAN) << "\n\n";

    std::cout << c("MODEL", YELLOW) << "\n" << c(std::string(36, '-'), DIM) << "\n";
    std::cout << "  Engine:      " << c(cfg.engine, GREEN) << "\n";
    std::cout << "  Variant:     " << (have_meta ? meta.variant : "?") << "\n";
    std::cout << "  Precision:   " << (have_meta ? meta.precision : "?") << "\n";
    std::cout << "  Task:        " << cfg.task << "\n";
    std::cout << "  Input type:  " << input_type << "\n";
    std::cout << "  Device:      " << c(cfg.device, GREEN) << "\n";
    std::cout << "  TRT:         " << trt_ver() << "  CUDA: " << cuda_ver()
              << "  Driver: " << driver_ver() << "\n\n";

    std::cout << c("PERFORMANCE", YELLOW) << "\n" << c(std::string(36, '-'), DIM) << "\n";
    std::cout << std::fixed;
    std::cout << "  Warmup:      " << std::setprecision(1) << m.warmup_ms << " ms  ("
              << cfg.warmup << " iters)\n";
    std::cout << "  Frames:      " << m.frame_count << "\n";
    std::cout << "  FPS:         " << c(std::to_string(static_cast<int>(m.fps)), GREEN) << "\n\n";

    std::cout << c("LATENCY (wall — preprocess + infer + postprocess)  ms", YELLOW) << "\n";
    std::cout << c(std::string(36, '-'), DIM) << "\n";
    std::cout << "  Avg:   " << std::setprecision(3) << m.latency.avg
              << "   StdDev: " << m.latency.stddev << "\n";
    std::cout << "  Min:   " << m.latency.min << "   Max: " << m.latency.max << "\n";
    std::cout << "  P50:   " << m.latency.p50 << "   P90: " << m.latency.p90
              << "   P95: " << m.latency.p95 << "   P99: " << m.latency.p99 << "\n\n";

    std::cout << c("LATENCY (GPU only — cudaEvent)  ms", YELLOW) << "\n";
    std::cout << c(std::string(36, '-'), DIM) << "\n";
    std::cout << "  Avg:   " << m.gpu_latency.avg
              << "   StdDev: " << m.gpu_latency.stddev << "\n";
    std::cout << "  P50:   " << m.gpu_latency.p50 << "   P90: " << m.gpu_latency.p90
              << "   P99: " << m.gpu_latency.p99 << "\n\n";

    std::cout << c("RESOURCE USAGE", YELLOW) << "\n" << c(std::string(36, '-'), DIM) << "\n";
    std::cout << "  CPU usage avg:  " << std::setprecision(1) << m.cpu_usage_avg << "%\n";
    std::cout << "  GPU usage avg:  " << m.gpu_usage_avg << "%\n";
    std::cout << "  GPU memory:     " << m.gpu_memory_mb << " MB\n";
    std::cout << "  Avg detections: " << m.avg_detections << "\n";

    std::cout << "\n" << c(std::string(72, '='), CYAN) << "\n\n";
}

static nlohmann::json lat_json(const LatencyStats& s) {
    return {{"avg", s.avg}, {"stddev", s.stddev}, {"min", s.min}, {"max", s.max},
            {"p50", s.p50}, {"p90", s.p90}, {"p95", s.p95}, {"p99", s.p99}};
}

static void export_json(const std::string& path, const BenchmarkConfig& cfg,
                        const BenchmarkMetrics& m, const rfdetr::EngineMeta& meta,
                        bool have_meta, const std::string& input_type) {
    fs::create_directories(fs::path(path).parent_path());
    nlohmann::json j;
    j["device"]         = cfg.device;
    j["variant"]        = have_meta ? meta.variant   : "unknown";
    j["precision"]      = have_meta ? meta.precision : "unknown";
    j["task"]           = cfg.task;
    j["input_type"]     = input_type;
    j["batch"]          = 1;
    j["trt_version"]    = trt_ver();
    j["cuda_version"]   = cuda_ver();
    j["driver_version"] = driver_ver();
    j["warmup_iters"]   = cfg.warmup;
    j["measure_iters"]  = cfg.iters;
    j["latency_ms"]     = lat_json(m.latency);
    j["gpu_ms"]         = lat_json(m.gpu_latency);
    j["throughput_fps"] = m.fps;
    j["cpu_usage_avg"]  = m.cpu_usage_avg;
    j["gpu_usage_avg"]  = m.gpu_usage_avg;
    j["gpu_memory_mb"]  = m.gpu_memory_mb;
    j["avg_detections"] = m.avg_detections;

    std::ofstream f(path);
    f << j.dump(2) << "\n";
    std::printf("  JSON written -> %s\n", path.c_str());
}

static void export_csv(const std::string& path, const BenchmarkConfig& cfg,
                       const BenchmarkMetrics& m, const rfdetr::EngineMeta& meta,
                       bool have_meta, const std::string& input_type) {
    fs::create_directories(fs::path(path).parent_path());
    bool exists = fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!exists) {
        f << "timestamp,model_type,task_type,input_type,device,precision,input_shape,"
             "load_ms,warmup_ms,frames,"
             "latency_avg,latency_stddev,latency_min,latency_max,"
             "latency_p50,latency_p90,latency_p95,latency_p99,"
             "gpu_latency_avg,gpu_latency_p50,gpu_latency_p99,"
             "fps,throughput,"
             "cpu_usage,gpu_usage,gpu_memory_mb,"
             "avg_detections,AP50,mAP5095,"
             "trt_version,cuda_version\n";
    }

    std::string input_shape = "?";
    if (have_meta)
        input_shape = std::to_string(meta.input_h) + "x" + std::to_string(meta.input_w);

    f << std::time(nullptr) << ","
      << (have_meta ? meta.variant   : "unknown") << ","
      << cfg.task << ","
      << input_type << ","
      << cfg.device << ","
      << (have_meta ? meta.precision : "unknown") << ","
      << input_shape << ","
      << std::fixed << std::setprecision(3)
      << m.load_ms << "," << m.warmup_ms << "," << m.frame_count << ","
      << m.latency.avg    << "," << m.latency.stddev << ","
      << m.latency.min    << "," << m.latency.max    << ","
      << m.latency.p50    << "," << m.latency.p90    << ","
      << m.latency.p95    << "," << m.latency.p99    << ","
      << m.gpu_latency.avg << "," << m.gpu_latency.p50 << "," << m.gpu_latency.p99 << ","
      << std::setprecision(2)
      << m.fps << "," << m.fps << ","    // throughput == fps for batch=1
      << std::setprecision(1)
      << m.cpu_usage_avg << "," << m.gpu_usage_avg << "," << m.gpu_memory_mb << ","
      << m.avg_detections << ","
      << "N/A,N/A,"
      << trt_ver() << "," << cuda_ver() << "\n";
    std::printf("  CSV appended -> %s\n", path.c_str());
}

static int run_det(rfdetr::RFDetrDetector& det, const cv::Mat& f, float thr) {
    return static_cast<int>(det.detect(f, thr).size());
}

static int run_seg(rfdetr::RFDetrSegmenter& seg, const cv::Mat& f, float thr) {
    return static_cast<int>(seg.segment(f, thr).size());
}

// Dispatch helper: builds either a detector or segmenter and returns a
// closure invoking the right run_* per frame. Keeps the bench_* code paths
// identical between tasks.
template <typename RunFn>
BenchmarkMetrics run_with_engine(const BenchmarkConfig& cfg,
                                  const std::vector<cv::Mat>& frames,
                                  RunFn&& configure) {
    if (cfg.task == "segmentation") {
        rfdetr::RFDetrSegmenter seg(cfg.engine);
        return configure([&](const cv::Mat& f) { return run_seg(seg, f, cfg.threshold); });
    }
    rfdetr::RFDetrDetector det(cfg.engine);
    return configure([&](const cv::Mat& f) { return run_det(det, f, cfg.threshold); });
}

static BenchmarkMetrics bench_image(const BenchmarkConfig& cfg, const cv::Mat& frame) {
    const int pool = std::max(1, std::min(8, cfg.iters));
    std::vector<cv::Mat> frames(static_cast<std::size_t>(pool), frame);

    auto t0 = std::chrono::steady_clock::now();
    BenchmarkMetrics m = run_with_engine(cfg, frames, [&](auto infer) {
        return run_loop(infer, frames, cfg);
    });
    m.load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() - m.warmup_ms - m.latency.avg * cfg.iters;
    return m;
}

static BenchmarkMetrics bench_video(const BenchmarkConfig& cfg, const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + path);

    std::vector<cv::Mat> frames;
    cv::Mat f;
    while (static_cast<int>(frames.size()) < cfg.max_video_frames && cap.read(f))
        if (!f.empty()) frames.push_back(f.clone());

    if (frames.empty()) throw std::runtime_error("no frames in video");
    std::printf("  Loaded %zu frames from video\n", frames.size());

    return run_with_engine(cfg, frames, [&](auto infer) {
        return run_loop(infer, frames, cfg);
    });
}

static BenchmarkMetrics bench_camera(const BenchmarkConfig& cfg, int cam_id) {
    cv::VideoCapture cap(cam_id);
    if (!cap.isOpened())
        throw std::runtime_error("cannot open camera " + std::to_string(cam_id));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::vector<cv::Mat> frames;
    cv::Mat f;
    std::printf("  Capturing %d seconds from camera %d...\n", cfg.camera_seconds, cam_id);
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(cfg.camera_seconds);
    while (std::chrono::steady_clock::now() < end && cap.read(f))
        if (!f.empty()) frames.push_back(f.clone());

    if (frames.empty()) throw std::runtime_error("no frames from camera");

    return run_with_engine(cfg, frames, [&](auto infer) {
        return run_loop(infer, frames, cfg);
    });
}

static void usage(const char* prog) {
    std::fprintf(stderr,
        "RF-DETR Benchmark Suite\n\n"
        "Usage:\n"
        "  %s quick  <engine> <image>  [options]  — quick run with defaults\n"
        "  %s image  <engine> <image>  [options]\n"
        "  %s video  <engine> <video>  [options]\n"
        "  %s camera <engine> <cam_id> [options]\n\n"
        "Options:\n"
        "  --task TYPE        detection (default) | segmentation\n"
        "  --device TAG       Label for CSV/JSON (default: unknown)\n"
        "  --warmup N         Warm-up iterations (default: 50)\n"
        "  --iters N          Measurement iterations (default: 500)\n"
        "  --threshold F      Detection threshold (default: 0.5)\n"
        "  --output-dir PATH  Results directory (default: benchmarks/results)\n"
        "  --json             Write per-run JSON file\n"
        "  --verbose / --quiet\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") {
        usage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string mode = argv[1];

    // Quick mode: rfdetr_bench quick <engine> <image>
    if (mode == "quick") {
        if (argc < 4) { usage(argv[0]); return 1; }
        BenchmarkConfig cfg;
        cfg.engine  = argv[2];
        cfg.warmup  = 10;
        cfg.iters   = 100;
        cfg.verbose = true;
        // parse any extra flags
        for (int i = 4; i < argc; ++i) {
            std::string_view a = argv[i];
            if (starts_with(a, "--device"))    cfg.device = next_value(argc, argv, i, "--device");
            else if (starts_with(a, "--task")) cfg.task   = next_value(argc, argv, i, "--task");
        }
        std::string img_path = argv[3];
        cv::Mat frame = cv::imread(img_path);
        if (frame.empty()) { std::fprintf(stderr, "error: cannot read %s\n", img_path.c_str()); return 1; }

        rfdetr::EngineMeta meta; bool have_meta = false;
        std::string mp = cfg.engine + ".json";
        if (fs::is_regular_file(mp)) { meta = rfdetr::EngineMeta::from_json_file(mp); have_meta = true; }

        if (cfg.task == "detection" && have_meta && meta.has_masks) {
            cfg.task = "segmentation";
        }

        std::printf("\n%s\n  Engine: %s\n  Quick mode: %d warm-up, %d iters\n\n",
                    colors::c("RF-DETR Quick Benchmark", colors::BOLD).c_str(),
                    cfg.engine.c_str(), cfg.warmup, cfg.iters);

        auto m = bench_image(cfg, frame);
        print_metrics(cfg, m, meta, have_meta, "image");
        return 0;
    }

    // Standard modes: image / video / camera
    if (mode != "image" && mode != "video" && mode != "camera") {
        std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
        usage(argv[0]);
        return 1;
    }
    if (argc < 4) { usage(argv[0]); return 1; }

    BenchmarkConfig cfg;
    cfg.engine  = argv[2];
    std::string input = argv[3];

    for (int i = 4; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--verbose")                    cfg.verbose = true;
        else if (a == "--quiet")                 cfg.verbose = false;
        else if (a == "--json")                  cfg.json_output = true;
        else if (starts_with(a, "--device"))     cfg.device     = next_value(argc, argv, i, "--device");
        else if (starts_with(a, "--task"))       cfg.task       = next_value(argc, argv, i, "--task");
        else if (starts_with(a, "--warmup"))     cfg.warmup     = parse_int(next_value(argc, argv, i, "--warmup"), "--warmup");
        else if (starts_with(a, "--iters"))      cfg.iters      = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
        else if (starts_with(a, "--threshold"))  cfg.threshold  = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
        else if (starts_with(a, "--output-dir")) cfg.output_dir = next_value(argc, argv, i, "--output-dir");
        else { std::fprintf(stderr, "unknown flag: %s\n", std::string(a).c_str()); return 1; }
    }

    rfdetr::EngineMeta meta; bool have_meta = false;
    std::string mp = cfg.engine + ".json";
    if (fs::is_regular_file(mp)) { meta = rfdetr::EngineMeta::from_json_file(mp); have_meta = true; }

    // Auto-detect task from the engine sidecar if --task wasn't explicitly set.
    if (cfg.task == "detection" && have_meta && meta.has_masks) {
        cfg.task = "segmentation";
    }
    if (cfg.task != "detection" && cfg.task != "segmentation") {
        std::fprintf(stderr, "error: --task must be 'detection' or 'segmentation', got '%s'\n",
                     cfg.task.c_str());
        return 1;
    }

    std::printf("\n%s\n", colors::c(std::string(72, '='), colors::CYAN).c_str());
    std::printf("  %s\n", colors::c("RF-DETR BENCHMARK", colors::BOLD).c_str());
    std::printf("%s\n\n", colors::c(std::string(72, '='), colors::CYAN).c_str());
    std::printf("  Engine:    %s\n", cfg.engine.c_str());
    std::printf("  Mode:      %s\n", mode.c_str());
    std::printf("  Task:      %s\n", cfg.task.c_str());
    std::printf("  Input:     %s\n", input.c_str());
    std::printf("  Device:    %s\n", cfg.device.c_str());
    std::printf("  Warmup:    %d   Iters: %d\n\n", cfg.warmup, cfg.iters);

    BenchmarkMetrics m;
    try {
        if (mode == "image") {
            cv::Mat frame = cv::imread(input);
            if (frame.empty()) { std::fprintf(stderr, "error: cannot read %s\n", input.c_str()); return 1; }
            m = bench_image(cfg, frame);
        } else if (mode == "video") {
            m = bench_video(cfg, input);
        } else {
            m = bench_camera(cfg, parse_int(input.c_str(), "camera"));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    print_metrics(cfg, m, meta, have_meta, mode);

    // CSV always
    std::string stem = (have_meta ? meta.variant : "model") + "_" + (have_meta ? meta.precision : "fp32");
    export_csv(cfg.output_dir + "/benchmark_results.csv", cfg, m, meta, have_meta, mode);

    // JSON if requested
    if (cfg.json_output) {
        std::string jpath = cfg.output_dir + "/" + cfg.device + "_" + stem + "_" +
                            std::to_string(std::time(nullptr)) + ".json";
        export_json(jpath, cfg, m, meta, have_meta, mode);
    }

    return 0;
}
