// rfdetr_detect — single-image detection via the RFDetrDetector public API.
//
// usage:
//   rfdetr_detect --engine ENGINE --image IMAGE
//                 [--threshold 0.5] [--out PATH] [--iters 1] [--cuda-graph]

#include "cli_helpers.hpp"
#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/version.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::filesystem::path engine;
    std::filesystem::path image;
    std::filesystem::path out;
    float threshold{0.5f};
    int   iters{1};
    bool  cuda_graph{false};
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s --engine ENGINE --image IMAGE\n"
                 "       [--threshold 0.5] [--out PATH] [--iters 1] [--cuda-graph]\n"
                 "  rfdetr v%s\n",
                 a0, rfdetr::version());
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help")    { usage(argv[0]); std::exit(0); }
        else if (starts_with(arg, "--engine"))    a.engine    = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--image"))     a.image     = next_value(argc, argv, i, "--image");
        else if (starts_with(arg, "--out"))       a.out       = next_value(argc, argv, i, "--out");
        else if (starts_with(arg, "--threshold")) a.threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
        else if (starts_with(arg, "--iters"))     a.iters     = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
        else if (arg == "--cuda-graph")           a.cuda_graph = true;
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty() || a.image.empty()) { usage(argv[0]); std::exit(2); }
    if (a.iters <= 0) a.iters = 1;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try { args = parse(argc, argv); }
    catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n", e.what());
        usage(argv[0]);
        return 2;
    }

    try {
        cv::Mat image = cv::imread(args.image.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::fprintf(stderr, "error: could not read image: %s\n", args.image.c_str());
            return 1;
        }

        rfdetr::DetectorOptions opts;
        opts.threshold      = args.threshold;
        opts.use_cuda_graph = args.cuda_graph;
        rfdetr::RFDetrDetector detector(args.engine, opts);

        std::printf("engine:  %s\n", args.engine.c_str());
        std::printf("variant: %s  input=%dx%d  queries=%d  classes=%d\n",
                    detector.variant().c_str(),
                    detector.input_w(), detector.input_h(),
                    detector.num_queries(), detector.num_classes());
        std::printf("image:   %s  %dx%d\n", args.image.c_str(), image.cols, image.rows);
        if (args.cuda_graph) std::printf("mode:    CUDA Graph\n");

        // Warm-up (also captures graph if --cuda-graph).
        detector.detect(image, args.threshold);

        rfdetr::Detections results;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < args.iters; ++k) {
            results = detector.detect(image, args.threshold);
        }
        const auto t1  = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const auto& t = detector.last_timings();
        std::printf("timing:  total=%.3f ms  pre=%.3f  infer=%.3f  post=%.3f\n",
                    t.total_ms, t.preprocess_ms, t.infer_ms, t.postprocess_ms);
        std::printf("loop x%d: total=%.3f ms  mean=%.3f ms/iter\n",
                    args.iters, ms, ms / args.iters);
        std::printf("detections (threshold=%.2f): %zu\n", args.threshold, results.size());

        for (const auto& d : results) {
            std::printf("  %-15s %.3f  bbox=[%.1f, %.1f, %.1f, %.1f]\n",
                        rfdetr::coco_label(d.class_id), d.score,
                        d.box.x1, d.box.y1, d.box.x2, d.box.y2);
        }

        if (!args.out.empty()) {
            rfdetr::draw_detections(image, results, &rfdetr::coco_label);
            std::filesystem::create_directories(args.out.parent_path().empty()
                                                ? std::filesystem::path(".")
                                                : args.out.parent_path());
            if (!cv::imwrite(args.out.string(), image)) {
                std::fprintf(stderr, "error: cv::imwrite failed: %s\n", args.out.c_str());
                return 1;
            }
            std::printf("wrote %s\n", args.out.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
