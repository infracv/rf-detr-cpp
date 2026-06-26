// rfdetr_batch — batch inference demo for RF-DETR.
//
// Repeats a single test image N times to fill the batch, runs the engine once,
// and measures throughput. Use an engine built with --max-batch N.
//
// usage:
//   rfdetr_batch --engine ENGINE --image IMAGE --batch N
//                [--threshold 0.5] [--iters 20] [--out-dir DIR]

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
#include <vector>

namespace {

struct Args {
    std::filesystem::path engine;
    std::filesystem::path image;
    std::filesystem::path out_dir;
    float threshold{0.5f};
    int   batch{2};
    int   iters{20};
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s --engine ENGINE --image IMAGE --batch N [options]\n"
        "\n"
        "  --engine  ENGINE   TensorRT engine (built with --max-batch >= N)\n"
        "  --image   IMAGE    Test image to replicate across the batch\n"
        "  --batch   N        Batch size (default: 2)\n"
        "  --threshold 0.5    Detection score threshold\n"
        "  --iters   N        Timing iterations (default: 20)\n"
        "  --out-dir DIR      Write annotated images to DIR/batch_0.jpg ...\n"
        "\n"
        "  rfdetr v%s\n",
        a0, rfdetr::version());
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if      (arg == "-h" || arg == "--help")   { usage(argv[0]); std::exit(0); }
        else if (starts_with(arg, "--engine"))     a.engine    = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--image"))      a.image     = next_value(argc, argv, i, "--image");
        else if (starts_with(arg, "--out-dir"))    a.out_dir   = next_value(argc, argv, i, "--out-dir");
        else if (starts_with(arg, "--threshold"))  a.threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
        else if (starts_with(arg, "--batch"))      a.batch     = parse_int(next_value(argc, argv, i, "--batch"), "--batch");
        else if (starts_with(arg, "--iters"))      a.iters     = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty() || a.image.empty()) { usage(argv[0]); std::exit(2); }
    if (a.batch <= 0) throw std::runtime_error("--batch must be >= 1");
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
        cv::Mat src = cv::imread(args.image.string(), cv::IMREAD_COLOR);
        if (src.empty()) {
            std::fprintf(stderr, "error: cannot read image: %s\n", args.image.c_str());
            return 1;
        }

        // Fill batch with copies of the same image.
        std::vector<cv::Mat> batch(static_cast<std::size_t>(args.batch), src);

        rfdetr::DetectorOptions opts;
        opts.threshold = args.threshold;
        rfdetr::RFDetrDetector detector(args.engine, opts);

        std::printf("engine:  %s  variant=%s  input=%dx%d\n",
                    args.engine.c_str(), detector.variant().c_str(),
                    detector.input_w(), detector.input_h());
        std::printf("image:   %s  %dx%d\n", args.image.c_str(), src.cols, src.rows);
        std::printf("batch:   %d\n", args.batch);

        // Warm-up.
        detector.detect_batch(batch, args.threshold);

        // Timing loop.
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        std::vector<rfdetr::Detections> results;
        for (int k = 0; k < args.iters; ++k) {
            results = detector.detect_batch(batch, args.threshold);
        }
        const auto t1 = Clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double per_batch_ms = total_ms / args.iters;
        const double per_image_ms = per_batch_ms / args.batch;
        const double throughput   = 1000.0 / per_image_ms;

        std::printf("\nbatch=%d  iters=%d\n", args.batch, args.iters);
        std::printf("  total        : %.3f ms\n",  total_ms);
        std::printf("  per batch    : %.3f ms\n",  per_batch_ms);
        std::printf("  per image    : %.3f ms\n",  per_image_ms);
        std::printf("  throughput   : %.1f img/s\n", throughput);

        // Print detections for first image in batch.
        std::printf("\ndetections[0] (threshold=%.2f): %zu\n",
                    args.threshold, results[0].size());
        for (const auto& d : results[0]) {
            std::printf("  %-15s %.3f  bbox=[%.1f, %.1f, %.1f, %.1f]\n",
                        rfdetr::coco_label(d.class_id), d.score,
                        d.box.x1, d.box.y1, d.box.x2, d.box.y2);
        }

        // Optionally write annotated images.
        if (!args.out_dir.empty()) {
            std::filesystem::create_directories(args.out_dir);
            for (int i = 0; i < args.batch; ++i) {
                cv::Mat annotated = batch[static_cast<std::size_t>(i)].clone();
                rfdetr::draw_detections(annotated, results[static_cast<std::size_t>(i)],
                                        &rfdetr::coco_label);
                const auto out_path = args.out_dir / cv::format("batch_%d.jpg", i);
                cv::imwrite(out_path.string(), annotated);
                std::printf("wrote %s  (%zu dets)\n", out_path.c_str(),
                            results[static_cast<std::size_t>(i)].size());
            }
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
