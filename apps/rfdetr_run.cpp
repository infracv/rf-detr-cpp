// rfdetr_run — full RF-DETR inference: image -> CUDA preprocess -> TRT engine ->
// CPU postprocess -> annotated output. End-to-end smoke test for steps 3 + 4.
//
// usage:
//   rfdetr_run --engine ENGINE --image IMAGE [--meta META]
//              [--threshold 0.5] [--out out.jpg] [--iters 1]

#include "cli_helpers.hpp"
#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/cuda_check.hpp"
#include "rfdetr/core/cuda_preprocess.cuh"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/postprocess.hpp"
#include "rfdetr/core/trt_session.hpp"
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
    std::filesystem::path meta;
    std::filesystem::path out;
    float threshold{0.5f};
    int   iters{1};
};

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s --engine ENGINE --image IMAGE [--meta META]\n"
                 "       [--threshold 0.5] [--out PATH] [--iters 1]\n"
                 "  rfdetr v%s\n",
                 a0, rfdetr::version());
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else if (starts_with(arg, "--engine"))    a.engine = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--image"))     a.image  = next_value(argc, argv, i, "--image");
        else if (starts_with(arg, "--meta"))      a.meta   = next_value(argc, argv, i, "--meta");
        else if (starts_with(arg, "--out"))       a.out    = next_value(argc, argv, i, "--out");
        else if (starts_with(arg, "--threshold")) a.threshold = parse_float(next_value(argc, argv, i, "--threshold"), "--threshold");
        else if (starts_with(arg, "--iters"))     a.iters  = parse_int(next_value(argc, argv, i, "--iters"), "--iters");
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty() || a.image.empty()) {
        usage(argv[0]);
        std::exit(2);
    }
    if (a.meta.empty()) a.meta = a.engine.string() + ".json";
    if (a.iters <= 0)   a.iters = 1;
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception& e) {
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
        std::printf("image: %s  %dx%d\n", args.image.c_str(), image.cols, image.rows);

        rfdetr::TrtSession sess(args.engine);
        rfdetr::EngineMeta meta;
        if (std::filesystem::is_regular_file(args.meta)) {
            meta = rfdetr::EngineMeta::from_json_file(args.meta);
            std::printf("meta:  variant=%s  HxW=%dx%d  num_queries=%d  num_classes=%d\n",
                        meta.variant.c_str(), meta.input_h, meta.input_w, meta.num_queries,
                        meta.num_classes);
        } else {
            std::fprintf(stderr, "warning: no meta sidecar at %s — assuming defaults\n",
                         args.meta.c_str());
            const auto* in_b = sess.find("input");
            if (!in_b || in_b->shape.nbDims < 4) {
                throw std::runtime_error("cannot infer input dims; pass --meta");
            }
            meta.input_h = in_b->shape.d[2];
            meta.input_w = in_b->shape.d[3];
            meta.num_classes = 90;
            meta.color_order = "RGB";
        }
        if (meta.input_h != meta.input_w) {
            throw std::runtime_error("non-square inputs are not supported yet");
        }

        const rfdetr::BindingInfo* b_dets   = sess.find("dets");
        const rfdetr::BindingInfo* b_labels = sess.find("labels");
        if (!b_dets || !b_labels) {
            const auto& outs = sess.output_indices();
            if (outs.size() < 2) throw std::runtime_error("engine has < 2 outputs");
            b_dets   = &sess.bindings()[outs[0]];
            b_labels = &sess.bindings()[outs[1]];
        }

        // shape: dets (B, N, 4); labels (B, N, C). C may be 91 (COCO) or num_classes+1.
        const int N = b_dets->shape.d[b_dets->shape.nbDims - 2];
        const int C = b_labels->shape.d[b_labels->shape.nbDims - 1];
        std::printf("output: dets=[*,%d,4]  labels=[*,%d,%d]\n", N, N, C);

        rfdetr::ImagePreprocessor prep(meta.input_h,
                                       /*bgr_to_rgb=*/meta.color_order == "RGB");
        prep.set_mean(meta.mean[0], meta.mean[1], meta.mean[2]);
        prep.set_std(meta.std[0], meta.std[1], meta.std[2]);

        float* d_input = static_cast<float*>(sess.device_buffer("input"));
        std::vector<float> dets(static_cast<std::size_t>(N) * 4);
        std::vector<float> labels(static_cast<std::size_t>(N) * static_cast<std::size_t>(C));

        rfdetr::PostprocessParams pp;
        pp.num_queries          = N;
        pp.num_classes_with_bg  = C;
        pp.topk                 = N;
        pp.threshold            = args.threshold;
        pp.bg_class_index       = 0;

        // Warm-up.
        prep.process(image, d_input, sess.stream());
        sess.infer();
        sess.get_output_f32(b_dets->name,   dets.data(),   dets.size());
        sess.get_output_f32(b_labels->name, labels.data(), labels.size());

        rfdetr::Detections results;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < args.iters; ++k) {
            prep.process(image, d_input, sess.stream());
            sess.infer();
            sess.get_output_f32(b_dets->name,   dets.data(),   dets.size());
            sess.get_output_f32(b_labels->name, labels.data(), labels.size());
            results = rfdetr::decode_detections(dets.data(), labels.data(), image.cols,
                                                image.rows, pp);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("infer x%d: total=%.3f ms  mean=%.3f ms/iter\n", args.iters, total_ms,
                    total_ms / args.iters);
        std::printf("detections (threshold=%.2f): %zu\n", args.threshold, results.size());

        for (const auto& d : results) {
            const char* name = rfdetr::coco_label(d.class_id);
            std::printf("  %-15s %.3f  bbox=[%.1f, %.1f, %.1f, %.1f]\n", name, d.score,
                        d.box.x1, d.box.y1, d.box.x2, d.box.y2);
        }

        if (!args.out.empty()) {
            rfdetr::draw_detections(image, results, &rfdetr::coco_label);
            std::filesystem::create_directories(args.out.parent_path().empty()
                                                ? std::filesystem::path(".")
                                                : args.out.parent_path());
            if (!cv::imwrite(args.out.string(), image)) {
                std::fprintf(stderr, "error: cv::imwrite failed for %s\n", args.out.c_str());
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
