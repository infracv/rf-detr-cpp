// rfdetr_seg — run RF-DETR segmentation on an image or video.
//
// Usage:
//   rfdetr_seg --engine trt-files/onnx/rf-detr-seg-nano-fp32.engine \
//              --image  test_img.jpg  --out out/seg_result.jpg
//
//   rfdetr_seg --engine trt-files/onnx/rf-detr-seg-nano-fp32.engine \
//              --video  test_video.mp4  --out out/seg_video.mp4

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/segmenter.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::string engine;
    std::string image;
    std::string video;
    std::string out;
    float       threshold{0.5f};
    int         iters{1};
    bool        show{false};
};

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

const char* next_value(int argc, char** argv, int& i, std::string_view flag) {
    std::string_view a = argv[i];
    if (a.size() > flag.size() + 1 && a[flag.size()] == '=')
        return argv[i] + flag.size() + 1;
    if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + std::string(flag));
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::fprintf(stderr,
                "usage: %s --engine PATH (--image PATH | --video PATH) [options]\n"
                "\n"
                "  --engine PATH      TRT engine built from a seg-* variant\n"
                "  --image  PATH      Input image (JPEG/PNG/...)\n"
                "  --video  PATH      Input video file (or camera index e.g. 0)\n"
                "  --out    PATH      Output image/video path\n"
                "  --threshold F      Score threshold (default 0.5)\n"
                "  --iters N          Repeat N times for timing (image mode only, default 1)\n"
                "  --show             Display window (press q/ESC to quit)\n",
                argv[0]);
            std::exit(0);
        }
        else if (starts_with(arg, "--engine"))    a.engine    = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--image"))     a.image     = next_value(argc, argv, i, "--image");
        else if (starts_with(arg, "--video"))     a.video     = next_value(argc, argv, i, "--video");
        else if (starts_with(arg, "--out"))       a.out       = next_value(argc, argv, i, "--out");
        else if (starts_with(arg, "--threshold")) a.threshold = std::stof(next_value(argc, argv, i, "--threshold"));
        else if (starts_with(arg, "--iters"))     a.iters     = std::atoi(next_value(argc, argv, i, "--iters"));
        else if (arg == "--show")                 a.show      = true;
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.image.empty() && a.video.empty()) throw std::runtime_error("--image or --video is required");
    return a;
}

// ------- image mode -------

void run_image(rfdetr::RFDetrSegmenter& seg, const Args& args) {
    cv::Mat image = cv::imread(args.image);
    if (image.empty()) throw std::runtime_error("cannot read image: " + args.image);

    rfdetr::Detections dets;
    for (int i = 0; i < args.iters; ++i) {
        dets = seg.segment(image, args.threshold);
    }

    const auto& t = seg.last_timings();
    std::printf("detections: %zu\n", dets.size());
    std::printf("timing (last iter): pre=%.2f ms  infer=%.2f ms  post=%.2f ms  total=%.2f ms\n",
                t.preprocess_ms, t.infer_ms, t.postprocess_ms, t.total_ms);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        std::printf("  [%zu] cls=%d (%s) score=%.3f box=[%.1f %.1f %.1f %.1f] mask=%s\n",
                    i, d.class_id,
                    rfdetr::coco_label(d.class_id),
                    d.score,
                    d.box.x1, d.box.y1, d.box.x2, d.box.y2,
                    d.mask.empty() ? "none" : "ok");
    }

    cv::Mat vis = image.clone();
    rfdetr::draw_segmentations(vis, dets, rfdetr::coco_label);

    if (!args.out.empty()) {
        std::filesystem::create_directories(
            std::filesystem::path(args.out).parent_path());
        cv::imwrite(args.out, vis);
        std::printf("wrote %s\n", args.out.c_str());
    }
    if (args.show) {
        cv::imshow("rfdetr_seg", vis);
        cv::waitKey(0);
    }
}

// ------- video mode -------

void draw_hud(cv::Mat& frame, double fps, double latency_ms, int det_count) {
    const std::string text1 = "FPS: " + std::to_string(static_cast<int>(fps + 0.5));
    const std::string text2 = std::to_string(static_cast<int>(latency_ms + 0.5)) + " ms";
    const std::string text3 = std::to_string(det_count) + " objs";
    constexpr int kFontFace = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double kScale = 0.55;
    int baseline = 0;
    const cv::Size s1 = cv::getTextSize(text1, kFontFace, kScale, 1, &baseline);
    const int pad = 6, lh = s1.height + pad + 2;
    const int bw = std::max({s1.width,
                              cv::getTextSize(text2, kFontFace, kScale, 1, &baseline).width,
                              cv::getTextSize(text3, kFontFace, kScale, 1, &baseline).width}) + pad * 2;
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(bw, lh * 3 + pad),
                  cv::Scalar(0, 0, 0, 180), cv::FILLED);
    auto put = [&](const std::string& t, int row) {
        cv::putText(frame, t, cv::Point(pad, row * lh + s1.height + pad / 2),
                    kFontFace, kScale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    };
    put(text1, 0); put(text2, 1); put(text3, 2);
}

void run_video(rfdetr::RFDetrSegmenter& seg, const Args& args) {
    cv::VideoCapture cap;
    // Numeric string -> camera index, else file path.
    bool is_num = !args.video.empty() &&
        args.video.find_first_not_of("0123456789") == std::string::npos;
    if (is_num) cap.open(std::stoi(args.video));
    else        cap.open(args.video);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + args.video);

    const double src_fps = cap.get(cv::CAP_PROP_FPS);
    const int    width   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int    height  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int    total   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer;
    if (!args.out.empty()) {
        std::filesystem::create_directories(
            std::filesystem::path(args.out).parent_path());
        writer.open(args.out, cv::VideoWriter::fourcc('m','p','4','v'),
                    src_fps > 0 ? src_fps : 30.0, cv::Size(width, height));
        if (!writer.isOpened())
            throw std::runtime_error("cannot open output video: " + args.out);
    }

    std::deque<double> fps_buf;
    constexpr int kFpsWindow = 30;
    int frame_idx = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        ++frame_idx;
        const auto t0 = std::chrono::steady_clock::now();
        auto dets = seg.segment(frame, args.threshold);
        const auto t1 = std::chrono::steady_clock::now();
        const double latency = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fps_buf.push_back(latency);
        if (static_cast<int>(fps_buf.size()) > kFpsWindow) fps_buf.pop_front();
        double avg = 0; for (auto v : fps_buf) avg += v;
        const double fps = fps_buf.empty() ? 0.0 : 1000.0 * fps_buf.size() / avg;

        rfdetr::draw_segmentations(frame, dets, rfdetr::coco_label);
        draw_hud(frame, fps, latency, static_cast<int>(dets.size()));

        if (writer.isOpened()) writer.write(frame);

        if (frame_idx % 100 == 0 || frame_idx == 1) {
            std::printf("  frame %d/%d  fps=%.1f  latency=%.2f ms\n",
                        frame_idx, total, fps, latency);
            std::fflush(stdout);
        }

        if (args.show) {
            cv::imshow("rfdetr_seg", frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        }
    }

    if (writer.isOpened()) {
        writer.release();
        std::printf("wrote %s\n", args.out.c_str());
    }
    std::printf("done: processed %d frames\n", frame_idx);
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

    try {
        rfdetr::SegmenterOptions opts;
        opts.threshold = args.threshold;
        rfdetr::RFDetrSegmenter seg(args.engine, opts);

        std::printf("[rfdetr_seg] engine  : %s\n", args.engine.c_str());
        std::printf("[rfdetr_seg] variant : %s\n", seg.variant().c_str());
        std::printf("[rfdetr_seg] input   : %dx%d\n", seg.input_w(), seg.input_h());
        std::printf("[rfdetr_seg] queries : %d   classes: %d\n",
                    seg.num_queries(), seg.num_classes());
        std::printf("[rfdetr_seg] mask sz : %dx%d\n", seg.mask_w(), seg.mask_h());

        if (!args.image.empty()) run_image(seg, args);
        else                     run_video(seg, args);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
