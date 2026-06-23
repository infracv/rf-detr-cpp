// rfdetr_video — run RF-DETR detection on a video file or camera stream.
//
// usage:
//   rfdetr_video --engine ENGINE --input INPUT
//                [--threshold 0.5] [--out PATH] [--show] [--fps-overlay]
//                [--skip N] [--max-frames N]
//
// INPUT can be:
//   - a video file path  (e.g. video.mp4)
//   - a camera index     (e.g. 0, 1)
//   - an RTSP/HTTP URL

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/version.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
    std::string           input;          // file path, camera index, or URL
    std::filesystem::path out;            // output video file (empty = no write)
    float threshold{0.5f};
    bool  show{false};                    // display window
    bool  fps_overlay{true};             // draw FPS/latency HUD
    int   skip{0};                        // process every (skip+1)-th frame
    int   max_frames{0};                  // 0 = unlimited
    bool  cuda_graph{false};
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s --engine ENGINE --input INPUT [options]\n"
        "\n"
        "  --engine  ENGINE      TensorRT engine file\n"
        "  --input   INPUT       Video file, camera index (0,1,...), or RTSP URL\n"
        "  --threshold 0.5       Detection score threshold\n"
        "  --out     PATH        Write annotated video to file\n"
        "  --show                Open a display window (requires GUI)\n"
        "  --no-fps-overlay      Disable FPS/latency HUD\n"
        "  --skip    N           Skip N frames between detections (default: 0)\n"
        "  --max-frames N        Stop after N frames (default: unlimited)\n"
        "  --cuda-graph          Attempt CUDA Graph capture\n"
        "\n"
        "  rfdetr v%s\n",
        a0, rfdetr::version());
}

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

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if      (arg == "-h" || arg == "--help")    { usage(argv[0]); std::exit(0); }
        else if (starts_with(arg, "--engine"))      a.engine    = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--input"))       a.input     = next_value(argc, argv, i, "--input");
        else if (starts_with(arg, "--out"))         a.out       = next_value(argc, argv, i, "--out");
        else if (starts_with(arg, "--threshold"))   a.threshold = std::stof(next_value(argc, argv, i, "--threshold"));
        else if (starts_with(arg, "--skip"))        a.skip      = std::atoi(next_value(argc, argv, i, "--skip"));
        else if (starts_with(arg, "--max-frames"))  a.max_frames= std::atoi(next_value(argc, argv, i, "--max-frames"));
        else if (arg == "--show")                   a.show       = true;
        else if (arg == "--no-fps-overlay")         a.fps_overlay= false;
        else if (arg == "--cuda-graph")             a.cuda_graph = true;
        else throw std::runtime_error("unknown arg: " + std::string(arg));
    }
    if (a.engine.empty() || a.input.empty()) { usage(argv[0]); std::exit(2); }
    return a;
}

// Open a cv::VideoCapture from either a numeric camera index or a file/URL.
cv::VideoCapture open_capture(const std::string& input) {
    // Check if the string is a pure integer (camera index).
    bool is_index = !input.empty();
    for (char c : input) {
        if (c < '0' || c > '9') { is_index = false; break; }
    }
    if (is_index) {
        cv::VideoCapture cap(std::stoi(input));
        return cap;
    }
    return cv::VideoCapture(input);
}

// Draw a semi-transparent HUD showing FPS and per-phase latency.
void draw_hud(cv::Mat& frame, double fps, const rfdetr::RFDetrDetector::Timings& t,
              int det_count) {
    const std::string line1 = cv::format("%.1f FPS  |  total %.2f ms", fps, t.total_ms);
    const std::string line2 = cv::format("pre %.2f  infer %.2f  post %.2f ms",
                                          t.preprocess_ms, t.infer_ms, t.postprocess_ms);
    const std::string line3 = cv::format("detections: %d", det_count);

    const int font     = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.55;
    const int thick    = 1;
    int baseline       = 0;

    auto text_sz = [&](const std::string& s) {
        return cv::getTextSize(s, font, scale, thick, &baseline);
    };

    const cv::Size s1 = text_sz(line1);
    const cv::Size s2 = text_sz(line2);
    const cv::Size s3 = text_sz(line3);
    const int pad = 6;
    const int box_w = std::max({s1.width, s2.width, s3.width}) + pad * 2;
    const int box_h = s1.height + s2.height + s3.height + baseline * 3 + pad * 4;

    // Semi-transparent background.
    cv::Mat roi = frame(cv::Rect(0, 0, std::min(box_w, frame.cols),
                                       std::min(box_h, frame.rows)));
    cv::Mat overlay = roi.clone();
    overlay.setTo(cv::Scalar(0, 0, 0));
    cv::addWeighted(overlay, 0.5, roi, 0.5, 0, roi);

    const cv::Scalar white(255, 255, 255);
    int y = pad + s1.height;
    cv::putText(frame, line1, {pad, y}, font, scale, white, thick, cv::LINE_AA);
    y += s2.height + pad;
    cv::putText(frame, line2, {pad, y}, font, scale, white, thick, cv::LINE_AA);
    y += s3.height + pad;
    cv::putText(frame, line3, {pad, y}, font, scale, white, thick, cv::LINE_AA);
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
        // Open source.
        cv::VideoCapture cap = open_capture(args.input);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "error: cannot open input: %s\n", args.input.c_str());
            return 1;
        }

        const int src_w  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int src_h  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double src_fps = cap.get(cv::CAP_PROP_FPS);
        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::printf("input:   %s\n", args.input.c_str());
        std::printf("source:  %dx%d @ %.1f fps  frames=%d\n",
                    src_w, src_h, src_fps, total_frames);

        // Build detector.
        rfdetr::DetectorOptions opts;
        opts.threshold      = args.threshold;
        opts.use_cuda_graph = args.cuda_graph;
        rfdetr::RFDetrDetector detector(args.engine, opts);

        std::printf("engine:  %s  variant=%s  input=%dx%d\n",
                    args.engine.c_str(), detector.variant().c_str(),
                    detector.input_w(), detector.input_h());

        // Optional output writer.
        cv::VideoWriter writer;
        if (!args.out.empty()) {
            // Use src_fps for output; fall back to 25 if source doesn't report it.
            const double out_fps = (src_fps > 0.0) ? src_fps : 25.0;
            const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(args.out.string(), fourcc, out_fps, {src_w, src_h});
            if (!writer.isOpened()) {
                std::fprintf(stderr, "warning: cannot open output writer: %s\n",
                             args.out.c_str());
            } else {
                std::printf("output:  %s  (%.1f fps)\n", args.out.c_str(), out_fps);
            }
        }

        if (args.show) {
            cv::namedWindow("rfdetr", cv::WINDOW_NORMAL);
        }

        // Warm up.
        {
            cv::Mat tmp;
            if (cap.read(tmp) && !tmp.empty()) {
                detector.detect(tmp, args.threshold);
                // Rewind file; cameras can't rewind so just discard the warm-up frame.
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
        }

        using Clock = std::chrono::steady_clock;
        int  frame_idx   = 0;
        int  proc_count  = 0;
        rfdetr::Detections last_results;
        rfdetr::RFDetrDetector::Timings last_t;

        // Rolling FPS over the last 30 processed frames.
        auto fps_start = Clock::now();
        double display_fps = 0.0;

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;
            if (args.max_frames > 0 && frame_idx >= args.max_frames) break;

            const bool do_infer = (frame_idx % (args.skip + 1) == 0);

            if (do_infer) {
                last_results = detector.detect(frame, args.threshold);
                last_t       = detector.last_timings();
                ++proc_count;

                // Update FPS every 30 processed frames.
                if (proc_count % 30 == 0) {
                    auto now = Clock::now();
                    display_fps = 30.0 /
                        std::chrono::duration<double>(now - fps_start).count();
                    fps_start = now;
                }
            }

            // Draw detections on frame copy (avoids modifying source for writer).
            rfdetr::draw_detections(frame, last_results, &rfdetr::coco_label);
            if (args.fps_overlay) {
                draw_hud(frame, display_fps, last_t,
                         static_cast<int>(last_results.size()));
            }

            if (writer.isOpened()) writer.write(frame);
            if (args.show) {
                cv::imshow("rfdetr", frame);
                // q or ESC to quit.
                const int key = cv::waitKey(1);
                if (key == 'q' || key == 27) break;
            }

            ++frame_idx;

            // Progress every 100 frames for file sources.
            if (total_frames > 0 && frame_idx % 100 == 0) {
                std::printf("  frame %d/%d  fps=%.1f  last_latency=%.2f ms\n",
                            frame_idx, total_frames, display_fps, last_t.total_ms);
            }
        }

        std::printf("done: processed %d / %d frames\n", proc_count, frame_idx);
        if (writer.isOpened()) {
            writer.release();
            std::printf("wrote %s\n", args.out.c_str());
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
