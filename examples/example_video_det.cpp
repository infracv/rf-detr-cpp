// example_video_det — RF-DETR detection on a video file.
//
// Usage:
//   ./example_video_det <engine> <video> [threshold]
//
// Example:
//   ./example_video_det rf-detr-nano-fp16.engine video.mp4 0.5

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "utils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine> <video> [threshold=0.5]\n", argv[0]);
        return 1;
    }

    const std::string engine_path = argv[1];
    const std::string video_path  = argv[2];
    const float threshold = (argc >= 4) ? std::stof(argv[3]) : 0.5f;

    try {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "error: cannot open video: %s\n", video_path.c_str());
            return 1;
        }

        const int    width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps    = cap.get(cv::CAP_PROP_FPS);
        const int    total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::printf("video:   %s  %dx%d @ %.1f fps  frames=%d\n",
                    video_path.c_str(), width, height, fps, total);

        rfdetr::RFDetrDetector detector(engine_path);
        std::printf("variant: %s  input=%dx%d\n",
                    detector.variant().c_str(), detector.input_w(), detector.input_h());

        // Output writer
        const std::string out_path = "out/video_det_" + utils::timestamp() + ".mp4";
        std::filesystem::create_directories("out");
        cv::VideoWriter writer(out_path,
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               fps > 0 ? fps : 25.0, {width, height});
        if (!writer.isOpened())
            std::fprintf(stderr, "warning: cannot open output writer: %s\n", out_path.c_str());
        else
            std::printf("output:  %s\n", out_path.c_str());

        // Warm-up
        cv::Mat tmp;
        if (cap.read(tmp)) detector.detect(tmp, threshold);
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        int frame_idx = 0;
        double total_ms = 0.0;
        cv::Mat frame;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            const auto t0 = std::chrono::steady_clock::now();
            rfdetr::Detections results = detector.detect(frame, threshold);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;
            ++frame_idx;

            rfdetr::draw_detections(frame, results, &rfdetr::coco_label);

            // FPS overlay
            const std::string hud = cv::format("FPS: %.1f  dets: %zu",
                                                1000.0 / ms, results.size());
            cv::putText(frame, hud, {8, 24}, cv::FONT_HERSHEY_SIMPLEX,
                        0.6, {0, 0, 0}, 2, cv::LINE_AA);
            cv::putText(frame, hud, {8, 24}, cv::FONT_HERSHEY_SIMPLEX,
                        0.6, {255, 255, 255}, 1, cv::LINE_AA);

            if (writer.isOpened()) writer.write(frame);

            if (frame_idx % 100 == 0)
                std::printf("  frame %d/%d  avg %.2f ms\n",
                            frame_idx, total, total_ms / frame_idx);
        }

        if (frame_idx > 0)
            utils::print_metrics("video_det", total_ms / frame_idx, 1000.0 / (total_ms / frame_idx));
        std::printf("processed %d frames\n", frame_idx);
        if (writer.isOpened()) {
            writer.release();
            std::printf("saved: %s\n", out_path.c_str());
        }

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
