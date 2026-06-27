// example_camera_det — RF-DETR live detection from a camera or RTSP stream.
//
// Usage:
//   ./example_camera_det <engine> [camera_id=0] [threshold=0.5]
//
// Example:
//   ./example_camera_det rf-detr-nano-fp16.engine 0 0.5
//
// Controls:
//   q / ESC  — quit
//   s        — save current frame to outputs/

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "utils.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <engine> [camera_id=0] [threshold=0.5]\n", argv[0]);
        return 1;
    }

    const std::string engine_path = argv[1];
    const int camera_id = (argc >= 3) ? std::atoi(argv[2]) : 0;
    const float threshold = (argc >= 4) ? std::stof(argv[3]) : 0.5f;

    try {
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "error: cannot open camera %d\n", camera_id);
            return 1;
        }
        std::printf("camera:  %d  %dx%d\n",
                    camera_id,
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                    static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        rfdetr::RFDetrDetector detector(engine_path);
        std::printf("variant: %s  input=%dx%d\n",
                    detector.variant().c_str(), detector.input_w(), detector.input_h());
        std::printf("controls: q/ESC = quit  |  s = save snapshot\n");

        cv::namedWindow("rf-detr", cv::WINDOW_NORMAL);

        // Rolling FPS over last 30 frames
        using Clock = std::chrono::steady_clock;
        auto fps_tp = Clock::now();
        double display_fps = 0.0;
        int proc = 0;

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            const auto t0 = Clock::now();
            rfdetr::Detections results = detector.detect(frame, threshold);
            const auto t1 = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++proc;

            if (proc % 30 == 0) {
                display_fps = 30.0 / std::chrono::duration<double>(t1 - fps_tp).count();
                fps_tp = t1;
            }

            rfdetr::draw_detections(frame, results, &rfdetr::coco_label);

            // HUD
            const std::string hud = cv::format("%.1f FPS  %.2f ms  dets: %zu",
                                                display_fps, ms, results.size());
            cv::putText(frame, hud, {8, 28}, cv::FONT_HERSHEY_SIMPLEX,
                        0.65, {0, 0, 0}, 2, cv::LINE_AA);
            cv::putText(frame, hud, {8, 28}, cv::FONT_HERSHEY_SIMPLEX,
                        0.65, {255, 255, 255}, 1, cv::LINE_AA);

            cv::imshow("rf-detr", frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
            if (key == 's') {
                const std::string saved = utils::save_image(frame, "camera_snapshot");
                std::printf("snapshot saved: %s\n", saved.c_str());
            }
        }

        std::printf("done: %d frames processed\n", proc);

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
