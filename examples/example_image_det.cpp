// example_image_det — minimal RF-DETR detection on a single image.
//
// Usage:
//   ./example_image_det <engine> <image> [threshold]
//
// Example:
//   ./example_image_det rf-detr-nano-fp16.engine dog.jpg 0.5

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/drawing.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "utils.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <engine> <image> [threshold=0.5]\n", argv[0]);
        return 1;
    }

    const std::string engine_path = argv[1];
    const std::string image_path  = argv[2];
    const float threshold = (argc >= 4) ? std::stof(argv[3]) : 0.5f;

    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::fprintf(stderr, "error: cannot read image: %s\n", image_path.c_str());
            return 1;
        }

        rfdetr::RFDetrDetector detector(engine_path);
        std::printf("variant: %s  input=%dx%d\n",
                    detector.variant().c_str(), detector.input_w(), detector.input_h());

        // Warm-up
        detector.detect(image, threshold);

        const auto t0 = std::chrono::steady_clock::now();
        rfdetr::Detections results = detector.detect(image, threshold);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        utils::print_metrics("image_det", ms);
        std::printf("detections: %zu\n", results.size());
        for (const auto& d : results)
            std::printf("  %-15s %.3f  [%.0f %.0f %.0f %.0f]\n",
                        rfdetr::coco_label(d.class_id), d.score,
                        d.box.x1, d.box.y1, d.box.x2, d.box.y2);

        rfdetr::draw_detections(image, results, &rfdetr::coco_label);
        const std::string out = utils::save_image(image, image_path);
        std::printf("saved: %s\n", out.c_str());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
