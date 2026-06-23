#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace rfdetr {

// Bounding box in original-image pixel coordinates (xyxy).
struct Box {
    float x1{0};
    float y1{0};
    float x2{0};
    float y2{0};

    float width()  const noexcept { return x2 - x1; }
    float height() const noexcept { return y2 - y1; }
    float area()   const noexcept { return width() * height(); }
};

struct Detection {
    Box box;
    int class_id{-1};   // 0..num_classes-1, foreground only (background already shifted out)
    float score{0.0f};

    // Filled by RFDetrSegmenter (otherwise empty). CV_8UC1, original-image size, 0 or 255.
    cv::Mat mask;
};

using Detections = std::vector<Detection>;

}  // namespace rfdetr
