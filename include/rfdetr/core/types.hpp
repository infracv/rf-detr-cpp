#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace rfdetr {

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
    int class_id{-1};
    float score{0.0f};

    // Filled by RFDetrSegmenter. Empty for detection-only.
    // CV_8UC1, same size as the input image, values 0 or 255.
    cv::Mat mask;
};

using Detections = std::vector<Detection>;

}  // namespace rfdetr
