#pragma once

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
};

using Detections = std::vector<Detection>;

}  // namespace rfdetr
