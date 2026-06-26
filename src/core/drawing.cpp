#include "rfdetr/core/drawing.hpp"

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <string>

namespace rfdetr {

namespace {

// Deterministic per-class color in BGR. Hash on class id.
cv::Scalar color_for_class(int class_id) noexcept {
    const auto u = static_cast<unsigned int>(class_id * 2654435761u);
    return cv::Scalar((u >> 0) & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF);
}

}  // namespace

void draw_detections(cv::Mat& image, const Detections& dets,
                     const char* (*label_for_class)(int)) {
    constexpr int      thickness   = 2;
    constexpr double   font_scale  = 0.55;
    constexpr int      font_face   = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Scalar   text_color  = cv::Scalar(255, 255, 255);

    for (const auto& d : dets) {
        const cv::Scalar color = color_for_class(d.class_id);
        const cv::Point  p1(static_cast<int>(d.box.x1), static_cast<int>(d.box.y1));
        const cv::Point  p2(static_cast<int>(d.box.x2), static_cast<int>(d.box.y2));
        cv::rectangle(image, p1, p2, color, thickness);

        char label[128];
        const char* name = label_for_class ? label_for_class(d.class_id) : "?";
        std::snprintf(label, sizeof(label), "%s %.2f", name, d.score);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, font_face, font_scale, 1, &baseline);
        const cv::Point tl(p1.x, std::max(0, p1.y - ts.height - 4));
        cv::rectangle(image, tl, cv::Point(tl.x + ts.width + 6, tl.y + ts.height + 4),
                      color, cv::FILLED);
        cv::putText(image, label, cv::Point(tl.x + 3, tl.y + ts.height + 1), font_face,
                    font_scale, text_color, 1, cv::LINE_AA);
    }
}

}  // namespace rfdetr
