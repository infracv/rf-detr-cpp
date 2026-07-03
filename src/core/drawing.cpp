#include "rfdetr/core/drawing.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <string>

namespace rfdetr {

namespace {

// Deterministic per-class color in BGR. Hash on class id.
cv::Scalar color_for_class(int class_id) noexcept {
    const auto u = static_cast<unsigned int>(class_id * 2654435761u);
    return cv::Scalar((u >> 0) & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF);
}

// Scale fonts/strokes with image size. Matches supervision/ultralytics.
double adaptive_font_scale(const cv::Size& image_size) noexcept {
    const int min_dim = std::min(image_size.width, image_size.height);
    return std::max(0.4, min_dim * 0.0008);
}

}  // namespace

void draw_detections(cv::Mat& image, const Detections& dets,
                     const char* (*label_for_class)(int)) {
    constexpr int    font_face  = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Scalar text_color = cv::Scalar(255, 255, 255);

    const double font_scale = adaptive_font_scale(image.size());
    const int    thickness  = std::max(1, static_cast<int>(font_scale * 3.0));

    for (const auto& d : dets) {
        const cv::Scalar color = color_for_class(d.class_id);
        const cv::Point  p1(static_cast<int>(d.box.x1), static_cast<int>(d.box.y1));
        const cv::Point  p2(static_cast<int>(d.box.x2), static_cast<int>(d.box.y2));
        // LINE_8 for rectangles — antialiasing is wasted on axis-aligned edges.
        cv::rectangle(image, p1, p2, color, thickness, cv::LINE_8);

        char label[128];
        const char* name = label_for_class ? label_for_class(d.class_id) : "?";
        std::snprintf(label, sizeof(label), "%s %.2f", name, d.score);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, font_face, font_scale, 1, &baseline);

        // Flip the label below the box if it would clip the top edge.
        const bool flip_below = p1.y < ts.height + 4;
        const int  tl_y = flip_below ? p1.y : (p1.y - ts.height - 4);
        const cv::Point tl(p1.x, tl_y);
        const cv::Point br(tl.x + ts.width + 6, tl.y + ts.height + 4);
        cv::rectangle(image, tl, br, color, cv::FILLED, cv::LINE_8);
        cv::putText(image, label, cv::Point(tl.x + 3, tl.y + ts.height + 1), font_face,
                    font_scale, text_color, 1, cv::LINE_AA);
    }
}

void draw_segmentations(cv::Mat& image, const Detections& dets, float alpha) {
    if (image.empty() || dets.empty()) return;
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    // Per-detection: clip work to the mask's bounding box, then use OpenCV's
    // SIMD-vectorized addWeighted on the cropped ROI. Detections typically
    // cover a small fraction of the frame, so blending only inside the bbox
    // beats both a full-image blend and a scalar pixel loop.
    for (const auto& d : dets) {
        if (d.mask.empty()) continue;
        if (d.mask.size() != image.size() || d.mask.type() != CV_8UC1) continue;

        cv::Rect roi(
            std::max(0, static_cast<int>(d.box.x1)),
            std::max(0, static_cast<int>(d.box.y1)),
            0, 0);
        roi.width  = std::min(image.cols - roi.x,
                              static_cast<int>(d.box.x2) - roi.x + 1);
        roi.height = std::min(image.rows - roi.y,
                              static_cast<int>(d.box.y2) - roi.y + 1);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat image_roi = image(roi);
        cv::Mat mask_roi  = d.mask(roi);
        cv::Mat color_roi(image_roi.size(), image_roi.type(), color_for_class(d.class_id));
        cv::Mat blended_roi;
        cv::addWeighted(image_roi, 1.0 - alpha, color_roi, alpha, 0.0, blended_roi);
        blended_roi.copyTo(image_roi, mask_roi);
    }
}

}  // namespace rfdetr
