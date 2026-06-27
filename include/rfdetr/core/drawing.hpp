#pragma once

#include "rfdetr/core/types.hpp"

#include <opencv2/core.hpp>

namespace rfdetr {

// Draw all detections onto `image` in-place: rectangle + "<label> <score>" text.
// `label_for_class` may be null, in which case the class id is shown.
void draw_detections(cv::Mat& image, const Detections& dets,
                     const char* (*label_for_class)(int) = nullptr);

// Draw instance segmentation masks: per-class color, alpha-blended fill.
// Skips detections with empty `.mask`. `alpha` controls fill opacity (0..1).
// Call before draw_detections() so boxes and labels render on top.
void draw_segmentations(cv::Mat& image, const Detections& dets, float alpha = 0.5f);

}  // namespace rfdetr
