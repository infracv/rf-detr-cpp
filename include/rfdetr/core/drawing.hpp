#pragma once

#include "rfdetr/core/types.hpp"

#include <opencv2/core.hpp>

namespace rfdetr {

// Draw all detections onto `image` in-place: rectangle + "<label> <score>" text.
// `label_for_class` may be null, in which case the class id is shown.
void draw_detections(cv::Mat& image, const Detections& dets,
                     const char* (*label_for_class)(int) = nullptr);

}  // namespace rfdetr
