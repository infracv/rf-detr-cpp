#pragma once

#include "rfdetr/core/types.hpp"

#include <opencv2/core.hpp>

namespace rfdetr {

// Draw all detections onto `image` in-place: rectangle + "<label> <score>" text.
// `label_for_class` may be null, in which case the class id is shown.
void draw_detections(cv::Mat& image, const Detections& dets,
                     const char* (*label_for_class)(int) = nullptr);

// Draw segmentation results: filled semi-transparent colored mask overlay for
// each detection that has a non-empty `.mask`, then bounding box + label text.
// `alpha` controls mask opacity (0=invisible, 1=opaque, default 0.45).
void draw_segmentations(cv::Mat& image, const Detections& dets,
                        const char* (*label_for_class)(int) = nullptr,
                        float alpha = 0.45f);

}  // namespace rfdetr
