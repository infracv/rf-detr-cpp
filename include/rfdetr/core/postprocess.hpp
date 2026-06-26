#pragma once

#include "rfdetr/core/types.hpp"

#include <cstddef>

namespace rfdetr {

// Parameters for RF-DETR detection decoding.
struct PostprocessParams {
    int   num_queries{300};       // model output dim 1 (e.g. 300 for nano/small/medium/base/large)
    int   num_classes_with_bg{91};// model output dim 2 (e.g. 91 = 90 fg + 1 bg)
    int   topk{300};              // num_select; default = num_queries (Python reference)
    float threshold{0.5f};        // score threshold
    int   bg_class_index{0};      // raw class id treated as background; -1 disables filtering
};

// Decode RF-DETR detection outputs (CPU implementation).
//
// Inputs (host pointers, contiguous):
//   dets   : (num_queries, 4) float32, cxcywh normalized to [0,1]
//   labels : (num_queries, num_classes_with_bg) float32, raw logits (focal-style head)
//
// img_w / img_h: original-image dims for box rescaling.
//
// Output: dense Detections with xyxy boxes in original-image pixel coordinates.
Detections decode_detections(const float* dets, const float* labels, int img_w,
                              int img_h, const PostprocessParams& params);

}  // namespace rfdetr
