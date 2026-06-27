#pragma once

// Keep this header includable from plain .cpp files — no CUDA runtime types.
// cudaStream_t is passed as void* to avoid pulling in cuda_runtime_api.h here.

#include "rfdetr/core/types.hpp"

#include <vector>

namespace rfdetr {

// GPU bilinear upsample + threshold for RF-DETR segmentation masks.
// Allocates GPU scratch buffers internally and writes CV_8UC1 masks back into
// detections[i].mask.
//
// h_masks_logits : host, [num_queries, mH, mW] float32 raw logits
// query_indices  : host, [num_dets] — which query row each detection uses (-1 = skip)
// mH, mW         : source mask resolution
// imgH, imgW     : output mask resolution (original image dims)
// detections     : detections whose `.mask` field will be populated
// cuda_stream    : opaque cudaStream_t (void* to keep header CUDA-free)
void gpu_decode_masks(const float*            h_masks_logits,
                      const std::vector<int>& query_indices,
                      int mH, int mW,
                      int imgH, int imgW,
                      Detections&             detections,
                      void*                   cuda_stream);

}  // namespace rfdetr
