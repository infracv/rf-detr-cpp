#pragma once

// Keep this header includable from plain .cpp files — no CUDA runtime types.
// cudaStream_t is passed as void* to avoid pulling in cuda_runtime_api.h here.

#include "rfdetr/core/types.hpp"

#include <memory>
#include <vector>

namespace rfdetr {

// Reusable pinned-host + device staging for the mask decode. Page-locking the
// full-frame mask buffer costs more than the decode itself at 1080p, so a
// segmenter creates one of these at init and reuses it across frames. Buffers
// grow to the high-water mark and are never shrunk.
//
// Opaque by design: the definition lives in mask_decode.cu so this header stays
// includable from plain .cpp files.
struct MaskDecodeScratch;

struct MaskDecodeScratchDeleter {
    void operator()(MaskDecodeScratch* p) const noexcept;
};
using MaskDecodeScratchPtr = std::unique_ptr<MaskDecodeScratch, MaskDecodeScratchDeleter>;

[[nodiscard]] MaskDecodeScratchPtr make_mask_decode_scratch();

// GPU bilinear upsample + threshold for RF-DETR segmentation masks.
// Writes CV_8UC1 masks back into detections[i].mask.
//
// `scratch` may be null, in which case per-call buffers are allocated and freed
// (convenient, but ~10x slower at 1080p — prefer passing a persistent scratch).
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
                      MaskDecodeScratch*      scratch,
                      void*                   cuda_stream);

}  // namespace rfdetr
