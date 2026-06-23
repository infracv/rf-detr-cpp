#pragma once

#include "rfdetr/core/types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rfdetr {

struct SegmenterOptions {
    float threshold{0.5f};
    bool  use_cuda_graph{false};
    int   graph_warmup_iters{1};
};

// Public segmenter API.  Hides all TRT/CUDA headers behind PIMPL.
// Works with any RF-DETR seg-* engine that has three outputs:
//   dets   (B, num_queries, 4)          cxcywh normalized
//   labels (B, num_queries, num_classes) raw logits
//   masks  (B, num_queries, mH, mW)     per-query mask logits
//
// Each returned Detection has its `.mask` field filled:
//   CV_8UC1, same size as the input image, values 0 or 255.
//
// Usage:
//   rfdetr::RFDetrSegmenter seg("rf-detr-seg-nano-fp32.engine");
//   auto results = seg.segment(image, 0.5f);
//   for (auto& d : results) { /* d.mask is a binary CV_8UC1 */ }
class RFDetrSegmenter {
   public:
    explicit RFDetrSegmenter(const std::filesystem::path& engine_path,
                             const SegmenterOptions& opts = {});

    RFDetrSegmenter(const std::filesystem::path& engine_path,
                    const std::filesystem::path& meta_path,
                    const SegmenterOptions& opts = {});

    ~RFDetrSegmenter();

    RFDetrSegmenter(const RFDetrSegmenter&)            = delete;
    RFDetrSegmenter& operator=(const RFDetrSegmenter&) = delete;
    RFDetrSegmenter(RFDetrSegmenter&&) noexcept;
    RFDetrSegmenter& operator=(RFDetrSegmenter&&) noexcept;

    // Run segmentation on a single image. Returns detections with masks filled.
    Detections segment(const cv::Mat& image, float threshold = -1.0f);

    // Batch segmentation. Engine must be built with max_batch >= images.size().
    std::vector<Detections> segment_batch(const std::vector<cv::Mat>& images,
                                          float threshold = -1.0f);

    const std::string& variant()     const noexcept;
    int                input_h()     const noexcept;
    int                input_w()     const noexcept;
    int                num_queries() const noexcept;
    int                num_classes() const noexcept;
    int                mask_h()      const noexcept;
    int                mask_w()      const noexcept;

    struct Timings {
        double preprocess_ms{0};
        double infer_ms{0};
        double postprocess_ms{0};  // includes mask decode
        double total_ms{0};
    };
    const Timings& last_timings() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void init_(const std::filesystem::path& engine_path,
               const std::filesystem::path& meta_path,
               const SegmenterOptions& opts);
};

}  // namespace rfdetr
