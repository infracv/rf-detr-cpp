#pragma once

#include "rfdetr/core/types.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>

namespace rfdetr {

struct DetectorOptions {
    float threshold{0.5f};
    // Enable CUDA Graph capture for lower dispatch latency (~0.1–0.3 ms).
    bool use_cuda_graph{false};
    // Warm-up iterations before graph capture (1 is sufficient).
    int  graph_warmup_iters{1};
};

// Public single-class detector API.  Hides all TRT/CUDA headers from users.
// Wraps TrtSession + ImagePreprocessor + decode_detections in a PIMPL.
//
// Thread safety: not thread-safe.  Use one instance per thread.
class RFDetrDetector {
   public:
    // Load engine + sidecar from `engine_path` (sidecar = `engine_path` + ".json").
    explicit RFDetrDetector(const std::filesystem::path& engine_path,
                            const DetectorOptions& opts = {});

    // Load engine with an explicit sidecar path.
    RFDetrDetector(const std::filesystem::path& engine_path,
                   const std::filesystem::path& meta_path,
                   const DetectorOptions& opts = {});

    ~RFDetrDetector();

    RFDetrDetector(const RFDetrDetector&)            = delete;
    RFDetrDetector& operator=(const RFDetrDetector&) = delete;
    RFDetrDetector(RFDetrDetector&&) noexcept;
    RFDetrDetector& operator=(RFDetrDetector&&) noexcept;

    // Detect objects in a single BGR (or RGB if meta says so) cv::Mat.
    // `threshold` overrides the option set at construction; pass < 0 to use default.
    [[nodiscard]] Detections detect(const cv::Mat& image, float threshold = -1.0f);

    // Batch detect: images must all have the same dimensions. Returns a flat
    // vector of Detections per image (results[i] = detections for images[i]).
    // Requires an engine built with max_batch >= images.size().
    [[nodiscard]] std::vector<Detections> detect_batch(const std::vector<cv::Mat>& images,
                                                        float threshold = -1.0f);

    // Accessors for inspection / benchmarking.
    [[nodiscard]] const std::string& variant()    const noexcept;
    [[nodiscard]] int                input_h()    const noexcept;
    [[nodiscard]] int                input_w()    const noexcept;
    [[nodiscard]] int                num_queries() const noexcept;
    [[nodiscard]] int                num_classes() const noexcept;
    // True if CUDA Graph was successfully captured at construction.
    // False means enqueueV3 is used every call (still correct, slightly higher latency).
    [[nodiscard]] bool               cuda_graph_active() const noexcept;

    // Timings from the last detect() call, in milliseconds.
    struct Timings {
        double preprocess_ms{0};
        double infer_ms{0};
        double postprocess_ms{0};
        double total_ms{0};
    };
    [[nodiscard]] const Timings& last_timings() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void init_(const std::filesystem::path& engine_path,
               const std::filesystem::path& meta_path,
               const DetectorOptions& opts);
};

}  // namespace rfdetr
