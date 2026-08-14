#pragma once

#include "cuda_raii.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>

namespace rfdetr {

// Launch the fused square-resize + (optional) BGR->RGB + /255 + ImageNet normalize
// + HWC->NCHW kernel. The kernel writes directly into `d_dst` on `stream`.
//
//   d_src       : device pointer to a contiguous HWC uint8 image, src_pitch bytes/row
//   src_h, src_w: source image dims in pixels
//   src_pitch   : bytes per source row (typically src_w * 3)
//   d_dst       : device pointer to (3 * dst_R * dst_R) float32, planar CHW
//   dst_R       : square output side (H == W == R)
//   src_is_bgr  : if true, swaps channels 0/2 to produce RGB (OpenCV default is BGR)
//   mean,std    : per-channel ImageNet-style normalization constants (RGB order)
//   stream      : CUDA stream the kernel runs on
//
// All inputs/outputs live on the device. The caller is responsible for the lifetime
// of d_src and d_dst, and for synchronizing with the stream as needed.
void launch_square_resize_normalize(cudaStream_t stream,
                                    const std::uint8_t* d_src, int src_h, int src_w,
                                    int src_pitch, float* d_dst, int dst_R,
                                    bool src_is_bgr, const float mean[3],
                                    const float std[3]);

// Convenience preprocessor: owns a pinned host staging buffer + a device source
// buffer that grows on demand. Each `process()` call uploads `image` to the device
// asynchronously and runs the fused kernel, writing into `d_dst`.
//
// Designed to be reused across many frames — the buffers are not freed between calls.
class ImagePreprocessor {
   public:
    ImagePreprocessor(int dst_R, bool src_is_bgr = true);
    ~ImagePreprocessor();

    ImagePreprocessor(const ImagePreprocessor&)            = delete;
    ImagePreprocessor& operator=(const ImagePreprocessor&) = delete;

    void set_mean(float r, float g, float b) noexcept;
    void set_std(float r, float g, float b) noexcept;

    // image: HxWx3 uint8 cv::Mat. d_dst: 3*R*R floats on device.
    void process(const cv::Mat& image, float* d_dst, cudaStream_t stream);

    int   target_size() const noexcept { return R_; }
    bool  source_is_bgr() const noexcept { return src_is_bgr_; }

   private:
    void ensure_capacity_(std::size_t bytes);

    int  R_{0};
    bool src_is_bgr_{true};
    float mean_[3]{0.485f, 0.456f, 0.406f};
    float std_[3]{0.229f, 0.224f, 0.225f};
    DevPtr        d_src_;
    HostPtr       h_pinned_;
    std::size_t   capacity_{0};
    // Signals that the previous call has finished reading the staging buffers.
    // Consecutive process() calls on one stream (batch preprocessing) would
    // otherwise overwrite — or reallocate — staging that is still in flight.
    cudaEvent_t   staging_done_{nullptr};
};

}  // namespace rfdetr
