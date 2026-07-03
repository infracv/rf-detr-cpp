#include "internal/cuda_preprocess.cuh"

#include "internal/cuda_check.hpp"
#include "internal/cuda_raii.hpp"

#include <cstring>
#include <stdexcept>

namespace rfdetr {

namespace {

// Fused preprocess kernel — one pass does bilinear resize, optional BGR->RGB
// channel swap, /255 + per-channel (x - mean) / std, and HWC->CHW transpose.
// All loop-invariants (scale factors, folded normalization constants) are
// precomputed on the host so the kernel body is pure per-pixel work.
__global__ void squareResizeNormalizeKernel(const std::uint8_t* __restrict__ src,
                                             int src_h, int src_w, int src_pitch,
                                             float* __restrict__ dst, int R,
                                             bool src_is_bgr,
                                             float scale_x, float scale_y,
                                             float3 pre_mul, float3 pre_sub) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= R || y >= R) return;

    // Map output pixel center to source coordinates (OpenCV INTER_LINEAR semantics).
    const float sx_f = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    const float sy_f = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

    const int sx0 = max(0, min(src_w - 1, static_cast<int>(floorf(sx_f))));
    const int sy0 = max(0, min(src_h - 1, static_cast<int>(floorf(sy_f))));
    const int sx1 = min(src_w - 1, sx0 + 1);
    const int sy1 = min(src_h - 1, sy0 + 1);
    const float fx = fmaxf(0.0f, fminf(1.0f, sx_f - sx0));
    const float fy = fmaxf(0.0f, fminf(1.0f, sy_f - sy0));
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = fx * (1.0f - fy);
    const float w10 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    const std::uint8_t* p00 = src + sy0 * src_pitch + sx0 * 3;
    const std::uint8_t* p01 = src + sy0 * src_pitch + sx1 * 3;
    const std::uint8_t* p10 = src + sy1 * src_pitch + sx0 * 3;
    const std::uint8_t* p11 = src + sy1 * src_pitch + sx1 * 3;

    // __ldg routes through the read-only data cache (texture path on SM≥35).
    // No-op on modern Blackwell with __restrict__ but explicit avoids
    // depending on the compiler to infer it.
    const float c0 = __ldg(p00 + 0) * w00 + __ldg(p01 + 0) * w01
                   + __ldg(p10 + 0) * w10 + __ldg(p11 + 0) * w11;
    const float c1 = __ldg(p00 + 1) * w00 + __ldg(p01 + 1) * w01
                   + __ldg(p10 + 1) * w10 + __ldg(p11 + 1) * w11;
    const float c2 = __ldg(p00 + 2) * w00 + __ldg(p01 + 2) * w01
                   + __ldg(p10 + 2) * w10 + __ldg(p11 + 2) * w11;

    // OpenCV channel order is BGR by default. ImageNet mean/std assume RGB.
    const float r = src_is_bgr ? c2 : c0;
    const float g = c1;
    const float b = src_is_bgr ? c0 : c2;

    // Normalization is folded on the host:
    //   nx = (x/255 - mean) / std   =   x * pre_mul - pre_sub
    //   where pre_mul = inv_std / 255   and   pre_sub = mean / std
    const float nr = r * pre_mul.x - pre_sub.x;
    const float ng = g * pre_mul.y - pre_sub.y;
    const float nb = b * pre_mul.z - pre_sub.z;

    const int hw  = R * R;
    const int idx = y * R + x;
    dst[0 * hw + idx] = nr;
    dst[1 * hw + idx] = ng;
    dst[2 * hw + idx] = nb;
}

}  // namespace

void launch_square_resize_normalize(cudaStream_t stream, const std::uint8_t* d_src,
                                     int src_h, int src_w, int src_pitch, float* d_dst,
                                     int dst_R, bool src_is_bgr, const float mean[3],
                                     const float std[3]) {
    if (dst_R <= 0 || src_h <= 0 || src_w <= 0) {
        throw std::runtime_error("rfdetr: launch_square_resize_normalize bad dims");
    }

    // Precompute scale factors and folded normalization constants on the host
    // once per launch — keeps the kernel body branch-free and FMA-friendly.
    constexpr float kInv255 = 1.0f / 255.0f;
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_R);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_R);
    const float3 pre_mul{kInv255 / std[0], kInv255 / std[1], kInv255 / std[2]};
    const float3 pre_sub{mean[0] / std[0], mean[1] / std[1], mean[2] / std[2]};

    const dim3 block(16, 16);
    const dim3 grid((dst_R + block.x - 1) / block.x, (dst_R + block.y - 1) / block.y);
    squareResizeNormalizeKernel<<<grid, block, 0, stream>>>(d_src, src_h, src_w,
                                                             src_pitch, d_dst, dst_R,
                                                             src_is_bgr, scale_x, scale_y,
                                                             pre_mul, pre_sub);
    RFDETR_CUDA_CHECK(cudaGetLastError());
}

ImagePreprocessor::ImagePreprocessor(int dst_R, bool src_is_bgr)
    : R_(dst_R), src_is_bgr_(src_is_bgr) {}

// DevPtr/HostPtr destructors handle cudaFree/cudaFreeHost automatically.
ImagePreprocessor::~ImagePreprocessor() = default;

void ImagePreprocessor::set_mean(float r, float g, float b) noexcept {
    mean_[0] = r;
    mean_[1] = g;
    mean_[2] = b;
}

void ImagePreprocessor::set_std(float r, float g, float b) noexcept {
    std_[0] = r;
    std_[1] = g;
    std_[2] = b;
}

void ImagePreprocessor::ensure_capacity_(std::size_t bytes) {
    if (bytes <= capacity_) return;
    // RAII reset frees the old buffers before re-allocating.
    d_src_.reset();
    h_pinned_.reset();
    void* dp = nullptr;
    void* hp = nullptr;
    RFDETR_CUDA_CHECK(cudaMalloc(&dp, bytes));
    RFDETR_CUDA_CHECK(cudaMallocHost(&hp, bytes));
    d_src_.reset(dp);
    h_pinned_.reset(hp);
    capacity_ = bytes;
}

void ImagePreprocessor::process(const cv::Mat& image, float* d_dst, cudaStream_t stream) {
    if (image.empty()) {
        throw std::runtime_error("rfdetr: ImagePreprocessor::process given empty image");
    }
    if (image.type() != CV_8UC3) {
        throw std::runtime_error(
            "rfdetr: ImagePreprocessor expects CV_8UC3 (HxWx3 uint8); convert first");
    }

    const int rows = image.rows;
    const int cols = image.cols;
    const std::size_t bytes_per_row = static_cast<std::size_t>(cols) * 3;
    const std::size_t total_bytes   = bytes_per_row * static_cast<std::size_t>(rows);
    ensure_capacity_(total_bytes);

    // Copy host cv::Mat (which may have step != cols*3) into a contiguous pinned buffer.
    auto* dst_pinned = static_cast<std::uint8_t*>(h_pinned_.get());
    auto* d_src_raw  = static_cast<std::uint8_t*>(d_src_.get());
    if (image.isContinuous() && static_cast<std::size_t>(image.step) == bytes_per_row) {
        std::memcpy(dst_pinned, image.data, total_bytes);
    } else {
        for (int r = 0; r < rows; ++r) {
            std::memcpy(dst_pinned + r * bytes_per_row, image.ptr<std::uint8_t>(r),
                        bytes_per_row);
        }
    }
    RFDETR_CUDA_CHECK(cudaMemcpyAsync(d_src_raw, dst_pinned, total_bytes,
                                       cudaMemcpyHostToDevice, stream));

    launch_square_resize_normalize(stream, d_src_raw, rows, cols,
                                    static_cast<int>(bytes_per_row), d_dst, R_,
                                    src_is_bgr_, mean_, std_);
}

}  // namespace rfdetr
