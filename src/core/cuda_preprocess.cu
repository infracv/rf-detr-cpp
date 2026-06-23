#include "rfdetr/core/cuda_preprocess.cuh"

#include "rfdetr/core/cuda_check.hpp"

#include <cstring>
#include <stdexcept>

namespace rfdetr {

namespace {

__global__ void squareResizeNormalizeKernel(const std::uint8_t* __restrict__ src,
                                             int src_h, int src_w, int src_pitch,
                                             float* __restrict__ dst, int R,
                                             bool src_is_bgr, float3 mean,
                                             float3 inv_std) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= R || y >= R) return;

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(R);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(R);
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

    const float c0 = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
    const float c1 = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
    const float c2 = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;

    // OpenCV channel order is BGR by default. ImageNet mean/std assume RGB.
    const float r = src_is_bgr ? c2 : c0;
    const float g = c1;
    const float b = src_is_bgr ? c0 : c2;

    constexpr float k1_255 = 1.0f / 255.0f;
    const float nr = (r * k1_255 - mean.x) * inv_std.x;
    const float ng = (g * k1_255 - mean.y) * inv_std.y;
    const float nb = (b * k1_255 - mean.z) * inv_std.z;

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
    const float3 mean3{mean[0], mean[1], mean[2]};
    const float3 inv_std3{1.0f / std[0], 1.0f / std[1], 1.0f / std[2]};

    const dim3 block(16, 16);
    const dim3 grid((dst_R + block.x - 1) / block.x, (dst_R + block.y - 1) / block.y);
    squareResizeNormalizeKernel<<<grid, block, 0, stream>>>(d_src, src_h, src_w,
                                                             src_pitch, d_dst, dst_R,
                                                             src_is_bgr, mean3, inv_std3);
    RFDETR_CUDA_CHECK(cudaGetLastError());
}

ImagePreprocessor::ImagePreprocessor(int dst_R, bool src_is_bgr)
    : R_(dst_R), src_is_bgr_(src_is_bgr) {}

ImagePreprocessor::~ImagePreprocessor() {
    if (d_src_) cudaFree(d_src_);
    if (h_pinned_) cudaFreeHost(h_pinned_);
}

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
    if (d_src_) {
        cudaFree(d_src_);
        d_src_ = nullptr;
    }
    if (h_pinned_) {
        cudaFreeHost(h_pinned_);
        h_pinned_ = nullptr;
    }
    RFDETR_CUDA_CHECK(cudaMalloc(&d_src_, bytes));
    RFDETR_CUDA_CHECK(cudaMallocHost(&h_pinned_, bytes));
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
    auto* dst_pinned = static_cast<std::uint8_t*>(h_pinned_);
    if (image.isContinuous() && static_cast<std::size_t>(image.step) == bytes_per_row) {
        std::memcpy(dst_pinned, image.data, total_bytes);
    } else {
        for (int r = 0; r < rows; ++r) {
            std::memcpy(dst_pinned + r * bytes_per_row, image.ptr<std::uint8_t>(r),
                        bytes_per_row);
        }
    }
    RFDETR_CUDA_CHECK(cudaMemcpyAsync(d_src_, dst_pinned, total_bytes,
                                       cudaMemcpyHostToDevice, stream));

    launch_square_resize_normalize(stream, d_src_, rows, cols,
                                    static_cast<int>(bytes_per_row), d_dst, R_,
                                    src_is_bgr_, mean_, std_);
}

}  // namespace rfdetr
