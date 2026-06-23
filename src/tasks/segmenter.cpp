#include "rfdetr/tasks/segmenter.hpp"

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/cuda_check.hpp"
#include "rfdetr/core/cuda_preprocess.cuh"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/postprocess.hpp"
#include "rfdetr/core/trt_session.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace rfdetr {

// ---------------------------------------------------------------------------
// Mask postprocess helpers (CPU)
// ---------------------------------------------------------------------------

namespace {

// Sigmoid a single value.
inline float sigmoidf(float x) noexcept {
    return x >= 0.0f ? 1.0f / (1.0f + std::exp(-x))
                     : std::exp(x) / (1.0f + std::exp(x));
}

// Bilinear resize of a float32 planar mask (mH x mW) -> (dstH x dstW),
// then threshold to binary CV_8UC1.
//
// src layout: row-major float32, mH rows x mW cols (logits).
// Output: CV_8UC1 Mat of size (dstH, dstW), values 0 or 255.
cv::Mat resize_and_binarize(const float* src, int mH, int mW, int dstH, int dstW) {
    // Upsample using OpenCV's INTER_LINEAR on a 32FC1 intermediate.
    cv::Mat logit_small(mH, mW, CV_32FC1, const_cast<float*>(src));
    cv::Mat logit_full;
    cv::resize(logit_small, logit_full, cv::Size(dstW, dstH), 0, 0, cv::INTER_LINEAR);

    // Threshold: logit > 0 -> 255 (sigmoid > 0.5).
    cv::Mat binary(dstH, dstW, CV_8UC1);
    for (int r = 0; r < dstH; ++r) {
        const float* src_row = logit_full.ptr<float>(r);
        uint8_t*     dst_row = binary.ptr<uint8_t>(r);
        for (int c = 0; c < dstW; ++c) {
            dst_row[c] = src_row[c] > 0.0f ? 255u : 0u;
        }
    }
    return binary;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct RFDetrSegmenter::Impl {
    TrtSession                         session;
    EngineMeta                         meta;
    std::unique_ptr<ImagePreprocessor> preprocessor;
    SegmenterOptions                   opts;

    const BindingInfo* b_dets{nullptr};
    const BindingInfo* b_labels{nullptr};
    const BindingInfo* b_masks{nullptr};

    int N{0};   // num_queries
    int C{0};   // num_classes_with_bg
    int mH{0};  // mask height at engine output
    int mW{0};  // mask width  at engine output

    std::vector<float> h_dets;
    std::vector<float> h_labels;
    std::vector<float> h_masks;

    PostprocessParams pp;
    Timings           timings;

    Impl(const std::filesystem::path& engine_path,
         const std::filesystem::path& meta_path,
         const SegmenterOptions& options)
        : session(engine_path)
    {
        opts = options;

        if (std::filesystem::is_regular_file(meta_path)) {
            meta = EngineMeta::from_json_file(meta_path);
        } else {
            const auto* in_b = session.find("input");
            if (!in_b || in_b->shape.nbDims < 4) {
                throw std::runtime_error(
                    "rfdetr: RFDetrSegmenter: no meta sidecar at " + meta_path.string() +
                    " and cannot infer input dims");
            }
            meta.input_h     = in_b->shape.d[2];
            meta.input_w     = in_b->shape.d[3];
            meta.num_classes = 90;
            meta.color_order = "RGB";
            meta.has_masks   = true;
        }

        if (!meta.has_masks) {
            throw std::runtime_error(
                "rfdetr: RFDetrSegmenter: engine sidecar says has_masks=false — "
                "use a seg-* variant engine (rf-detr-seg-nano, seg-small, ...)");
        }
        if (meta.input_h != meta.input_w) {
            throw std::runtime_error("rfdetr: non-square input not supported");
        }

        preprocessor = std::make_unique<ImagePreprocessor>(
            meta.input_h, /*src_is_bgr=*/meta.color_order != "BGR");
        preprocessor->set_mean(meta.mean[0], meta.mean[1], meta.mean[2]);
        preprocessor->set_std (meta.std[0],  meta.std[1],  meta.std[2]);

        // Resolve output bindings: dets, labels, masks.
        b_dets   = session.find("dets");
        b_labels = session.find("labels");
        b_masks  = session.find("masks");
        if (!b_dets || !b_labels || !b_masks) {
            const auto& outs = session.output_indices();
            if (outs.size() < 3) {
                throw std::runtime_error(
                    "rfdetr: RFDetrSegmenter: engine has fewer than 3 outputs — "
                    "need dets, labels, masks");
            }
            b_dets   = &session.bindings()[outs[0]];
            b_labels = &session.bindings()[outs[1]];
            b_masks  = &session.bindings()[outs[2]];
        }

        N  = b_dets->shape.d[b_dets->shape.nbDims - 2];
        C  = b_labels->shape.d[b_labels->shape.nbDims - 1];
        mH = b_masks->shape.d[b_masks->shape.nbDims - 2];
        mW = b_masks->shape.d[b_masks->shape.nbDims - 1];

        h_dets.resize(static_cast<std::size_t>(N) * 4);
        h_labels.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(C));
        h_masks.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(mH) *
                       static_cast<std::size_t>(mW));

        pp.num_queries         = N;
        pp.num_classes_with_bg = C;
        pp.topk                = N;
        pp.threshold           = opts.threshold;
        pp.bg_class_index      = 0;
    }

    // Decode detections + gather + upsample masks for one image in a batch.
    Detections decode_one(const float* dets_ptr, const float* labels_ptr,
                          const float* masks_ptr, int img_w, int img_h,
                          float threshold) {
        pp.threshold = threshold;
        Detections dets = decode_detections(dets_ptr, labels_ptr, img_w, img_h, pp);

        // For every detection, we need the *original* query index before the top-K
        // reordering.  decode_detections doesn't expose it, so we re-run the
        // candidate selection inline here.  N and C are small (<=300, <=91), so
        // the redundant pass is negligible.
        //
        // We match by (class_id, score) — the combination is unique in practice.
        // Build a map: (query_idx, raw_class) -> score for the top-K candidates.
        struct Cand { float score; int q; int raw_class; };
        std::vector<Cand> cands;
        cands.reserve(static_cast<std::size_t>(N * C));
        for (int q = 0; q < N; ++q) {
            const float* lq = labels_ptr + q * C;
            for (int c = 0; c < C; ++c) {
                cands.push_back({lq[c], q, c});
            }
        }
        const int K = std::min(N, static_cast<int>(cands.size()));
        std::partial_sort(cands.begin(), cands.begin() + K, cands.end(),
                          [](const Cand& a, const Cand& b){ return a.score > b.score; });

        // For each decoded detection, find its query_idx from the sorted candidates.
        // We iterate in the same order decode_detections does: skip bg, skip below threshold.
        std::size_t cand_ptr = 0;
        for (auto& det : dets) {
            // Advance cand_ptr to the next above-threshold non-bg candidate.
            while (cand_ptr < static_cast<std::size_t>(K)) {
                const auto& ck = cands[cand_ptr];
                if (ck.raw_class == pp.bg_class_index) { ++cand_ptr; continue; }
                const float s = sigmoidf(ck.score);
                if (s < pp.threshold)                  { ++cand_ptr; continue; }
                break;
            }
            if (cand_ptr >= static_cast<std::size_t>(K)) break;

            const int query_idx = cands[cand_ptr].q;
            ++cand_ptr;

            // Gather mask logits for this query: shape (mH, mW), row-major.
            const float* mask_ptr = masks_ptr +
                static_cast<std::size_t>(query_idx) * mH * mW;
            det.mask = resize_and_binarize(mask_ptr, mH, mW, img_h, img_w);
        }
        return dets;
    }

    Detections run(const cv::Mat& image, float threshold) {
        using Clock = std::chrono::steady_clock;

        float* d_input = static_cast<float*>(session.device_buffer("input"));

        const auto t0 = Clock::now();
        preprocessor->process(image, d_input, session.stream());
        RFDETR_CUDA_CHECK(cudaStreamSynchronize(session.stream()));
        const auto t1 = Clock::now();

        session.infer();

        const auto t2 = Clock::now();

        session.get_output_f32(b_dets->name,   h_dets.data(),   h_dets.size());
        session.get_output_f32(b_labels->name, h_labels.data(), h_labels.size());
        session.get_output_f32(b_masks->name,  h_masks.data(),  h_masks.size());

        const float thr = (threshold >= 0.0f) ? threshold : opts.threshold;
        Detections results = decode_one(h_dets.data(), h_labels.data(), h_masks.data(),
                                        image.cols, image.rows, thr);
        const auto t3 = Clock::now();

        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timings.preprocess_ms  = ms(t0, t1);
        timings.infer_ms       = ms(t1, t2);
        timings.postprocess_ms = ms(t2, t3);
        timings.total_ms       = ms(t0, t3);
        return results;
    }

    std::vector<Detections> run_batch(const std::vector<cv::Mat>& images, float threshold) {
        const int B = static_cast<int>(images.size());
        if (B == 0) return {};

        const int R = meta.input_h;
        const std::size_t single_floats = static_cast<std::size_t>(3) * R * R;

        if (meta.dynamic_batch) {
            nvinfer1::Dims4 dims{B, 3, R, R};
            session.set_input_shape("input", dims);
        } else if (B != 1) {
            throw std::runtime_error(
                "rfdetr: segment_batch: engine is static-batch (batch=1). "
                "Rebuild with --min-batch 1 --opt-batch N --max-batch N for batch > 1.");
        }

        float* d_input = static_cast<float*>(session.device_buffer("input"));
        for (int i = 0; i < B; ++i) {
            preprocessor->process(images[i], d_input + i * single_floats, session.stream());
        }
        RFDETR_CUDA_CHECK(cudaStreamSynchronize(session.stream()));

        session.infer();

        const std::size_t dets_per_batch   = static_cast<std::size_t>(B) * N * 4;
        const std::size_t labels_per_batch = static_cast<std::size_t>(B) * N * C;
        const std::size_t masks_per_batch  = static_cast<std::size_t>(B) * N * mH * mW;

        if (h_dets.size()   < dets_per_batch)   h_dets.resize(dets_per_batch);
        if (h_labels.size() < labels_per_batch) h_labels.resize(labels_per_batch);
        if (h_masks.size()  < masks_per_batch)  h_masks.resize(masks_per_batch);

        session.get_output_f32(b_dets->name,   h_dets.data(),   dets_per_batch);
        session.get_output_f32(b_labels->name, h_labels.data(), labels_per_batch);
        session.get_output_f32(b_masks->name,  h_masks.data(),  masks_per_batch);

        const float thr = (threshold >= 0.0f) ? threshold : opts.threshold;
        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        for (int i = 0; i < B; ++i) {
            const float* d = h_dets.data()   + i * N * 4;
            const float* l = h_labels.data() + i * N * C;
            const float* m = h_masks.data()  + i * N * mH * mW;
            results.push_back(decode_one(d, l, m, images[i].cols, images[i].rows, thr));
        }
        return results;
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void RFDetrSegmenter::init_(const std::filesystem::path& engine_path,
                             const std::filesystem::path& meta_path,
                             const SegmenterOptions& opts) {
    impl_ = std::make_unique<Impl>(engine_path, meta_path, opts);
}

RFDetrSegmenter::RFDetrSegmenter(const std::filesystem::path& engine_path,
                                  const SegmenterOptions& opts) {
    init_(engine_path, engine_path.string() + ".json", opts);
}

RFDetrSegmenter::RFDetrSegmenter(const std::filesystem::path& engine_path,
                                  const std::filesystem::path& meta_path,
                                  const SegmenterOptions& opts) {
    init_(engine_path, meta_path, opts);
}

RFDetrSegmenter::~RFDetrSegmenter() = default;
RFDetrSegmenter::RFDetrSegmenter(RFDetrSegmenter&&) noexcept = default;
RFDetrSegmenter& RFDetrSegmenter::operator=(RFDetrSegmenter&&) noexcept = default;

Detections RFDetrSegmenter::segment(const cv::Mat& image, float threshold) {
    if (!impl_) throw std::runtime_error("rfdetr: RFDetrSegmenter: moved-from object");
    if (image.empty()) throw std::runtime_error("rfdetr: RFDetrSegmenter::segment: empty image");
    return impl_->run(image, threshold);
}

std::vector<Detections> RFDetrSegmenter::segment_batch(const std::vector<cv::Mat>& images,
                                                        float threshold) {
    if (!impl_) throw std::runtime_error("rfdetr: RFDetrSegmenter: moved-from object");
    if (images.empty()) return {};
    for (const auto& img : images) {
        if (img.empty()) throw std::runtime_error("rfdetr: segment_batch: image in batch is empty");
    }
    return impl_->run_batch(images, threshold);
}

const std::string& RFDetrSegmenter::variant()     const noexcept { return impl_->meta.variant; }
int                RFDetrSegmenter::input_h()      const noexcept { return impl_->meta.input_h; }
int                RFDetrSegmenter::input_w()      const noexcept { return impl_->meta.input_w; }
int                RFDetrSegmenter::num_queries()  const noexcept { return impl_->N; }
int                RFDetrSegmenter::num_classes()  const noexcept { return impl_->meta.num_classes; }
int                RFDetrSegmenter::mask_h()       const noexcept { return impl_->mH; }
int                RFDetrSegmenter::mask_w()       const noexcept { return impl_->mW; }

const RFDetrSegmenter::Timings& RFDetrSegmenter::last_timings() const noexcept {
    return impl_->timings;
}

}  // namespace rfdetr
