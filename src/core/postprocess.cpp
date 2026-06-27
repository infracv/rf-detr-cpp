#include "rfdetr/core/postprocess.hpp"

#include "../internal/mask_decode.cuh"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rfdetr {

namespace {

inline float sigmoid(float x) noexcept {
    // Numerically stable: clamp to avoid overflow on extreme logits.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

// (score, query_idx, raw_class) — sorted by score desc.
struct Candidate {
    float score;
    int   query_idx;
    int   raw_class;
};

}  // namespace

namespace {

void decode_impl(const float* dets, const float* labels, int img_w, int img_h,
                 const PostprocessParams& p, Detections& out,
                 std::vector<int>* out_query_idx) {
    if (!dets || !labels || p.num_queries <= 0 || p.num_classes_with_bg <= 0) {
        return;
    }
    const int N = p.num_queries;
    const int C = p.num_classes_with_bg;
    const int total = N * C;
    const int K = std::min(p.topk > 0 ? p.topk : N, total);

    std::vector<Candidate> cand;
    cand.reserve(static_cast<std::size_t>(total));
    for (int q = 0; q < N; ++q) {
        const float* lq = labels + q * C;
        for (int c = 0; c < C; ++c) {
            cand.push_back({lq[c], q, c});
        }
    }

    std::partial_sort(cand.begin(), cand.begin() + K, cand.end(),
                      [](const Candidate& a, const Candidate& b) noexcept {
                          return a.score > b.score;
                      });

    const float W = static_cast<float>(img_w);
    const float H = static_cast<float>(img_h);

    out.reserve(out.size() + static_cast<std::size_t>(K));
    if (out_query_idx) out_query_idx->reserve(out_query_idx->size() + static_cast<std::size_t>(K));

    for (int i = 0; i < K; ++i) {
        const Candidate& cnd = cand[i];
        if (p.bg_class_index >= 0 && cnd.raw_class == p.bg_class_index) continue;

        const float score = sigmoid(cnd.score);
        if (score < p.threshold) {
            continue;
        }

        const float* db = dets + cnd.query_idx * 4;
        const float cx = db[0];
        const float cy = db[1];
        const float w  = db[2];
        const float h  = db[3];

        Detection d;
        d.box.x1   = (cx - 0.5f * w) * W;
        d.box.y1   = (cy - 0.5f * h) * H;
        d.box.x2   = (cx + 0.5f * w) * W;
        d.box.y2   = (cy + 0.5f * h) * H;
        d.box.x1   = std::max(0.0f, std::min(W, d.box.x1));
        d.box.y1   = std::max(0.0f, std::min(H, d.box.y1));
        d.box.x2   = std::max(0.0f, std::min(W, d.box.x2));
        d.box.y2   = std::max(0.0f, std::min(H, d.box.y2));
        d.score    = score;
        d.class_id = dense_class_from_raw(cnd.raw_class, p.bg_class_index);
        if (d.box.width() > 0.0f && d.box.height() > 0.0f) {
            out.push_back(std::move(d));
            if (out_query_idx) out_query_idx->push_back(cnd.query_idx);
        }
    }
}

}  // namespace

Detections decode_detections(const float* dets, const float* labels, int img_w, int img_h,
                              const PostprocessParams& p) {
    Detections out;
    decode_impl(dets, labels, img_w, img_h, p, out, nullptr);
    return out;
}

Detections decode_detections_with_queries(const float* dets, const float* labels, int img_w,
                                           int img_h, const PostprocessParams& p,
                                           std::vector<int>& out_query_idx) {
    Detections out;
    out_query_idx.clear();
    decode_impl(dets, labels, img_w, img_h, p, out, &out_query_idx);
    return out;
}

void decode_masks(const float* masks_logits, int mask_h, int mask_w,
                  const std::vector<int>& query_indices,
                  int img_w, int img_h, Detections& detections,
                  void* stream) {
    if (!masks_logits || mask_h <= 0 || mask_w <= 0 || img_w <= 0 || img_h <= 0) return;
    if (query_indices.size() != detections.size()) return;
    if (detections.empty()) return;

    if (stream) {
        // GPU path: bilinear upsample + threshold entirely on device.
        gpu_decode_masks(masks_logits, query_indices, mask_h, mask_w,
                         img_h, img_w, detections, stream);
        return;
    }

    // CPU fallback — hoist scratch buffers outside the loop so cv::resize
    // reuses logit_full's allocation each iteration (same output dims).
    const std::size_t plane = static_cast<std::size_t>(mask_h) * mask_w;
    cv::Mat logit_full;
    cv::Mat binary;
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const int q = query_indices[i];
        if (q < 0) continue;

        const float* src = masks_logits + static_cast<std::size_t>(q) * plane;
        const cv::Mat logit_small(mask_h, mask_w, CV_32FC1, const_cast<float*>(src));

        cv::resize(logit_small, logit_full, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
        cv::compare(logit_full, 0.0f, binary, cv::CMP_GT);
        detections[i].mask = std::move(binary);
    }
}

}  // namespace rfdetr
