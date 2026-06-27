#include "rfdetr/core/postprocess.hpp"

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

Detections decode_detections(const float* dets, const float* labels, int img_w, int img_h,
                              const PostprocessParams& p) {
    if (!dets || !labels || p.num_queries <= 0 || p.num_classes_with_bg <= 0) {
        return {};
    }
    const int N = p.num_queries;
    const int C = p.num_classes_with_bg;
    const int total = N * C;
    const int K = std::min(p.topk > 0 ? p.topk : N, total);

    // Sigmoid is monotonic: sort by raw logit, then sigmoid only the survivors.
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

    Detections out;
    out.reserve(static_cast<std::size_t>(K));

    const float W = static_cast<float>(img_w);
    const float H = static_cast<float>(img_h);

    for (int i = 0; i < K; ++i) {
        const Candidate& cnd = cand[i];
        if (p.bg_class_index >= 0 && cnd.raw_class == p.bg_class_index) continue;

        const float score = sigmoid(cnd.score);
        if (score < p.threshold) {
            // Sorted desc, so anything beyond also fails — but background hits are
            // sparse near the top, so don't break: they'd otherwise mask later hits.
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
        d.class_id = (p.bg_class_index >= 0) ? (cnd.raw_class - 1) : cnd.raw_class;
        if (d.box.width() > 0.0f && d.box.height() > 0.0f) {
            out.push_back(std::move(d));
        }
    }
    return out;
}

}  // namespace rfdetr
