#include "rfdetr/tasks/segmenter.hpp"

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/log.hpp"
#include "rfdetr/core/postprocess.hpp"
#include "../internal/cuda_check.hpp"
#include "../internal/cuda_preprocess.cuh"
#include "../internal/mask_decode.cuh"
#include "../internal/trt_session.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace rfdetr {

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
    int mH{0};  // mask height
    int mW{0};  // mask width

    std::vector<float> h_dets;
    std::vector<float> h_labels;
    std::vector<float> h_masks;

    PostprocessParams pp;
    Timings           timings;

    // Reused across frames — see MaskDecodeScratch.
    MaskDecodeScratchPtr mask_scratch{make_mask_decode_scratch()};

    bool            graph_captured{false};
    cudaGraph_t     graph{nullptr};
    cudaGraphExec_t graph_exec{nullptr};

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
                    " and cannot infer input dims — pass explicit meta path");
            }
            meta.input_h     = in_b->shape.d[2];
            meta.input_w     = in_b->shape.d[3];
            meta.num_classes = 90;
            meta.color_order = "RGB";
        }
        if (meta.input_h != meta.input_w) {
            throw std::runtime_error("rfdetr: non-square input not supported");
        }

        preprocessor = std::make_unique<ImagePreprocessor>(
            meta.input_h, /*bgr_to_rgb=*/meta.color_order == "RGB");
        preprocessor->set_mean(meta.mean[0], meta.mean[1], meta.mean[2]);
        preprocessor->set_std (meta.std[0],  meta.std[1],  meta.std[2]);

        b_dets   = session.find("dets");
        b_labels = session.find("labels");
        b_masks  = session.find("masks");
        if (!b_dets || !b_labels || !b_masks) {
            // Index-order fallback: some ONNX exports rename outputs
            // (e.g. pred_logits / pred_boxes / pred_masks). Pick by position
            // and warn so the user can verify.
            const auto& outs = session.output_indices();
            if (outs.size() < 3) {
                throw std::runtime_error(
                    "rfdetr: RFDetrSegmenter requires an engine with three outputs "
                    "(dets, labels, masks). This engine has " +
                    std::to_string(outs.size()) + " output(s) — likely detection-only.");
            }
            log_message(LogSeverity::kWarning,
                "expected output names 'dets'/'labels'/'masks' not found — "
                "falling back to index order [0,1,2]");
            b_dets   = &session.bindings()[outs[0]];
            b_labels = &session.bindings()[outs[1]];
            b_masks  = &session.bindings()[outs[2]];
            // Re-validate the masks binding located by fallback index.
            if (b_masks->shape.nbDims < 4) {
                throw std::runtime_error(
                    "rfdetr: RFDetrSegmenter: fallback 'masks' tensor (index 2) has "
                    "fewer than 4 dims — engine output order may differ from expected");
            }
        }

        N = b_dets->shape.d[b_dets->shape.nbDims - 2];
        C = b_labels->shape.d[b_labels->shape.nbDims - 1];
        // masks shape: [B, num_queries, mH, mW] — read last two dims.
        const auto& ms = b_masks->shape;
        if (ms.nbDims < 4) {
            throw std::runtime_error(
                "rfdetr: RFDetrSegmenter: 'masks' tensor has fewer than 4 dims");
        }
        mH = ms.d[ms.nbDims - 2];
        mW = ms.d[ms.nbDims - 1];
        if (mH <= 0 || mW <= 0) {
            throw std::runtime_error(
                "rfdetr: RFDetrSegmenter: 'masks' tensor has unresolved spatial dims");
        }

        h_dets.resize(static_cast<std::size_t>(N) * 4);
        h_labels.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(C));
        h_masks.resize(static_cast<std::size_t>(N) *
                       static_cast<std::size_t>(mH) *
                       static_cast<std::size_t>(mW));

        pp.num_queries         = N;
        pp.num_classes_with_bg = C;
        pp.topk                = N;
        pp.threshold           = opts.threshold;
        pp.bg_class_index      = 0;

        if (opts.use_cuda_graph) {
            if (!meta.cuda_graph_compat) {
                log_message(LogSeverity::kWarning,
                    "use_cuda_graph=true but engine was not built with --cuda-graph. "
                    "Disabling CUDA Graph for this session.");
            } else {
                capture_graph_();
            }
        }
    }

    ~Impl() {
        destroy_graph_();
    }

    void destroy_graph_() {
        if (graph_exec) { cudaGraphExecDestroy(graph_exec); graph_exec = nullptr; }
        if (graph)      { cudaGraphDestroy(graph);          graph      = nullptr; }
        graph_captured = false;
    }

    void capture_graph_() {
        destroy_graph_();

        auto* ctx = session.context();
        ctx->setEnqueueEmitsProfile(false);

        float* d_input = static_cast<float*>(session.device_buffer("input"));
        cv::Mat dummy(meta.input_h, meta.input_w, CV_8UC3, cv::Scalar(128));

        for (int i = 0; i < opts.graph_warmup_iters + 2; ++i) {
            preprocessor->process(dummy, d_input, session.stream());
            session.infer();
            session.get_output_f32(b_dets->name,   h_dets.data(),   h_dets.size());
            session.get_output_f32(b_labels->name, h_labels.data(), h_labels.size());
            session.get_output_f32(b_masks->name,  h_masks.data(),  h_masks.size());
        }

        RFDETR_CUDA_CHECK(cudaStreamBeginCapture(session.stream(), cudaStreamCaptureModeGlobal));
        bool ok = ctx->enqueueV3(session.stream());
        if (ok) {
            cudaMemcpyAsync(h_dets.data(), session.device_buffer(b_dets->name),
                            h_dets.size() * sizeof(float), cudaMemcpyDeviceToHost, session.stream());
            cudaMemcpyAsync(h_labels.data(), session.device_buffer(b_labels->name),
                            h_labels.size() * sizeof(float), cudaMemcpyDeviceToHost, session.stream());
            cudaMemcpyAsync(h_masks.data(), session.device_buffer(b_masks->name),
                            h_masks.size() * sizeof(float), cudaMemcpyDeviceToHost, session.stream());
        }
        cudaStreamEndCapture(session.stream(), &graph);
        cudaGetLastError();

        if (ok && graph && cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess) {
            graph_captured = true;
            log_message(LogSeverity::kInfo, "CUDA Graph captured successfully (segmenter)");
        } else {
            log_message(LogSeverity::kWarning,
                "CUDA Graph capture failed (segmenter) — using enqueueV3");
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        }
    }

    Detections run(const cv::Mat& image, float threshold) {
        using Clock = std::chrono::steady_clock;

        // A dynamic-batch engine resolves no shapes until one is set, so the
        // input binding has no device buffer yet — and a prior segment_batch()
        // may have left the profile at B>1. set_input_shape early-returns when
        // the shape is already correct, so the steady-state cost is a compare.
        if (meta.dynamic_batch) {
            const int R = meta.input_h;
            session.set_input_shape("input", nvinfer1::Dims4{1, 3, R, R});
        }

        float* d_input = static_cast<float*>(session.device_buffer("input"));

        // Preprocess queued on the session stream — no host sync; infer sees
        // the work in stream order. preprocess timing merges into infer_ms.
        const auto t0 = Clock::now();
        preprocessor->process(image, d_input, session.stream());

        if (graph_captured) {
            RFDETR_CUDA_CHECK(cudaGraphLaunch(graph_exec, session.stream()));
            RFDETR_CUDA_CHECK(cudaStreamSynchronize(session.stream()));
        } else {
            session.infer();
            // Explicit batch-1 counts: segment_batch() may have grown h_*, and
            // get_output_f32 requires an exact element count.
            session.get_output_f32(b_dets->name,   h_dets.data(),
                                   static_cast<std::size_t>(N) * 4);
            session.get_output_f32(b_labels->name, h_labels.data(),
                                   static_cast<std::size_t>(N) * C);
            session.get_output_f32(b_masks->name,  h_masks.data(),
                                   static_cast<std::size_t>(N) * mH * mW);
        }
        const auto t1 = t0;  // preprocess merged into infer_ms
        const auto t2 = Clock::now();

        pp.threshold = (threshold >= 0.0f) ? threshold : opts.threshold;
        std::vector<int> query_idx;
        Detections results = decode_detections_with_queries(
            h_dets.data(), h_labels.data(), image.cols, image.rows, pp, query_idx);
        gpu_decode_masks(h_masks.data(), query_idx, mH, mW, image.rows, image.cols,
                         results, mask_scratch.get(), session.stream());
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

        const int R       = meta.input_h;
        const std::size_t single_floats = static_cast<std::size_t>(3) * R * R;

        if (meta.dynamic_batch) {
            if (B > meta.max_batch) {
                throw std::runtime_error("rfdetr: segment_batch: batch size " +
                    std::to_string(B) + " exceeds engine max_batch " +
                    std::to_string(meta.max_batch));
            }
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

        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        pp.threshold = (threshold >= 0.0f) ? threshold : opts.threshold;
        const std::size_t mask_stride = static_cast<std::size_t>(N) * mH * mW;
        for (int i = 0; i < B; ++i) {
            const float* d = h_dets.data()   + i * N * 4;
            const float* l = h_labels.data() + i * N * C;
            const float* m = h_masks.data()  + i * mask_stride;
            std::vector<int> q;
            Detections r = decode_detections_with_queries(d, l, images[i].cols,
                                                           images[i].rows, pp, q);
            gpu_decode_masks(m, q, mH, mW, images[i].rows, images[i].cols, r,
                             mask_scratch.get(), session.stream());
            results.push_back(std::move(r));
        }
        return results;
    }
};

RFDetrSegmenter::RFDetrSegmenter(const std::filesystem::path& engine_path,
                                  const SegmenterOptions& opts) {
    init_(engine_path, engine_path.string() + ".json", opts);
}

RFDetrSegmenter::RFDetrSegmenter(const std::filesystem::path& engine_path,
                                  const std::filesystem::path& meta_path,
                                  const SegmenterOptions& opts) {
    init_(engine_path, meta_path, opts);
}

void RFDetrSegmenter::init_(const std::filesystem::path& engine_path,
                             const std::filesystem::path& meta_path,
                             const SegmenterOptions& opts) {
    impl_ = std::make_unique<Impl>(engine_path, meta_path, opts);
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

const std::string& RFDetrSegmenter::variant()          const noexcept { return impl_->meta.variant; }
int                RFDetrSegmenter::input_h()           const noexcept { return impl_->meta.input_h; }
int                RFDetrSegmenter::input_w()           const noexcept { return impl_->meta.input_w; }
int                RFDetrSegmenter::num_queries()       const noexcept { return impl_->N; }
int                RFDetrSegmenter::num_classes()       const noexcept { return impl_->meta.num_classes; }
int                RFDetrSegmenter::mask_h()            const noexcept { return impl_->mH; }
int                RFDetrSegmenter::mask_w()            const noexcept { return impl_->mW; }
bool               RFDetrSegmenter::cuda_graph_active() const noexcept { return impl_->graph_captured; }

const RFDetrSegmenter::Timings& RFDetrSegmenter::last_timings() const noexcept {
    return impl_->timings;
}

}  // namespace rfdetr
