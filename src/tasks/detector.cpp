#include "rfdetr/tasks/detector.hpp"

#include "rfdetr/core/coco_classes.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/log.hpp"
#include "rfdetr/core/postprocess.hpp"
#include "../internal/cuda_check.hpp"
#include "../internal/cuda_preprocess.cuh"
#include "../internal/trt_session.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace rfdetr {

struct RFDetrDetector::Impl {
    TrtSession                         session;
    EngineMeta                         meta;
    std::unique_ptr<ImagePreprocessor> preprocessor;
    DetectorOptions                    opts;

    // Output tensor metadata (resolved once at init).
    const BindingInfo* b_dets{nullptr};
    const BindingInfo* b_labels{nullptr};
    int N{0};  // num_queries
    int C{0};  // num_classes_with_bg

    std::vector<float> h_dets;
    std::vector<float> h_labels;

    PostprocessParams pp;
    Timings           timings;

    bool            graph_captured{false};
    cudaGraph_t     graph{nullptr};
    cudaGraphExec_t graph_exec{nullptr};

    Impl(const std::filesystem::path& engine_path,
         const std::filesystem::path& meta_path,
         const DetectorOptions& options)
        : session(engine_path)
    {
        opts = options;

        // Load meta sidecar.
        if (std::filesystem::is_regular_file(meta_path)) {
            meta = EngineMeta::from_json_file(meta_path);
        } else {
            // Fallback: infer dims from engine bindings.
            const auto* in_b = session.find("input");
            if (!in_b || in_b->shape.nbDims < 4) {
                throw std::runtime_error(
                    "rfdetr: RFDetrDetector: no meta sidecar at " + meta_path.string() +
                    " and cannot infer input dims — pass explicit meta path");
            }
            meta.input_h    = in_b->shape.d[2];
            meta.input_w    = in_b->shape.d[3];
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
        if (!b_dets || !b_labels) {
            const auto& outs = session.output_indices();
            if (outs.size() < 2) {
                throw std::runtime_error("rfdetr: engine has fewer than 2 outputs");
            }
            b_dets   = &session.bindings()[outs[0]];
            b_labels = &session.bindings()[outs[1]];
        }

        N = b_dets->shape.d[b_dets->shape.nbDims - 2];
        C = b_labels->shape.d[b_labels->shape.nbDims - 1];
        h_dets.resize(static_cast<std::size_t>(N) * 4);
        h_labels.resize(static_cast<std::size_t>(N) * static_cast<std::size_t>(C));

        pp.num_queries         = N;
        pp.num_classes_with_bg = C;
        pp.topk                = N;
        pp.threshold           = opts.threshold;
        pp.bg_class_index      = 0;

        // Attempt CUDA Graph capture if requested.
        if (opts.use_cuda_graph) {
            if (!meta.cuda_graph_compat) {
                log_message(LogSeverity::kWarning,
                    "use_cuda_graph=true but engine was not built with --cuda-graph "
                    "(kCUDA_GRAPH_COMPATIBLE). Rebuild with `rfdetr_build --onnx <onnx> "
                    "--cuda-graph`. Disabling CUDA Graph for this session.");
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

    // Capture a CUDA Graph covering enqueueV3 + D2H.
    // H2D stays outside the graph (image bytes differ per frame).
    // Falls back silently if the engine has data-dependent internal shapes
    // (e.g. RF-DETR cross-attention — common with transformer networks).
    void capture_graph_() {
        destroy_graph_();

        auto* ctx = session.context();
        ctx->setEnqueueEmitsProfile(false);

        float* d_input = static_cast<float*>(session.device_buffer("input"));
        cv::Mat dummy(meta.input_h, meta.input_w, CV_8UC3, cv::Scalar(128));

        // Warm up: flush lazy CUDA init + any TRT timing caches.
        for (int i = 0; i < opts.graph_warmup_iters + 2; ++i) {
            preprocessor->process(dummy, d_input, session.stream());
            session.infer();
            session.get_output_f32(b_dets->name,   h_dets.data(),   h_dets.size());
            session.get_output_f32(b_labels->name, h_labels.data(), h_labels.size());
        }

        RFDETR_CUDA_CHECK(cudaStreamBeginCapture(session.stream(), cudaStreamCaptureModeGlobal));
        bool ok = ctx->enqueueV3(session.stream());
        if (ok) {
            cudaMemcpyAsync(h_dets.data(), session.device_buffer(b_dets->name),
                            h_dets.size() * sizeof(float), cudaMemcpyDeviceToHost, session.stream());
            cudaMemcpyAsync(h_labels.data(), session.device_buffer(b_labels->name),
                            h_labels.size() * sizeof(float), cudaMemcpyDeviceToHost, session.stream());
        }
        cudaStreamEndCapture(session.stream(), &graph);
        cudaGetLastError();  // clear any sticky error from inside capture

        if (ok && graph && cudaGraphInstantiate(&graph_exec, graph, 0) == cudaSuccess) {
            graph_captured = true;
            log_message(LogSeverity::kInfo, "CUDA Graph captured successfully");
        } else {
            log_message(LogSeverity::kWarning,
                "CUDA Graph capture failed (engine has dynamic internal shapes) — "
                "using enqueueV3");
            if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
        }
    }

    // Batch inference: upload B images, run one engine call, decode each result.
    std::vector<Detections> run_batch(const std::vector<cv::Mat>& images, float threshold) {
        const int B = static_cast<int>(images.size());
        if (B == 0) return {};

        // Resize device buffer to hold B × (3 × H × W) floats.
        const int R       = meta.input_h;
        const std::size_t single_floats = static_cast<std::size_t>(3) * R * R;
        const std::size_t batch_bytes   = static_cast<std::size_t>(B) * single_floats * sizeof(float);

        // For dynamic-batch engines, update the input shape before inference.
        if (meta.dynamic_batch) {
            if (B > meta.max_batch) {
                throw std::runtime_error("rfdetr: detect_batch: batch size " +
                    std::to_string(B) + " exceeds engine max_batch " +
                    std::to_string(meta.max_batch));
            }
            nvinfer1::Dims4 dims{B, 3, R, R};
            session.set_input_shape("input", dims);
        } else if (B != 1) {
            throw std::runtime_error(
                "rfdetr: detect_batch: engine is static-batch (batch=1). "
                "Rebuild with --min-batch 1 --opt-batch N --max-batch N for batch > 1.");
        }

        // Upload each image into the contiguous device input buffer.
        float* d_input = static_cast<float*>(session.device_buffer("input"));
        for (int i = 0; i < B; ++i) {
            preprocessor->process(images[i], d_input + i * single_floats, session.stream());
        }

        // Single engine call covers the whole batch.
        session.infer();

        // Resize host output buffers for B-batch.
        const std::size_t dets_per_batch   = static_cast<std::size_t>(B) * N * 4;
        const std::size_t labels_per_batch = static_cast<std::size_t>(B) * N * C;
        if (h_dets.size()   < dets_per_batch)   h_dets.resize(dets_per_batch);
        if (h_labels.size() < labels_per_batch) h_labels.resize(labels_per_batch);

        session.get_output_f32(b_dets->name,   h_dets.data(),   dets_per_batch);
        session.get_output_f32(b_labels->name, h_labels.data(), labels_per_batch);

        // Decode per image.
        std::vector<Detections> results;
        results.reserve(static_cast<std::size_t>(B));
        pp.threshold = (threshold >= 0.0f) ? threshold : opts.threshold;
        for (int i = 0; i < B; ++i) {
            const float* d = h_dets.data()   + i * N * 4;
            const float* l = h_labels.data() + i * N * C;
            results.push_back(decode_detections(d, l,
                                                images[i].cols, images[i].rows, pp));
        }
        return results;
    }

    Detections run(const cv::Mat& image, float threshold) {
        using Clock = std::chrono::steady_clock;

        // A dynamic-batch engine resolves no shapes until one is set, so the
        // input binding has no device buffer yet — and a prior detect_batch()
        // may have left the profile at B>1. set_input_shape early-returns when
        // the shape is already correct, so the steady-state cost is a compare.
        if (meta.dynamic_batch) {
            const int R = meta.input_h;
            session.set_input_shape("input", nvinfer1::Dims4{1, 3, R, R});
        }

        float* d_input = static_cast<float*>(session.device_buffer("input"));

        // Preprocess (H2D + CUDA kernel) queued on the session stream — no host
        // sync here. infer()/cudaGraphLaunch enqueue on the same stream and TRT
        // sees the prior work in order, so async pipelining is preserved.
        // The host-side timing below collapses preprocess+infer into infer_ms;
        // this is the right trade for ~0.1–0.3 ms saved per call.
        const auto t0 = Clock::now();
        preprocessor->process(image, d_input, session.stream());

        if (graph_captured) {
            // H2D was queued above (outside graph), now launch graph (infer + D2H).
            RFDETR_CUDA_CHECK(cudaGraphLaunch(graph_exec, session.stream()));
            RFDETR_CUDA_CHECK(cudaStreamSynchronize(session.stream()));
        } else {
            session.infer();
            // Explicit batch-1 counts: detect_batch() may have grown h_*, and
            // get_output_f32 requires an exact element count.
            session.get_output_f32(b_dets->name,   h_dets.data(),
                                   static_cast<std::size_t>(N) * 4);
            session.get_output_f32(b_labels->name, h_labels.data(),
                                   static_cast<std::size_t>(N) * C);
        }
        const auto t1 = t0;  // preprocess merged into infer_ms
        const auto t2 = Clock::now();

        // Decode.
        pp.threshold = (threshold >= 0.0f) ? threshold : opts.threshold;
        Detections results = decode_detections(h_dets.data(), h_labels.data(),
                                               image.cols, image.rows, pp);
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
};

RFDetrDetector::RFDetrDetector(const std::filesystem::path& engine_path,
                                const DetectorOptions& opts) {
    init_(engine_path, engine_path.string() + ".json", opts);
}

RFDetrDetector::RFDetrDetector(const std::filesystem::path& engine_path,
                                const std::filesystem::path& meta_path,
                                const DetectorOptions& opts) {
    init_(engine_path, meta_path, opts);
}

void RFDetrDetector::init_(const std::filesystem::path& engine_path,
                            const std::filesystem::path& meta_path,
                            const DetectorOptions& opts) {
    impl_ = std::make_unique<Impl>(engine_path, meta_path, opts);
}

RFDetrDetector::~RFDetrDetector() = default;
RFDetrDetector::RFDetrDetector(RFDetrDetector&&) noexcept = default;
RFDetrDetector& RFDetrDetector::operator=(RFDetrDetector&&) noexcept = default;

Detections RFDetrDetector::detect(const cv::Mat& image, float threshold) {
    if (!impl_) throw std::runtime_error("rfdetr: RFDetrDetector: moved-from object");
    if (image.empty()) throw std::runtime_error("rfdetr: RFDetrDetector::detect: empty image");
    return impl_->run(image, threshold);
}

const std::string& RFDetrDetector::variant()          const noexcept { return impl_->meta.variant; }
int                RFDetrDetector::input_h()           const noexcept { return impl_->meta.input_h; }
int                RFDetrDetector::input_w()           const noexcept { return impl_->meta.input_w; }
int                RFDetrDetector::num_queries()       const noexcept { return impl_->N; }
int                RFDetrDetector::num_classes()       const noexcept { return impl_->meta.num_classes; }
bool               RFDetrDetector::cuda_graph_active() const noexcept { return impl_->graph_captured; }

const RFDetrDetector::Timings& RFDetrDetector::last_timings() const noexcept {
    return impl_->timings;
}

std::vector<Detections> RFDetrDetector::detect_batch(const std::vector<cv::Mat>& images,
                                                      float threshold) {
    if (!impl_) throw std::runtime_error("rfdetr: RFDetrDetector: moved-from object");
    if (images.empty()) return {};
    for (const auto& img : images) {
        if (img.empty()) throw std::runtime_error("rfdetr: detect_batch: image in batch is empty");
    }
    return impl_->run_batch(images, threshold);
}

}  // namespace rfdetr
