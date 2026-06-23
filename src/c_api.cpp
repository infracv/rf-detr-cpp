#include "rfdetr/c_api.h"

#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/tasks/segmenter.hpp"
#include "rfdetr/version.hpp"

#include <opencv2/core.hpp>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Thread-local error storage
// ---------------------------------------------------------------------------

static thread_local std::string t_last_error;

static void set_error(const std::string& msg) noexcept {
    t_last_error = msg;
}

// ---------------------------------------------------------------------------
// Log callback
// ---------------------------------------------------------------------------

static rfdetr_log_fn_t g_log_callback = nullptr;
static std::mutex      g_log_mutex;

static void log_message(const std::string& msg) noexcept {
    rfdetr_log_fn_t cb = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_log_mutex);
        cb = g_log_callback;
    }
    if (cb) {
        cb(msg.c_str());
    } else {
        std::fputs(msg.c_str(), stderr);
        std::fputs("\n", stderr);
    }
}

// ---------------------------------------------------------------------------
// Opaque handle structs
// ---------------------------------------------------------------------------

struct rfdetr_detector_s {
    rfdetr::RFDetrDetector obj;
    explicit rfdetr_detector_s(rfdetr::RFDetrDetector&& d) : obj(std::move(d)) {}
};

struct rfdetr_segmenter_s {
    rfdetr::RFDetrSegmenter obj;
    explicit rfdetr_segmenter_s(rfdetr::RFDetrSegmenter&& s) : obj(std::move(s)) {}
};

// ---------------------------------------------------------------------------
// Helper: convert rfdetr::Detections -> heap-allocated rfdetr_detections_t
// ---------------------------------------------------------------------------

static rfdetr_detections_t* pack_detections(const rfdetr::Detections& dets) {
    auto* out = new rfdetr_detections_t;
    out->count      = static_cast<int>(dets.size());
    out->detections = (out->count > 0)
                          ? new rfdetr_detection_t[static_cast<std::size_t>(out->count)]
                          : nullptr;

    for (int i = 0; i < out->count; ++i) {
        const auto& d  = dets[static_cast<std::size_t>(i)];
        auto&       od = out->detections[i];
        od.box.x1   = d.box.x1;
        od.box.y1   = d.box.y1;
        od.box.x2   = d.box.x2;
        od.box.y2   = d.box.y2;
        od.class_id = d.class_id;
        od.score    = d.score;

        if (!d.mask.empty()) {
            const int bytes = d.mask.rows * d.mask.cols;
            od.mask_data   = new uint8_t[static_cast<std::size_t>(bytes)];
            od.mask_width  = d.mask.cols;
            od.mask_height = d.mask.rows;
            std::memcpy(od.mask_data, d.mask.data, static_cast<std::size_t>(bytes));
        } else {
            od.mask_data   = nullptr;
            od.mask_width  = 0;
            od.mask_height = 0;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Helper: wrap raw BGR bytes as a cv::Mat without copying
// ---------------------------------------------------------------------------

static cv::Mat wrap_bgr(const uint8_t* bgr_data, int width, int height, int step) {
    return cv::Mat(height, width, CV_8UC3,
                   const_cast<void*>(static_cast<const void*>(bgr_data)),
                   static_cast<std::size_t>(step));
}

// ---------------------------------------------------------------------------
// Public C API implementation
// ---------------------------------------------------------------------------

extern "C" {

const char* rfdetr_last_error(void) {
    return t_last_error.c_str();
}

void rfdetr_set_log_callback(rfdetr_log_fn_t callback) {
    std::lock_guard<std::mutex> lk(g_log_mutex);
    g_log_callback = callback;
}

const char* rfdetr_version(void) {
    return rfdetr::version();
}

// ----- Detector -----

rfdetr_detector_t* rfdetr_detector_create(const char* engine_path, const char* meta_path) {
    t_last_error.clear();
    if (!engine_path) { set_error("rfdetr_detector_create: engine_path is NULL"); return nullptr; }
    try {
        rfdetr::RFDetrDetector det =
            meta_path
                ? rfdetr::RFDetrDetector(engine_path, meta_path)
                : rfdetr::RFDetrDetector(engine_path);
        return new rfdetr_detector_s(std::move(det));
    } catch (const std::exception& e) {
        set_error(e.what());
        log_message(std::string("rfdetr_detector_create failed: ") + e.what());
        return nullptr;
    }
}

void rfdetr_detector_destroy(rfdetr_detector_t* det) {
    delete det;
}

const char* rfdetr_detector_variant(const rfdetr_detector_t* det) {
    if (!det) return "";
    return det->obj.variant().c_str();
}
int rfdetr_detector_input_width(const rfdetr_detector_t* det) {
    return det ? det->obj.input_w() : 0;
}
int rfdetr_detector_input_height(const rfdetr_detector_t* det) {
    return det ? det->obj.input_h() : 0;
}
int rfdetr_detector_num_queries(const rfdetr_detector_t* det) {
    return det ? det->obj.num_queries() : 0;
}
int rfdetr_detector_num_classes(const rfdetr_detector_t* det) {
    return det ? det->obj.num_classes() : 0;
}

rfdetr_detections_t* rfdetr_detector_detect(rfdetr_detector_t* det,
                                             const uint8_t* bgr_data,
                                             int width, int height, int step,
                                             float threshold) {
    t_last_error.clear();
    if (!det)      { set_error("rfdetr_detector_detect: det is NULL");      return nullptr; }
    if (!bgr_data) { set_error("rfdetr_detector_detect: bgr_data is NULL"); return nullptr; }
    try {
        cv::Mat img = wrap_bgr(bgr_data, width, height, step);
        auto dets   = det->obj.detect(img, threshold);
        return pack_detections(dets);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

// ----- Segmenter -----

rfdetr_segmenter_t* rfdetr_segmenter_create(const char* engine_path, const char* meta_path) {
    t_last_error.clear();
    if (!engine_path) { set_error("rfdetr_segmenter_create: engine_path is NULL"); return nullptr; }
    try {
        rfdetr::RFDetrSegmenter seg =
            meta_path
                ? rfdetr::RFDetrSegmenter(engine_path, meta_path)
                : rfdetr::RFDetrSegmenter(engine_path);
        return new rfdetr_segmenter_s(std::move(seg));
    } catch (const std::exception& e) {
        set_error(e.what());
        log_message(std::string("rfdetr_segmenter_create failed: ") + e.what());
        return nullptr;
    }
}

void rfdetr_segmenter_destroy(rfdetr_segmenter_t* seg) {
    delete seg;
}

const char* rfdetr_segmenter_variant(const rfdetr_segmenter_t* seg) {
    if (!seg) return "";
    return seg->obj.variant().c_str();
}
int rfdetr_segmenter_input_width(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.input_w() : 0;
}
int rfdetr_segmenter_input_height(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.input_h() : 0;
}
int rfdetr_segmenter_num_queries(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.num_queries() : 0;
}
int rfdetr_segmenter_num_classes(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.num_classes() : 0;
}
int rfdetr_segmenter_mask_width(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.mask_w() : 0;
}
int rfdetr_segmenter_mask_height(const rfdetr_segmenter_t* seg) {
    return seg ? seg->obj.mask_h() : 0;
}

rfdetr_detections_t* rfdetr_segmenter_segment(rfdetr_segmenter_t* seg,
                                               const uint8_t* bgr_data,
                                               int width, int height, int step,
                                               float threshold) {
    t_last_error.clear();
    if (!seg)      { set_error("rfdetr_segmenter_segment: seg is NULL");      return nullptr; }
    if (!bgr_data) { set_error("rfdetr_segmenter_segment: bgr_data is NULL"); return nullptr; }
    try {
        cv::Mat img = wrap_bgr(bgr_data, width, height, step);
        auto dets   = seg->obj.segment(img, threshold);
        return pack_detections(dets);
    } catch (const std::exception& e) {
        set_error(e.what());
        return nullptr;
    }
}

// ----- Memory management -----

void rfdetr_detections_free(rfdetr_detections_t* dets) {
    if (!dets) return;
    if (dets->detections) {
        for (int i = 0; i < dets->count; ++i) {
            delete[] dets->detections[i].mask_data;
        }
        delete[] dets->detections;
    }
    delete dets;
}

}  // extern "C"
