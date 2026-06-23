/*
 * rfdetr/c_api.h — stable C ABI for rf-detr-cpp.
 *
 * Opaque handles, raw-byte image input, heap-allocated result sets.
 * No C++ or OpenCV types in this header — safe to include from pure C,
 * Python ctypes, Rust FFI, or any other language with a C FFI layer.
 *
 * Compile the library with -DRFDETR_BUILD_C_API=ON to enable.
 *
 * Thread safety: each handle is NOT thread-safe. Create one handle per thread.
 *
 * Quick example (detection):
 *
 *   rfdetr_detector_t* det = rfdetr_detector_create("model.engine", NULL);
 *   if (!det) { fprintf(stderr, "%s\n", rfdetr_last_error()); return 1; }
 *
 *   rfdetr_detections_t* res = rfdetr_detector_detect(
 *       det, bgr_data, width, height, width * 3, 0.5f);
 *   for (int i = 0; i < res->count; ++i) {
 *       printf("cls=%d score=%.2f\n",
 *              res->detections[i].class_id, res->detections[i].score);
 *   }
 *   rfdetr_detections_free(res);
 *   rfdetr_detector_destroy(det);
 */

#ifndef RFDETR_C_API_H
#define RFDETR_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Visibility / export macros
 * ---------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef RFDETR_BUILDING_LIB
#    define RFDETR_API __declspec(dllexport)
#  else
#    define RFDETR_API __declspec(dllimport)
#  endif
#else
#  define RFDETR_API __attribute__((visibility("default")))
#endif

/* -------------------------------------------------------------------------
 * Opaque handle types
 * ---------------------------------------------------------------------- */
typedef struct rfdetr_detector_s  rfdetr_detector_t;   /* detection engine  */
typedef struct rfdetr_segmenter_s rfdetr_segmenter_t;  /* segmentation engine */

/* -------------------------------------------------------------------------
 * Data types
 * ---------------------------------------------------------------------- */

/* Bounding box in original-image pixel coordinates, xyxy format. */
typedef struct rfdetr_box_s {
    float x1, y1, x2, y2;
} rfdetr_box_t;

/*
 * Single detection result.
 *
 * class_id : foreground dense index (0 = first foreground class, e.g. "person").
 *            Add 1 to get the COCO-91 sparse id.
 *
 * mask_data : binary mask, CV_8UC1 flattened row-major (0 or 255).
 *             Non-NULL only when produced by rfdetr_segmenter_segment().
 *             Width = mask_width, Height = mask_height (same as input image).
 *             Memory is owned by the parent rfdetr_detections_t; freed by
 *             rfdetr_detections_free().  Do NOT free mask_data directly.
 */
typedef struct rfdetr_detection_s {
    rfdetr_box_t box;
    int          class_id;
    float        score;
    uint8_t*     mask_data;    /* NULL unless from segmenter */
    int          mask_width;
    int          mask_height;
} rfdetr_detection_t;

/*
 * Heap-allocated set of detection results.
 * Free with rfdetr_detections_free() when done.
 */
typedef struct rfdetr_detections_s {
    rfdetr_detection_t* detections;
    int                 count;
} rfdetr_detections_t;

/* -------------------------------------------------------------------------
 * Error reporting
 * ---------------------------------------------------------------------- */

/*
 * Returns a human-readable description of the last error on the calling
 * thread.  Valid until the next rfdetr_* call on the same thread.
 * Never returns NULL (returns "" if no error has occurred).
 */
RFDETR_API const char* rfdetr_last_error(void);

/* -------------------------------------------------------------------------
 * Log callback
 * ---------------------------------------------------------------------- */

typedef void (*rfdetr_log_fn_t)(const char* message);

/*
 * Override the default stderr logger.  Pass NULL to restore the default.
 * The callback is called with a null-terminated UTF-8 string from any thread.
 */
RFDETR_API void rfdetr_set_log_callback(rfdetr_log_fn_t callback);

/* -------------------------------------------------------------------------
 * Detector — rfdetr_detector_t
 * ---------------------------------------------------------------------- */

/*
 * Create a detector from a TensorRT engine file.
 *
 * engine_path : path to the .engine file (UTF-8, null-terminated).
 * meta_path   : path to the .json sidecar, or NULL to use
 *               "<engine_path>.json" automatically.
 *
 * Returns a non-NULL handle on success, NULL on failure.
 * Call rfdetr_last_error() for a diagnostic on failure.
 */
RFDETR_API rfdetr_detector_t* rfdetr_detector_create(const char* engine_path,
                                                      const char* meta_path);

/* Destroy a detector and free all associated resources. */
RFDETR_API void rfdetr_detector_destroy(rfdetr_detector_t* det);

/* Introspection. */
RFDETR_API const char* rfdetr_detector_variant(const rfdetr_detector_t* det);
RFDETR_API int         rfdetr_detector_input_width(const rfdetr_detector_t* det);
RFDETR_API int         rfdetr_detector_input_height(const rfdetr_detector_t* det);
RFDETR_API int         rfdetr_detector_num_queries(const rfdetr_detector_t* det);
RFDETR_API int         rfdetr_detector_num_classes(const rfdetr_detector_t* det);

/*
 * Run detection on a single image.
 *
 * bgr_data  : pointer to the first byte of the image in BGR uint8 interleaved format.
 * width     : image width in pixels.
 * height    : image height in pixels.
 * step      : row stride in bytes (typically width * 3; use width * 3 if contiguous).
 * threshold : score threshold in [0,1]; pass <=0 to use the default (0.5).
 *
 * Returns a heap-allocated rfdetr_detections_t* on success (may have count=0),
 * or NULL on failure.  Call rfdetr_detections_free() when done.
 */
RFDETR_API rfdetr_detections_t* rfdetr_detector_detect(rfdetr_detector_t* det,
                                                        const uint8_t* bgr_data,
                                                        int width, int height, int step,
                                                        float threshold);

/* -------------------------------------------------------------------------
 * Segmenter — rfdetr_segmenter_t
 * ---------------------------------------------------------------------- */

/*
 * Create a segmenter from a seg-* TRT engine.
 * Arguments identical to rfdetr_detector_create.
 */
RFDETR_API rfdetr_segmenter_t* rfdetr_segmenter_create(const char* engine_path,
                                                        const char* meta_path);

RFDETR_API void rfdetr_segmenter_destroy(rfdetr_segmenter_t* seg);

RFDETR_API const char* rfdetr_segmenter_variant(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_input_width(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_input_height(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_num_queries(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_num_classes(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_mask_width(const rfdetr_segmenter_t* seg);
RFDETR_API int         rfdetr_segmenter_mask_height(const rfdetr_segmenter_t* seg);

/*
 * Run segmentation on a single image.
 * Each returned detection has mask_data filled (non-NULL, same size as the input image).
 * Arguments and return value identical to rfdetr_detector_detect.
 */
RFDETR_API rfdetr_detections_t* rfdetr_segmenter_segment(rfdetr_segmenter_t* seg,
                                                          const uint8_t* bgr_data,
                                                          int width, int height, int step,
                                                          float threshold);

/* -------------------------------------------------------------------------
 * Result memory management
 * ---------------------------------------------------------------------- */

/*
 * Free a detection result set returned by rfdetr_detector_detect() or
 * rfdetr_segmenter_segment().  Safe to call with NULL.
 * Also frees any mask_data pointers inside the detections.
 */
RFDETR_API void rfdetr_detections_free(rfdetr_detections_t* dets);

/* -------------------------------------------------------------------------
 * Version
 * ---------------------------------------------------------------------- */

/* Returns the library version string, e.g. "0.0.1". */
RFDETR_API const char* rfdetr_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RFDETR_C_API_H */
