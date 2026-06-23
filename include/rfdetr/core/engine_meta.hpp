#pragma once

#include <array>
#include <filesystem>
#include <string>

namespace rfdetr {

// Sidecar metadata stored alongside a built TensorRT engine as `<engine>.json`.
// Removes the need for fragile shape-based variant autodetection at runtime.
//
// Defaults match RF-DETR upstream:
//   color_order = "RGB", mean = ImageNet, std = ImageNet, num_classes = 90.
struct EngineMeta {
    std::string variant;            // canonical name, e.g. "small", "seg-large"
    int input_h{0};
    int input_w{0};
    int num_queries{0};
    int num_classes{0};
    bool has_masks{false};
    int mask_h{0};                  // 0 if !has_masks, else input_h / 4
    int mask_w{0};

    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
    std::string color_order{"RGB"};

    std::string precision{"fp16"};  // "fp32" / "fp16" / "int8"
    int patch_size{0};
    bool dynamic_batch{false};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph_compat{false};  // built with kCUDA_GRAPH_COMPATIBLE

    static EngineMeta from_json_file(const std::filesystem::path& path);
    void to_json_file(const std::filesystem::path& path) const;
};

}  // namespace rfdetr
