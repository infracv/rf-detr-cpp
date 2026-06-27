#pragma once

#include <array>
#include <filesystem>
#include <string>

namespace rfdetr {

// Bump when the JSON schema gains fields that would break older readers.
inline constexpr int kEngineMetaSchemaVersion = 1;

struct EngineMeta {
    int schema_version{kEngineMetaSchemaVersion};
    std::string variant;            // canonical name, e.g. "small", "large"
    int input_h{0};
    int input_w{0};
    int num_queries{0};
    int num_classes{0};

    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
    std::string color_order{"RGB"};

    std::string precision{"fp16"};  // "fp32" / "fp16" / "int8"
    int patch_size{0};
    bool dynamic_batch{false};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph_compat{false};  // meta label set by rfdetr_build --cuda-graph

    // Segmentation outputs (filled for seg-* variants, zero otherwise).
    bool has_masks{false};
    int  mask_h{0};
    int  mask_w{0};

    static EngineMeta from_json_file(const std::filesystem::path& path);
    void to_json_file(const std::filesystem::path& path) const;
};

}  // namespace rfdetr
