#include "rfdetr/core/engine_meta.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace rfdetr {

using nlohmann::json;

EngineMeta EngineMeta::from_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("rfdetr: cannot open meta sidecar: " + path.string());
    }
    json j;
    in >> j;

    EngineMeta m;
    m.schema_version = j.value("schema_version", 0);
    m.variant     = j.value("variant", std::string{});
    m.input_h     = j.value("input_h", 0);
    m.input_w     = j.value("input_w", 0);
    m.num_queries = j.value("num_queries", 0);
    m.num_classes = j.value("num_classes", 0);

    if (j.contains("mean") && j["mean"].is_array() && j["mean"].size() == 3) {
        m.mean = j["mean"].get<std::array<float, 3>>();
    }
    if (j.contains("std") && j["std"].is_array() && j["std"].size() == 3) {
        m.std = j["std"].get<std::array<float, 3>>();
    }

    m.color_order   = j.value("color_order", std::string{"RGB"});
    m.precision     = j.value("precision", std::string{"fp16"});
    m.patch_size    = j.value("patch_size", 0);
    m.dynamic_batch      = j.value("dynamic_batch", false);
    m.min_batch          = j.value("min_batch", 1);
    m.opt_batch          = j.value("opt_batch", 1);
    m.max_batch          = j.value("max_batch", 1);
    m.cuda_graph_compat  = j.value("cuda_graph_compat", false);
    m.has_masks          = j.value("has_masks", false);
    m.mask_h             = j.value("mask_h", 0);
    m.mask_w             = j.value("mask_w", 0);
    return m;
}

void EngineMeta::to_json_file(const std::filesystem::path& path) const {
    json j = {
        {"schema_version", schema_version},
        {"variant", variant},
        {"input_h", input_h},
        {"input_w", input_w},
        {"num_queries", num_queries},
        {"num_classes", num_classes},
        {"mean", mean},
        {"std", std},
        {"color_order", color_order},
        {"precision", precision},
        {"patch_size", patch_size},
        {"dynamic_batch", dynamic_batch},
        {"min_batch", min_batch},
        {"opt_batch", opt_batch},
        {"max_batch", max_batch},
        {"cuda_graph_compat", cuda_graph_compat},
        {"has_masks", has_masks},
        {"mask_h", mask_h},
        {"mask_w", mask_w},
    };
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("rfdetr: cannot write meta sidecar: " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace rfdetr
