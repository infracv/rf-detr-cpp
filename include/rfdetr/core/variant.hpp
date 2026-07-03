#pragma once

#include <string_view>

namespace rfdetr {

// RF-DETR variants exported by the upstream `roboflow/rf-detr` repo.
enum class Variant {
    Unknown,
    // Detection (Apache-2.0)
    Nano,
    Small,
    Medium,
    Base,
    Large,
    // Instance segmentation (Apache-2.0)
    SegNano,
    SegSmall,
    SegMedium,
    SegLarge,
    SegXLarge,
    Seg2XLarge,
};

struct VariantMeta {
    Variant variant;
    std::string_view name;
    int input_size;
    int num_queries;
    int num_classes;   // 90 across all upstream variants (COCO 91-id sparse, ID 0 = bg)
    int patch_size;    // 12 / 14 / 16
};

// Look up the metadata table by enum or by canonical name.
// On miss, returns the Unknown row (variant == Variant::Unknown, all numeric fields 0).
const VariantMeta& variant_meta(Variant v) noexcept;
const VariantMeta& variant_meta(std::string_view name) noexcept;

Variant variant_from_name(std::string_view name) noexcept;
std::string_view variant_to_name(Variant v) noexcept;

}  // namespace rfdetr
