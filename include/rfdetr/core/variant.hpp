#pragma once

#include <string_view>

namespace rfdetr {

// RF-DETR variants exported by the upstream `roboflow/rf-detr` repo.
// Sources: src/rfdetr/variants.py, src/rfdetr/config.py.
enum class Variant {
    Unknown,
    // Detection (Apache-2.0)
    Nano,        // 384, 300 queries, patch 16
    Small,       // 512, 300 queries, patch 16
    Medium,      // 576, 300 queries, patch 16
    Base,        // 560, 300 queries, patch 14 (DINOv2 default)
    Large,       // 704, 300 queries, patch 16
    // Segmentation (Apache-2.0)
    SegNano,     // 312, 100 queries
    SegSmall,    // 384, 100 queries
    SegMedium,   // 432, 200 queries
    SegLarge,    // 504, 200 queries
    SegXL,       // 624, 300 queries
    Seg2XL,      // 768, 300 queries
    SegPreview,  // 432, 200 queries
};

struct VariantMeta {
    Variant variant;
    std::string_view name;
    int input_size;    // square: H == W
    int num_queries;   // 100 / 200 / 300
    int num_classes;   // 90 across all upstream variants (COCO 91-id sparse, ID 0 = bg)
    int patch_size;    // 12 / 14 / 16
    bool has_masks;    // true for segmentation variants
};

// Look up the metadata table by enum or by canonical name.
// On miss, returns the Unknown row (variant == Variant::Unknown, all numeric fields 0).
const VariantMeta& variant_meta(Variant v) noexcept;
const VariantMeta& variant_meta(std::string_view name) noexcept;

Variant variant_from_name(std::string_view name) noexcept;
std::string_view variant_to_name(Variant v) noexcept;

// Output mask resolution at the network head: input_size / 4 (Mask2Former-style).
inline int mask_resolution(const VariantMeta& m) noexcept {
    return m.has_masks ? m.input_size / 4 : 0;
}

}  // namespace rfdetr
