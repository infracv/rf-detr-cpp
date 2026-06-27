#include "rfdetr/core/variant.hpp"

#include <array>

namespace rfdetr {

namespace {

// Single source of truth for RF-DETR variant metadata.
// num_classes is 90 across all upstream variants (COCO 91-id sparse, ID 0 = background).
constexpr std::array<VariantMeta, 12> kVariants = {{
    {Variant::Unknown,    "unknown",     0,   0,  0,  0},
    // Detection
    {Variant::Nano,       "nano",       384, 300, 90, 16},
    {Variant::Small,      "small",      512, 300, 90, 16},
    {Variant::Medium,     "medium",     576, 300, 90, 16},
    {Variant::Base,       "base",       560, 300, 90, 14},
    {Variant::Large,      "large",      704, 300, 90, 16},
    // Instance segmentation (mask_h = mask_w = input_size / 4)
    {Variant::SegNano,    "seg-nano",   312, 100, 90, 12},
    {Variant::SegSmall,   "seg-small",  384, 100, 90, 12},
    {Variant::SegMedium,  "seg-medium", 432, 200, 90, 12},
    {Variant::SegLarge,   "seg-large",  504, 200, 90, 12},
    {Variant::SegXLarge,  "seg-xlarge", 624, 300, 90, 12},
    {Variant::Seg2XLarge, "seg-2xlarge",768, 300, 90, 12},
}};

}  // namespace

const VariantMeta& variant_meta(Variant v) noexcept {
    for (const auto& m : kVariants) {
        if (m.variant == v) return m;
    }
    return kVariants[0];  // Unknown
}

const VariantMeta& variant_meta(std::string_view name) noexcept {
    for (const auto& m : kVariants) {
        if (m.name == name) return m;
    }
    return kVariants[0];
}

Variant variant_from_name(std::string_view name) noexcept {
    return variant_meta(name).variant;
}

std::string_view variant_to_name(Variant v) noexcept {
    return variant_meta(v).name;
}

}  // namespace rfdetr
