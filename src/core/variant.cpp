#include "rfdetr/core/variant.hpp"

#include <array>

namespace rfdetr {

namespace {

// Single source of truth for RF-DETR variant metadata.
// num_classes is 90 across all upstream variants (COCO 91-id sparse, ID 0 = background).
constexpr std::array<VariantMeta, 13> kVariants = {{
    {Variant::Unknown,    "unknown",     0,   0,  0,  0, false},
    // Detection
    {Variant::Nano,       "nano",        384, 300, 90, 16, false},
    {Variant::Small,      "small",       512, 300, 90, 16, false},
    {Variant::Medium,     "medium",      576, 300, 90, 16, false},
    {Variant::Base,       "base",        560, 300, 90, 14, false},
    {Variant::Large,      "large",       704, 300, 90, 16, false},
    // Segmentation (all share patch_size = 12)
    {Variant::SegNano,    "seg-nano",    312, 100, 90, 12, true},
    {Variant::SegSmall,   "seg-small",   384, 100, 90, 12, true},
    {Variant::SegMedium,  "seg-medium",  432, 200, 90, 12, true},
    {Variant::SegLarge,   "seg-large",   504, 200, 90, 12, true},
    {Variant::SegXL,      "seg-xl",      624, 300, 90, 12, true},
    {Variant::Seg2XL,     "seg-2xl",     768, 300, 90, 12, true},
    {Variant::SegPreview, "seg-preview", 432, 200, 90, 12, true},
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
