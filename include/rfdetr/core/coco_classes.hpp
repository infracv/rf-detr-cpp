#pragma once

namespace rfdetr {

// COCO 91-id sparse class labels. Index 0 is the background "no-object" class
// emitted by the DETR-style class head. Indices 12, 26, 29, 30, 45, 66, 68, 69,
// 71, 83 are reserved gaps in the COCO 91-id mapping and are never produced by
// the trained model — kept as placeholders so the array is index-aligned.
//
// Use raw_class (0..90) when indexing this array. Detection.class_id is the
// dense foreground index (0..89), so the user-facing lookup is
// kCocoLabels91[detection.class_id + 1].
inline constexpr const char* kCocoLabels91[91] = {
    "__background__",  // 0
    "person",          // 1
    "bicycle",         // 2
    "car",             // 3
    "motorcycle",      // 4
    "airplane",        // 5
    "bus",             // 6
    "train",           // 7
    "truck",           // 8
    "boat",            // 9
    "traffic light",   // 10
    "fire hydrant",    // 11
    "__unused_12__",   // 12
    "stop sign",       // 13
    "parking meter",   // 14
    "bench",           // 15
    "bird",            // 16
    "cat",             // 17
    "dog",             // 18
    "horse",           // 19
    "sheep",           // 20
    "cow",             // 21
    "elephant",        // 22
    "bear",            // 23
    "zebra",           // 24
    "giraffe",         // 25
    "__unused_26__",   // 26
    "backpack",        // 27
    "umbrella",        // 28
    "__unused_29__",   // 29
    "__unused_30__",   // 30
    "handbag",         // 31
    "tie",             // 32
    "suitcase",        // 33
    "frisbee",         // 34
    "skis",            // 35
    "snowboard",       // 36
    "sports ball",     // 37
    "kite",            // 38
    "baseball bat",    // 39
    "baseball glove",  // 40
    "skateboard",      // 41
    "surfboard",       // 42
    "tennis racket",   // 43
    "bottle",          // 44
    "__unused_45__",   // 45
    "wine glass",      // 46
    "cup",             // 47
    "fork",            // 48
    "knife",           // 49
    "spoon",           // 50
    "bowl",            // 51
    "banana",          // 52
    "apple",           // 53
    "sandwich",        // 54
    "orange",          // 55
    "broccoli",        // 56
    "carrot",          // 57
    "hot dog",         // 58
    "pizza",           // 59
    "donut",           // 60
    "cake",            // 61
    "chair",           // 62
    "couch",           // 63
    "potted plant",    // 64
    "bed",             // 65
    "__unused_66__",   // 66
    "dining table",    // 67
    "__unused_68__",   // 68
    "__unused_69__",   // 69
    "toilet",          // 70
    "__unused_71__",   // 71
    "tv",              // 72
    "laptop",          // 73
    "mouse",           // 74
    "remote",          // 75
    "keyboard",        // 76
    "cell phone",      // 77
    "microwave",       // 78
    "oven",            // 79
    "toaster",         // 80
    "sink",            // 81
    "refrigerator",    // 82
    "__unused_83__",   // 83
    "book",            // 84
    "clock",           // 85
    "vase",            // 86
    "scissors",        // 87
    "teddy bear",      // 88
    "hair drier",      // 89
    "toothbrush",      // 90
};

// Helper: returns the human-readable name for a foreground (dense) class id.
// dense_id is in [0..89]; mapped to raw_class = dense_id + 1.
inline const char* coco_label(int dense_id) noexcept {
    if (dense_id < 0 || dense_id + 1 >= 91) return "?";
    return kCocoLabels91[dense_id + 1];
}

}  // namespace rfdetr
