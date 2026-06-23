#pragma once

#define RFDETR_VERSION_MAJOR 0
#define RFDETR_VERSION_MINOR 0
#define RFDETR_VERSION_PATCH 1
#define RFDETR_VERSION_STRING "0.0.1"

namespace rfdetr {

constexpr const char* version() noexcept { return RFDETR_VERSION_STRING; }

}  // namespace rfdetr
