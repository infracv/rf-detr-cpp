#pragma once

#include <string>

namespace rfdetr {

// Severity ordering matches TensorRT's nvinfer1::ILogger::Severity:
//   FATAL > ERROR > WARN > INFO > VERBOSE
// A message at level X is delivered only if X has lower numeric value than
// the configured minimum.
enum class LogSeverity : int {
    kFatal   = 0,
    kError   = 1,
    kWarning = 2,
    kInfo    = 3,
    kVerbose = 4,
};

// User-supplied logging callback. Receives the formatted message text with no
// trailing newline. The callback must be reentrant — concurrent calls from
// multiple threads are possible.
using LogCallback = void (*)(LogSeverity, const char*) noexcept;

// Install a custom log callback. Pass nullptr to revert to the default
// (stderr) handler. The callback may be installed/uninstalled at any time;
// the change is visible to subsequent log calls.
void set_log_callback(LogCallback callback) noexcept;

// Filter messages: nothing more verbose than `min` is forwarded to the
// callback (or stderr). Default is kWarning.
void set_log_severity(LogSeverity min) noexcept;

// Library-internal: dispatch a message to the active callback / stderr.
// Skipped if `severity` is more verbose than the configured minimum.
void log_message(LogSeverity severity, const char* msg) noexcept;

// Convenience overload for std::string.
inline void log_message(LogSeverity severity, const std::string& msg) noexcept {
    log_message(severity, msg.c_str());
}

}  // namespace rfdetr
