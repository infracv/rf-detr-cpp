#pragma once

#include <NvInferRuntime.h>

#include <cstdio>

namespace rfdetr {

// Minimal nvinfer1::ILogger that writes to stderr. Severity threshold defaults
// to WARNING; pass kVERBOSE to debug engine builds.
class TrtLogger : public nvinfer1::ILogger {
   public:
    using Severity = nvinfer1::ILogger::Severity;

    explicit TrtLogger(Severity min = Severity::kWARNING) noexcept : min_severity_(min) {}

    void log(Severity severity, const char* msg) noexcept override {
        if (severity > min_severity_) return;
        const char* tag = "INFO ";
        switch (severity) {
            case Severity::kINTERNAL_ERROR: tag = "FATAL"; break;
            case Severity::kERROR:          tag = "ERROR"; break;
            case Severity::kWARNING:        tag = "WARN "; break;
            case Severity::kINFO:           tag = "INFO "; break;
            case Severity::kVERBOSE:        tag = "VERB "; break;
        }
        std::fprintf(stderr, "[TRT][%s] %s\n", tag, msg);
    }

    void set_min_severity(Severity s) noexcept { min_severity_ = s; }
    Severity min_severity() const noexcept { return min_severity_; }

   private:
    Severity min_severity_;
};

}  // namespace rfdetr
