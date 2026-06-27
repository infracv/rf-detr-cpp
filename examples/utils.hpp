#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace utils {

// Returns "YYYYMMDD_HHMMSS" timestamp string.
inline std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

// Saves image to output_dir/<stem>_result_<timestamp>.jpg and returns the path.
inline std::string save_image(const cv::Mat& image,
                               const std::filesystem::path& input_path,
                               const std::filesystem::path& output_dir = "out") {
    std::filesystem::create_directories(output_dir);
    const std::string name = input_path.stem().string()
                           + "_result_" + timestamp() + ".jpg";
    const std::string out = (output_dir / name).string();
    if (!cv::imwrite(out, image))
        throw std::runtime_error("utils::save_image: cv::imwrite failed: " + out);
    return out;
}

// Prints inference metrics to stdout.
inline void print_metrics(const std::string& label, double duration_ms, double fps = -1.0) {
    if (fps > 0)
        std::printf("[%s]  %.2f ms  |  %.1f FPS\n", label.c_str(), duration_ms, fps);
    else
        std::printf("[%s]  %.2f ms\n", label.c_str(), duration_ms);
}

// Returns true for common image file extensions.
inline bool is_image(const std::filesystem::path& p) {
    std::string lext = p.extension().string();
    std::transform(lext.begin(), lext.end(), lext.begin(), ::tolower);
    return lext == ".jpg" || lext == ".jpeg" || lext == ".png"
        || lext == ".bmp" || lext == ".tiff" || lext == ".tif";
}

// Returns true for common video file extensions.
inline bool is_video(const std::filesystem::path& p) {
    std::string lext = p.extension().string();
    std::transform(lext.begin(), lext.end(), lext.begin(), ::tolower);
    return lext == ".mp4" || lext == ".avi" || lext == ".mov"
        || lext == ".mkv" || lext == ".flv" || lext == ".wmv";
}

}  // namespace utils
