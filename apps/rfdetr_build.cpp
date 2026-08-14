// rfdetr_build — build a TensorRT .engine from an ONNX model entirely in C++.
//
// Replaces the typical Python+trtexec workflow for FP32/FP16/INT8 engine builds.
// Reads the optional <onnx>.json sidecar produced by export_onnx.py to copy
// variant metadata into the output <engine>.json.
//
// INT8 requires a calibration cache produced by int8_calibrate.py:
//   python trt-files/scripts/int8_calibrate.py \
//       --onnx trt-files/onnx/rf-detr-nano.onnx \
//       --images /path/to/calib_images \
//       --output trt-files/onnx/rf-detr-nano.calib
//
// usage:
//   rfdetr_build --onnx model.onnx [--engine model.engine] [--meta model.json]
//                [--precision fp32|fp16|int8] [--calib model.calib]
//                [--workspace-mib 4096]
//                [--input-name input] [--min-batch 1 --opt-batch 1 --max-batch 1]
//                [--verbose]

#include "cli_helpers.hpp"
#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/trt_logger.hpp"
#include "rfdetr/version.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// INT8 calibrator: reads the .calib cache produced by int8_calibrate.py.
// Only available on TRT < 11 (implicit quantization API).
// TRT 11+ uses QDQ-based explicit quantization; use build_engine.py + trtexec.
// ---------------------------------------------------------------------------
#if NV_TENSORRT_MAJOR < 11
#include <NvInferRuntime.h>

class CacheOnlyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    explicit CacheOnlyCalibrator(const std::filesystem::path& cache_path) {
        std::ifstream f(cache_path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("cannot open calib cache: " + cache_path.string());
        const auto sz = f.tellg();
        f.seekg(0);
        cache_.resize(static_cast<std::size_t>(sz));
        f.read(cache_.data(), sz);
    }
    int         getBatchSize()     const noexcept override { return 1; }
    bool        getBatch(void*[], const char*[], int) noexcept override { return false; }
    const void* readCalibrationCache(std::size_t& length) noexcept override {
        length = cache_.size(); return cache_.data();
    }
    void writeCalibrationCache(const void*, std::size_t) noexcept override {}
private:
    std::vector<char> cache_;
};
#endif  // NV_TENSORRT_MAJOR < 11

namespace {

struct Args {
    std::filesystem::path onnx;
    std::filesystem::path engine;            // default: <onnx-stem>-<precision>.engine
    std::filesystem::path meta_in;           // default: <onnx>.json
    std::filesystem::path meta_out;          // default: <engine>.json
    std::filesystem::path calib;             // INT8 calibration cache (.calib)
    std::string precision{"fp32"};           // fp32 / fp16 / int8
    int workspace_mib{4096};
    std::string input_name{"input"};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph_compat{false};
    int  slim{-1};                           // -1 auto (on for fp32), 0 off, 1 on
    bool verbose{false};
};

void usage(const char* argv0) {
    std::fprintf(
        stderr,
        "usage: %s --onnx PATH [options]\n"
        "\n"
        "Required:\n"
        "  --onnx PATH               Input ONNX model.\n"
        "\n"
        "Output paths (defaults derived from --onnx and --precision):\n"
        "  --engine PATH             Output .engine     (default: <onnx-stem>-<precision>.engine,\n"
        "                                                e.g. rf-detr-nano-fp32.engine)\n"
        "  --meta PATH               Input meta JSON    (default: <onnx>.json)\n"
        "  --meta-out PATH           Output meta JSON   (default: <engine>.json)\n"
        "\n"
        "Build:\n"
        "  --precision fp32|fp16|int8  Default: fp32.\n"
        "  --calib PATH              INT8 calibration cache (.calib) produced by\n"
        "                            trt-files/scripts/int8_calibrate.py.\n"
        "                            Required when --precision int8.\n"
        "  --workspace-mib N         Workspace pool in MiB (default: 4096)\n"
        "  --input-name NAME         Input tensor name (default: input)\n"
        "  --min-batch N             Dynamic profile min  (default: 1)\n"
        "  --opt-batch N             Dynamic profile opt  (default: 1)\n"
        "  --max-batch N             Dynamic profile max  (default: 1)\n"
        "                            If max > 1 (or ONNX has dynamic batch) an\n"
        "                            optimization profile is added.\n"
        "  --slim / --no-slim        Simplify the ONNX with onnxslim before building.\n"
        "                            On by default for fp32 (lossless, faster engine);\n"
        "                            off for fp16/int8. --no-slim opts out.\n"
        "  --verbose                 TRT logger at VERBOSE severity.\n"
        "\n"
        "rfdetr v%s\n",
        argv0, rfdetr::version());
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "-h" || arg == "--help") { usage(argv[0]); std::exit(0); }
        else if (starts_with(arg, "--onnx"))         a.onnx = next_value(argc, argv, i, "--onnx");
        else if (starts_with(arg, "--engine"))       a.engine = next_value(argc, argv, i, "--engine");
        else if (starts_with(arg, "--meta-out"))     a.meta_out = next_value(argc, argv, i, "--meta-out");
        else if (starts_with(arg, "--meta"))         a.meta_in = next_value(argc, argv, i, "--meta");
        else if (starts_with(arg, "--precision"))    a.precision = next_value(argc, argv, i, "--precision");
        else if (starts_with(arg, "--calib"))        a.calib     = next_value(argc, argv, i, "--calib");
        else if (starts_with(arg, "--workspace-mib"))a.workspace_mib = parse_int(next_value(argc, argv, i, "--workspace-mib"), "--workspace-mib");
        else if (starts_with(arg, "--input-name"))   a.input_name = next_value(argc, argv, i, "--input-name");
        else if (starts_with(arg, "--min-batch"))    a.min_batch = parse_int(next_value(argc, argv, i, "--min-batch"), "--min-batch");
        else if (starts_with(arg, "--opt-batch"))    a.opt_batch = parse_int(next_value(argc, argv, i, "--opt-batch"), "--opt-batch");
        else if (starts_with(arg, "--max-batch"))    a.max_batch = parse_int(next_value(argc, argv, i, "--max-batch"), "--max-batch");
        else if (arg == "--cuda-graph")              a.cuda_graph_compat = true;
        else if (arg == "--slim")                    a.slim = 1;
        else if (arg == "--no-slim")                 a.slim = 0;
        else if (arg == "--verbose")                 a.verbose = true;
        else {
            throw std::runtime_error("unknown arg: " + std::string(arg));
        }
    }

    if (a.onnx.empty()) {
        usage(argv[0]);
        std::exit(2);
    }
    if (a.precision != "fp32" && a.precision != "fp16" && a.precision != "int8") {
        throw std::runtime_error("--precision must be fp32, fp16, or int8");
    }
    if (a.slim == 1 && a.precision != "fp32") {
        throw std::runtime_error(
            "--slim is FP32 only (onnxslim degrades FP16/INT8 accuracy)");
    }
    if (a.precision == "int8" && a.calib.empty()) {
        throw std::runtime_error(
            "--precision int8 requires --calib <cache.calib>\n"
            "  generate with: python trt-files/scripts/int8_calibrate.py "
            "--onnx <onnx> --images <dir> --output <cache.calib>");
    }
    if (!a.calib.empty() && !std::filesystem::is_regular_file(a.calib)) {
        throw std::runtime_error("calibration cache not found: " + a.calib.string());
    }
    if (a.engine.empty()) {
        // e.g. trt-files/onnx/rf-detr-nano.onnx + fp32 -> trt-files/onnx/rf-detr-nano-fp32.engine
        a.engine = a.onnx.parent_path() /
                   (a.onnx.stem().string() + "-" + a.precision + ".engine");
    }
    if (a.meta_in.empty()) {
        a.meta_in = a.onnx;
        a.meta_in.replace_extension(".json");
    }
    if (a.meta_out.empty()) {
        // Append ".json" rather than replacing the extension, so the engine sidecar
        // (e.g. "rf-detr-nano-fp32.engine.json") never collides with the ONNX sidecar
        // (e.g. "rf-detr-nano.json") in the same directory.
        a.meta_out = a.engine.string() + ".json";
    }
    if (a.min_batch <= 0 || a.opt_batch <= 0 || a.max_batch <= 0) {
        throw std::runtime_error("--*-batch values must be positive");
    }
    if (a.min_batch > a.opt_batch || a.opt_batch > a.max_batch) {
        throw std::runtime_error("require min-batch <= opt-batch <= max-batch");
    }
    return a;
}

void print_dims(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) std::fputc(',', stdout);
        std::printf("%ld", static_cast<long>(d.d[i]));
    }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n", e.what());
        usage(argv[0]);
        return 2;
    }

    if (!std::filesystem::is_regular_file(args.onnx)) {
        std::fprintf(stderr, "error: onnx not found: %s\n", args.onnx.c_str());
        return 1;
    }

    // TRT 11+ FP16: the kFP16 builder flag was removed. Auto-convert ONNX weights
    // to FP16 via convert_fp16.py, then feed the converted ONNX to the TRT parser.
    // The user still just runs:  rfdetr_build --onnx model.onnx --precision fp16
#if NV_TENSORRT_MAJOR >= 11
    std::filesystem::path fp16_tmp_onnx;
    if (args.precision == "fp16") {
        const std::filesystem::path script{"trt-files/scripts/convert_fp16.py"};
        if (!std::filesystem::is_regular_file(script)) {
            std::fprintf(stderr,
                "error: trt-files/scripts/convert_fp16.py not found.\n"
                "       Run rfdetr_build from the project root directory.\n");
            return 1;
        }
        fp16_tmp_onnx = args.onnx.parent_path() /
                        (args.onnx.stem().string() + "-fp16-tmp.onnx");
        std::printf("[rfdetr_build] TRT 11+: converting ONNX weights to FP16...\n");
        const std::string cmd = "python \"" + script.string() + "\""
                              + " --onnx \"" + args.onnx.string() + "\""
                              + " --out \""  + fp16_tmp_onnx.string() + "\"";
        if (std::system(cmd.c_str()) != 0 ||
                !std::filesystem::is_regular_file(fp16_tmp_onnx)) {
            std::fprintf(stderr, "error: FP16 ONNX conversion failed\n");
            return 1;
        }
        args.onnx = fp16_tmp_onnx;
    }
#endif

    // FP32: simplify the ONNX with onnxslim before parsing. On by default for FP32
    // (halves the graph for a lossless ~6-9% speedup); never for fp16/int8, where
    // onnxslim's fusions reduce numerical headroom and hurt accuracy. Mirrors the
    // convert_fp16.py shell-out above.
    std::filesystem::path slim_tmp_onnx;
    const bool do_slim = (args.slim == 1) ||
                         (args.slim < 0 && args.precision == "fp32");
    if (do_slim) {
        const std::filesystem::path script{"trt-files/scripts/slim_onnx.py"};
        if (!std::filesystem::is_regular_file(script)) {
            std::fprintf(stderr,
                "error: trt-files/scripts/slim_onnx.py not found.\n"
                "       Run rfdetr_build from the project root directory.\n");
            return 1;
        }
        slim_tmp_onnx = args.onnx.parent_path() /
                        (args.onnx.stem().string() + "-slim-tmp.onnx");
        std::printf("[rfdetr_build] onnxslim: simplifying ONNX graph (FP32)...\n");
        const std::string cmd = "python \"" + script.string() + "\""
                              + " --onnx \"" + args.onnx.string() + "\""
                              + " --out \""  + slim_tmp_onnx.string() + "\"";
        if (std::system(cmd.c_str()) == 0 &&
                std::filesystem::is_regular_file(slim_tmp_onnx)) {
            args.onnx = slim_tmp_onnx;
        } else if (args.slim == 1) {
            std::fprintf(stderr,
                "error: onnxslim step failed (install it: pip install onnxslim)\n");
            return 1;
        } else {
            // Default-on must not break a build when onnxslim is unavailable.
            std::fprintf(stderr,
                "[rfdetr_build] onnxslim unavailable; building without it "
                "(pip install onnxslim for a faster FP32 engine)\n");
        }
    }

    rfdetr::TrtLogger logger{args.verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                                          : nvinfer1::ILogger::Severity::kWARNING};

    std::printf("[rfdetr_build] onnx        : %s\n", args.onnx.c_str());
    std::printf("[rfdetr_build] engine out  : %s\n", args.engine.c_str());
    std::printf("[rfdetr_build] precision   : %s\n", args.precision.c_str());
    std::printf("[rfdetr_build] workspace   : %d MiB\n", args.workspace_mib);
    std::printf("[rfdetr_build] batch range : min=%d opt=%d max=%d\n",
                args.min_batch, args.opt_batch, args.max_batch);

    rfdetr::EngineMeta meta;
    bool have_meta = false;
    if (std::filesystem::is_regular_file(args.meta_in)) {
        try {
            meta = rfdetr::EngineMeta::from_json_file(args.meta_in);
            have_meta = true;
            std::printf("[rfdetr_build] meta in     : %s (variant=%s, %dx%d)\n",
                        args.meta_in.c_str(), meta.variant.c_str(), meta.input_h, meta.input_w);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "warning: ignoring meta sidecar: %s\n", e.what());
        }
    } else {
        std::printf("[rfdetr_build] meta in     : (none — will infer shape from ONNX)\n");
    }

    try {
        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
        if (!builder) throw std::runtime_error("createInferBuilder failed");

        std::unique_ptr<nvinfer1::INetworkDefinition> network{
            builder->createNetworkV2(0)};  // explicit batch (default in TRT 10+)
        if (!network) throw std::runtime_error("createNetworkV2 failed");

        std::unique_ptr<nvonnxparser::IParser> parser{
            nvonnxparser::createParser(*network, logger)};
        if (!parser) throw std::runtime_error("createParser failed");

        const auto severity = args.verbose ? static_cast<int>(nvinfer1::ILogger::Severity::kVERBOSE)
                                           : static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
        if (!parser->parseFromFile(args.onnx.c_str(), severity)) {
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                std::fprintf(stderr, "  ONNX error: %s\n", parser->getError(i)->desc());
            }
            throw std::runtime_error("ONNX parsing failed");
        }
        std::printf("[rfdetr_build] parsed ONNX, %d input(s), %d output(s)\n",
                    network->getNbInputs(), network->getNbOutputs());

        // Locate the input tensor: try by --input-name, else first input.
        nvinfer1::ITensor* input = nullptr;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            if (args.input_name == network->getInput(i)->getName()) {
                input = network->getInput(i);
                break;
            }
        }
        if (!input && network->getNbInputs() > 0) {
            input = network->getInput(0);
            if (args.input_name != input->getName()) {
                std::printf("[rfdetr_build] input '%s' not found, using '%s'\n",
                            args.input_name.c_str(), input->getName());
                args.input_name = input->getName();
            }
        }
        if (!input) throw std::runtime_error("network has no inputs");

        nvinfer1::Dims in_dims = input->getDimensions();
        std::printf("[rfdetr_build] input '%s' shape=", input->getName());
        print_dims(in_dims);
        std::putchar('\n');

        const bool has_dynamic_dim = [&]() {
            for (int i = 0; i < in_dims.nbDims; ++i) {
                if (in_dims.d[i] < 0) return true;
            }
            return false;
        }();
        const bool need_profile = has_dynamic_dim || args.max_batch > 1;

        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config) throw std::runtime_error("createBuilderConfig failed");

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   static_cast<std::size_t>(args.workspace_mib) << 20);

        // TRT < 11: set kFP16 builder flag on the original FP32 ONNX.
        // TRT 11+: ONNX weights were already converted to FP16 above; no flag needed.
#if NV_TENSORRT_MAJOR < 11
        if (args.precision == "fp16") {
            if (!builder->platformHasFastFp16()) {
                std::fprintf(stderr,
                    "warning: GPU does not report fast FP16; building FP16 anyway.\n");
            }
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::printf("[rfdetr_build] FP16 mode enabled\n");
        }
#endif

        // INT8 calibration — implicit quantization (TRT < 11 only).
        // TRT 11+ uses QDQ-based explicit quantization via convert_int8.py.
#if NV_TENSORRT_MAJOR < 11
        std::unique_ptr<CacheOnlyCalibrator> calibrator;
        if (args.precision == "int8") {
            calibrator = std::make_unique<CacheOnlyCalibrator>(args.calib);
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
            if (builder->platformHasFastFp16()) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            config->setInt8Calibrator(calibrator.get());
            std::printf("[rfdetr_build] INT8 mode, calib cache: %s\n", args.calib.c_str());
        }
#else
        if (args.precision == "int8") {
            std::fprintf(stderr,
                "error: --precision int8 is not supported by rfdetr_build on TensorRT 11+.\n"
                "       TRT 11 uses QDQ-based explicit quantization. Insert QDQ nodes first:\n"
                "         python trt-files/scripts/convert_int8.py \\\n"
                "             --onnx %s --images val2017/ \\\n"
                "             --out  trt-files/onnx/rf-detr-nano-int8.onnx\n"
                "         ./build/rfdetr_build \\\n"
                "             --onnx trt-files/onnx/rf-detr-nano-int8.onnx --precision fp32\n",
                args.onnx.c_str());
            return 1;
        }
#endif

        if (args.cuda_graph_compat) {
            // No special TRT 11 builder flag needed for CUDA Graph — the graph
            // compatibility is achieved at inference time via setAuxStreams().
            // This flag only labels the meta sidecar so rfdetr_detect knows to attempt capture.
            std::printf("[rfdetr_build] CUDA Graph compatible: yes (meta label only)\n");
        }

        if (need_profile) {
            // Decide H/W: prefer sidecar, fall back to network introspection.
            int H = (have_meta && meta.input_h > 0) ? meta.input_h : -1;
            int W = (have_meta && meta.input_w > 0) ? meta.input_w : -1;
            if (in_dims.nbDims == 4) {
                if (in_dims.d[2] > 0) H = in_dims.d[2];
                if (in_dims.d[3] > 0) W = in_dims.d[3];
            }
            if (H <= 0 || W <= 0) {
                throw std::runtime_error(
                    "cannot determine input H/W for dynamic profile; "
                    "pass --meta with input_h/input_w or use a static-shape ONNX");
            }
            const int C = (in_dims.nbDims == 4 && in_dims.d[1] > 0) ? in_dims.d[1] : 3;

            auto* profile = builder->createOptimizationProfile();
            const nvinfer1::Dims4 dmin{args.min_batch, C, H, W};
            const nvinfer1::Dims4 dopt{args.opt_batch, C, H, W};
            const nvinfer1::Dims4 dmax{args.max_batch, C, H, W};
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dmin);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dopt);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dmax);
            config->addOptimizationProfile(profile);
            std::printf("[rfdetr_build] profile     : min=[%d,%d,%d,%d] opt=[%d,%d,%d,%d] max=[%d,%d,%d,%d]\n",
                        args.min_batch, C, H, W, args.opt_batch, C, H, W,
                        args.max_batch, C, H, W);
        }

        std::printf("[rfdetr_build] building... (this can take 10-60s)\n");
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        const auto t1 = std::chrono::steady_clock::now();
        if (!plan) throw std::runtime_error("buildSerializedNetwork returned null");
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[rfdetr_build] built in %.1fs, plan size = %.1f MiB\n", secs,
                    static_cast<double>(plan->size()) / (1024.0 * 1024.0));

        std::ofstream out(args.engine, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open engine output: " + args.engine.string());
        out.write(static_cast<const char*>(plan->data()),
                  static_cast<std::streamsize>(plan->size()));
        out.close();
        std::printf("[rfdetr_build] wrote %s\n", args.engine.c_str());

        // Update + write the meta sidecar.
        if (!have_meta) {
            // Synthesize a minimal meta from network introspection.
            meta.num_classes = 90;  // RF-DETR default
            meta.color_order = "RGB";
        }
        if (in_dims.nbDims == 4) {
            if (in_dims.d[2] > 0) meta.input_h = in_dims.d[2];
            if (in_dims.d[3] > 0) meta.input_w = in_dims.d[3];
        }
        // If the ONNX was pre-converted to FP16 or INT8-QDQ but the builder flag
        // is fp32 (TRT 11 strongly-typed path), preserve the sidecar precision so
        // the runtime and benchmark tools report the correct precision.
        if (!(args.precision == "fp32" && have_meta &&
              (meta.precision == "fp16" || meta.precision == "int8"))) {
            meta.precision = args.precision;
        }
        meta.dynamic_batch      = need_profile;
        meta.cuda_graph_compat  = args.cuda_graph_compat;
        meta.min_batch     = args.min_batch;
        meta.opt_batch     = args.opt_batch;
        meta.max_batch     = args.max_batch;

        meta.to_json_file(args.meta_out);
        std::printf("[rfdetr_build] wrote %s\n", args.meta_out.c_str());

    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

#if NV_TENSORRT_MAJOR >= 11
    if (!fp16_tmp_onnx.empty() && std::filesystem::is_regular_file(fp16_tmp_onnx)) {
        std::filesystem::remove(fp16_tmp_onnx);
    }
#endif

    return 0;
}
