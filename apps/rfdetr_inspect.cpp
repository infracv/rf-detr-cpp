// rfdetr_inspect — load a serialized TensorRT engine and print its IO bindings.
// Optional second arg: a meta-sidecar JSON, which is parsed and cross-checked
// against the engine.

#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/trt_logger.hpp"
#include "rfdetr/core/variant.hpp"
#include "rfdetr/version.hpp"

#include <NvInferRuntime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <vector>

namespace {

std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("could not open: " + path.string());
    }
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!in.read(buf.data(), size)) {
        throw std::runtime_error("short read: " + path.string());
    }
    return buf;
}

const char* mode_name(nvinfer1::TensorIOMode m) noexcept {
    switch (m) {
        case nvinfer1::TensorIOMode::kINPUT:  return "INPUT ";
        case nvinfer1::TensorIOMode::kOUTPUT: return "OUTPUT";
        default:                              return "NONE  ";
    }
}

const char* dtype_name(nvinfer1::DataType d) noexcept {
    switch (d) {
        case nvinfer1::DataType::kFLOAT:  return "fp32";
        case nvinfer1::DataType::kHALF:   return "fp16";
        case nvinfer1::DataType::kINT8:   return "int8";
        case nvinfer1::DataType::kINT32:  return "int32";
        case nvinfer1::DataType::kBOOL:   return "bool";
        case nvinfer1::DataType::kUINT8:  return "uint8";
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:    return "fp8";
        case nvinfer1::DataType::kBF16:   return "bf16";
        case nvinfer1::DataType::kINT64:  return "int64";
#endif
        default:                          return "?";
    }
}

void print_shape(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) std::fputc(',', stdout);
        std::printf("%ld", static_cast<long>(d.d[i]));
    }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr,
                     "usage: %s <engine.plan> [meta.json]\n"
                     "  rfdetr v%s\n",
                     argv[0], rfdetr::version());
        return 2;
    }

    const std::filesystem::path engine_path{argv[1]};
    std::vector<char> blob;
    try {
        blob = read_file(engine_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    rfdetr::TrtLogger logger{nvinfer1::ILogger::Severity::kWARNING};
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        std::fprintf(stderr, "error: createInferRuntime returned null\n");
        return 1;
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(blob.data(), blob.size())};
    if (!engine) {
        std::fprintf(stderr, "error: deserializeCudaEngine returned null\n");
        return 1;
    }

    std::printf("engine: %s\n", engine_path.c_str());
    std::printf("  trt header: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                NV_TENSORRT_PATCH);
    std::printf("  bindings:\n");
    const int nb = engine->getNbIOTensors();
    int input_count = 0;
    int output_count = 0;
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        const auto dtype = engine->getTensorDataType(name);
        const auto shape = engine->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) ++input_count;
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) ++output_count;

        std::printf("    [%d] %-12s  %s  %-5s  shape=", i, name, mode_name(mode),
                    dtype_name(dtype));
        print_shape(shape);
        std::putchar('\n');
    }

    if (argc == 3) {
        const std::filesystem::path meta_path{argv[2]};
        rfdetr::EngineMeta meta;
        try {
            meta = rfdetr::EngineMeta::from_json_file(meta_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error reading meta: %s\n", e.what());
            return 1;
        }

        std::printf("\nmeta sidecar: %s\n", meta_path.c_str());
        std::printf("  variant      = %s\n", meta.variant.c_str());
        std::printf("  input_hxw    = %dx%d\n", meta.input_h, meta.input_w);
        std::printf("  num_queries  = %d\n", meta.num_queries);
        std::printf("  num_classes  = %d\n", meta.num_classes);
        std::printf("  has_masks    = %s\n", meta.has_masks ? "true" : "false");
        if (meta.has_masks) {
            std::printf("  mask_hxw     = %dx%d\n", meta.mask_h, meta.mask_w);
        }
        std::printf("  color_order  = %s\n", meta.color_order.c_str());
        std::printf("  precision    = %s\n", meta.precision.c_str());
        std::printf("  mean         = [%.4f, %.4f, %.4f]\n", meta.mean[0], meta.mean[1],
                    meta.mean[2]);
        std::printf("  std          = [%.4f, %.4f, %.4f]\n", meta.std[0], meta.std[1], meta.std[2]);

        const auto& vmeta = rfdetr::variant_meta(meta.variant);
        if (vmeta.variant == rfdetr::Variant::Unknown) {
            std::printf("  WARNING: variant '%s' not in built-in variant table\n",
                        meta.variant.c_str());
        } else if (vmeta.input_size != meta.input_h || vmeta.input_size != meta.input_w) {
            std::printf("  WARNING: variant '%s' expects %dx%d, sidecar says %dx%d\n",
                        meta.variant.c_str(), vmeta.input_size, vmeta.input_size, meta.input_h,
                        meta.input_w);
        }

        const int expected_outputs = meta.has_masks ? 3 : 2;
        if (output_count != expected_outputs) {
            std::printf("  WARNING: engine has %d outputs, meta implies %d (has_masks=%s)\n",
                        output_count, expected_outputs, meta.has_masks ? "true" : "false");
        }
        if (input_count != 1) {
            std::printf("  WARNING: engine has %d inputs, expected 1\n", input_count);
        }
    }

    return 0;
}
