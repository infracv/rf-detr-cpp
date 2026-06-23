#include "rfdetr/core/trt_session.hpp"

#include "rfdetr/core/cuda_check.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace rfdetr {

namespace {

std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("rfdetr: cannot open engine file: " + path.string());
    }
    const auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<std::size_t>(size));
    if (!in.read(buf.data(), size)) {
        throw std::runtime_error("rfdetr: short read from engine file: " + path.string());
    }
    return buf;
}

}  // namespace

TrtSession::TrtSession(const std::filesystem::path& engine_path,
                       nvinfer1::ILogger::Severity log_severity)
    : logger_(log_severity) {
    load_engine_(engine_path);
    parse_bindings_();
    RFDETR_CUDA_CHECK(cudaStreamCreate(&stream_));
    allocate_buffers_();
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        bind_address_(static_cast<int>(i));
    }
}

TrtSession::~TrtSession() {
    free_buffers_();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void TrtSession::load_engine_(const std::filesystem::path& path) {
    const auto blob = read_file(path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("rfdetr: createInferRuntime failed");

    // Warn when the runtime TRT version differs from the compile-time headers.
    // Engines are not portable across TRT major/minor versions — rebuild after
    // any TRT upgrade.
    {
        const int rt_major  = getInferLibMajorVersion();
        const int rt_minor  = getInferLibMinorVersion();
        const int hdr_major = NV_TENSORRT_MAJOR;
        const int hdr_minor = NV_TENSORRT_MINOR;
        if (rt_major != hdr_major || rt_minor != hdr_minor) {
            const std::string msg =
                "TRT runtime " + std::to_string(rt_major) + "." + std::to_string(rt_minor) +
                " != compile-time headers " + std::to_string(hdr_major) + "." +
                std::to_string(hdr_minor) +
                " — engine may fail to deserialize; rebuild with rfdetr_build";
            logger_.log(nvinfer1::ILogger::Severity::kWARNING, msg.c_str());
        }
    }

    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("rfdetr: deserializeCudaEngine failed");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("rfdetr: createExecutionContext failed");
}

void TrtSession::parse_bindings_() {
    const int n = engine_->getNbIOTensors();
    bindings_.reserve(n);
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        BindingInfo b;
        b.name = name;
        b.dtype = engine_->getTensorDataType(name);
        b.is_input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        // Use context shape: for static engines this is fully resolved at this point.
        // For dynamic-shape inputs, dims may contain -1 until set_input_shape is called.
        b.shape = context_->getTensorShape(name);
        b.element_count = volume(b.shape);
        b.bytes = (b.element_count > 0)
                      ? static_cast<std::size_t>(b.element_count) * dtype_bytes(b.dtype)
                      : 0;
        bindings_.push_back(std::move(b));
        if (bindings_.back().is_input) input_indices_.push_back(i);
        else                           output_indices_.push_back(i);
    }
}

void TrtSession::allocate_buffers_() {
    device_buffers_.assign(bindings_.size(), nullptr);
    host_buffers_.assign(bindings_.size(), nullptr);
    buffer_capacity_.assign(bindings_.size(), 0);
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        const std::size_t bytes = bindings_[i].bytes;
        if (bytes == 0) continue;  // dynamic, deferred until set_input_shape
        RFDETR_CUDA_CHECK(cudaMalloc(&device_buffers_[i], bytes));
        RFDETR_CUDA_CHECK(cudaMallocHost(&host_buffers_[i], bytes));
        buffer_capacity_[i] = bytes;
    }
}

void TrtSession::free_buffers_() noexcept {
    for (auto* p : device_buffers_) {
        if (p) cudaFree(p);
    }
    for (auto* p : host_buffers_) {
        if (p) cudaFreeHost(p);
    }
    device_buffers_.clear();
    host_buffers_.clear();
    buffer_capacity_.clear();
}

void TrtSession::bind_address_(int idx) {
    if (!device_buffers_[idx]) return;  // dynamic, no buffer yet
    if (!context_->setTensorAddress(bindings_[idx].name.c_str(), device_buffers_[idx])) {
        throw std::runtime_error("rfdetr: setTensorAddress failed for: " + bindings_[idx].name);
    }
}

int TrtSession::find_index(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        if (bindings_[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

const BindingInfo* TrtSession::find(std::string_view name) const noexcept {
    const int i = find_index(name);
    return (i >= 0) ? &bindings_[i] : nullptr;
}

void TrtSession::update_binding_shape_(int idx, const nvinfer1::Dims& dims) {
    BindingInfo& b = bindings_[idx];
    b.shape = dims;
    b.element_count = volume(dims);
    b.bytes = (b.element_count > 0)
                  ? static_cast<std::size_t>(b.element_count) * dtype_bytes(b.dtype)
                  : 0;

    if (b.bytes > buffer_capacity_[idx]) {
        if (device_buffers_[idx]) {
            cudaFree(device_buffers_[idx]);
            device_buffers_[idx] = nullptr;
        }
        if (host_buffers_[idx]) {
            cudaFreeHost(host_buffers_[idx]);
            host_buffers_[idx] = nullptr;
        }
        if (b.bytes > 0) {
            RFDETR_CUDA_CHECK(cudaMalloc(&device_buffers_[idx], b.bytes));
            RFDETR_CUDA_CHECK(cudaMallocHost(&host_buffers_[idx], b.bytes));
            buffer_capacity_[idx] = b.bytes;
            bind_address_(idx);
        } else {
            buffer_capacity_[idx] = 0;
        }
    }
}

void TrtSession::set_input_shape(std::string_view name, const nvinfer1::Dims& dims) {
    const int idx = find_index(name);
    if (idx < 0) {
        throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    }
    if (!bindings_[idx].is_input) {
        throw std::runtime_error("rfdetr: not an input: " + bindings_[idx].name);
    }
    if (!context_->setInputShape(bindings_[idx].name.c_str(), dims)) {
        throw std::runtime_error("rfdetr: setInputShape rejected for: " + bindings_[idx].name);
    }
    // Output shapes may now be resolved differently — refresh every binding.
    for (std::size_t i = 0; i < bindings_.size(); ++i) {
        update_binding_shape_(static_cast<int>(i),
                              context_->getTensorShape(bindings_[i].name.c_str()));
    }
}

void TrtSession::set_input(std::string_view name, const void* host_data, std::size_t bytes) {
    const int idx = find_index(name);
    if (idx < 0) {
        throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    }
    const auto& b = bindings_[idx];
    if (!b.is_input) {
        throw std::runtime_error("rfdetr: tensor is not an input: " + b.name);
    }
    if (b.bytes == 0) {
        throw std::runtime_error("rfdetr: input '" + b.name +
                                 "' has unresolved shape; call set_input_shape first");
    }
    if (bytes != b.bytes) {
        std::ostringstream os;
        os << "rfdetr: set_input(" << b.name << ") size mismatch: got " << bytes
           << " bytes, binding expects " << b.bytes;
        throw std::runtime_error(os.str());
    }
    std::memcpy(host_buffers_[idx], host_data, bytes);
    RFDETR_CUDA_CHECK(cudaMemcpyAsync(device_buffers_[idx], host_buffers_[idx], bytes,
                                       cudaMemcpyHostToDevice, stream_));
}

void TrtSession::infer() {
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("rfdetr: enqueueV3 failed");
    }
    RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void* TrtSession::device_buffer(std::string_view name) {
    const int idx = find_index(name);
    if (idx < 0) {
        throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    }
    return device_buffers_[idx];
}

const void* TrtSession::device_buffer(std::string_view name) const {
    const int idx = find_index(name);
    if (idx < 0) {
        throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    }
    return device_buffers_[idx];
}

void TrtSession::get_output(std::string_view name, void* host_data, std::size_t bytes) {
    const int idx = find_index(name);
    if (idx < 0) {
        throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    }
    const auto& b = bindings_[idx];
    if (b.is_input) {
        throw std::runtime_error("rfdetr: tensor is not an output: " + b.name);
    }
    if (bytes != b.bytes) {
        std::ostringstream os;
        os << "rfdetr: get_output(" << b.name << ") size mismatch: got " << bytes
           << " bytes, binding produces " << b.bytes;
        throw std::runtime_error(os.str());
    }
    RFDETR_CUDA_CHECK(cudaMemcpyAsync(host_buffers_[idx], device_buffers_[idx], bytes,
                                       cudaMemcpyDeviceToHost, stream_));
    RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::memcpy(host_data, host_buffers_[idx], bytes);
}

void TrtSession::get_output_f32(std::string_view name, float* host_float32,
                                 std::size_t element_count) {
    const int idx = find_index(name);
    if (idx < 0) throw std::runtime_error("rfdetr: no such binding: " + std::string(name));
    const auto& b = bindings_[idx];
    if (b.is_input) throw std::runtime_error("rfdetr: tensor is not an output: " + b.name);
    if (static_cast<std::size_t>(b.element_count) != element_count) {
        std::ostringstream os;
        os << "rfdetr: get_output_f32(" << b.name << ") element count mismatch: got "
           << element_count << ", binding has " << b.element_count;
        throw std::runtime_error(os.str());
    }

    if (b.dtype == nvinfer1::DataType::kFLOAT) {
        // Native FP32 — use the normal path.
        RFDETR_CUDA_CHECK(cudaMemcpyAsync(host_buffers_[idx], device_buffers_[idx], b.bytes,
                                           cudaMemcpyDeviceToHost, stream_));
        RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream_));
        std::memcpy(host_float32, host_buffers_[idx], b.bytes);
    } else if (b.dtype == nvinfer1::DataType::kHALF) {
        // FP16 — copy raw bytes then convert on CPU.
        RFDETR_CUDA_CHECK(cudaMemcpyAsync(host_buffers_[idx], device_buffers_[idx], b.bytes,
                                           cudaMemcpyDeviceToHost, stream_));
        RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream_));
        // Convert fp16 (stored as uint16_t) to float32.
        // Uses the standard bit-cast trick via __half if CUDA headers are available,
        // otherwise a portable software fallback.
        const auto* src = static_cast<const std::uint16_t*>(host_buffers_[idx]);
        for (std::size_t i = 0; i < element_count; ++i) {
            // Portable IEEE 754 fp16->fp32 conversion.
            const std::uint16_t h = src[i];
            const std::uint32_t sign     = (h >> 15u) & 1u;
            const std::uint32_t exponent = (h >> 10u) & 0x1Fu;
            const std::uint32_t mantissa =  h         & 0x3FFu;
            std::uint32_t f;
            if (exponent == 0u) {
                if (mantissa == 0u) {
                    f = sign << 31u;  // ±0
                } else {
                    // Subnormal: renormalize.
                    std::uint32_t e = 0u, m = mantissa;
                    while (!(m & 0x400u)) { m <<= 1u; ++e; }
                    f = (sign << 31u) | ((127u - 14u - e) << 23u) | ((m & 0x3FFu) << 13u);
                }
            } else if (exponent == 31u) {
                f = (sign << 31u) | 0x7F800000u | (mantissa << 13u);  // Inf or NaN
            } else {
                f = (sign << 31u) | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
            }
            std::memcpy(&host_float32[i], &f, 4);
        }
    } else {
        throw std::runtime_error("rfdetr: get_output_f32: unsupported dtype for " +
                                  std::string(name) + " (only fp32/fp16 supported)");
    }
}

std::size_t TrtSession::dtype_bytes(nvinfer1::DataType d) noexcept {
    switch (d) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:   return 1;
        case nvinfer1::DataType::kBF16:  return 2;
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: return 0;
    }
}

int64_t TrtSession::volume(const nvinfer1::Dims& dims) noexcept {
    if (dims.nbDims <= 0) return 0;
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return 0;
        v *= dims.d[i];
    }
    return v;
}

const char* TrtSession::dtype_name(nvinfer1::DataType d) noexcept {
    switch (d) {
        case nvinfer1::DataType::kFLOAT: return "fp32";
        case nvinfer1::DataType::kHALF:  return "fp16";
        case nvinfer1::DataType::kINT8:  return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL:  return "bool";
        case nvinfer1::DataType::kUINT8: return "uint8";
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:   return "fp8";
        case nvinfer1::DataType::kBF16:  return "bf16";
        case nvinfer1::DataType::kINT64: return "int64";
#endif
        default: return "?";
    }
}

}  // namespace rfdetr
