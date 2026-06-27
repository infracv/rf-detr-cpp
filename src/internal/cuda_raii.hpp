#pragma once

#include <cuda_runtime_api.h>
#include <memory>

namespace rfdetr {
namespace detail {

struct DevFree {
    void operator()(void* p) const noexcept { if (p) cudaFree(p); }
};

struct HostFree {
    void operator()(void* p) const noexcept { if (p) cudaFreeHost(p); }
};

}  // namespace detail

// Owning device-memory pointer.  Drop-in for void* managed with cudaMalloc/cudaFree.
using DevPtr  = std::unique_ptr<void, detail::DevFree>;

// Owning pinned-host-memory pointer.  Managed with cudaMallocHost/cudaFreeHost.
using HostPtr = std::unique_ptr<void, detail::HostFree>;

}  // namespace rfdetr
