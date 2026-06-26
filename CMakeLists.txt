cmake_minimum_required(VERSION 3.20)

project(rfdetr
    VERSION 0.0.1
    DESCRIPTION "C++/TensorRT inference library for RF-DETR"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to release build if not specified.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architectures: 87 = Orin (Orin NX), 101 = Thor, 120 = Blackwell (RTX 50xx).
# Override at configure time with -DCMAKE_CUDA_ARCHITECTURES="<list>".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 101 120)
endif()

# Build options.
option(RFDETR_BUILD_SHARED      "Build the rfdetr library as SHARED (vs STATIC)" ON)
option(RFDETR_BUILD_APPS        "Build CLI applications under apps/"             ON)
option(RFDETR_BUILD_EXAMPLES    "Build example programs under examples/"         OFF)
option(RFDETR_BUILD_BENCHMARKS  "Build benchmark harness under benchmarks/"      OFF)
option(RFDETR_BUILD_TESTS       "Build tests under tests/"                       OFF)
option(RFDETR_BUILD_C_API       "Build C ABI (rfdetr_c) target"                  OFF)
option(RFDETR_USE_OPENCV_CUDA   "Use OpenCV CUDA module for GPU pre/post"        OFF)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# ------------------------------------------------------------------------------
# Library target
# ------------------------------------------------------------------------------
if(RFDETR_BUILD_SHARED)
    add_library(rfdetr SHARED)
else()
    add_library(rfdetr STATIC)
endif()
add_library(rfdetr::rfdetr ALIAS rfdetr)

target_sources(rfdetr
    PRIVATE
        src/core/variant.cpp
        src/core/engine_meta.cpp
        src/core/trt_session.cpp
        src/core/cuda_preprocess.cu
        src/core/postprocess.cpp
        src/core/drawing.cpp
        src/tasks/detector.cpp
)

target_include_directories(rfdetr
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(rfdetr
    PUBLIC
        TensorRT::TensorRT
        CUDA::cudart
        ${OpenCV_LIBS}
    PRIVATE
        nlohmann_json::nlohmann_json
)

target_compile_features(rfdetr PUBLIC cxx_std_17)

set_target_properties(rfdetr PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(RFDETR_USE_OPENCV_CUDA)
    target_compile_definitions(rfdetr PUBLIC RFDETR_USE_OPENCV_CUDA=1)
endif()

# ------------------------------------------------------------------------------
# Apps
# ------------------------------------------------------------------------------
if(RFDETR_BUILD_APPS)
    add_executable(rfdetr_inspect apps/rfdetr_inspect.cpp)
    target_link_libraries(rfdetr_inspect PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_smoke apps/rfdetr_smoke.cpp)
    target_link_libraries(rfdetr_smoke PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_build apps/rfdetr_build.cpp)
    target_link_libraries(rfdetr_build PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_run apps/rfdetr_run.cpp)
    target_link_libraries(rfdetr_run PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_detect apps/rfdetr_detect.cpp)
    target_link_libraries(rfdetr_detect PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_video apps/rfdetr_video.cpp)
    target_link_libraries(rfdetr_video PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_batch apps/rfdetr_batch.cpp)
    target_link_libraries(rfdetr_batch PRIVATE rfdetr::rfdetr)

    add_executable(rfdetr_bench apps/rfdetr_bench.cpp)
    target_link_libraries(rfdetr_bench PRIVATE rfdetr::rfdetr nlohmann_json::nlohmann_json CUDA::cudart)
endif()

# ------------------------------------------------------------------------------
# C ABI shared library
# ------------------------------------------------------------------------------
if(RFDETR_BUILD_C_API)
    add_library(rfdetr_c SHARED src/c_api.cpp)
    add_library(rfdetr::rfdetr_c ALIAS rfdetr_c)
    target_link_libraries(rfdetr_c PRIVATE rfdetr::rfdetr)
    target_include_directories(rfdetr_c
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(rfdetr_c PRIVATE RFDETR_BUILDING_LIB=1)
    set_target_properties(rfdetr_c PROPERTIES
        VERSION   ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# ------------------------------------------------------------------------------
# Tests / benchmarks (placeholders, wired in later steps)
# ------------------------------------------------------------------------------
if(RFDETR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(RFDETR_BUILD_TESTS)
    enable_testing()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
endif()

if(RFDETR_BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/CMakeLists.txt)
        add_subdirectory(benchmarks)
    endif()
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "rfdetr ${PROJECT_VERSION}")
message(STATUS "  build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  cuda archs:        ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  cuda toolkit:      ${CUDAToolkit_VERSION} (${CUDAToolkit_BIN_DIR})")
message(STATUS "  tensorrt include:  ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  tensorrt nvinfer:  ${TENSORRT_NVINFER_LIB}")
message(STATUS "  opencv:            ${OpenCV_VERSION}")
message(STATUS "  shared library:    ${RFDETR_BUILD_SHARED}")
message(STATUS "  apps:              ${RFDETR_BUILD_APPS}")
message(STATUS "  examples:          ${RFDETR_BUILD_EXAMPLES}")
message(STATUS "  benchmarks:        ${RFDETR_BUILD_BENCHMARKS}")
message(STATUS "  tests:             ${RFDETR_BUILD_TESTS}")
message(STATUS "  c api:             ${RFDETR_BUILD_C_API}")
message(STATUS "  opencv-cuda:       ${RFDETR_USE_OPENCV_CUDA}")
