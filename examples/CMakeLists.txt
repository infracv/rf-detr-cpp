cmake_minimum_required(VERSION 3.20)

# When built standalone (not via parent -DRFDETR_BUILD_EXAMPLES=ON),
# bootstrap the full project so rfdetr::rfdetr is available.
if(NOT TARGET rfdetr::rfdetr)
    project(rfdetr_examples LANGUAGES CXX CUDA)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 89 120)
    endif()
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
    add_subdirectory(.. rfdetr_parent)
endif()

foreach(example image_det video_det camera_det image_seg video_seg)
    add_executable(example_${example} example_${example}.cpp)
    target_link_libraries(example_${example} PRIVATE rfdetr::rfdetr)
    target_include_directories(example_${example} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
