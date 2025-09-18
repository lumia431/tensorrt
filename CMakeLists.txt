cmake_minimum_required(VERSION 3.16)
project(YOLOv5TensorRT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT paths
set(TensorRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TensorRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

# Include directories
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Link directories
link_directories(${TensorRT_LIB_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Add executable
add_executable(yolo_tensorrt yolo_tensorrt.cpp)

# Link libraries
target_link_libraries(yolo_tensorrt
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    cudart
)

# Set CUDA architectures
set_property(TARGET yolo_tensorrt PROPERTY CUDA_ARCHITECTURES 86)

# Add chrono header
target_compile_features(yolo_tensorrt PRIVATE cxx_std_14)