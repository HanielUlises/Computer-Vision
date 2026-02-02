cmake_minimum_required(VERSION 3.16)
project(descriptor_learning LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(train
    src/train.cpp
    src/PatchDataset.cpp
    src/DescriptorNet.cpp
)

target_include_directories(train PRIVATE
    include
)

target_link_libraries(train
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

target_compile_options(train PRIVATE ${TORCH_CXX_FLAGS})

set_target_properties(train PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
    INSTALL_RPATH "/opt/libtorch/lib"
)

if (TORCH_CUDA_FOUND)
    target_compile_definitions(train PRIVATE USE_CUDA)
endif()
