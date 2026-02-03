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
    src/patch_dataset.cpp
    src/torch_descriptor.cpp
)

target_include_directories(train PRIVATE
    include
    ${TORCH_INCLUDE_DIRS}
)

if (TARGET Torch::Torch)
    target_link_libraries(train Torch::Torch)
else()
    target_link_libraries(train ${TORCH_LIBRARIES})
endif()

target_link_libraries(train ${OpenCV_LIBS})

target_compile_options(train PRIVATE ${TORCH_CXX_FLAGS})
target_link_options(train PRIVATE ${TORCH_CXX_FLAGS})

set_target_properties(train PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
    INSTALL_RPATH "/opt/libtorch/lib"
)

if (TORCH_CUDA_FOUND)
    target_compile_definitions(train PRIVATE TORCH_USE_CUDA)
endif()
