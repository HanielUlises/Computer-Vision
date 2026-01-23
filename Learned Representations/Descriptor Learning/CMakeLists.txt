cmake_minimum_required(VERSION 3.16)
project(descriptor_learning)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(train
    src/train.cpp
    src/patch_dataset.cpp
)

target_include_directories(train PRIVATE include)

target_link_libraries(train
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

set_property(TARGET train PROPERTY CXX_STANDARD 23)
