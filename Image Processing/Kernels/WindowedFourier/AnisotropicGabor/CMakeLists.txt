cmake_minimum_required(VERSION 3.10)
project(AnisotropicGaborTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(
    anisotropic_gabor_test
    main.cpp
    AnisotropicGabor.cpp
)

target_include_directories(
    anisotropic_gabor_test
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anisotropic_gabor_test
    PRIVATE
    ${OpenCV_LIBS}
)
