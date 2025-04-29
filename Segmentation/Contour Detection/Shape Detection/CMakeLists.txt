cmake_minimum_required(VERSION 3.10)
project(contour_detection_from_scratch)

find_package(OpenCV REQUIRED)

add_executable(contour_detection_from_scratch
    main.cpp
    contour_detector.cpp
)

target_include_directories(contour_detection_from_scratch PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(contour_detection_from_scratch PRIVATE ${OpenCV_LIBS})
