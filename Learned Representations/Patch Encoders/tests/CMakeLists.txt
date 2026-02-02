add_executable(encoder_shapes
    encoder_shapes.cpp
    ../src/CNN_Encoder.cpp
)

add_executable(encoder_device
    encoder_device.cpp
    ../src/CNN_Encoder.cpp
)

target_include_directories(encoder_shapes PRIVATE
    ../include
)

target_include_directories(encoder_device PRIVATE
    ../include
)

target_link_libraries(encoder_shapes
    ${TORCH_LIBRARIES}
)

target_link_libraries(encoder_device
    ${TORCH_LIBRARIES}
)

target_compile_options(encoder_shapes PRIVATE ${TORCH_CXX_FLAGS})
target_compile_options(encoder_device PRIVATE ${TORCH_CXX_FLAGS})

set_target_properties(encoder_shapes PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
)

set_target_properties(encoder_device PROPERTIES
    BUILD_RPATH "/opt/libtorch/lib"
)
