add_executable(sobel src/sobel.cu src/stb_image.h)
target_compile_definitions(sobel
    PRIVATE
        SOBEL_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/asset"
        STB_IMAGE_IMPLEMENTATION
        STB_IMAGE_WRITE_IMPLEMENTATION)
target_link_libraries(sobel common)