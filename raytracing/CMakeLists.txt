add_executable(raytracing src/raytracing.cu src/vec3.h src/ray.h src/hittable.h src/sphere.h src/hittable_list.h src/rtweekend.h src/camera.h src/material.h)
target_compile_features(raytracing PUBLIC cxx_std_20)
target_compile_definitions(raytracing PRIVATE STB_IMAGE_WRITE_IMPLEMENTATION)
target_link_libraries(raytracing common)