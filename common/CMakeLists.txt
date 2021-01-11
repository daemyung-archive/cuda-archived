
add_library(common STATIC include/common/util.h src/util.cu)
target_include_directories(common PUBLIC include PRIVATE include/common src)
target_compile_features(common PUBLIC cuda_std_14)