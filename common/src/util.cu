//
// Created by djang on 2021-01-09.
//

#include "util.h"

#include <cuda_runtime.h>
#include <iostream>

void cuda_validate(cudaError_t err, const char *file, int line) {
    if (err) {
        std::cerr << "CUDA: " << file << "(" << line << "): "
                  << "error: " << cudaGetErrorString(err) << " : return code '0x" << std::hex << err << std::dec
                  << "'\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
