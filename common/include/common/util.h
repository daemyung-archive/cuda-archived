//
// Created by djang on 2021-01-09.
//

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda.h>

#define CUDA_VALIDATE(err) cuda_validate(err, __FILE__, __LINE__)
extern void cuda_validate(cudaError_t err, const char* file, int line);

#endif //CUDA_UTIL_H
