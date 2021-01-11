//
// Created by djang on 2021-01-11.
//

#ifndef CUDA_RTWEEKEND_H
#define CUDA_RTWEEKEND_H

#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// constants
constexpr double infinity = std::numeric_limits<float>::max();
constexpr double pi = 3.1415926535897932385;

// utilities
__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ inline double random_double(curandState* rand_state) {
    return curand_uniform_double(rand_state);
}

__device__ inline double random_double(curandState* rand_state, double min, double max) {
    return min + (max - min) * random_double(rand_state);
}

#endif //CUDA_RTWEEKEND_H
