//
// Created by djang on 2021-01-10.
//

#ifndef CUDA_RAY_H
#define CUDA_RAY_H

#include "vec3.h"

class ray {
public:
    __device__ ray() {}

    __device__ ray(const point3 &o, const vec3 &d) :
            origin(o), direction(d) {
    }

    __device__ inline point3 at(double t) const {
        return origin + t * direction;
    }

public:
    point3 origin;
    vec3 direction;
};

#endif //CUDA_RAY_H
