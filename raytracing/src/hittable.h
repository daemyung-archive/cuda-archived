//
// Created by djang on 2021-01-10.
//

#ifndef CUDA_HITTABLE_H
#define CUDA_HITTABLE_H

#include <cuda_runtime.h>
#include "ray.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    __device__ virtual ~hittable() {};
    __device__ virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) = 0;
};

#endif //CUDA_HITTABLE_H
