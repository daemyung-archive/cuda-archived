//
// Created by djang on 2021-01-10.
//

#ifndef CUDA_SPHERE_H
#define CUDA_SPHERE_H

#include "hittable.h"
#include "material.h"

class sphere : public hittable {
public:
    __device__ sphere() {}

    __device__ sphere(const point3 &c, double r, material *m) :
            center(c), radius(r), mat_ptr(m) {
    }

    __device__ ~sphere() {
        delete mat_ptr;
    }

    __device__ virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) override {

        vec3 oc = r.origin - center;
        double a = r.direction.length_squared();
        double half_b = dot(oc, r.direction);
        double c = oc.length_squared() - radius * radius;
        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        double root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root) {
                return false;
            }
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }

public:
    point3 center;
    double radius;
    material *mat_ptr;
};

#endif //CUDA_SPHERE_H
