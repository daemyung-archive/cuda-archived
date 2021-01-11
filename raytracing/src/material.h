//
// Created by djang on 2021-01-11.
//

#ifndef CUDA_MATERIAL_H
#define CUDA_MATERIAL_H

#include "rtweekend.h"
#include "hittable.h"

class material {
public:
    __device__ virtual ~material() {}

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                                    curandState *rand_state) const = 0;
};

class lambertian : public material {
public:
    __device__ lambertian(const color &a) :
            albedo(a) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState *rand_state) const override {
        auto scattered_direction = rec.normal + random_unit_vector(rand_state);

        if (scattered_direction.near_zero())
            scattered_direction = rec.normal;

        scattered = ray(rec.p, scattered_direction);
        attenuation = albedo;
        return true;
    }

public:
    color albedo;
};

class metal : public material {
public:
    __device__ metal(const color &a, double f) :
            albedo(a), fuzz(f < 1 ? f : 1) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState *rand_state) const override {
        auto reflected = reflect(unit_vector(r_in.direction), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state));
        attenuation = albedo;
        return dot(scattered.direction, rec.normal) > 0;
    }

public:
    color albedo;
    double fuzz;
};

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

class dielectric : public material {
public:
    __device__ dielectric(double index_of_refraction) :
            ir(index_of_refraction) {
    }

    __device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,
                            curandState *rand_state) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction);
        double cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    __device__ static double reflectance(double cosine, double ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5.0);
    }

public:
    double ir;
};

#endif //CUDA_MATERIAL_H
