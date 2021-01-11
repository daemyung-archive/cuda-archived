//
// Created by djang on 2021-01-11.
//

#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include "rtweekend.h"

class camera {
public:
    __host__ __device__ camera(point3 from, point3 at, vec3 up,
                               double vfov, double aspect_ratio, double aperture, double focus_dist) :
            origin(from) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = unit_vector(from - at);
        u = unit_vector(cross(up, w));
        v = cross(w, u);

        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        lens_radius = aperture / 2;
    }

    __device__ inline ray get_ray(double s, double t, curandState *rand_state) const {
        vec3 rd = lens_radius * random_in_unit_disk(rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset,
                   lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

private:
    point3 origin;
    point3 lower_left_corner;
    vec3 vertical;
    vec3 horizontal;
    vec3 u, v, w;
    double lens_radius;
};

#endif //CUDA_CAMERA_H
