//
// Created by djang on 2021-01-08.
//

#include <common/util.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "stb_image_write.h"

__global__ void vec3_to_uint8(uint8_t *dst, const vec3 *src, int w, int h, int s) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= w || y >= h)
        return;

    int i = x + y * blockDim.x * gridDim.x;
    const double scale = 1.0 / s;
    double r = sqrt(src[i].x() * scale);
    double g = sqrt(src[i].y() * scale);
    double b = sqrt(src[i].z() * scale);

    i = i * 3;
    dst[i + 0] = 256 * clamp(r, 0.0, 0.999);
    dst[i + 1] = 256 * clamp(g, 0.0, 0.999);
    dst[i + 2] = 256 * clamp(b, 0.0, 0.999);
}

void write_image(const vec3 *data, int w, int h, int s) {
    uint8_t *tmp;
    CUDA_VALIDATE(cudaMallocManaged(&tmp, w * h * sizeof(vec3)));

    dim3 blocks((w + 8 - 1) / 8, (h + 8 - 1) / 8);
    dim3 threads(8, 8);
    vec3_to_uint8<<<blocks, threads>>>(tmp, data, w, h, s);
    CUDA_VALIDATE(cudaDeviceSynchronize());

    stbi_flip_vertically_on_write(1);
    stbi_write_png("raytracing.png", w, h, 3, tmp, w * 3);
    stbi_flip_vertically_on_write(0);
    CUDA_VALIDATE(cudaFree(tmp));
}

__global__ void random_scene(hittable **world) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    curandState rand_state;
    curand_init(1984, 0, 0, &rand_state);

    hittable_list *list = new hittable_list();

    auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
    list->add(new sphere(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double(&rand_state);
            point3 center(a + 0.9 * random_double(&rand_state), 0.2, b + 0.9 * random_double(&rand_state));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material *sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random(&rand_state) * color::random(&rand_state);
                    sphere_material = new lambertian(albedo);
                    list->add(new sphere(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(&rand_state, 0.5, 1);
                    auto fuzz = random_double(&rand_state, 0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                    list->add(new sphere(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);
                    list->add(new sphere(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = new dielectric(1.5);
    list->add(new sphere(point3(0, 1, 0), 1.0, material1));

    auto material2 = new lambertian(color(0.4, 0.2, 0.1));
    list->add(new sphere(point3(-4, 1, 0), 1.0, material2));

    auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
    list->add(new sphere(point3(4, 1, 0), 1.0, material3));

    *world = list;
}

__global__ void deinit_world(hittable **world) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    delete *world;
}

__global__ void init_rand_state(curandState_t *rand_states, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= w || y >= h)
        return;

    int index = x + y * blockDim.x * gridDim.x;
    curand_init(1984, index, 0, &rand_states[index]);
}

__device__ color ray_color(const ray &r, hittable **world, curandState *rand_state, int depth) {
    ray cur_ray = r;
    color cur_attenuation(1, 1, 1);
    for (int d = 0; d != depth; ++d) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001, FLT_MAX, rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return {0, 0, 0};
            }
        } else {
            break;
        }
    }

    vec3 unit_direction = unit_vector(r.direction);
    double t = 0.5 * (unit_direction.y() + 1.0);
    vec3 c = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
    return cur_attenuation * c;
}

__global__ void render(vec3 *pixels, int w, int h, int s, int d,
                       camera cam, hittable **world, curandState_t *rand_states) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= w || y >= h)
        return;

    int index = x + y * blockDim.x * gridDim.x;
    curandState *rand_state = &rand_states[index];
    color pixel_color(0, 0, 0);
    for (int i = 0; i != s; ++i) {
        double u = (x + random_double(rand_state)) / (w - 1);
        double v = (y + random_double(rand_state)) / (h - 1);
        ray r = cam.get_ray(u, v, rand_state);
        pixel_color += ray_color(r, world, rand_state, d);
    }
    pixels[index] = pixel_color;
}

int main(int argc, char *argv[]) {
    // image
    const auto aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int sample_per_pixels = 500;
    const int max_depth = 50;

    // world
    hittable **world;
    CUDA_VALIDATE(cudaMalloc(&world, sizeof(hittable *)));
    random_scene<<<1, 1>>>(world);

    // camera
    point3 from(13, 2, 3);
    point3 at(0, 0, 0);
    vec3 up(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;

    camera cam(from, at, up, 20, aspect_ratio, aperture, dist_to_focus);

    int num_pixels = image_width * image_height;

    curandState_t *rand_states;
    CUDA_VALIDATE(cudaMalloc(&rand_states, num_pixels * sizeof(curandState_t)));

    vec3 *pixels;
    CUDA_VALIDATE(cudaMallocManaged(&pixels, num_pixels * sizeof(vec3)));

    cudaEvent_t start, end;
    CUDA_VALIDATE(cudaEventCreate(&start));
    CUDA_VALIDATE(cudaEventCreate(&end));
    CUDA_VALIDATE(cudaEventRecord(start, 0));

    // render
    std::cerr << "Start\n";
    int tx = 8, ty = 8;
    dim3 blocks((image_width + tx - 1) / tx, (image_height + ty - 1) / ty);
    dim3 threads(tx, ty);
    init_rand_state<<<blocks, threads>>>(rand_states, image_width, image_height);
    render<<<blocks, threads>>>(pixels, image_width, image_height, sample_per_pixels, max_depth,
                                cam, world, rand_states);
    CUDA_VALIDATE(cudaEventRecord(end, 0));
    CUDA_VALIDATE(cudaEventSynchronize(end));

    float elapsed_time;
    CUDA_VALIDATE(cudaEventElapsedTime(&elapsed_time, start, end));
    std::cerr << "End: " << elapsed_time << " ms\n";

    deinit_world<<<1, 1>>>(world);
    CUDA_VALIDATE(cudaDeviceSynchronize());
    CUDA_VALIDATE(cudaFree(world));

    CUDA_VALIDATE(cudaEventDestroy(start));
    CUDA_VALIDATE(cudaEventDestroy(end));

    write_image(pixels, image_width, image_height, sample_per_pixels);
    CUDA_VALIDATE(cudaFree(pixels));
    // https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/

    return EXIT_SUCCESS;
}
