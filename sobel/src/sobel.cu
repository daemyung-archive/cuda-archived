//
// Created by djang on 2021-01-14.
//

#include "stb_image.h"
#include "stb_image_write.h"
#include <common/util.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

__global__ void rgb_to_gray(uint8_t *dst, uint8_t *src, int size, int c) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    while (tid < size) {
        auto offset = tid * c;
        dst[tid] = (src[offset + 0] + src[offset + 1] + src[offset + 2]) / 3;
        tid += stride;
    }
}

texture<uint8_t, 2> tex;
__global__ void sobel_filter(uint8_t *dst, int w, int h) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (w <= x || h <= y)
        return;

    int dx = 1 * tex2D(tex, x - 1, y - 1) + -1 * tex2D(tex, x + 1, y - 1) +
             2 * tex2D(tex, x - 1, y    ) + -2 * tex2D(tex, x + 1, y    ) +
             1 * tex2D(tex, x - 1, y + 1) + -1 * tex2D(tex, x + 1, y + 1);
    int dy = 1 * tex2D(tex, x - 1, y - 1) + -1 * tex2D(tex, x - 1, y + 1) +
             2 * tex2D(tex, x    , y - 1) + -2 * tex2D(tex, x    , y + 1) +
             1 * tex2D(tex, x + 1, y - 1) + -1 * tex2D(tex, x + 1, y + 1);

    int offset = x + y * blockDim.x * gridDim.x;
    dst[offset] = sqrt(static_cast<double>(dx * dx + dy * dy));
}

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    const std::string path(SOBEL_ASSET_DIR"/steam_engine.png");
    int x, y, c;
    uint8_t *img = stbi_load(path.c_str(), &x, &y, &c, 0);
    const auto img_size = x * y;
    const auto img_byte_size = img_size * c;
    const auto img_stride = y * c;

    uint8_t *src;
    CUDA_VALIDATE(cudaMalloc(&src, img_byte_size));
    CUDA_VALIDATE(cudaMemcpy(src, img, img_byte_size, cudaMemcpyHostToDevice));

    stbi_image_free(img);

    uint8_t *dst;
    CUDA_VALIDATE(cudaMalloc(&dst, img_size));

    rgb_to_gray<<<prop.maxGridSize[0], 64>>>(dst, src, img_size, c);

    src = dst;
    CUDA_VALIDATE(cudaBindTexture2D(nullptr, tex, src, x, y, x));
    CUDA_VALIDATE(cudaMalloc(&dst, img_size));

    dim3 threads(16, 16);
    dim3 blocks((x + threads.x - 1) / threads.x, (y + threads.y - 1) / threads.y);
    sobel_filter<<<blocks, threads>>>(dst, x, y);

    CUDA_VALIDATE(cudaDeviceSynchronize());

    uint8_t *tmp = new uint8_t[x * y];
    CUDA_VALIDATE(cudaMemcpy(tmp, dst, x * y, cudaMemcpyDeviceToHost));
    stbi_write_png("result.png", x, y, 1, tmp, x);
    delete[] tmp;

    CUDA_VALIDATE(cudaUnbindTexture(tex));
    CUDA_VALIDATE(cudaFree(src));
    CUDA_VALIDATE(cudaFree(dst));

    return EXIT_SUCCESS;
}
