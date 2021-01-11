//
// Created by djang on 2021-01-11.
//

#ifndef CUDA_HITTABLE_LIST_H
#define CUDA_HITTABLE_LIST_H

#include <vector>
#include <memory>
#include "hittable.h"

class hittable_list : public hittable {
public:
    __device__ hittable_list() :
            objects(new hittable *[16]), size(0), capacity(16) {
    };

    __device__ ~hittable_list() {
        clear();
    }

    __device__ hittable_list(hittable *object) :
            objects(new hittable *[16]), size(0), capacity(16) {
        add(object);
    }

    __device__ void clear() {
        for (int i = 0; i != size; ++i) {
            delete objects[i];
        }
        size = 0;
    }

    __device__ void add(hittable *object) {
        if (size + 1 >= capacity) {
            capacity *= 2;
            hittable **tmp = new hittable *[capacity];
            memcpy(tmp, objects, size * sizeof(hittable *));
            delete[] objects;
            objects = tmp;
        }

        objects[size++] = object;
    }

    __device__ virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) override {
        hit_record tmp_rec;
        bool hit_anything = false;
        auto closest_so_far = t_max;

        for (int i = 0; i != size; ++i) {
            if (objects[i]->hit(r, t_min, closest_so_far, tmp_rec)) {
                hit_anything = true;
                closest_so_far = tmp_rec.t;
                rec = tmp_rec;
            }
        }

        return hit_anything;
    }

public:
    hittable **objects;
    int size;
    int capacity;
};

#endif //CUDA_HITTABLE_LIST_H
