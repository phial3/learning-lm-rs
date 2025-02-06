#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>


const int BLOCK_SIZE = 1024;

template <typename T>
__device__ void dot(
    const T* x,
    const T* y,
    T* result,
    int n
) {
    __shared__ T shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    T val = 0.0f;
    if (gid < n) {
        val = x[gid] * y[gid];
    }

    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared[0]);
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void dot_##RUST_NAME( \
        const TYPENAME* x, \
        const TYPENAME* y, \
        TYPENAME* result, \
        int n \
    ) { \
        dot<TYPENAME>(x, y, result, n); \
    } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)