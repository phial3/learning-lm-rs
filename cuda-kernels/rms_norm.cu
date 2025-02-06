#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

using bf16 =  __nv_bfloat16;
using f16 = __half;

const int BLOCK_SIZE = 1024;

template<typename T>
__device__ void rms_norm(
    T* y,
    const T* x,
    const T* w,
    T epsilon,
    int m,
    int n
) {
    __shared__ T shared[BLOCK_SIZE];

    int i = blockIdx.x;
    if (i >= m) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    T sum_val = 0.0f;
    for (int j = tid; j < n; j += stride) {
        T val = x[i * n + j];
        sum_val += val * val;
    }

    shared[tid] = sum_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    T sum_squares = shared[0];
    T rms = sqrtf(sum_squares / T(n) + epsilon);

    for (int j = tid; j < n; j += stride) {
        T normalized = x[i * n + j] / rms;
        y[i * n + j] = normalized * w[j];
    }
}

#define OPS(TYPENAME, RUST_NAME) \
    extern "C" __global__ void rms_norm_##RUST_NAME( \
        TYPENAME* y, \
        const TYPENAME* x, \
        const TYPENAME* w, \
        TYPENAME epsilon, \
        int m, \
        int n \
    ) { \
        rms_norm<TYPENAME>(y, x, w, epsilon, m, n); \
    } \


#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)